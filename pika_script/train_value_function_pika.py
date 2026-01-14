#!/usr/bin/env python3
"""
基于 SigLIP + Gemma 3 270M 的 Value Function 训练脚本 (Pika HDF5 数据版本)

用法:
    # 划分数据集
    python train_value_function_pika.py --mode split

    # 训练
    python train_value_function_pika.py --mode train

    # 断点续训（从 last.pt 或 best_model.pt 或 checkpoint_epoch_xx.pt）
    python scripts/train_value_function_pika.py --mode train --resume ./checkpoints/value_pika/run_20260113_152257/last.pt 

    # TensorBoard
    tensorboard --logdir ./checkpoints/value_pika/run_xxx/tb
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

import matplotlib
matplotlib.use('Agg')
import random
import argparse
import h5py
import cv2
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from torch.utils.tensorboard import SummaryWriter

# ===== 新增：用 PIL + SiglipImageProcessor 做正规预处理 =====
from PIL import Image

# 导入 transformers（必需）
from transformers import (
    SiglipVisionModel,
    SiglipImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    GemmaForCausalLM,
)

# =========================
# 字体（保持你原来的）
# =========================
chinese_font_candidates = ['WenQuanYi Micro Hei', 'SimHei', 'Noto Sans CJK SC',
                          'Source Han Sans CN', 'Microsoft YaHei', 'STHeiti']
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_font = None

for font_name in chinese_font_candidates:
    if font_name in available_fonts:
        chinese_font = font_name
        break

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'sans-serif']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False

# =========================
# 断点续训：工具函数
# =========================
def save_checkpoint_atomic(state: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

# ============================================
# 配置（保持你原来的）
# ============================================
NUM_BINS = 200
VALUE_MIN = -1.0
VALUE_MAX = 0.0

PIKA_TASK_PROMPT = "Take out the bread and hand it over to the plate"

CAMERA_CONFIGS = {
    "fisheye": {
        "left": "pikaFisheyeCamera_l",
        "right": "pikaFisheyeCamera_r",
        "description": "FisheyeCamera",
    },
    "depth": {
        "left": "pikaDepthCamera_l",
        "right": "pikaDepthCamera_r",
        "description": "DepthCamera",
    },
}

def get_task_max_len(data_root: str) -> int:
    task_dir = Path(data_root)
    max_len = 0
    for ep_dir in task_dir.glob("episode*"):
        hdf5_path = ep_dir / "data.hdf5"
        if not hdf5_path.exists():
            continue
        try:
            with h5py.File(hdf5_path, "r") as f:
                L = int(f["size"][()])
            if L > max_len:
                max_len = L
        except Exception:
            continue
    return max_len

# ============================================
# Value 离散化工具函数（保持你原来的）
# ============================================
def value_to_bin(value: float) -> int:
    value = np.clip(value, VALUE_MIN, VALUE_MAX)
    bin_idx = int((value - VALUE_MIN) / (VALUE_MAX - VALUE_MIN) * (NUM_BINS - 1))
    return np.clip(bin_idx, 0, NUM_BINS - 1)

def bin_to_value(bin_idx: int) -> float:
    return VALUE_MIN + (bin_idx / (NUM_BINS - 1)) * (VALUE_MAX - VALUE_MIN)

def bins_to_value_soft(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    bin_values = torch.linspace(VALUE_MIN, VALUE_MAX, NUM_BINS, device=logits.device)
    return torch.sum(probs * bin_values, dim=-1)

# ============================================
# 数据集（只改“送入 SigLIP 的图像预处理”）
# ============================================
class PikaHDF5Dataset(Dataset):
    def __init__(
        self,
        episode_dirs: List[str],
        image_size: int = 224,
        prompt: str = PIKA_TASK_PROMPT,
        camera_type: str = "fisheye",
        data_dir: str = "",
        image_processor: Optional[SiglipImageProcessor] = None,  # 新增：SigLIP processor
    ):
        self.data_dir = data_dir
        self.episode_dirs = episode_dirs
        self.image_size = image_size
        self.prompt = prompt
        self.camera_type = camera_type
        self.samples: List[Dict] = []

        # 新增：SigLIP 正规预处理器（resize/normalize 等由它负责）
        if image_processor is None:
            raise ValueError("image_processor 不能为空，请在构造 Dataset 时传入 model.image_processor")
        self.image_processor = image_processor

        if camera_type not in CAMERA_CONFIGS:
            raise ValueError(f"不支持的相机类型: {camera_type}，支持的类型: {list(CAMERA_CONFIGS.keys())}")

        self.left_camera = CAMERA_CONFIGS[camera_type]["left"]
        self.right_camera = CAMERA_CONFIGS[camera_type]["right"]

        print(f"使用相机类型: {camera_type} ({CAMERA_CONFIGS[camera_type]['description']})")
        print(f"  左相机: {self.left_camera}")
        print(f"  右相机: {self.right_camera}")

        self._load_data()
        print(f"加载了 {len(self.samples)} 个样本，来自 {len(episode_dirs)} 个 episode")

    def _load_data(self):
        for episode_dir in tqdm(self.episode_dirs, desc="加载 episodes"):
            try:
                hdf5_path = os.path.join(episode_dir, "data.hdf5")
                if not os.path.exists(hdf5_path):
                    print(f"警告: {hdf5_path} 不存在，跳过")
                    continue

                with h5py.File(hdf5_path, 'r') as f:
                    episode_len = int(f['size'][()])
                    if episode_len < 2:
                        continue

                    left_cam_key = f'camera/color/{self.left_camera}'
                    right_cam_key = f'camera/color/{self.right_camera}'

                    if left_cam_key not in f or right_cam_key not in f:
                        print(f"警告: {episode_dir} 中缺少相机数据，跳过")
                        continue

                    left_cam_paths = [f[left_cam_key][i].decode('utf-8') for i in range(episode_len)]
                    right_cam_paths = [f[right_cam_key][i].decode('utf-8') for i in range(episode_len)]

                # 你这里默认全成功
                is_success = True

                # --- 关键改动：按任务最大长度归一化（保持你现在的写法） ---
                T = episode_len - 1
                Lmax = int(get_task_max_len(self.data_dir))
                Lmax = max(1, Lmax)
                Cfail = 50

                normalized_values = []
                for t in range(episode_len):
                    remaining = (T - t)
                    ret = -remaining
                    if (t == T) and (not is_success):
                        ret -= Cfail
                    v = ret / Lmax
                    v = max(-1.0, min(0.0, v))
                    normalized_values.append(v)

                for t in range(episode_len):
                    value_bin = value_to_bin(normalized_values[t])
                    left_img_path = os.path.join(episode_dir, left_cam_paths[t])
                    right_img_path = os.path.join(episode_dir, right_cam_paths[t])

                    self.samples.append({
                        'left_image_path': left_img_path,
                        'right_image_path': right_img_path,
                        'prompt': self.prompt,
                        'value_target': normalized_values[t],
                        'value_bin': value_bin,
                        'is_success': is_success,
                        'episode_dir': episode_dir,
                        'frame_idx': t,
                    })

            except Exception as e:
                print(f"加载 {episode_dir} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    def __len__(self) -> int:
        return len(self.samples)

    # ========= 关键：正规 SigLIP 图像预处理 =========
    def _load_image(self, img_path: str) -> torch.Tensor:
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
        except Exception as e:
            raise ValueError(f"无法加载图像: {img_path}, err={e}")

        # processor 负责 resize/crop/normalize，输出 pixel_values: (1,3,H,W)
        # 这里 return 单张: (3,H,W)
        out = self.image_processor(images=im, return_tensors="pt")
        pixel_values = out["pixel_values"][0]  # (3,H,W)
        return pixel_values

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        left_tensor = self._load_image(sample['left_image_path'])
        right_tensor = self._load_image(sample['right_image_path'])

        return {
            'image': left_tensor,
            'wrist_image': right_tensor,
            'prompt': sample['prompt'],
            'value_target': torch.tensor(sample['value_target'], dtype=torch.float32),
            'value_bin': torch.tensor(sample['value_bin'], dtype=torch.long),
        }

# ============================================
# 模型（保持你原来的）
# ============================================
SIGLIP_MODELS = {
    "so400m": "google/siglip-so400m-patch14-384",
}
GEMMA_MODELS = {
    "gemma3-270m": "google/gemma-3-270m-it",
}

class SigLIPGemmaValueFunction(nn.Module):
    def __init__(
        self,
        num_bins: int = NUM_BINS,
        siglip_variant: str = "so400m",
        gemma_variant: str = "gemma3-270m",
        freeze_vision: bool = False,
        freeze_llm: bool = False,
        hidden_dim: int = 512,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim

        print("=" * 60)
        print("初始化 SigLIP + Gemma Value Function")
        print("输入: 左右两张图像 + prompt")
        print("=" * 60)

        self._init_vision_encoder(siglip_variant, freeze_vision)
        self._init_language_model(gemma_variant, freeze_llm, use_flash_attention)

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        fusion_input_dim = 3 * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_bins),
        )

        self._print_param_stats()

    def _init_vision_encoder(self, variant: str, freeze: bool):
        model_name = SIGLIP_MODELS.get(variant, SIGLIP_MODELS["so400m"])
        print(f"加载 SigLIP: {model_name}")
        self.vision_encoder = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.vision_dim = self.vision_encoder.config.hidden_size
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
        print(f"  - 隐藏维度: {self.vision_dim}")
        print(f"  - Patch size: {self.vision_encoder.config.patch_size}")

        if freeze:
            print("  - 冻结视觉编码器参数")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def _init_language_model(self, variant: str, freeze: bool, use_flash_attention: bool):
        model_name = GEMMA_MODELS.get(variant, GEMMA_MODELS["gemma3-270m"])
        print(f"加载 Gemma: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        self.text_dim = self.language_model.config.hidden_size
        print(f"  - 隐藏维度: {self.text_dim}")
        print(f"  - 层数: {self.language_model.config.num_hidden_layers}")

        if freeze:
            print("  - 冻结语言模型参数")
            for param in self.language_model.parameters():
                param.requires_grad = False

    def _print_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 60)
        print(f"总参数量: {total_params / 1e6:.1f}M")
        print(f"可训练参数量: {trainable_params / 1e6:.1f}M")
        print("=" * 60)

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(images)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        return features

    def _encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(device)

        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.language_model.parameters())):
            outputs = self.language_model(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        features = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return features

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor, prompts: List[str]):
        device = left_image.device

        left_features = self._encode_images(left_image)
        right_features = self._encode_images(right_image)

        left_proj = self.vision_proj(left_features)
        right_proj = self.vision_proj(right_features)

        text_features = self._encode_text(prompts, device)
        text_proj = self.text_proj(text_features)

        combined = torch.cat([left_proj, right_proj, text_proj], dim=-1)
        fused = self.fusion(combined)

        logits = self.value_head(fused)
        value = bins_to_value_soft(logits)
        return logits, value

PaliGemmaValueFunction = SigLIPGemmaValueFunction

# ============================================
# 数据集划分（保持你原来的）
# ============================================
def find_all_episodes(data_dir: str) -> List[str]:
    data_path = Path(data_dir)
    episode_dirs = []
    for d in sorted(data_path.iterdir()):
        if d.is_dir() and d.name.startswith("episode"):
            hdf5_path = d / "data.hdf5"
            if hdf5_path.exists():
                episode_dirs.append(str(d))
    return episode_dirs

def check_dataset_split(data_dir: str) -> Tuple[bool, Optional[Dict]]:
    data_path = Path(data_dir)
    split_file = data_path / "split_info.txt"
    if not split_file.exists():
        return False, None

    split_info = {"train": [], "test": []}
    current_split = None

    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line in ["[train]", "[test]"]:
                current_split = line[1:-1]
            elif line and current_split and current_split in split_info:
                episode_path = data_path / line
                if episode_path.exists():
                    split_info[current_split].append(str(episode_path))

    if len(split_info["train"]) == 0:
        return False, None

    print(f"数据集已划分: train={len(split_info['train'])}, test={len(split_info['test'])}")
    return True, split_info

def split_dataset_episodes(
    data_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
    force: bool = False,
) -> Dict[str, List[str]]:
    data_path = Path(data_dir)

    is_split, split_info = check_dataset_split(data_dir)
    if not force and is_split:
        print("数据集已经划分好，使用现有划分")
        return split_info

    episode_dirs = find_all_episodes(data_dir)
    if not episode_dirs:
        raise ValueError(f"在 {data_path} 中未找到 episode 目录")

    print(f"找到 {len(episode_dirs)} 个 episode")
    random.seed(seed)
    random.shuffle(episode_dirs)

    n = len(episode_dirs)
    n_train = int(n * train_ratio)

    split_info = {
        "train": episode_dirs[:n_train],
        "test": episode_dirs[n_train:],
    }

    split_file = data_path / "split_info.txt"
    with open(split_file, 'w') as f:
        for split_name in ["train", "test"]:
            f.write(f"[{split_name}]\n")
            for ep_dir in split_info[split_name]:
                ep_name = Path(ep_dir).name
                f.write(f"{ep_name}\n")
            f.write("\n")

    print(f"划分结果: train={len(split_info['train'])} (90%), test={len(split_info['test'])} (10%)")
    print(f"划分信息已保存到: {split_file}")
    return split_info

# ============================================
# 训练（仅在构造 Dataset 时注入 processor；其它不动）
# ============================================
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ========== 输出目录：resume 则沿用原 run 目录 ==========
    if args.resume:
        output_dir = Path(args.resume).resolve().parent
        print(f"[Resume] Continue in: {output_dir}")
    else:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path(args.output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # ========== TB ==========
    log_dir = output_dir / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = None  # 等拿到 global_step 再创建

    # ========== 数据划分 ==========
    is_split, split_info = check_dataset_split(args.data_dir)
    if not is_split:
        print("数据集未划分，自动进行划分...")
        split_info = split_dataset_episodes(args.data_dir, seed=args.seed)

    train_episodes = split_info["train"]
    test_episodes = split_info["test"]

    if args.max_episodes:
        train_episodes = train_episodes[:args.max_episodes]

    # ========== 模型 ==========
    print("创建 SigLIP + Gemma Value Function...")
    model = SigLIPGemmaValueFunction(
        num_bins=NUM_BINS,
        siglip_variant=args.siglip_variant,
        gemma_variant=args.gemma_variant,
        freeze_vision=args.freeze_vision,
        freeze_llm=args.freeze_llm,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # ===== 关键：把 SigLIP 的 processor 注入 Dataset（其余不动）=====
    train_dataset = PikaHDF5Dataset(
        episode_dirs=train_episodes,
        image_size=args.image_size,
        prompt=PIKA_TASK_PROMPT,
        camera_type=args.camera_type,
        data_dir=args.data_dir,
        image_processor=model.image_processor,   # 新增
    )

    test_dataset = PikaHDF5Dataset(
        episode_dirs=test_episodes,
        image_size=args.image_size,
        prompt=PIKA_TASK_PROMPT,
        camera_type=args.camera_type,
        data_dir=args.data_dir,
        image_processor=model.image_processor,   # 新增
    )

    def collate_fn(batch):
        return {
            'image': torch.stack([b['image'] for b in batch]),
            'wrist_image': torch.stack([b['wrist_image'] for b in batch]),
            'prompt': [b['prompt'] for b in batch],
            'value_target': torch.stack([b['value_target'] for b in batch]),
            'value_bin': torch.stack([b['value_bin'] for b in batch]),
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ========== Resume：恢复训练状态 ==========
    start_epoch = 0
    global_step = 0
    best_val_mae = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))

        print(f"[Resume] Loaded: {args.resume}")
        print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}, best_val_mae={best_val_mae:.6f}")

    # TB：purge_step 确保续训曲线不乱
    writer = SummaryWriter(log_dir=str(log_dir), purge_step=global_step)
    print(f"[TensorBoard] logdir: {log_dir} (purge_step={global_step})")

    train_losses = []

    # ========== 训练循环 ==========
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            images = batch['image'].to(device, non_blocking=True)
            wrist_images = batch['wrist_image'].to(device, non_blocking=True)
            prompts = batch['prompt']
            value_bins = batch['value_bin'].to(device, non_blocking=True)

            if scaler:
                with torch.cuda.amp.autocast():
                    logits, _ = model(images, wrist_images, prompts)
                    loss = criterion(logits, value_bins)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(images, wrist_images, prompts)
                loss = criterion(logits, value_bins)
                loss.backward()
                optimizer.step()

            pred_bins = logits.argmax(dim=-1)
            acc = (pred_bins == value_bins).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # ===== TensorBoard：step 级 =====
            if args.tb_log_interval > 0 and (global_step % args.tb_log_interval == 0):
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/acc_step", acc.item(), global_step)
                writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)

            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.4f}'})

        scheduler.step()

        train_loss = epoch_loss / max(1, len(train_loader))
        train_acc = epoch_acc / max(1, len(train_loader))
        train_losses.append(train_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

        # =========================
        # 验证（你原来的 test_loader 当 val）
        # =========================
        model.eval()
        val_ce = 0.0
        val_top1 = 0.0
        val_mae = 0.0
        val_huber = 0.0
        val_acc1 = 0.0
        val_acc2 = 0.0
        val_entropy = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="验证", leave=False):
                images = batch["image"].to(device, non_blocking=True)
                wrist_images = batch["wrist_image"].to(device, non_blocking=True)
                prompts = batch["prompt"]
                value_bins = batch["value_bin"].to(device, non_blocking=True)
                value_targets = batch["value_target"].to(device, non_blocking=True)

                logits, pred_values = model(images, wrist_images, prompts)

                loss_ce = criterion(logits, value_bins)

                pred_bins = logits.argmax(dim=-1)
                diff = (pred_bins - value_bins).abs()

                top1 = (diff == 0).float().mean()
                acc1 = (diff <= 1).float().mean()
                acc2 = (diff <= 2).float().mean()

                mae = (pred_values - value_targets).abs().mean()
                huber = F.smooth_l1_loss(pred_values, value_targets)

                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

                val_ce += loss_ce.item()
                val_top1 += top1.item()
                val_acc1 += acc1.item()
                val_acc2 += acc2.item()
                val_mae += mae.item()
                val_huber += huber.item()
                val_entropy += entropy.item()

        denom = max(1, len(test_loader))
        val_ce /= denom
        val_top1 /= denom
        val_acc1 /= denom
        val_acc2 /= denom
        val_mae /= denom
        val_huber /= denom
        val_entropy /= denom

        print(
            f"Epoch {epoch+1}: "
            f"Train CE={train_loss:.4f}, Train Top1={train_acc:.4f}, "
            f"Val CE={val_ce:.4f}, Val Top1={val_top1:.4f}, "
            f"Val Acc@1={val_acc1:.4f}, Val Acc@2={val_acc2:.4f}, "
            f"Val MAE={val_mae:.4f}, Val Huber={val_huber:.4f}, "
            f"Val Entropy={val_entropy:.4f}"
        )

        # ===== TensorBoard：epoch 级 =====
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/acc_epoch", train_acc, epoch)
        writer.add_scalar("val/ce_epoch", val_ce, epoch)
        writer.add_scalar("val/top1_epoch", val_top1, epoch)
        writer.add_scalar("val/acc1_epoch", val_acc1, epoch)
        writer.add_scalar("val/acc2_epoch", val_acc2, epoch)
        writer.add_scalar("val/mae_epoch", val_mae, epoch)
        writer.add_scalar("val/huber_epoch", val_huber, epoch)
        writer.add_scalar("val/entropy_epoch", val_entropy, epoch)

        # ===== best_model.pt（按 MAE）=====
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_path = output_dir / "best_model.pt"
            save_checkpoint_atomic({
                "epoch": epoch,
                "global_step": global_step,
                "best_val_mae": best_val_mae,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "camera_type": args.camera_type,
            }, best_path)
            print(f"  保存最佳模型 (Val MAE: {val_mae:.4f}) -> {best_path}")

        # ===== checkpoint_epoch_xx.pt（保持你原来的每 5 个 epoch）=====
        if (epoch + 1) % 5 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint_atomic({
                "epoch": epoch,
                "global_step": global_step,
                "best_val_mae": best_val_mae,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "camera_type": args.camera_type,
            }, ckpt_path)
            print(f"[Checkpoint] saved: {ckpt_path}")

        # ===== last.pt：每个 epoch 都覆盖保存，用于 resume =====
        last_path = output_dir / "last.pt"
        save_checkpoint_atomic({
            "epoch": epoch,
            "global_step": global_step,
            "best_val_mae": best_val_mae,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "camera_type": args.camera_type,
        }, last_path)

    # ===== final_model.pt（保留你原来的）=====
    final_path = output_dir / "final_model.pt"
    save_checkpoint_atomic({
        'epoch': args.num_epochs - 1,
        'global_step': global_step,
        'best_val_mae': best_val_mae,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'train_loss': train_losses[-1] if train_losses else None,
        'camera_type': args.camera_type,
    }, final_path)

    # 绘制损失曲线（保留你原来的）
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    writer.flush()
    writer.close()

    print(f"\n训练完成！best_val_mae={best_val_mae:.6f}")
    print(f"模型保存至: {output_dir}")

# ============================================
# 评估（只改“送入 SigLIP 的图像预处理”，其它不动）
# ============================================
def evaluate(args):
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    camera_type = checkpoint.get('camera_type', args.camera_type)
    print(f"使用相机类型: {camera_type}")

    if camera_type not in CAMERA_CONFIGS:
        raise ValueError(f"不支持的相机类型: {camera_type}")

    left_camera = CAMERA_CONFIGS[camera_type]["left"]
    right_camera = CAMERA_CONFIGS[camera_type]["right"]

    model = SigLIPGemmaValueFunction(
        num_bins=NUM_BINS,
        siglip_variant=args.siglip_variant,
        gemma_variant=args.gemma_variant,
        freeze_vision=True,
        freeze_llm=True,
        hidden_dim=args.hidden_dim,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    episode_dir = Path(args.episode_path)
    hdf5_path = episode_dir / "data.hdf5"
    if not hdf5_path.exists():
        raise ValueError(f"HDF5 文件不存在: {hdf5_path}")

    print(f"加载轨迹: {episode_dir}")

    with h5py.File(hdf5_path, 'r') as f:
        episode_len = int(f['size'][()])
        left_cam_paths = [f[f'camera/color/{left_camera}'][i].decode('utf-8') for i in range(episode_len)]
        right_cam_paths = [f[f'camera/color/{right_camera}'][i].decode('utf-8') for i in range(episode_len)]

    is_success = True
    prompt = PIKA_TASK_PROMPT

    C_FAIL = 50.0
    T_task = get_task_max_len(args.data_dir)
    denom = max(1, T_task - 1)
    T = episode_len - 1

    true_values = []
    for t in range(episode_len):
        remaining = T - t
        v = -float(remaining)
        if (t == T) and (not is_success):
            v -= float(C_FAIL)
        v = v / float(denom)
        v = max(-1.0, min(0.0, v))
        true_values.append(v)
    true_values = np.asarray(true_values, dtype=np.float32)

    print(f"轨迹结果: {'成功' if is_success else '失败'}")
    print(f"轨迹长度: {episode_len}")
    print(f"任务描述: {prompt}")

    pred_values = []
    pred_bins = []
    left_images = []
    right_images = []

    with torch.no_grad():
        for t in tqdm(range(episode_len), desc="预测"):
            left_img_path = str(episode_dir / left_cam_paths[t])
            left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
            if left_img is None:
                print(f"警告: 无法加载图像 {left_img_path}")
                continue
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_images.append(left_img.copy())

            # ===== 关键改动：用 SigLIP processor 生成 pixel_values =====
            with Image.open(left_img_path) as im:
                im = im.convert("RGB")
            left_tensor = model.image_processor(images=im, return_tensors="pt")["pixel_values"].to(device)  # (1,3,H,W)

            right_img_path = str(episode_dir / right_cam_paths[t])
            right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
            if right_img is None:
                print(f"警告: 无法加载图像 {right_img_path}")
                continue
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_images.append(right_img.copy())

            # ===== 关键改动：用 SigLIP processor 生成 pixel_values =====
            with Image.open(right_img_path) as im:
                im = im.convert("RGB")
            right_tensor = model.image_processor(images=im, return_tensors="pt")["pixel_values"].to(device)  # (1,3,H,W)

            logits, value = model(left_tensor, right_tensor, [prompt])
            pred_values.append(value.item())
            pred_bins.append(logits.argmax(dim=-1).item())

    pred_values = np.array(pred_values)
    true_values = np.array(true_values[:len(pred_values)])
    true_bins = np.array([value_to_bin(v) for v in true_values])
    pred_bins = np.array(pred_bins)

    mae = np.mean(np.abs(pred_values - true_values))
    acc = np.mean(pred_bins == true_bins)
    corr = np.corrcoef(pred_values, true_values)[0, 1] if len(pred_values) > 1 else 0.0

    print(f"\n评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Bin Accuracy: {acc:.4f}")
    print(f"  Correlation: {corr:.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    frames = np.arange(len(pred_values))
    ax.plot(frames, pred_values, 'r-', label='Predicted Value', linewidth=2)
    ax.fill_between(frames, pred_values, -1, alpha=0.2, color='red')
    ax.plot(frames, true_values, 'g--', label='True Value', linewidth=2)
    ax.fill_between(frames, true_values, -1, alpha=0.2, color='green')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Value')
    ax.set_ylim(-1.1, 0.1)
    status_text = "Success" if is_success else "Failure"
    ax.set_title(f'Value Function Prediction | Status: {status_text}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(args.checkpoint).parent / f"eval_{episode_dir.name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"评估图保存至: {output_path}")

    if args.save_video:
        print("\n生成评估视频...")
        video_path = Path(args.checkpoint).parent / f"eval_{episode_dir.name}.mp4"
        actual_len = len(pred_values)

        camera_desc = CAMERA_CONFIGS[camera_type]["description"]
        left_title = f'Left Camera ({camera_desc})'
        right_title = f'Right Camera ({camera_desc})'

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.5], height_ratios=[1, 1])

        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[1, 0])
        ax_value = fig.add_subplot(gs[:, 1])

        im_left = ax_left.imshow(left_images[0])
        ax_left.set_title(left_title, fontsize=12, fontweight='bold')
        ax_left.axis('off')

        im_right = ax_right.imshow(right_images[0])
        ax_right.set_title(right_title, fontsize=12, fontweight='bold')
        ax_right.axis('off')

        ax_value.set_xlim(0, actual_len - 1)
        ax_value.set_ylim(-1.1, 0.1)
        ax_value.set_xlabel('Frame', fontsize=11)
        ax_value.set_ylabel('Value', fontsize=11)
        ax_value.grid(True, alpha=0.3)

        line_pred, = ax_value.plot([], [], 'r-', label='Predicted Value', linewidth=2)
        vline = ax_value.axvline(x=0, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        scatter_pred = ax_value.scatter([], [], c='red', s=100, zorder=5, edgecolors='white', linewidths=2)

        ax_value.legend(loc='lower right', fontsize=10)

        title = fig.suptitle(
            f'Task: {prompt}\nFrame: 0/{actual_len-1} | Status: {status_text}',
            fontsize=11, fontweight='bold'
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        def update(frame):
            im_left.set_array(left_images[frame])
            im_right.set_array(right_images[frame])

            line_pred.set_data(frames[:frame+1], pred_values[:frame+1])
            vline.set_xdata([frame, frame])
            scatter_pred.set_offsets([[frame, pred_values[frame]]])

            title.set_text(f'Task: {prompt}\nFrame: {frame}/{actual_len-1} | Status: {status_text}')
            return im_left, im_right, line_pred, vline, scatter_pred, title

        anim = FuncAnimation(fig, update, frames=actual_len, interval=50, blit=False)
        anim.save(str(video_path), writer='ffmpeg', fps=20, dpi=100, bitrate=2000)
        plt.close()
        print(f"视频保存至: {video_path}")

# ============================================
# 主函数（只加 resume + tb_log_interval）
# ============================================
def main():
    parser = argparse.ArgumentParser(description="SigLIP + Gemma Value Function Training (Pika HDF5)")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["split", "train", "eval"],
                        help="运行模式")
    parser.add_argument("--data_dir", type=str, default="./data/task3_handover_bread",
                        help="数据目录（包含 episode* 子目录）")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/value_pika",
                        help="输出目录")

    # 模型配置
    parser.add_argument("--siglip_variant", type=str, default="so400m")
    parser.add_argument("--gemma_variant", type=str, default="gemma3-270m")
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_llm", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=512)

    # 相机配置
    parser.add_argument("--camera_type", type=str, default="fisheye",
                        choices=["fisheye", "depth"])

    # 训练配置
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # ===== 只新增：resume + tensorboard 频率 =====
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 继续训练（如 .../last.pt 或 best_model.pt）")
    parser.add_argument("--tb_log_interval", type=int, default=50,
                        help="每隔多少 step 写一次 TensorBoard(train step)")

    # 评估配置
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/value_pika/run_20260113_152257/best_model.pt")
    parser.add_argument("--episode_path", type=str, default="./data/task3_handover_bread/episode0")
    parser.add_argument("--save_video", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "split":
        split_dataset_episodes(args.data_dir, seed=args.seed)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        if args.checkpoint is None or args.episode_path is None:
            print("评估模式需要 --checkpoint 和 --episode_path")
            return
        evaluate(args)

if __name__ == "__main__":
    main()
