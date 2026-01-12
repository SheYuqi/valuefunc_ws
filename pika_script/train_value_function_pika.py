#!/usr/bin/env python3
"""
基于 SigLIP + Gemma 3 270M 的 Value Function 训练脚本 (Pika HDF5 数据版本)

按照 PaliGemma 的架构设计：
- 视觉编码器：SigLIP (google/siglip-base-patch16-224)
- 语言模型：Gemma 3 270M 
- Value Head：200 bin 分类

使用 Pika 机器人 HDF5 数据格式，输入为左右两张 pikaFisheyeCamera 图像。

使用方法:
    # 划分数据集(90%用于训练)
    python scripts/train_value_function_pika.py --mode split 

    # 训练
    python scripts/train_value_function_pika.py --mode train 

    # 评估
    python scripts/train_value_function_pika.py --mode eval --save_video
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

import matplotlib
matplotlib.use('Agg')
import sys
import glob
import random
import shutil
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
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

# 导入 transformers（必需）
from transformers import (
    SiglipVisionModel,
    SiglipImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    GemmaForCausalLM,
)

# ============================================
# 配置
# ============================================
NUM_BINS = 200  # Value 分成 200 个 bin
VALUE_MIN = -1.0  # Value 最小值
VALUE_MAX = 0.0   # Value 最大值

# Pika 任务的固定 prompt
PIKA_TASK_PROMPT = "Take out the bread and hand it over to the plate"

# 相机名称配置（支持鱼眼相机和深度相机两种类型）
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


# ============================================
# Value 离散化工具函数
# ============================================
def value_to_bin(value: float) -> int:
    """将连续 value 转换为 bin 索引"""
    value = np.clip(value, VALUE_MIN, VALUE_MAX)
    bin_idx = int((value - VALUE_MIN) / (VALUE_MAX - VALUE_MIN) * (NUM_BINS - 1))
    return np.clip(bin_idx, 0, NUM_BINS - 1)


def bin_to_value(bin_idx: int) -> float:
    """将 bin 索引转换回连续 value"""
    return VALUE_MIN + (bin_idx / (NUM_BINS - 1)) * (VALUE_MAX - VALUE_MIN)


def bins_to_value_soft(logits: torch.Tensor) -> torch.Tensor:
    """使用 softmax 加权求和将 logits 转换为连续 value"""
    probs = F.softmax(logits, dim=-1)
    bin_values = torch.linspace(VALUE_MIN, VALUE_MAX, NUM_BINS, device=logits.device)
    return torch.sum(probs * bin_values, dim=-1)


# ============================================
# 数据集
# ============================================
class PikaHDF5Dataset(Dataset):
    """用于 VLM Value Function 的 Pika HDF5 数据集
    
    读取 pika_data/task3_handover_bread 目录下的 HDF5 文件，
    使用左右两张相机图像作为输入（支持鱼眼相机或深度相机）。
    """
    
    def __init__(
        self,
        episode_dirs: List[str],
        image_size: int = 224,
        prompt: str = PIKA_TASK_PROMPT,
        camera_type: str = "fisheye",
    ):
        """
        Args:
            episode_dirs: episode 目录列表（每个目录包含 data.hdf5 文件）
            image_size: 图像大小
            prompt: 任务 prompt
            camera_type: 相机类型，"fisheye"（鱼眼相机）或 "depth"（深度相机/普通相机）
        """
        self.episode_dirs = episode_dirs
        self.image_size = image_size
        self.prompt = prompt
        self.camera_type = camera_type
        self.samples: List[Dict] = []
        
        # 获取相机配置
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
        """加载所有 episode 的数据"""
        gamma = 0.99
        
        for episode_dir in tqdm(self.episode_dirs, desc="加载 episodes"):
            try:
                hdf5_path = os.path.join(episode_dir, "data.hdf5")
                if not os.path.exists(hdf5_path):
                    print(f"警告: {hdf5_path} 不存在，跳过")
                    continue
                
                with h5py.File(hdf5_path, 'r') as f:
                    # 获取帧数
                    episode_len = int(f['size'][()])
                    
                    if episode_len < 2:
                        continue
                    
                    # 检查相机数据是否存在
                    left_cam_key = f'camera/color/{self.left_camera}'
                    right_cam_key = f'camera/color/{self.right_camera}'
                    
                    if left_cam_key not in f or right_cam_key not in f:
                        print(f"警告: {episode_dir} 中缺少相机数据，跳过")
                        continue
                    
                    # 获取左右相机图像路径
                    left_cam_paths = [
                        f[left_cam_key][i].decode('utf-8')
                        for i in range(episode_len)
                    ]
                    right_cam_paths = [
                        f[right_cam_key][i].decode('utf-8')
                        for i in range(episode_len)
                    ]
                
                # 假设所有 episode 都是成功的（Pika 数据通常都是成功的演示）
                is_success = True
                
                # 计算奖励
                rewards = []
                for t in range(episode_len):
                    if t == episode_len - 1:
                        rewards.append(0.0 if is_success else -50.0)
                    else:
                        rewards.append(-1.0)
                
                # 计算 Value（折扣累积奖励）
                values = []
                for t in range(episode_len):
                    v = 0.0
                    for i in range(t, episode_len):
                        v += (gamma ** (i - t)) * rewards[i]
                    values.append(v)
                
                # 归一化到 [-1, 0]，max 固定为 0
                min_value = min(values)
                max_value = 0.0
                
                if abs(max_value - min_value) < 1e-6:
                    normalized_values = [0.0] * episode_len
                else:
                    normalized_values = [
                        (v - min_value) / (max_value - min_value) - 1.0
                        for v in values
                    ]
                    normalized_values = [max(-1.0, min(0.0, v)) for v in normalized_values]
                
                # 保存样本
                for t in range(episode_len):
                    value_bin = value_to_bin(normalized_values[t])
                    
                    # 构建完整的图像路径
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
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """加载并预处理图像"""
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载左右相机图像
        left_image = self._load_image(sample['left_image_path'])
        right_image = self._load_image(sample['right_image_path'])
        
        # 转换为 tensor，归一化到 [0, 1]
        left_tensor = torch.tensor(left_image, dtype=torch.float32) / 255.0
        right_tensor = torch.tensor(right_image, dtype=torch.float32) / 255.0
        
        # 转换为 [C, H, W] 格式
        left_tensor = left_tensor.permute(2, 0, 1)
        right_tensor = right_tensor.permute(2, 0, 1)
        
        return {
            'image': left_tensor,           # 左相机图像（作为主图像）
            'wrist_image': right_tensor,    # 右相机图像（作为第二图像）
            'prompt': sample['prompt'],
            'value_target': torch.tensor(sample['value_target'], dtype=torch.float32),
            'value_bin': torch.tensor(sample['value_bin'], dtype=torch.long),
        }


# ============================================
# 基于 SigLIP + Gemma 3 的 Value Function
# ============================================

# SigLIP 模型选择（按大小排列）
SIGLIP_MODELS = {
    "so400m": "google/siglip-so400m-patch14-384",  # ~400M, 1152 dim, patch=14, img=384 (PI0/PI0.5 使用)
}

# Gemma 模型选择（按大小排列）
GEMMA_MODELS = {
    "gemma3-270m": "google/gemma-3-270m-it",       # 270M params 
}


class SigLIPGemmaValueFunction(nn.Module):
    """
    基于 SigLIP + Gemma 的 Value Function
    
    按照 PaliGemma 架构设计：
    - 视觉编码器：SigLIP (独立预训练)
    - 语言模型：Gemma 3 (独立预训练)
    - 融合层：将视觉、语言特征融合
    - Value Head：输出 200 bin 分类
    
    输入：左右两张相机图像 + 文本 prompt
    """
    
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
        
        # ==================== SigLIP 视觉编码器 ====================
        self._init_vision_encoder(siglip_variant, freeze_vision)
        
        # ==================== Gemma 语言模型 ====================
        self._init_language_model(gemma_variant, freeze_llm, use_flash_attention)
        
        # ==================== 投影层 ====================
        # 将不同维度的特征投影到统一维度
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
        
        print(f"视觉投影: {self.vision_dim} -> {hidden_dim}")
        print(f"文本投影: {self.text_dim} -> {hidden_dim}")
        
        # ==================== 融合层 ====================
        # 2张图像 + 文本 = 3 * hidden_dim
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
        print(f"融合层: {fusion_input_dim} -> {hidden_dim}")
        
        # ==================== Value Head ====================
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_bins),
        )
        print(f"Value Head: {hidden_dim} -> {num_bins} bins")
        
        # 打印参数统计
        self._print_param_stats()
    
    def _init_vision_encoder(self, variant: str, freeze: bool):
        """初始化 SigLIP 视觉编码器"""
        model_name = SIGLIP_MODELS.get(variant, SIGLIP_MODELS["so400m"])
        
        print(f"加载 SigLIP: {model_name}")
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.vision_dim = self.vision_encoder.config.hidden_size
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
        print(f"  - 隐藏维度: {self.vision_dim}")
        print(f"  - Patch size: {self.vision_encoder.config.patch_size}")
        
        if freeze:
            print("  - 冻结视觉编码器参数")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def _init_language_model(self, variant: str, freeze: bool, use_flash_attention: bool):
        """初始化 Gemma 语言模型"""
        model_name = GEMMA_MODELS.get(variant, GEMMA_MODELS["gemma3-270m"])
        
        print(f"加载 Gemma: {model_name}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
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
        """打印参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 60)
        print(f"总参数量: {total_params / 1e6:.1f}M")
        print(f"可训练参数量: {trainable_params / 1e6:.1f}M")
        print("=" * 60)
    
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像，返回 [batch, hidden_dim]"""
        # SigLIP 输出: [batch, num_patches, hidden_dim]
        outputs = self.vision_encoder(images)
        # 取 CLS token 或 mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output  # [batch, hidden_dim]
        else:
            features = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_dim]
        
        return features
    
    def _encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """编码文本，返回 [batch, hidden_dim]"""
        # Gemma tokenization
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        
        # 通过 Gemma 获取隐藏状态
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.language_model.parameters())):
            outputs = self.language_model(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                output_hidden_states=True,
            )
        
        # 使用最后一层隐藏状态的平均值
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]
        
        # 使用 attention mask 进行加权平均
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        features = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        return features
    
    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            left_image: [batch, C, H, W] 左相机图像
            right_image: [batch, C, H, W] 右相机图像
            prompts: 文本提示列表
            
        Returns:
            logits: [batch, num_bins] 分类 logits
            value: [batch] 预测的连续 value
        """
        device = left_image.device
        
        # 编码两张图像
        left_features = self._encode_images(left_image)     # [batch, vision_dim]
        right_features = self._encode_images(right_image)   # [batch, vision_dim]
        
        # 投影视觉特征
        left_proj = self.vision_proj(left_features)         # [batch, hidden_dim]
        right_proj = self.vision_proj(right_features)       # [batch, hidden_dim]
        
        # 编码文本
        text_features = self._encode_text(prompts, device)  # [batch, text_dim]
        text_proj = self.text_proj(text_features)           # [batch, hidden_dim]
        
        # 融合特征
        combined = torch.cat([
            left_proj,
            right_proj,
            text_proj,
        ], dim=-1)  # [batch, 3 * hidden_dim]
        
        fused = self.fusion(combined)  # [batch, hidden_dim]
        
        # Value Head
        logits = self.value_head(fused)  # [batch, num_bins]
        
        # 计算连续 value
        value = bins_to_value_soft(logits)  # [batch]
        
        return logits, value


# 为了兼容性，保留别名
PaliGemmaValueFunction = SigLIPGemmaValueFunction


# ============================================
# 数据集划分
# ============================================
def find_all_episodes(data_dir: str) -> List[str]:
    """查找所有 episode 目录"""
    data_path = Path(data_dir)
    episode_dirs = []
    
    # 查找所有包含 data.hdf5 的 episode* 目录
    for d in sorted(data_path.iterdir()):
        if d.is_dir() and d.name.startswith("episode"):
            hdf5_path = d / "data.hdf5"
            if hdf5_path.exists():
                episode_dirs.append(str(d))
    
    return episode_dirs


def check_dataset_split(data_dir: str) -> Tuple[bool, Optional[Dict]]:
    """检查数据集是否已经划分好
    
    Returns:
        (is_split, split_info): 是否已划分，以及划分信息
    """
    data_path = Path(data_dir)
    split_file = data_path / "split_info.txt"
    
    if not split_file.exists():
        return False, None
    
    # 读取划分信息
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
    """将数据集按 episode 划分为训练集和测试集
    
    Args:
        data_dir: 数据目录，包含 episode* 子目录
        train_ratio: 训练集比例（默认 90%）
        seed: 随机种子
        force: 是否强制重新划分（即使已经划分好）
        
    Returns:
        划分结果字典 {"train": [...], "test": [...]}
    """
    data_path = Path(data_dir)
    
    # 检查是否已经划分好
    is_split, split_info = check_dataset_split(data_dir)
    if not force and is_split:
        print("数据集已经划分好，使用现有划分")
        return split_info
    
    # 查找所有 episode 目录
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
    
    # 保存划分信息
    split_file = data_path / "split_info.txt"
    with open(split_file, 'w') as f:
        for split_name in ["train", "test"]:
            f.write(f"[{split_name}]\n")
            for ep_dir in split_info[split_name]:
                # 只保存相对路径（episode 目录名）
                ep_name = Path(ep_dir).name
                f.write(f"{ep_name}\n")
            f.write("\n")
    
    print(f"划分结果: train={len(split_info['train'])} (90%), test={len(split_info['test'])} (10%)")
    print(f"划分信息已保存到: {split_file}")
    
    return split_info


# ============================================
# 训练
# ============================================
def train(args):
    """训练 PaliGemma Value Function"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 检查并自动划分数据集
    is_split, split_info = check_dataset_split(args.data_dir)
    if not is_split:
        print("数据集未划分，自动进行划分...")
        split_info = split_dataset_episodes(args.data_dir, seed=args.seed)
    
    # 限制 episode 数量（如果指定）
    train_episodes = split_info["train"]
    
    if args.max_episodes:
        train_episodes = train_episodes[:args.max_episodes]
    
    train_dataset = PikaHDF5Dataset(
        episode_dirs=train_episodes,
        image_size=args.image_size,
        prompt=PIKA_TASK_PROMPT,
        camera_type=args.camera_type,
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
    
    # 创建模型
    print("创建 SigLIP + Gemma Value Function...")
    model = SigLIPGemmaValueFunction(
        num_bins=NUM_BINS,
        siglip_variant=args.siglip_variant,
        gemma_variant=args.gemma_variant,
        freeze_vision=args.freeze_vision,
        freeze_llm=args.freeze_llm,
        hidden_dim=args.hidden_dim,
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    best_train_loss = float('inf')
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 训练循环
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            images = batch['image'].to(device)
            wrist_images = batch['wrist_image'].to(device)
            prompts = batch['prompt']
            value_bins = batch['value_bin'].to(device)
            
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
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.4f}',
            })
        
        scheduler.step()
        
        train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_acc / len(train_loader)
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        
        # 保存最佳模型（基于训练损失）
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'camera_type': args.camera_type,
            }, output_dir / "best_model.pt")
            print(f"  保存最佳模型 (Train Loss: {train_loss:.4f})")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'camera_type': args.camera_type,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    # 保存最终模型
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'camera_type': args.camera_type,
    }, output_dir / "final_model.pt")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n训练完成！最佳训练损失: {best_train_loss:.4f}")
    print(f"模型保存至: {output_dir}")


# ============================================
# 评估
# ============================================
def evaluate(args):
    """评估单个 episode，可选生成视频
    
    对于 Pika 数据，episode_path 应该是 episode 目录的路径
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    
    # 从 checkpoint 获取相机类型（如果有）
    camera_type = checkpoint.get('camera_type', args.camera_type)
    print(f"使用相机类型: {camera_type}")
    
    # 获取相机配置
    if camera_type not in CAMERA_CONFIGS:
        raise ValueError(f"不支持的相机类型: {camera_type}")
    
    left_camera = CAMERA_CONFIGS[camera_type]["left"]
    right_camera = CAMERA_CONFIGS[camera_type]["right"]
    
    model = SigLIPGemmaValueFunction(
        num_bins=NUM_BINS,
        siglip_variant=args.siglip_variant,
        gemma_variant=args.gemma_variant,
        freeze_vision=True,  # 评估时冻结
        freeze_llm=True,     # 评估时冻结
        hidden_dim=args.hidden_dim,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 加载 Pika HDF5 数据
    episode_dir = Path(args.episode_path)
    hdf5_path = episode_dir / "data.hdf5"
    
    if not hdf5_path.exists():
        raise ValueError(f"HDF5 文件不存在: {hdf5_path}")
    
    print(f"加载轨迹: {episode_dir}")
    
    with h5py.File(hdf5_path, 'r') as f:
        episode_len = int(f['size'][()])
        
        # 获取左右相机图像路径
        left_cam_paths = [
            f[f'camera/color/{left_camera}'][i].decode('utf-8')
            for i in range(episode_len)
        ]
        right_cam_paths = [
            f[f'camera/color/{right_camera}'][i].decode('utf-8')
            for i in range(episode_len)
        ]
    
    # 假设所有 Pika 数据都是成功的
    is_success = True
    prompt = PIKA_TASK_PROMPT
    
    # 计算真实 Value
    gamma = 0.99
    rewards = []
    for t in range(episode_len):
        if t == episode_len - 1:
            rewards.append(0.0 if is_success else -50.0)
        else:
            rewards.append(-1.0)
    
    raw_values = []
    for t in range(episode_len):
        v = sum((gamma ** (i - t)) * rewards[i] for i in range(t, episode_len))
        raw_values.append(v)
    
    min_value = min(raw_values)
    max_value = 0.0
    if abs(max_value - min_value) < 1e-6:
        true_values = [0.0] * episode_len
    else:
        true_values = [(v - min_value) / (max_value - min_value) - 1.0 for v in raw_values]
        true_values = [max(-1.0, min(0.0, v)) for v in true_values]
    
    print(f"轨迹结果: {'成功' if is_success else '失败'}")
    print(f"轨迹长度: {episode_len}")
    print(f"任务描述: {prompt}")
    
    # 收集预测结果和图像
    pred_values = []
    pred_bins = []
    left_images = []
    right_images = []
    
    with torch.no_grad():
        for t in tqdm(range(episode_len), desc="预测"):
            # 加载左相机图像
            left_img_path = str(episode_dir / left_cam_paths[t])
            left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
            if left_img is None:
                print(f"警告: 无法加载图像 {left_img_path}")
                continue
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_images.append(left_img.copy())  # 保存原始图像用于视频
            
            left_img_resized = cv2.resize(left_img, (args.image_size, args.image_size))
            left_tensor = torch.tensor(left_img_resized, dtype=torch.float32) / 255.0
            left_tensor = left_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 加载右相机图像
            right_img_path = str(episode_dir / right_cam_paths[t])
            right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
            if right_img is None:
                print(f"警告: 无法加载图像 {right_img_path}")
                continue
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_images.append(right_img.copy())  # 保存原始图像用于视频
            
            right_img_resized = cv2.resize(right_img, (args.image_size, args.image_size))
            right_tensor = torch.tensor(right_img_resized, dtype=torch.float32) / 255.0
            right_tensor = right_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            logits, value = model(left_tensor, right_tensor, [prompt])
            pred_values.append(value.item())
            pred_bins.append(logits.argmax(dim=-1).item())
    
    pred_values = np.array(pred_values)
    true_values = np.array(true_values[:len(pred_values)])  # 确保长度匹配
    true_bins = np.array([value_to_bin(v) for v in true_values])
    pred_bins = np.array(pred_bins)
    
    mae = np.mean(np.abs(pred_values - true_values))
    acc = np.mean(pred_bins == true_bins)
    corr = np.corrcoef(pred_values, true_values)[0, 1] if len(pred_values) > 1 else 0.0
    
    print(f"\n评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Bin Accuracy: {acc:.4f}")
    print(f"  Correlation: {corr:.4f}")
    
    # 保存静态评估图（只显示预测值）
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    frames = np.arange(len(pred_values))
    ax.plot(frames, pred_values, 'r-', label='Predicted Value', linewidth=2)
    ax.fill_between(frames, pred_values, -1, alpha=0.2, color='red')
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
    
    # 如果需要生成视频
    if args.save_video:
        print("\n生成评估视频...")
        video_path = Path(args.checkpoint).parent / f"eval_{episode_dir.name}.mp4"
        
        actual_len = len(pred_values)
        
        # 根据相机类型设置标题
        camera_desc = CAMERA_CONFIGS[camera_type]["description"]
        left_title = f'Left Camera ({camera_desc})'
        right_title = f'Right Camera ({camera_desc})'
        
        # 创建图形布局
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.5], height_ratios=[1, 1])
        
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[1, 0])
        ax_value = fig.add_subplot(gs[:, 1])
        
        # 初始化图像显示
        im_left = ax_left.imshow(left_images[0])
        ax_left.set_title(left_title, fontsize=12, fontweight='bold')
        ax_left.axis('off')
        
        im_right = ax_right.imshow(right_images[0])
        ax_right.set_title(right_title, fontsize=12, fontweight='bold')
        ax_right.axis('off')
        
        # 初始化曲线图（只显示预测值）
        ax_value.set_xlim(0, actual_len - 1)
        ax_value.set_ylim(-1.1, 0.1)
        ax_value.set_xlabel('Frame', fontsize=11)
        ax_value.set_ylabel('Value', fontsize=11)
        ax_value.grid(True, alpha=0.3)
        
        # 预测值线（动态更新）
        line_pred, = ax_value.plot([], [], 'r-', label='Predicted Value', linewidth=2)
        
        # 当前帧标记
        vline = ax_value.axvline(x=0, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # 当前预测值点
        scatter_pred = ax_value.scatter([], [], c='red', s=100, zorder=5, edgecolors='white', linewidths=2)
        
        ax_value.legend(loc='lower right', fontsize=10)
        
        # 标题（动态更新）
        title = fig.suptitle(
            f'Task: {prompt}\nFrame: 0/{actual_len-1} | Status: {status_text}',
            fontsize=11, fontweight='bold'
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        def update(frame):
            # 更新图像
            im_left.set_array(left_images[frame])
            im_right.set_array(right_images[frame])
            
            # 更新预测曲线
            line_pred.set_data(frames[:frame+1], pred_values[:frame+1])
            
            # 更新当前帧标记
            vline.set_xdata([frame, frame])
            
            # 更新当前预测值点
            scatter_pred.set_offsets([[frame, pred_values[frame]]])
            
            # 更新标题
            title.set_text(
                f'Task: {prompt}\nFrame: {frame}/{actual_len-1} | Status: {status_text}'
            )
            
            return im_left, im_right, line_pred, vline, scatter_pred, title
        
        # 创建动画
        anim = FuncAnimation(
            fig, update,
            frames=actual_len,
            interval=50,  # 20 fps
            blit=False
        )
        
        # 保存视频
        anim.save(
            str(video_path),
            writer='ffmpeg',
            fps=20,
            dpi=100,
            bitrate=2000
        )
        plt.close()
        
        print(f"视频保存至: {video_path}")


# ============================================
# 主函数
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
    parser.add_argument("--siglip_variant", type=str, 
                        default="so400m",
                        help="SigLIP so400m(400M,384px)")
    parser.add_argument("--gemma_variant", type=str, 
                        default="gemma3-270m",
                        help="Gemma 模型变体: gemma3-270m")
    parser.add_argument("--freeze_vision", action="store_true",
                        help="冻结 SigLIP 视觉编码器")
    parser.add_argument("--freeze_llm", action="store_true",
                        help="冻结 Gemma 语言模型")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="隐藏层维度")
    
    # 相机配置
    parser.add_argument("--camera_type", type=str, default="fisheye",
                        choices=["fisheye", "depth"],
                        help="相机类型: fisheye(鱼眼相机/广角), depth(深度相机/普通相机图像)")
    
    # 训练配置
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--image_size", type=int, default=384,
                        help="图像大小 (so400m 需要 384, base 需要 224)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="最大 episode 数量")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # 评估配置
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/value_pika/run_20260112_110153/best_model.pt",
                        help="模型 checkpoint 路径")
    parser.add_argument("--episode_path", type=str, default="./data/task3_handover_bread/episode0",
                        help="评估的 episode 目录路径")
    parser.add_argument("--save_video", action="store_true",
                        help="生成评估视频")
    
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
