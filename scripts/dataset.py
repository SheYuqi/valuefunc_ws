# dataset.py
# ============================================
# 数据集
# ============================================

import os
from typing import List, Dict, Optional
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import SiglipImageProcessor

from config import CAMERA_CONFIGS
from valuefunc import value_to_bin


class PikaHDF5Dataset(Dataset):
    """
    episode_dirs: List[Dict]  (推荐) 来自 split_info["train"/"val"/"test"] 的元素
        每个 dict 至少包含：
            - episode_dir: str
            - prompt: str (可选，但建议有)
            - task_name: str (可选；否则从路径推断)
            - success: bool (可选；默认 True)
    """

    def __init__(
        self,
        episode_dirs: List[Dict],
        image_size: int = 224,
        camera_type: str = "fisheye",
        data_dir: str = "",
        image_processor: Optional[SiglipImageProcessor] = None,
        task_max_len: Optional[Dict[str, int]] = None,
    ):
        self.data_dir = data_dir
        self.episodes = episode_dirs  # FIX: 原来你写 self.episode_dirs 但 _load_data 用 self.episodes
        self.image_size = image_size
        self.camera_type = camera_type
        self.samples: List[Dict] = []

        if image_processor is None:
            raise ValueError("image_processor 不能为空，请在构造 Dataset 时传入 model.image_processor")
        self.image_processor = image_processor

        self.task_max_len = task_max_len or {}
        self.c_fail = 50.0

        if camera_type not in CAMERA_CONFIGS:
            raise ValueError(f"不支持的相机类型: {camera_type}，支持的类型: {list(CAMERA_CONFIGS.keys())}")

        self.left_camera = CAMERA_CONFIGS[camera_type]["left"]
        self.right_camera = CAMERA_CONFIGS[camera_type]["right"]

        print(f"[Dataset] camera_type: {camera_type} ({CAMERA_CONFIGS[camera_type]['description']})")
        print(f"[Dataset] left: {self.left_camera} | right: {self.right_camera}")

        self._load_data()
        print(f"[Dataset] loaded {len(self.samples)} samples from {len(self.episodes)} episodes")

    def _load_data(self):
        for ep in tqdm(self.episodes, desc="加载 episodes"):
            try:
                episode_dir = ep["episode_dir"]
                prompt = (ep.get("prompt") or "").strip()
                task_name = (ep.get("task_name") or Path(episode_dir).parent.name)
                is_success = bool(ep.get("success", True))

                hdf5_path = os.path.join(episode_dir, "data.hdf5")
                if not os.path.exists(hdf5_path):
                    print(f"[Dataset] 警告: {hdf5_path} 不存在，跳过")
                    continue

                with h5py.File(hdf5_path, "r") as f:
                    episode_len = int(f["size"][()])
                    if episode_len < 2:
                        continue

                    left_cam_key = f"camera/color/{self.left_camera}"
                    right_cam_key = f"camera/color/{self.right_camera}"
                    if left_cam_key not in f or right_cam_key not in f:
                        print(f"[Dataset] 警告: {episode_dir} 缺少相机 key，跳过")
                        continue

                    left_cam_paths = [f[left_cam_key][i].decode("utf-8") for i in range(episode_len)]
                    right_cam_paths = [f[right_cam_key][i].decode("utf-8") for i in range(episode_len)]

                # ===== 与 evaluate() 对齐的 value 归一化 =====
                # 成功：v_t = -(remaining_steps) / (T_task-1)，clip 到 [-1,0]
                # 失败：在终止帧额外 -C_FAIL，再归一化并 clip
                T = episode_len - 1
                T_task = int(self.task_max_len.get(task_name, 0))
                if T_task <= 1:
                    # 回退：至少保证 denom>=1
                    T_task = max(2, episode_len)
                denom = max(1, T_task - 1)

                normalized_values = []
                for t in range(episode_len):
                    remaining = (T - t)
                    v = -float(remaining)
                    if (t == T) and (not is_success):
                        v -= self.c_fail
                    v = v / float(denom)
                    v = max(-1.0, min(0.0, v))
                    normalized_values.append(v)

                for t in range(episode_len):
                    left_img_path = os.path.join(episode_dir, left_cam_paths[t])
                    right_img_path = os.path.join(episode_dir, right_cam_paths[t])

                    self.samples.append({
                        "left_image_path": left_img_path,
                        "right_image_path": right_img_path,
                        "prompt": prompt,
                        "value_target": normalized_values[t],
                        "value_bin": value_to_bin(normalized_values[t]),
                        "is_success": is_success,
                        "episode_dir": episode_dir,
                        "frame_idx": t,
                        "task_name": task_name,
                    })

            except Exception as e:
                print(f"[Dataset] 加载 episode 出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, img_path: str) -> torch.Tensor:
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
        except Exception as e:
            raise ValueError(f"无法加载图像: {img_path}, err={e}")

        # processor 输出 pixel_values: (1,3,H,W)
        out = self.image_processor(images=im, return_tensors="pt")
        return out["pixel_values"][0]  # (3,H,W)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        left_tensor = self._load_image(sample["left_image_path"])
        right_tensor = self._load_image(sample["right_image_path"])

        return {
            "image": left_tensor,
            "wrist_image": right_tensor,
            "prompt": sample["prompt"],
            "value_target": torch.tensor(sample["value_target"], dtype=torch.float32),
            "value_bin": torch.tensor(sample["value_bin"], dtype=torch.long),
        }
