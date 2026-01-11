
# ============================================
# 数据集
# ============================================

import os
from typing import List, Dict
from config import CAMERA_CONFIGS
import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from valuefunc import value_to_bin

class PikaHDF5Dataset(Dataset):
    """用于 VLM Value Function 的 Pika HDF5 多任务数据集
    
    episodes: List[Dict]，每个元素至少包含:
      - episode_dir: str
      - prompt: str
      - task_name: str
    """
    def __init__(
        self,
        episodes: List[Dict],
        image_size: int = 224,
        camera_type: str = "fisheye",
    ):
        self.episodes = episodes
        self.image_size = image_size
        self.camera_type = camera_type
        self.samples: List[Dict] = []

        # 获取相机配置
        if camera_type not in CAMERA_CONFIGS:
            raise ValueError(f"不支持的相机类型: {camera_type}，支持: {list(CAMERA_CONFIGS.keys())}")

        self.left_camera = CAMERA_CONFIGS[camera_type]["left"]
        self.right_camera = CAMERA_CONFIGS[camera_type]["right"]

        print(f"使用相机类型: {camera_type} ({CAMERA_CONFIGS[camera_type]['description']})")
        print(f"  左相机: {self.left_camera}")
        print(f"  右相机: {self.right_camera}")

        self._load_data()
        print(f"加载了 {len(self.samples)} 个样本，来自 {len(self.episodes)} 个 episode")

    def _load_data(self):
        gamma = 0.99

        for ep in tqdm(self.episodes, desc="加载 episodes"):
            episode_dir = ep["episode_dir"]
            prompt = ep.get("prompt")
            task_name = str(ep.get("task_name", "unknown"))
            is_success = ep.get("success")
            # print(f"处理 episode: {episode_dir}, 任务: {task_name}, 成功: {is_success}, prompt: {prompt}")
            if is_success is None:
                continue
            
            try:
                hdf5_path = os.path.join(episode_dir, "data.hdf5")
                if not os.path.exists(hdf5_path):
                    print(f"警告: {hdf5_path} 不存在，跳过")
                    continue

                with h5py.File(hdf5_path, "r") as f:
                    episode_len = int(f["size"][()])
                    if episode_len < 2:
                        continue

                    left_cam_key = f"camera/color/{self.left_camera}"
                    right_cam_key = f"camera/color/{self.right_camera}"
                    if left_cam_key not in f or right_cam_key not in f:
                        print(f"警告: {episode_dir} 缺少相机数据({left_cam_key}/{right_cam_key})，跳过")
                        continue

                    left_cam_paths = [f[left_cam_key][i].decode("utf-8") for i in range(episode_len)]
                    right_cam_paths = [f[right_cam_key][i].decode("utf-8") for i in range(episode_len)]

                rewards = [(-1.0) for _ in range(episode_len)]
                rewards[-1] = 0.0 if is_success else -50.0

                values = [0.0] * episode_len
                G = 0.0
                for t in reversed(range(episode_len)):
                    G = rewards[t] + gamma * G
                    values[t] = G


                # normalize to [-1, 0]
                min_value = min(values)
                max_value = 0.0
                if abs(max_value - min_value) < 1e-6:
                    normalized_values = [0.0] * episode_len
                else:
                    normalized_values = [(v - min_value) / (max_value - min_value) - 1.0 for v in values]
                    normalized_values = [max(-1.0, min(0.0, v)) for v in normalized_values]

                for t in range(episode_len):
                    value_bin = value_to_bin(normalized_values[t])
                    left_img_path = os.path.join(episode_dir, left_cam_paths[t])
                    right_img_path = os.path.join(episode_dir, right_cam_paths[t])

                    self.samples.append({
                        "left_image_path": left_img_path,
                        "right_image_path": right_img_path,
                        "prompt": prompt,
                        "task_name": task_name,
                        "value_target": normalized_values[t],
                        "value_bin": value_bin,
                        "is_success": is_success,
                        "episode_dir": episode_dir,
                        "frame_idx": t,
                    })

            except Exception as e:
                print(f"加载 {episode_dir} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    def __len__(self):
        return len(self.samples)

    def _load_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        # BGR -> RGB（用切片通常比 cvtColor 更轻量，但要保证 contiguous）
        img = img[..., ::-1]
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)  # HWC, uint8, contiguous

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        left = self._load_image(s["left_image_path"])
        right = self._load_image(s["right_image_path"])


        # 关键：用 from_numpy，避免 torch.tensor 的额外 copy
        left_t = torch.from_numpy(left).permute(2, 0, 1)   # CHW, uint8
        right_t = torch.from_numpy(right).permute(2, 0, 1)

        return {
            "image": left_t,                 # uint8
            "wrist_image": right_t,          # uint8
            "prompt": s["prompt"],
            "value_target": torch.tensor(s["value_target"], dtype=torch.float32),
            "value_bin": torch.tensor(s["value_bin"], dtype=torch.long),
        }
