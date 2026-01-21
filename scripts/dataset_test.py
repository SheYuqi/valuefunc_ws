import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from font.*")

import matplotlib
matplotlib.use("Agg")

import random
import argparse
import h5py
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from torch.utils.tensorboard import SummaryWriter

from dataset import PikaHDF5Dataset
from config import CAMERA_CONFIGS, NUM_BINS
from valuefunc import SigLIPGemmaValueFunction
from episode import  check_dataset_split, split_dataset_episodes, compute_task_max_len_from_path, get_task_max_len, scan_episodes, relpath_under, split_episodes_by_task

def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    log_dir = output_dir / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)

    # split
    is_split, split_info = check_dataset_split(args.data_dir)
    if not is_split:
        print("数据集未划分，自动进行划分...")
        split_info = split_dataset_episodes(args.data_dir, seed=args.seed)

    train_episodes = split_info["train"]
    val_episodes = split_info["val"]
    test_episodes = split_info["test"]

    all_episodes = train_episodes + val_episodes + test_episodes
    task_max_len = compute_task_max_len_from_path(all_episodes)
    print("[TaskMaxLen] 示例：", list(task_max_len.items())[:5])

    # if args.max_episodes:
    #     train_episodes = train_episodes[:args.max_episodes]

    # model
    model = SigLIPGemmaValueFunction(
        num_bins=NUM_BINS,
        siglip_variant=args.siglip_variant,
        gemma_variant=args.gemma_variant,
        freeze_vision=args.freeze_vision,
        freeze_llm=args.freeze_llm,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # dataset / loader
    train_dataset = PikaHDF5Dataset(
        episode_dirs=train_episodes,
        image_size=args.image_size,
        camera_type=args.camera_type,
        data_dir=args.data_dir,
        image_processor=model.image_processor,
        task_max_len=task_max_len,
    )
    val_dataset = PikaHDF5Dataset(
        episode_dirs=val_episodes,
        image_size=args.image_size,
        camera_type=args.camera_type,
        data_dir=args.data_dir,
        image_processor=model.image_processor,
        task_max_len=task_max_len,

    )
    test_dataset = PikaHDF5Dataset(
        episode_dirs=test_episodes,
        image_size=args.image_size,
        camera_type=args.camera_type,
        data_dir=args.data_dir,
        image_processor=model.image_processor,
        task_max_len=task_max_len,
    )
    


def main():
    import argparse

    parser = argparse.ArgumentParser(description="测试数据集脚本")
    parser.add_argument("--data_dir", type=str, default="data", help="数据根目录（包含各任务子目录）")
    parser.add_argument("--image_size", type=int, default=224, help="图像大小")
    parser.add_argument("--camera_type", type=str, default="fisheye", help="相机类型")
    parser.add_argument("--siglip_variant", type=str, default="base", help="SigLIP 变体")
    parser.add_argument("--gemma_variant", type=str, default="base", help="Gemma 变体")
    parser.add_argument("--freeze_vision", action="store_true", help="冻结视觉部分")
    parser.add_argument("--freeze_llm", action="store_true", help="冻结 LLM 部分")
    parser.add_argument("--hidden_dim", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--device", type=str, default="cuda", help="设备类型")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    test(args)

if __name__ == "__main__":
    main()