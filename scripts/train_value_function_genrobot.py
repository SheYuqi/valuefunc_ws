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

from dataset import PikaHDF5Dataset
from config import CAMERA_CONFIGS, NUM_BINS, VALUE_MIN, VALUE_MAX
from valuefunc import SigLIPGemmaValueFunction,value_to_bin
from episode import load_prompt_from_instructions,check_dataset_split,split_dataset_episodes

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
    val_episodes = split_info["val"]
    test_episodes = split_info["test"]
    if args.max_episodes:
        train_episodes = train_episodes[:args.max_episodes]

    train_dataset = PikaHDF5Dataset(
        episodes=train_episodes,
        image_size=args.image_size,
        camera_type=args.camera_type,
    )

    val_dataset = PikaHDF5Dataset(
        episodes=val_episodes,
        image_size=args.image_size,
        camera_type=args.camera_type,
    )
    test_dataset = PikaHDF5Dataset(
        episodes=test_episodes,
        image_size=args.image_size,
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    val_losses = []
    best_val_loss = float('inf')
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 训练循环
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            # images = batch['image'].to(device)
            # wrist_images = batch['wrist_image'].to(device)
            images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
            wrist_images = batch["wrist_image"].to(device, non_blocking=True).float().div_(255.0)

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
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证", leave=False):
                images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
                wrist_images = batch["wrist_image"].to(device, non_blocking=True).float().div_(255.0)
                prompts = batch['prompt']
                value_bins = batch["value_bin"].to(device, non_blocking=True)
                value_targets = batch["value_target"].to(device, non_blocking=True)
                
                logits, pred_values = model(images, wrist_images, prompts)
                loss = criterion(logits, value_bins)
                
                pred_bins = logits.argmax(dim=-1)
                acc = (pred_bins == value_bins).float().mean()
                mae = torch.abs(pred_values - value_targets).mean()
                
                val_loss += loss.item()
                val_acc += acc.item()
                val_mae += mae.item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_mae /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val MAE={val_mae:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_mae': val_mae,
            }, output_dir / "best_model.pt")
            print(f"  保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    # 保存最终模型
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'camera_type': args.camera_type,
    }, output_dir / "final_model.pt")

    def eval_loader(model, loader):
        model.eval()
        total_loss = total_acc = total_mae = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc="评估", leave=False):
                images = batch['image'].to(device)
                wrist_images = batch['wrist_image'].to(device)
                prompts = batch['prompt']
                value_bins = batch['value_bin'].to(device)
                value_targets = batch['value_target'].to(device)

                logits, pred_values = model(images, wrist_images, prompts)
                loss = criterion(logits, value_bins)

                pred_bins = logits.argmax(dim=-1)
                acc = (pred_bins == value_bins).float().mean()
                mae = torch.abs(pred_values - value_targets).mean()

                total_loss += loss.item()
                total_acc += acc.item()
                total_mae += mae.item()

        n = max(1, len(loader))
        return total_loss / n, total_acc / n, total_mae / n
 
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
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存至: {output_dir}")

    best_path = output_dir / "best_model.pt"
    if best_path.exists() and len(test_loader) > 0:
        print("\n使用 best_model.pt 在测试集上评估...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_loss, test_acc, test_mae = eval_loader(model, test_loader)
        print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, Test MAE={test_mae:.4f}")
    else:
        print("\n警告: best_model.pt 不存在或测试集为空，跳过测试集评估。")


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
    
    # is_success = True
    instr = load_prompt_from_instructions(episode_dir)
    prompt = None
    if isinstance(instr, dict):
        prompt = instr.get("prompt")
        is_success = instr.get("success")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"评估失败：{episode_dir/'instructions.json'} 缺少有效 prompt")
    prompt = prompt.strip()
        
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
            left_tensor = (
                torch.from_numpy(left_img_resized)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
                .permute(2, 0, 1)
                .contiguous()
                .unsqueeze(0)
            )
            
            # 加载右相机图像
            right_img_path = str(episode_dir / right_cam_paths[t])
            right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
            if right_img is None:
                print(f"警告: 无法加载图像 {right_img_path}")
                continue
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_images.append(right_img.copy())  # 保存原始图像用于视频

            right_img_resized = cv2.resize(right_img, (args.image_size, args.image_size))
            right_tensor = (
                torch.from_numpy(right_img_resized)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
                .permute(2, 0, 1)
                .contiguous()
                .unsqueeze(0)
            )
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
    parser.add_argument("--data_dir", type=str, default="data",
                    help="数据根目录（包含各任务子目录）")
    
    parser.add_argument("--output_dir", type=str, default="./checkpoints/",
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
    parser.add_argument("--checkpoint", type=str, default="./genrobot/checkpoints/value_genrobot/run_20260108_145315/best_model.pt",
                        help="模型 checkpoint 路径")
    parser.add_argument("--episode_path", type=str, default="data/clean_bowl/episode_90",
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
