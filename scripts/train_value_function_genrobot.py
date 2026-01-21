#!/usr/bin/env python3
"""
SigLIP + Gemma 3 270M Value Function Training (Pika HDF5)

修复点：
- Dataset: task_max_len / c_fail 对齐 evaluate 归一化
- global_step: 不再每个 epoch 重置
- 验证集: 使用 val_loader（不再用 test_loader 选 best）
- best_val_loss: 正确更新
- final_model: 不再被重复覆盖
- resume: optimizer/scheduler/scaler/global_step 恢复 + TensorBoard purge_step
"""

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
from valuefunc import SigLIPGemmaValueFunction, value_to_bin
from episode import load_prompt_from_instructions, check_dataset_split, split_dataset_episodes, compute_task_max_len_from_path, get_task_max_len, scan_episodes, relpath_under, split_episodes_by_task


# -----------------------------
# 字体（可选）
# -----------------------------
chinese_font_candidates = [
    "WenQuanYi Micro Hei", "SimHei", "Noto Sans CJK SC",
    "Source Han Sans CN", "Microsoft YaHei", "STHeiti"
]
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_font = next((fn for fn in chinese_font_candidates if fn in available_fonts), None)
if chinese_font:
    plt.rcParams["font.sans-serif"] = [chinese_font, "DejaVu Sans", "sans-serif"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# utils
# -----------------------------


def _get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    if state is None:
        return
    try:
        random.setstate(state.get("python"))
        np.random.set_state(state.get("numpy"))
        torch.set_rng_state(state.get("torch"))
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print(f"[Resume] RNG restore failed (ignored): {e}")


def save_checkpoint_atomic(state: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic-ish


# -----------------------------
# train
# -----------------------------
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # output_dir
    if args.resume:
        output_dir = Path(args.resume).resolve().parent
        print(f"[Resume] Continue in existing run dir: {output_dir}")
    else:
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

    if args.max_episodes:
        train_episodes = train_episodes[:args.max_episodes]

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

    def collate_fn(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "wrist_image": torch.stack([b["wrist_image"] for b in batch]),
            "prompt": [b["prompt"] for b in batch],
            "value_target": torch.stack([b["value_target"] for b in batch]),
            "value_bin": torch.stack([b["value_bin"] for b in batch]),
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )

    # params
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
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_val_mae = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))
        _set_rng_state(ckpt.get("rng_state", None))

        print(f"[Resume] Loaded: {args.resume}")
        print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}, best_val_loss={best_val_loss:.6f}, best_val_mae={best_val_mae:.6f}")

    writer = SummaryWriter(log_dir=str(log_dir), purge_step=global_step)
    print(f"[TensorBoard] logdir: {log_dir} (purge_step={global_step})")

    train_losses = []

    # sanity (可注释)
    if len(train_dataset.samples) > 0:
        vals = [float(s["value_target"]) for s in train_dataset.samples[:5000]]
        print(f"[Sanity] value_target min/max: {min(vals):.4f}/{max(vals):.4f}")
        bins = [int(s["value_bin"]) for s in train_dataset.samples[:5000]]
        print(f"[Sanity] value_bin unique: {len(set(bins))}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            images = batch["image"].to(device, non_blocking=True)
            wrist_images = batch["wrist_image"].to(device, non_blocking=True)
            prompts = batch["prompt"]
            value_bins = batch["value_bin"].to(device, non_blocking=True)

            if scaler is not None:
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

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.4f}"})

            # TB step
            if args.tb_log_interval > 0 and (global_step % args.tb_log_interval == 0):
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/acc_step", acc.item(), global_step)
                writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)

            # save step ckpt
            if args.save_interval > 0 and (global_step % args.save_interval == 0) and global_step > 0:
                ckpt_path = output_dir / f"last_step_{global_step}.pt"
                save_checkpoint_atomic({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "best_val_loss": best_val_loss,
                    "best_val_mae": best_val_mae,
                    "rng_state": _get_rng_state(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"[Checkpoint] saved: {ckpt_path}")

            global_step += 1

        scheduler.step()

        train_loss = epoch_loss / max(1, len(train_loader))
        train_acc = epoch_acc / max(1, len(train_loader))
        train_losses.append(train_loss)

        print(f"Epoch {epoch+1}: Train CE={train_loss:.4f}, Train Acc={train_acc:.4f}")

        # ---------- validate on val_loader (NO test leakage) ----------
        model.eval()
        val_ce = val_top1 = val_mae = val_huber = val_acc1 = val_acc2 = val_entropy = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证(Val)", leave=False):
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

        denom = max(1, len(val_loader))
        val_ce /= denom
        val_top1 /= denom
        val_acc1 /= denom
        val_acc2 /= denom
        val_mae /= denom
        val_huber /= denom
        val_entropy /= denom

        if val_ce < best_val_loss:
            best_val_loss = val_ce

        print(
            f"Epoch {epoch+1}: "
            f"Val CE={val_ce:.4f}, Val Top1={val_top1:.4f}, "
            f"Val Acc@1={val_acc1:.4f}, Val Acc@2={val_acc2:.4f}, "
            f"Val MAE={val_mae:.4f}, Val Huber={val_huber:.4f}, "
            f"Val Entropy={val_entropy:.4f}"
        )

        # TB epoch
        writer.add_scalar("train/ce_epoch", train_loss, epoch)
        writer.add_scalar("train/acc_epoch", train_acc, epoch)
        writer.add_scalar("val/ce_epoch", val_ce, epoch)
        writer.add_scalar("val/top1_epoch", val_top1, epoch)
        writer.add_scalar("val/acc1_epoch", val_acc1, epoch)
        writer.add_scalar("val/acc2_epoch", val_acc2, epoch)
        writer.add_scalar("val/mae_epoch", val_mae, epoch)
        writer.add_scalar("val/huber_epoch", val_huber, epoch)
        writer.add_scalar("val/entropy_epoch", val_entropy, epoch)

        # best by MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_path = output_dir / "best_model.pt"
            save_checkpoint_atomic({
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "best_val_mae": best_val_mae,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "rng_state": _get_rng_state(),
                "camera_type": args.camera_type,
                "args": vars(args),
            }, best_path)
            print(f"[Best] saved (Val MAE={val_mae:.4f}) -> {best_path}")

        # periodic epoch ckpt
        if (epoch + 1) % 5 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint_atomic({
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "best_val_mae": best_val_mae,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "rng_state": _get_rng_state(),
                "camera_type": args.camera_type,
                "args": vars(args),
            }, ckpt_path)
            print(f"[Checkpoint] saved: {ckpt_path}")

        # last for resume
        last_path = output_dir / "last.pt"
        save_checkpoint_atomic({
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "rng_state": _get_rng_state(),
            "camera_type": args.camera_type,
            "args": vars(args),
        }, last_path)

    # final
    final_path = output_dir / "final_model.pt"
    save_checkpoint_atomic({
        "epoch": args.num_epochs - 1,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": _get_rng_state(),
        "train_loss": train_losses[-1] if train_losses else None,
        "camera_type": args.camera_type,
        "args": vars(args),
    }, final_path)

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train CE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross Entropy)")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    writer.flush()
    writer.close()

    print(f"\n训练完成！best_val_loss={best_val_loss:.4f}, best_val_mae={best_val_mae:.4f}")
    print(f"模型保存至: {output_dir}")

    # （可选）最后做一次 test 评估（不选 best，只汇报）
    if args.eval_test_at_end:
        print("\n[Final] 在 Test 集上评估（不用于选模型）...")
        test_ce, test_top1, test_mae = eval_on_loader(model, test_loader, device, criterion)
        print(f"[Test] CE={test_ce:.4f}, Top1={test_top1:.4f}, MAE={test_mae:.4f}")


def eval_on_loader(model, loader, device, criterion):
    model.eval()
    ce = top1 = mae = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="TestEval", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            wrist_images = batch["wrist_image"].to(device, non_blocking=True)
            prompts = batch["prompt"]
            value_bins = batch["value_bin"].to(device, non_blocking=True)
            value_targets = batch["value_target"].to(device, non_blocking=True)

            logits, pred_values = model(images, wrist_images, prompts)
            loss_ce = criterion(logits, value_bins)

            pred_bins = logits.argmax(dim=-1)
            diff = (pred_bins - value_bins).abs()
            t1 = (diff == 0).float().mean()

            m = (pred_values - value_targets).abs().mean()

            ce += loss_ce.item()
            top1 += t1.item()
            mae += m.item()

    n = max(1, len(loader))
    return ce / n, top1 / n, mae / n


# -----------------------------
# evaluate single episode
# -----------------------------
def evaluate(args):
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    camera_type = checkpoint.get("camera_type", args.camera_type)
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

    with h5py.File(hdf5_path, "r") as f:
        episode_len = int(f["size"][()])
        left_cam_paths = [f[f"camera/color/{left_camera}"][i].decode("utf-8") for i in range(episode_len)]
        right_cam_paths = [f[f"camera/color/{right_camera}"][i].decode("utf-8") for i in range(episode_len)]

    instr = load_prompt_from_instructions(episode_dir)
    if not isinstance(instr, dict):
        raise ValueError(f"评估失败：{episode_dir/'instructions.json'} 解析失败")
    prompt = (instr.get("prompt") or "").strip()
    is_success = bool(instr.get("success", True))
    task_name = instr.get("task_name", Path(episode_dir).parent.name)

    if not prompt:
        raise ValueError(f"评估失败：{episode_dir/'instructions.json'} 缺少有效 prompt")

    # true values (aligned)
    C_FAIL = float(args.c_fail)
    T_task = get_task_max_len(args.data_dir, task_name)
    if T_task <= 1:
        raise ValueError(f"无法获得 task={task_name} 的 max_len（检查 data_dir/任务目录结构）")

    denom = max(1, T_task - 1)
    T = episode_len - 1

    true_values = []
    for t in range(episode_len):
        remaining = T - t
        v = -float(remaining)
        if (t == T) and (not is_success):
            v -= C_FAIL
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
            right_img_path = str(episode_dir / right_cam_paths[t])

            left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
            right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
            if left_img is None or right_img is None:
                print(f"警告: 无法加载图像 {left_img_path} 或 {right_img_path}")
                continue

            left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            left_images.append(left_img_rgb.copy())
            right_images.append(right_img_rgb.copy())

            with Image.open(left_img_path) as im:
                im = im.convert("RGB")
            left_tensor = model.image_processor(images=im, return_tensors="pt")["pixel_values"].to(device)

            with Image.open(right_img_path) as im:
                im = im.convert("RGB")
            right_tensor = model.image_processor(images=im, return_tensors="pt")["pixel_values"].to(device)

            logits, value = model(left_tensor, right_tensor, [prompt])
            pred_values.append(value.item())
            pred_bins.append(int(logits.argmax(dim=-1).item()))

    pred_values = np.array(pred_values, dtype=np.float32)
    true_values = np.array(true_values[:len(pred_values)], dtype=np.float32)

    true_bins = np.array([value_to_bin(v) for v in true_values], dtype=np.int64)
    pred_bins = np.array(pred_bins, dtype=np.int64)

    mae = float(np.mean(np.abs(pred_values - true_values))) if len(pred_values) > 0 else 0.0
    acc = float(np.mean(pred_bins == true_bins)) if len(pred_values) > 0 else 0.0
    corr = float(np.corrcoef(pred_values, true_values)[0, 1]) if len(pred_values) > 1 else 0.0

    print("\n评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Bin Accuracy: {acc:.4f}")
    print(f"  Correlation: {corr:.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    frames = np.arange(len(pred_values))
    ax.plot(frames, pred_values, "r-", label="Predicted Value", linewidth=2)
    ax.fill_between(frames, pred_values, -1, alpha=0.2, color="red")
    ax.plot(frames, true_values, "g--", label="True Value", linewidth=2)
    ax.fill_between(frames, true_values, -1, alpha=0.2, color="green")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.set_ylim(-1.1, 0.1)
    status_text = "Success" if is_success else "Failure"
    ax.set_title(f"Value Function Prediction | Status: {status_text}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(args.checkpoint).parent / f"eval_{task_name}_{episode_dir.name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"评估图保存至: {output_path}")

    if args.save_video:
        print("\n生成评估视频...")
        video_path = Path(args.checkpoint).parent / f"eval_{task_name}_{episode_dir.name}.mp4"

        actual_len = len(pred_values)
        if actual_len <= 0:
            print("没有有效帧，跳过视频生成")
            return

        camera_desc = CAMERA_CONFIGS[camera_type]["description"]
        left_title = f"Left Camera ({camera_desc})"
        right_title = f"Right Camera ({camera_desc})"

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.5], height_ratios=[1, 1])

        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[1, 0])
        ax_value = fig.add_subplot(gs[:, 1])

        im_left = ax_left.imshow(left_images[0])
        ax_left.set_title(left_title, fontsize=12, fontweight="bold")
        ax_left.axis("off")

        im_right = ax_right.imshow(right_images[0])
        ax_right.set_title(right_title, fontsize=12, fontweight="bold")
        ax_right.axis("off")

        ax_value.set_xlim(0, actual_len - 1)
        ax_value.set_ylim(-1.1, 0.1)
        ax_value.set_xlabel("Frame", fontsize=11)
        ax_value.set_ylabel("Value", fontsize=11)
        ax_value.grid(True, alpha=0.3)

        (line_pred,) = ax_value.plot([], [], "r-", label="Predicted Value", linewidth=2)
        vline = ax_value.axvline(x=0, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
        scatter_pred = ax_value.scatter([], [], c="red", s=100, zorder=5, edgecolors="white", linewidths=2)

        ax_value.legend(loc="lower right", fontsize=10)

        title = fig.suptitle(
            f"Task: {prompt}\nFrame: 0/{actual_len-1} | Status: {status_text}",
            fontsize=11, fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        def update(frame):
            im_left.set_array(left_images[frame])
            im_right.set_array(right_images[frame])
            line_pred.set_data(np.arange(frame + 1), pred_values[: frame + 1])
            vline.set_xdata([frame, frame])
            scatter_pred.set_offsets([[frame, pred_values[frame]]])
            title.set_text(f"Task: {prompt}\nFrame: {frame}/{actual_len-1} | Status: {status_text}")
            return im_left, im_right, line_pred, vline, scatter_pred, title

        anim = FuncAnimation(fig, update, frames=actual_len, interval=50, blit=False)
        anim.save(str(video_path), writer="ffmpeg", fps=20, dpi=100, bitrate=2000)
        plt.close()
        print(f"视频保存至: {video_path}")


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SigLIP + Gemma Value Function Training (Pika HDF5)")

    parser.add_argument("--mode", type=str, required=True, choices=["split", "train", "eval"], help="运行模式")
    parser.add_argument("--data_dir", type=str, default="data", help="数据根目录（包含各任务子目录）")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/", help="输出目录")

    # model
    parser.add_argument("--siglip_variant", type=str, default="so400m", help="SigLIP so400m(400M,384px)")
    parser.add_argument("--gemma_variant", type=str, default="gemma3-270m", help="Gemma: gemma3-270m")
    parser.add_argument("--freeze_vision", action="store_true", help="冻结 SigLIP 视觉编码器")
    parser.add_argument("--freeze_llm", action="store_true", help="冻结 Gemma 语言模型")
    parser.add_argument("--hidden_dim", type=int, default=512, help="隐藏层维度")

    # camera
    parser.add_argument("--camera_type", type=str, default="fisheye", choices=["fisheye", "depth"], help="相机类型")

    # train
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--max_episodes", type=int, default=None)  ## 限制训练集样本数（调试用）
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 继续训练（如 .../last.pt）")
    parser.add_argument("--tb_log_interval", type=int, default=50, help="每隔多少 step 写一次 TensorBoard")
    parser.add_argument("--save_interval", type=int, default=0, help="每隔多少 step 额外保存 last_step_xx.pt（0=关闭）")
    parser.add_argument("--c_fail", type=float, default=50.0, help="失败终止额外惩罚（与数据/评估保持一致）")
    parser.add_argument("--eval_test_at_end", action="store_true", help="训练结束在 test 集汇报一次（不用于选模型）")

    # eval
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/run_20260113_205201/best_model.pt", help="模型 checkpoint 路径")
    parser.add_argument("--episode_path", type=str, default="./data/", help="评估 episode 目录路径")
    parser.add_argument("--save_video", action="store_true", help="生成评估视频")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "split":
        split_dataset_episodes(args.data_dir, seed=args.seed)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        if not args.checkpoint or not args.episode_path:
            raise ValueError("评估模式需要 --checkpoint 和 --episode_path")
        evaluate(args)


if __name__ == "__main__":
    main()
