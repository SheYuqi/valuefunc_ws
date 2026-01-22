#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from font.*")

import matplotlib
matplotlib.use("Agg")

import random
import argparse
from datetime import datetime
from pathlib import Path


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

from dataset import HDF5Dataset
from config import NUM_BINS
from valuefunc import SigLIPGemmaValueFunction
from episode import check_dataset_split, split_dataset_episodes, compute_task_max_len_from_path


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
    train_dataset = HDF5Dataset(
        episode_dirs=train_episodes,
        image_size=args.image_size,
        camera_type=args.camera_type,
        data_dir=args.data_dir,
        image_processor=model.image_processor,
        task_max_len=task_max_len,
    )
    val_dataset = HDF5Dataset(
        episode_dirs=val_episodes,
        image_size=args.image_size,
        camera_type=args.camera_type,
        data_dir=args.data_dir,
        image_processor=model.image_processor,
        task_max_len=task_max_len,

    )
    test_dataset = HDF5Dataset(
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


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SigLIP + Gemma Value Function Training (Pika HDF5)")

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


    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    train(args)


if __name__ == "__main__":
    main()
