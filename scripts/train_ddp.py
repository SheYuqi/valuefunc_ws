#!/usr/bin/env python3
"""
Single-node Multi-GPU (DDP) Training Script

用法：
1) 单卡：
   python train_ddp.py --data_dir data --output_dir ./checkpoints --batch_size 16

2) 单机多卡（推荐）：
   torchrun --standalone --nproc_per_node=4 train_ddp.py \
     --data_dir data --output_dir ./checkpoints --batch_size 8

说明：
- batch_size 是“每张 GPU”的 batch。
- DDP 下：只在 rank0 写 TensorBoard / 保存 checkpoint / 打印日志。
"""

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

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torch.utils.tensorboard import SummaryWriter

# =========================
# 你仓库里的依赖
# =========================
from dataset import HDF5Dataset
from config import NUM_BINS
from valuefunc import SigLIPGemmaValueFunction
from episode import check_dataset_split, split_dataset_episodes, compute_task_max_len_from_path

# =========================
# ===== DDP CHANGE 1/8: imports for DDP =====
# =========================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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


# =========================
# ===== DDP CHANGE 2/8: ddp helpers =====
# =========================
def ddp_is_enabled():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def get_rank():
    return int(os.environ.get("RANK", "0"))

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))

def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))

def is_main_process():
    return get_rank() == 0

def ddp_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def setup_ddp():
    """Initialize process group for single-node DDP via torchrun env://"""
    if not ddp_is_enabled():
        return
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(get_local_rank())

def cleanup_ddp():
    if ddp_is_enabled() and dist.is_initialized():
        dist.destroy_process_group()

def unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def barrier():
    if ddp_is_enabled() and dist.is_initialized():
        dist.barrier()

def allreduce_sum(t: torch.Tensor):
    if ddp_is_enabled() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# -----------------------------
# utils (checkpoint)
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

def save_checkpoint_atomic(state: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic-ish


# =========================
# ===== DDP CHANGE 3/8: metric aggregation (val/test) =====
# =========================
@torch.no_grad()
def eval_on_loader(model, loader, device, criterion):
    """
    DDP 下：每个 rank 只看自己的 sampler 子集，然后 all_reduce 汇总成全局平均。
    返回：val_ce, val_top1, val_mae, val_huber, val_acc1, val_acc2, val_entropy
    """
    model.eval()

    # 使用“加权求和”方式统计：sum(metric * batch_size) / total_count
    sum_ce = 0.0
    sum_top1 = 0.0
    sum_acc1 = 0.0
    sum_acc2 = 0.0
    sum_mae = 0.0
    sum_huber = 0.0
    sum_entropy = 0.0
    total = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        wrist_images = batch["wrist_image"].to(device, non_blocking=True)
        prompts = batch["prompt"]
        value_bins = batch["value_bin"].to(device, non_blocking=True)
        value_targets = batch["value_target"].to(device, non_blocking=True)

        logits, pred_values = model(images, wrist_images, prompts)

        bs = int(value_bins.shape[0])
        total += bs

        loss_ce = criterion(logits, value_bins)  # mean over batch
        pred_bins = logits.argmax(dim=-1)
        diff = (pred_bins - value_bins).abs()

        top1 = (diff == 0).float().mean()
        acc1 = (diff <= 1).float().mean()
        acc2 = (diff <= 2).float().mean()

        mae = (pred_values - value_targets).abs().mean()
        huber = F.smooth_l1_loss(pred_values, value_targets)

        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

        sum_ce += float(loss_ce.item()) * bs
        sum_top1 += float(top1.item()) * bs
        sum_acc1 += float(acc1.item()) * bs
        sum_acc2 += float(acc2.item()) * bs
        sum_mae += float(mae.item()) * bs
        sum_huber += float(huber.item()) * bs
        sum_entropy += float(entropy.item()) * bs

    # 汇总到全局
    stats = torch.tensor(
        [sum_ce, sum_top1, sum_acc1, sum_acc2, sum_mae, sum_huber, sum_entropy, float(total)],
        device=device, dtype=torch.float64
    )
    allreduce_sum(stats)

    total_g = max(1.0, float(stats[-1].item()))
    ce = float(stats[0].item() / total_g)
    top1 = float(stats[1].item() / total_g)
    acc1 = float(stats[2].item() / total_g)
    acc2 = float(stats[3].item() / total_g)
    mae = float(stats[4].item() / total_g)
    huber = float(stats[5].item() / total_g)
    entropy = float(stats[6].item() / total_g)

    return ce, top1, mae, huber, acc1, acc2, entropy


# -----------------------------
# train
# -----------------------------
def train(args):
    # =========================
    # ===== DDP CHANGE 4/8: init ddp + device selection =====
    # =========================
    setup_ddp()
    try:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{get_local_rank()}" if ddp_is_enabled() else args.device)
        else:
            device = torch.device("cpu")
        ddp_print(f"使用设备: {device} | ddp={ddp_is_enabled()} | world_size={get_world_size()}")

        # seed（DDP 下每个 rank 不同，避免完全同随机序列）
        seed = int(args.seed) + (get_rank() if ddp_is_enabled() else 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # output_dir：DDP 下只让 rank0 创建并写 log
        if args.resume:
            output_dir = Path(args.resume).resolve().parent
            if is_main_process():
                ddp_print(f"[Resume] Continue in existing run dir: {output_dir}")
        else:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = Path(args.output_dir) / run_name
            if is_main_process():
                output_dir.mkdir(parents=True, exist_ok=True)
        barrier()
        ddp_print(f"输出目录: {output_dir}")

        log_dir = output_dir / "tb"
        if is_main_process():
            log_dir.mkdir(parents=True, exist_ok=True)

        # =========================
        # ===== DDP CHANGE 5/8: split only once on rank0 + broadcast barrier =====
        # =========================
        is_split, split_info = check_dataset_split(args.data_dir)
        if (not is_split) and is_main_process():
            ddp_print("数据集未划分，自动进行划分...")
            split_info = split_dataset_episodes(args.data_dir, seed=args.seed)
        barrier()
        # 其他 rank 重新读取 split_info（确保一致）
        is_split, split_info = check_dataset_split(args.data_dir)
        if not is_split:
            raise RuntimeError("数据集 split 仍不存在/不完整，请检查 split_dataset_episodes 输出。")

        train_episodes = split_info["train"]
        val_episodes = split_info["val"]
        test_episodes = split_info["test"]

        all_episodes = train_episodes + val_episodes + test_episodes
        task_max_len = compute_task_max_len_from_path(all_episodes)
        ddp_print("[TaskMaxLen] 示例：", list(task_max_len.items())[:5])

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

        # =========================
        # ===== DDP CHANGE 6/8: wrap with DDP (after .to(device)) =====
        # =========================
        if ddp_is_enabled():
            model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank(), broadcast_buffers=False)

        # dataset / loader
        train_dataset = HDF5Dataset(
            episode_dirs=train_episodes,
            image_size=args.image_size,
            camera_type=args.camera_type,
            data_dir=args.data_dir,
            image_processor=unwrap_model(model).image_processor,
            task_max_len=task_max_len,
        )
        val_dataset = HDF5Dataset(
            episode_dirs=val_episodes,
            image_size=args.image_size,
            camera_type=args.camera_type,
            data_dir=args.data_dir,
            image_processor=unwrap_model(model).image_processor,
            task_max_len=task_max_len,
        )
        test_dataset = HDF5Dataset(
            episode_dirs=test_episodes,
            image_size=args.image_size,
            camera_type=args.camera_type,
            data_dir=args.data_dir,
            image_processor=unwrap_model(model).image_processor,
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

        # =========================
        # ===== DDP CHANGE 7/8: DistributedSampler =====
        # =========================
        if ddp_is_enabled():
            train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=False)
            val_sampler = DistributedSampler(val_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False, drop_last=False)
            test_sampler = DistributedSampler(test_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False, drop_last=False)
            shuffle_train = False  # 交给 sampler
        else:
            train_sampler = val_sampler = test_sampler = None
            shuffle_train = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=(args.num_workers > 0),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=(args.num_workers > 0),
        )

        # params
        if is_main_process():
            total_params = sum(p.numel() for p in unwrap_model(model).parameters())
            trainable_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
            ddp_print(f"总参数: {total_params / 1e6:.2f}M")
            ddp_print(f"可训练参数: {trainable_params / 1e6:.2f}M")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unwrap_model(model).parameters()),
            lr=args.lr,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss()

        # AMP（新 API，避免 FutureWarning）
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        # resume
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        best_val_mae = float("inf")

        if args.resume:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(ckpt["model_state_dict"], strict=True)

            if ckpt.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict") is not None and scaler is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("global_step", 0))
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
            best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))

            ddp_print(f"[Resume] Loaded: {args.resume}")
            ddp_print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}, best_val_loss={best_val_loss:.6f}, best_val_mae={best_val_mae:.6f}")
        barrier()

        # TB：只在 rank0
        writer = None
        if is_main_process():
            writer = SummaryWriter(log_dir=str(log_dir), purge_step=global_step)
            ddp_print(f"[TensorBoard] logdir: {log_dir} (purge_step={global_step})")

        train_losses = []

        # sanity（只在 rank0）
        if is_main_process() and hasattr(train_dataset, "samples") and len(train_dataset.samples) > 0:
            vals = [float(s["value_target"]) for s in train_dataset.samples[:5000]]
            ddp_print(f"[Sanity] value_target min/max: {min(vals):.4f}/{max(vals):.4f}")
            bins = [int(s["value_bin"]) for s in train_dataset.samples[:5000]]
            ddp_print(f"[Sanity] value_bin unique: {len(set(bins))}")

        for epoch in range(start_epoch, args.num_epochs):
            # DDP: 每个 epoch 设置 sampler seed
            if ddp_is_enabled() and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            unwrap_model(model).train()

            # 训练统计：加权求和 + all_reduce
            sum_loss = 0.0
            sum_acc = 0.0
            total = 0

            # 只在 rank0 开 tqdm
            it = train_loader
            if is_main_process():
                from tqdm import tqdm
                it = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

            for batch in it:
                optimizer.zero_grad(set_to_none=True)

                images = batch["image"].to(device, non_blocking=True)
                wrist_images = batch["wrist_image"].to(device, non_blocking=True)
                prompts = batch["prompt"]
                value_bins = batch["value_bin"].to(device, non_blocking=True)

                bs = int(value_bins.shape[0])

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits, _ = model(images, wrist_images, prompts)
                    loss = criterion(logits, value_bins)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pred_bins = logits.argmax(dim=-1)
                acc = (pred_bins == value_bins).float().mean()

                sum_loss += float(loss.item()) * bs
                sum_acc += float(acc.item()) * bs
                total += bs

                if is_main_process():
                    if hasattr(it, "set_postfix"):
                        it.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.4f}"})

                # TB step：只在 rank0
                if writer is not None and args.tb_log_interval > 0 and (global_step % args.tb_log_interval == 0):
                    writer.add_scalar("train/loss_step", float(loss.item()), global_step)
                    writer.add_scalar("train/acc_step", float(acc.item()), global_step)
                    writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)

                # step ckpt：只在 rank0
                if is_main_process() and args.save_interval > 0 and (global_step % args.save_interval == 0) and global_step > 0:
                    ckpt_path = output_dir / f"last_step_{global_step}.pt"
                    save_checkpoint_atomic({
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                        "best_val_loss": best_val_loss,
                        "best_val_mae": best_val_mae,
                        "rng_state": _get_rng_state(),  # rank0 only
                        "camera_type": args.camera_type,
                        "args": vars(args),
                    }, ckpt_path)
                    ddp_print(f"[Checkpoint] saved: {ckpt_path}")

                global_step += 1

            scheduler.step()

            # 汇总 train 指标
            stats = torch.tensor([sum_loss, sum_acc, float(total)], device=device, dtype=torch.float64)
            allreduce_sum(stats)
            total_g = max(1.0, float(stats[2].item()))
            train_loss = float(stats[0].item() / total_g)
            train_acc = float(stats[1].item() / total_g)

            train_losses.append(train_loss)
            ddp_print(f"Epoch {epoch+1}: Train CE={train_loss:.4f}, Train Acc={train_acc:.4f}")

            # ========== Val（DDP 汇总）==========
            val_ce, val_top1, val_mae, val_huber, val_acc1, val_acc2, val_entropy = eval_on_loader(
                model, val_loader, device, criterion
            )

            if val_ce < best_val_loss:
                best_val_loss = val_ce

            ddp_print(
                f"Epoch {epoch+1}: "
                f"Val CE={val_ce:.4f}, Val Top1={val_top1:.4f}, "
                f"Val Acc@1={val_acc1:.4f}, Val Acc@2={val_acc2:.4f}, "
                f"Val MAE={val_mae:.4f}, Val Huber={val_huber:.4f}, "
                f"Val Entropy={val_entropy:.4f}"
            )

            # TB epoch：只在 rank0
            if writer is not None:
                writer.add_scalar("train/ce_epoch", train_loss, epoch)
                writer.add_scalar("train/acc_epoch", train_acc, epoch)
                writer.add_scalar("val/ce_epoch", val_ce, epoch)
                writer.add_scalar("val/top1_epoch", val_top1, epoch)
                writer.add_scalar("val/acc1_epoch", val_acc1, epoch)
                writer.add_scalar("val/acc2_epoch", val_acc2, epoch)
                writer.add_scalar("val/mae_epoch", val_mae, epoch)
                writer.add_scalar("val/huber_epoch", val_huber, epoch)
                writer.add_scalar("val/entropy_epoch", val_entropy, epoch)

            # best：只在 rank0 保存
            if is_main_process() and val_mae < best_val_mae:
                best_val_mae = val_mae
                best_path = output_dir / "best_model.pt"
                save_checkpoint_atomic({
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "best_val_mae": best_val_mae,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "rng_state": _get_rng_state(),
                    "camera_type": args.camera_type,
                    "args": vars(args),
                }, best_path)
                ddp_print(f"[Best] saved (Val MAE={val_mae:.4f}) -> {best_path}")

            # periodic epoch ckpt：只在 rank0
            if is_main_process() and (epoch + 1) % 5 == 0:
                ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
                save_checkpoint_atomic({
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "best_val_mae": best_val_mae,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "rng_state": _get_rng_state(),
                    "camera_type": args.camera_type,
                    "args": vars(args),
                }, ckpt_path)
                ddp_print(f"[Checkpoint] saved: {ckpt_path}")

            # last.pt：只在 rank0
            if is_main_process():
                last_path = output_dir / "last.pt"
                save_checkpoint_atomic({
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "best_val_mae": best_val_mae,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "rng_state": _get_rng_state(),
                    "camera_type": args.camera_type,
                    "args": vars(args),
                }, last_path)

        # final：只在 rank0
        if is_main_process():
            final_path = output_dir / "final_model.pt"
            save_checkpoint_atomic({
                "epoch": args.num_epochs - 1,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "best_val_mae": best_val_mae,
                "model_state_dict": unwrap_model(model).state_dict(),
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

            if writer is not None:
                writer.flush()
                writer.close()

            ddp_print(f"\n训练完成！best_val_loss={best_val_loss:.4f}, best_val_mae={best_val_mae:.4f}")
            ddp_print(f"模型保存至: {output_dir}")

        barrier()

        # （可选）训练结束 test 汇报（不用于选 best）
        if args.eval_test_at_end:
            test_ce, test_top1, test_mae, _, _, _, _ = eval_on_loader(model, test_loader, device, criterion)
            ddp_print(f"\n[Final Test] CE={test_ce:.4f}, Top1={test_top1:.4f}, MAE={test_mae:.4f}")

    finally:
        cleanup_ddp()


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SigLIP + Gemma Value Function Training (HDF5) - Single-node DDP ready")

    parser.add_argument("--data_dir", type=str, default="data", help="数据根目录（包含各任务子目录）")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/", help="输出目录")

    # model
    parser.add_argument("--siglip_variant", type=str, default="so400m", help="SigLIP variant")
    parser.add_argument("--gemma_variant", type=str, default="gemma3-270m", help="Gemma variant")
    parser.add_argument("--freeze_vision", action="store_true", help="冻结 SigLIP 视觉编码器")
    parser.add_argument("--freeze_llm", action="store_true", help="冻结 Gemma 语言模型")
    parser.add_argument("--hidden_dim", type=int, default=512, help="隐藏层维度")

    # camera
    parser.add_argument("--camera_type", type=str, default="fisheye", choices=["fisheye", "depth"], help="相机类型")

    # train
    parser.add_argument("--batch_size", type=int, default=16, help="每张 GPU 的 batch_size（DDP 下总 batch = batch_size * nGPU）")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--max_episodes", type=int, default=None, help="限制训练集 episode 数（调试用）")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0", help="单卡模式下使用的 device")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 继续训练（如 .../last.pt）")
    parser.add_argument("--tb_log_interval", type=int, default=50, help="每隔多少 step 写一次 TensorBoard（仅 rank0）")
    parser.add_argument("--save_interval", type=int, default=0, help="每隔多少 step 额外保存 last_step_xx.pt（仅 rank0，0=关闭）")
    parser.add_argument("--c_fail", type=float, default=50.0, help="保留参数位（你 dataset 里会用到的话）")
    parser.add_argument("--eval_test_at_end", action="store_true", help="训练结束在 test 集汇报一次（不用于选模型）")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
