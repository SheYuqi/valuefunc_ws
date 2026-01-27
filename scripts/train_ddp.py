#!/usr/bin/env python3
# ============================================================
# Minimal train_ddp.py
# DDP version of train.py for LeRobot v2.1 pack dataset
# - Uses LeRobotV21SigLIPDataset (from lerobot_dataset.py)
# - 3 cameras: head + left wrist + right wrist
# - No scheduler / no tensorboard / minimal args
# - Saves: last.pt (each epoch), best_model.pt (by val MAE), final_model.pt
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

from config import NUM_BINS
from valuefunc import SigLIPGemmaValueFunction
from episode import check_dataset_split, split_dataset_episodes
from lerobot_dataset import LeRobotV21SigLIPDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_torch(state: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def ddp_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def setup_ddp():
    if not ddp_is_enabled():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def cleanup_ddp():
    if ddp_is_enabled() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if ddp_is_enabled() and dist.is_initialized():
        dist.barrier()


def allreduce_sum(t: torch.Tensor) -> torch.Tensor:
    if ddp_is_enabled() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def collate_fn_three_cam(batch):
    # LeRobotV21SigLIPDataset returns:
    #   head_image, left_image, right_image, prompt, value_target, value_bin
    return {
        "head_image": torch.stack([b["head_image"] for b in batch], dim=0),
        "left_image": torch.stack([b["left_image"] for b in batch], dim=0),
        "right_image": torch.stack([b["right_image"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "value_target": torch.stack([b["value_target"] for b in batch], dim=0),
        "value_bin": torch.stack([b["value_bin"] for b in batch], dim=0),
    }


def model_forward(model, batch: Dict[str, Any], device: torch.device):
    head = batch["head_image"].to(device, non_blocking=True)
    left = batch["left_image"].to(device, non_blocking=True)
    right = batch["right_image"].to(device, non_blocking=True)
    prompts = batch["prompt"]

    # Prefer 4-arg signature: (head, left, right, prompts)
    try:
        return model(head, left, right, prompts)
    except TypeError:
        # Fallback: some models are 3-arg: (head, left, prompts)
        return model(head, left, prompts)


@torch.no_grad()
def evaluate_ddp(model, loader, device, criterion):
    model.eval()
    sum_ce = 0.0
    sum_acc = 0.0
    sum_mae = 0.0
    total = 0

    for batch in loader:
        y_bin = batch["value_bin"].to(device, non_blocking=True)
        y_val = batch["value_target"].to(device, non_blocking=True)

        logits, pred_values = model_forward(model, batch, device)
        loss = criterion(logits, y_bin)

        pred_bins = logits.argmax(dim=-1)
        acc = (pred_bins == y_bin).float().mean()
        mae = (pred_values - y_val).abs().mean()

        bs = int(y_bin.shape[0])
        total += bs
        sum_ce += float(loss.item()) * bs
        sum_acc += float(acc.item()) * bs
        sum_mae += float(mae.item()) * bs

    stats = torch.tensor(
        [sum_ce, sum_acc, sum_mae, float(total)], device=device, dtype=torch.float64
    )
    allreduce_sum(stats)

    total_g = float(stats[3].item())
    if total_g <= 0:
        return {"ce": 0.0, "acc": 0.0, "mae": 0.0}

    return {
        "ce": float(stats[0].item() / total_g),
        "acc": float(stats[1].item() / total_g),
        "mae": float(stats[2].item() / total_g),
    }


def train(args):
    setup_ddp()
    try:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{get_local_rank()}" if ddp_is_enabled() else args.device)
        else:
            device = torch.device("cpu")
        ddp_print(f"[Device] {device} | ddp={ddp_is_enabled()} | world_size={get_world_size()}")

        seed = int(args.seed) + (get_rank() if ddp_is_enabled() else 0)
        set_seed(seed)

        # Ensure splits.json exists (only rank0 writes)
        is_split, _ = check_dataset_split(args.data_dir)
        if (not is_split) and is_main_process():
            ddp_print("[Split] splits.json 不存在或不完整，自动划分...")
            split_dataset_episodes(args.data_dir, seed=args.seed)
        barrier()
        is_split, _ = check_dataset_split(args.data_dir)
        if not is_split:
            raise RuntimeError("数据集 split 仍不存在/不完整，请检查 split_dataset_episodes 输出。")

        # Output dir
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path(args.output_dir) / run_name
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier()
        ddp_print(f"[Output] {output_dir}")

        # Model
        model = SigLIPGemmaValueFunction(
            num_bins=NUM_BINS,
            siglip_variant=args.siglip_variant,
            gemma_variant=args.gemma_variant,
            freeze_vision=args.freeze_vision,
            freeze_llm=args.freeze_llm,
            hidden_dim=args.hidden_dim,
        ).to(device)

        if ddp_is_enabled():
            model = DDP(model, device_ids=[get_local_rank()] if device.type == "cuda" else None, broadcast_buffers=False)

        # 3 cameras: head + left wrist + right wrist
        camera_keys = [args.cam_head_key, args.cam_left_key, args.cam_right_key]

        train_dataset = LeRobotV21SigLIPDataset(
            data_dir=args.data_dir,
            split="train",
            camera_keys=camera_keys,
            image_processor=unwrap_model(model).image_processor,
            return_value=True,
            return_meta=False,
        )
        val_dataset = LeRobotV21SigLIPDataset(
            data_dir=args.data_dir,
            split="val",
            camera_keys=camera_keys,
            image_processor=unwrap_model(model).image_processor,
            return_value=True,
            return_meta=False,
        )

        if ddp_is_enabled():
            train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
            shuffle_train = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle_train = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_three_cam,
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_three_cam,
            persistent_workers=(args.num_workers > 0),
        )

        optimizer = torch.optim.AdamW(unwrap_model(model).parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        use_amp = (device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_mae = float("inf")

        if is_main_process():
            x0 = train_dataset[0]
            ddp_print(
                f"[Sanity] head={tuple(x0['head_image'].shape)} left={tuple(x0['left_image'].shape)} "
                f"right={tuple(x0['right_image'].shape)} prompt='{x0['prompt'][:80]}'"
            )

        for epoch in range(args.num_epochs):
            if ddp_is_enabled() and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            unwrap_model(model).train()
            sum_loss = 0.0
            sum_acc = 0.0
            total = 0

            it = train_loader
            if is_main_process():
                it = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

            for batch in it:
                optimizer.zero_grad(set_to_none=True)
                y_bin = batch["value_bin"].to(device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits, _ = model_forward(model, batch, device)
                        loss = criterion(logits, y_bin)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, _ = model_forward(model, batch, device)
                    loss = criterion(logits, y_bin)
                    loss.backward()
                    optimizer.step()

                pred_bins = logits.argmax(dim=-1)
                acc = (pred_bins == y_bin).float().mean().item()

                bs = int(y_bin.shape[0])
                sum_loss += float(loss.item()) * bs
                sum_acc += float(acc) * bs
                total += bs

                if is_main_process() and hasattr(it, "set_postfix"):
                    it.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

            stats = torch.tensor([sum_loss, sum_acc, float(total)], device=device, dtype=torch.float64)
            allreduce_sum(stats)
            total_g = max(1.0, float(stats[2].item()))
            train_ce = float(stats[0].item() / total_g)
            train_acc = float(stats[1].item() / total_g)

            val_metrics = evaluate_ddp(model, val_loader, device, criterion)
            ddp_print(
                f"[Epoch {epoch+1}] "
                f"Train CE={train_ce:.4f} Acc={train_acc:.4f} | "
                f"Val CE={val_metrics['ce']:.4f} Acc={val_metrics['acc']:.4f} MAE={val_metrics['mae']:.4f}"
            )

            if is_main_process():
                save_torch(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                        "val_metrics": val_metrics,
                    },
                    output_dir / "last.pt",
                )

                if val_metrics["mae"] < best_val_mae:
                    best_val_mae = val_metrics["mae"]
                    save_torch(
                        {
                            "epoch": epoch,
                            "model_state_dict": unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "args": vars(args),
                            "val_metrics": val_metrics,
                        },
                        output_dir / "best_model.pt",
                    )
                    ddp_print(f"[Best] updated: val_mae={best_val_mae:.4f} -> {output_dir/'best_model.pt'}")

        if is_main_process():
            save_torch(
                {
                    "epoch": args.num_epochs - 1,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                output_dir / "final_model.pt",
            )
            ddp_print(f"[Done] best_val_mae={best_val_mae:.4f}")
            ddp_print(f"[Saved] {output_dir}")

    finally:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser("Minimal VF training on LeRobot v2.1 (3 cameras) - DDP")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    # model
    parser.add_argument("--siglip_variant", type=str, default="so400m")
    parser.add_argument("--gemma_variant", type=str, default="gemma3-270m")
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_llm", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=512)

    # 3 cameras: head + left + right
    parser.add_argument("--cam_head_key", type=str, default="observation.images.cam_high_rgb")
    parser.add_argument("--cam_left_key", type=str, default="observation.images.cam_left_wrist_rgb")
    parser.add_argument("--cam_right_key", type=str, default="observation.images.cam_right_wrist_rgb")

    # train
    parser.add_argument("--batch_size", type=int, default=8, help="每张 GPU 的 batch_size（DDP 下总 batch = batch_size * nGPU）")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
