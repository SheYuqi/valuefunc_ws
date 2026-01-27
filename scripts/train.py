#!/usr/bin/env python3
# ============================================================
# Minimal train.py
# Train SigLIP+Gemma Value Function on LeRobot v2.1 pack dataset
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
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


def collate_fn_three_cam(batch):
    # LeRobotV21SigLIPDataset returns:
    #   image, wrist_image, third_image, prompt, value_target, value_bin
    return {
        "head_image": torch.stack([b["image"] for b in batch], dim=0),
        "left_image": torch.stack([b["wrist_image"] for b in batch], dim=0),
        "right_image": torch.stack([b["third_image"] for b in batch], dim=0),
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
def evaluate(model, loader, device, criterion):
    model.eval()
    total_ce = 0.0
    total_acc = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        y_bin = batch["value_bin"].to(device, non_blocking=True)
        y_val = batch["value_target"].to(device, non_blocking=True)

        logits, pred_values = model_forward(model, batch, device)
        loss = criterion(logits, y_bin)

        pred_bins = logits.argmax(dim=-1)
        acc = (pred_bins == y_bin).float().mean().item()

        mae = (pred_values - y_val).abs().mean().item()

        total_ce += float(loss.item())
        total_acc += float(acc)
        total_mae += float(mae)
        n_batches += 1

    if n_batches == 0:
        return {"ce": 0.0, "acc": 0.0, "mae": 0.0}

    return {
        "ce": total_ce / n_batches,
        "acc": total_acc / n_batches,
        "mae": total_mae / n_batches,
    }


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Ensure splits.json exists
    is_split, _ = check_dataset_split(args.data_dir)
    if not is_split:
        print("[Split] splits.json 不存在或不完整，自动划分...")
        split_dataset_episodes(args.data_dir, seed=args.seed)

    # Output dir
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] {output_dir}")

    # Model
    model = SigLIPGemmaValueFunction(
        num_bins=NUM_BINS,
        siglip_variant=args.siglip_variant,
        gemma_variant=args.gemma_variant,
        freeze_vision=args.freeze_vision,
        freeze_llm=args.freeze_llm,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # 3 cameras: head + left wrist + right wrist
    camera_keys = [args.cam_head_key, args.cam_left_key, args.cam_right_key]

    # Dataset (uses your already-working LeRobot wrapper; internally uses official LeRobotDataset)
    train_dataset = LeRobotV21SigLIPDataset(
        data_dir=args.data_dir,
        split="train",
        camera_keys=camera_keys,
        image_processor=model.image_processor,
        return_value=True,
        return_meta=False,
    )
    val_dataset = LeRobotV21SigLIPDataset(
        data_dir=args.data_dir,
        split="val",
        camera_keys=camera_keys,
        image_processor=model.image_processor,
        return_value=True,
        return_meta=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_three_cam,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_three_cam,
        persistent_workers=(args.num_workers > 0),
    )

    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_mae = float("inf")

    # Quick sanity
    x0 = train_dataset[0]
    print(
        f"[Sanity] head={tuple(x0['image'].shape)} left={tuple(x0['wrist_image'].shape)} "
        f"right={tuple(x0['third_image'].shape)} prompt='{x0['prompt'][:80]}'"
    )

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
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

            epoch_loss += float(loss.item())
            epoch_acc += float(acc)
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        train_ce = epoch_loss / max(1, n_batches)
        train_acc = epoch_acc / max(1, n_batches)

        val_metrics = evaluate(model, val_loader, device, criterion)
        print(
            f"[Epoch {epoch+1}] "
            f"Train CE={train_ce:.4f} Acc={train_acc:.4f} | "
            f"Val CE={val_metrics['ce']:.4f} Acc={val_metrics['acc']:.4f} MAE={val_metrics['mae']:.4f}"
        )

        # Save last
        save_torch(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
            },
            output_dir / "last.pt",
        )

        # Save best by MAE
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            save_torch(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )
            print(f"[Best] updated: val_mae={best_val_mae:.4f} -> {output_dir/'best_model.pt'}")

    # Save final
    save_torch(
        {
            "epoch": args.num_epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        output_dir / "final_model.pt",
    )
    print(f"[Done] best_val_mae={best_val_mae:.4f}")
    print(f"[Saved] {output_dir}")


def main():
    parser = argparse.ArgumentParser("Minimal VF training on LeRobot v2.1 (3 cameras)")

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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
