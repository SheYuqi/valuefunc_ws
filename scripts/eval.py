#!/usr/bin/env python3
# ============================================================
# eval_single_lerobot.py
# Evaluate SigLIP+Gemma Value Function on ONE LeRobot v2.1 episode
# - Input: LeRobot pack root (--data_dir) + one episode (--episode_id or --episode_index)
# - Outputs:
#   1) PNG: Value + TrueValue + Advantage + eps + I_t strip
#   2) MP4 (optional): 3-camera frames + curves + cursor + per-frame token/I_t/V/A
# - Computes:
#   - MAE / Bin Acc / Corr (pred vs true)
#   - Advantage (n-step): sum r + V_{t+N} - V_t
#   - I_t = (Advantage > eps), eps = percentile(pred_values, it_percentile)
#
# Requirements:
# - LeRobotV21SigLIPDataset must support return_meta=True and provide meta with:
#     episode_id (or similar), step index t (or similar), optionally success flag.
#   If your meta keys differ, edit META_KEY_CANDIDATES below.
#
# Notes:
# - Advantage needs per-step reward r_norm. We derive denom from true_values slope by default.
#   You can override with --denom.
# - If you want failure penalty, meta should include success flag; else success=Unknown => terminal reward 0.
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from font.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from config import NUM_BINS
from valuefunc import SigLIPGemmaValueFunction
from lerobot_dataset import LeRobotV21SigLIPDataset


# -----------------------------
# Meta key candidates (edit if needed)
# -----------------------------
META_KEY_CANDIDATES = {
    "episode_id": ["episode_id", "episode", "traj_id", "trajectory_id", "demo_id", "episode_name"],
    "step": ["t", "step", "frame", "index_in_episode", "step_idx", "time_idx"],
    "episode_len": ["episode_len", "T", "horizon", "traj_len", "trajectory_len", "len"],
    "success": ["success", "is_success", "done_success", "episode_success"],
    "task_name": ["task_name", "task", "task_id", "task_key", "env"],
}


# -----------------------------
# Helpers
# -----------------------------
def autocast_ctx(device: torch.device, use_amp: bool):
    if not use_amp:
        return torch.autocast(device_type="cpu", enabled=False)
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def meta_get(meta: Any, keys: List[str], default=None):
    if not isinstance(meta, dict):
        return default
    for k in keys:
        if k in meta:
            return meta[k]
    return default


def meta_episode_id(meta: Any) -> Optional[str]:
    v = meta_get(meta, META_KEY_CANDIDATES["episode_id"], None)
    return str(v) if v is not None else None


def meta_step(meta: Any) -> Optional[int]:
    v = meta_get(meta, META_KEY_CANDIDATES["step"], None)
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def meta_success(meta: Any) -> Optional[bool]:
    v = meta_get(meta, META_KEY_CANDIDATES["success"], None)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(int(v))
    if isinstance(v, str):
        return v.strip().lower() in ["1", "true", "yes", "y"]
    return None


def meta_task_name(meta: Any, prompt: str) -> str:
    v = meta_get(meta, META_KEY_CANDIDATES["task_name"], None)
    if v is not None:
        return str(v)
    return f"task_{abs(hash(prompt)) % 100000}"


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ("model_state_dict" in ckpt):
        return ckpt
    return {"model_state_dict": ckpt}


def infer_model_kwargs_from_ckpt(ckpt: Dict[str, Any], args) -> Dict[str, Any]:
    cfg = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}

    def pick(key, default):
        v = getattr(args, key, None)
        return v if v is not None else cfg.get(key, default)

    return {
        "num_bins": NUM_BINS,
        "siglip_variant": pick("siglip_variant", "so400m"),
        "gemma_variant": pick("gemma_variant", "gemma3-270m"),
        "freeze_vision": bool(pick("freeze_vision", True)),
        "freeze_llm": bool(pick("freeze_llm", True)),
        "hidden_dim": int(pick("hidden_dim", 512)),
    }


def to_uint8_rgb(img_chw: torch.Tensor, image_processor=None) -> np.ndarray:
    """
    img_chw: [3,H,W], float tensor (likely normalized)
    return: HxWx3 uint8 RGB
    """
    x = img_chw.detach().float().cpu()

    if image_processor is not None and hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
        mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
        std = torch.tensor(image_processor.image_std).view(3, 1, 1)
        x = x * std + mean

    x = x.clamp(0, 1)
    x = (x * 255.0).byte()
    return x.permute(1, 2, 0).numpy()  # HWC RGB


# -----------------------------
# Main episode eval
# -----------------------------
@torch.no_grad()
def eval_one_episode(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ----- load ckpt + build model -----
    ckpt = load_checkpoint(args.ckpt, device)
    model_kwargs = infer_model_kwargs_from_ckpt(ckpt, args)

    model = SigLIPGemmaValueFunction(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # ----- dataset -----
    camera_keys = [args.cam_head_key, args.cam_left_key, args.cam_right_key]
    ds = LeRobotV21SigLIPDataset(
        data_dir=args.data_dir,
        split=args.split,
        camera_keys=camera_keys,
        image_processor=model.image_processor,
        c_fail=args.c_fail,
        return_value=True,
        return_meta=True,   # needed
        video_backend="pyav",
        delta_timestamps=None,
        strict_split=True,
    )
    print(f"[Dataset] split={args.split} len={len(ds)}")

    # ----- collect episode indices -----
    if args.episode_id is None and args.episode_index is None:
        raise ValueError("Must provide --episode_id or --episode_index")

    indices: List[int] = []
    if args.episode_id is not None:
        target_id = str(args.episode_id)
        for i in tqdm(range(len(ds)), desc="Scan dataset", leave=False):
            meta = ds[i].get("meta", None)
            eid = meta_episode_id(meta)
            if eid == target_id:
                indices.append(i)
    else:
        # episode_index: based on first-seen episode_id order
        target_idx = int(args.episode_index)
        order: List[str] = []
        for i in tqdm(range(len(ds)), desc="Scan dataset", leave=False):
            meta = ds[i].get("meta", None)
            eid = meta_episode_id(meta)
            if eid is None:
                continue
            if eid not in order:
                order.append(eid)
            if (len(order) - 1) == target_idx:
                indices.append(i)
            elif len(order) - 1 > target_idx and len(indices) > 0:
                # finished collecting that episode
                break

    if len(indices) == 0:
        raise RuntimeError("No samples found for this episode. Check return_meta=True and meta episode_id fields.")

    # sort by step index if possible
    def _step(i):
        s = meta_step(ds[i].get("meta", None))
        return s if s is not None else 10**18

    indices.sort(key=_step)

    # apply stride and max_steps
    stride = max(1, int(args.video_stride))
    if args.max_steps is not None:
        indices = indices[: int(args.max_steps)]
    indices = indices[::stride]

    # ----- forward per-step -----
    head_imgs: List[np.ndarray] = []
    left_imgs: List[np.ndarray] = []
    right_imgs: List[np.ndarray] = []

    pred_values: List[float] = []
    pred_bins: List[int] = []
    true_values: List[float] = []
    true_bins: List[int] = []
    prompts: List[str] = []
    metas: List[Any] = []

    for idx in tqdm(indices, desc="Forward episode", leave=False):
        sample = ds[idx]
        meta = sample.get("meta", None)
        metas.append(meta)

        prompt = sample["prompt"]
        prompts.append(prompt)

        head = sample["image"].to(device).unsqueeze(0)
        left = sample["wrist_image"].to(device).unsqueeze(0)
        right = sample["third_image"].to(device).unsqueeze(0)

        with autocast_ctx(device, args.amp and device.type == "cuda"):
            logits, v = model(head, left, right, [prompt])
            v = v.view(-1)[0]

        pred_values.append(float(v.item()))
        pred_bins.append(int(logits.argmax(dim=-1).item()))

        tv = float(sample["value_target"].view(-1)[0].item())
        tb = int(sample["value_bin"].view(-1)[0].item())
        true_values.append(tv)
        true_bins.append(tb)

        head_imgs.append(to_uint8_rgb(sample["image"], model.image_processor))
        left_imgs.append(to_uint8_rgb(sample["wrist_image"], model.image_processor))
        right_imgs.append(to_uint8_rgb(sample["third_image"], model.image_processor))

    pred_values_np = np.asarray(pred_values, dtype=np.float32)
    pred_bins_np = np.asarray(pred_bins, dtype=np.int64)
    true_values_np = np.asarray(true_values, dtype=np.float32)
    true_bins_np = np.asarray(true_bins, dtype=np.int64)

    L = len(pred_values_np)
    frames = np.arange(L)

    prompt0 = prompts[0]
    meta0 = metas[0] if len(metas) else None
    eid = meta_episode_id(meta0) or (args.episode_id or f"episode_{args.episode_index}")
    is_success = meta_success(meta0)
    task_name = meta_task_name(meta0, prompt0)

    # ----- denom for advantage reward -----
    if args.denom is not None:
        denom = float(args.denom)
    else:
        # estimate from true_values slope: V(t+1)-V(t) ~= 1/denom
        deltas = true_values_np[1:] - true_values_np[:-1]
        deltas = deltas[(deltas > 1e-6) & np.isfinite(deltas)]
        denom = float(1.0 / np.median(deltas)) if len(deltas) > 0 else float(max(1, L - 1))
    denom = max(1.0, denom)

    # ----- compute advantage & I_t -----
    N = max(1, int(args.adv_n))
    r_norm = np.full((L,), -1.0 / float(denom), dtype=np.float32)

    # terminal reward (if success known)
    if is_success is None:
        r_norm[-1] = 0.0
    else:
        r_norm[-1] = 0.0 if is_success else (-float(args.c_fail) / float(denom))

    V_pad = np.concatenate([pred_values_np, np.asarray([0.0], dtype=np.float32)], axis=0)
    adv = np.zeros((L,), dtype=np.float32)
    for i in range(L):
        n = min(N, L - i)
        adv[i] = float(r_norm[i:i+n].sum() + V_pad[i+n] - V_pad[i])

    it_percentile = max(0.0, min(100.0, float(args.it_percentile)))
    eps = float(np.percentile(pred_values_np, it_percentile))
    It = adv > eps
    adv_token = np.where(It, "Advantage: positive", "Advantage: negative")

    # ----- metrics -----
    mae = float(np.mean(np.abs(pred_values_np - true_values_np)))
    acc = float(np.mean(pred_bins_np == true_bins_np))
    corr = float(np.corrcoef(pred_values_np, true_values_np)[0, 1]) if L > 1 else 0.0
    it_rate = float(It.mean())
    status_text = "Success" if (is_success is True) else ("Failure" if (is_success is False) else "Unknown")

    print("\n========== Episode Eval ==========")
    print(f"episode_id: {eid}")
    print(f"task_name : {task_name}")
    print(f"success   : {status_text}")
    print(f"len       : {L} (stride={stride})")
    print(f"prompt    : {prompt0}")
    print(f"denom     : {denom:.3f} (adv reward scale)")
    print(f"MAE       : {mae:.4f}")
    print(f"Bin Acc   : {acc:.4f}")
    print(f"Corr      : {corr:.4f}")
    print(f"Adv(n={N}) eps(p{it_percentile:.0f})={eps:.6f}, I_t True rate={it_rate:.3f}")
    print("=================================\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- save plot png -----
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(frames, pred_values_np, "r-", label="Pred Value", linewidth=2)
    ax.plot(frames, true_values_np, "g--", label="True Value", linewidth=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.set_ylim(-1.1, 0.1)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(frames, adv, "b-", label=f"Pred Advantage (n={N})", linewidth=1.5, alpha=0.9)
    ax2.axhline(y=eps, color="gray", linestyle=":", linewidth=1.2, alpha=0.8, label=f"eps(p{it_percentile:.0f})")
    ax2.set_ylabel("Advantage")

    it_colors = ["green" if b else "red" for b in It.tolist()]
    ax.scatter(frames, np.full_like(frames, -1.05, dtype=np.float32), c=it_colors, s=10, marker="s", alpha=0.8, label="I_t")

    ax.set_title(f"Value+Advantage+I_t | {task_name} | {eid} | Status: {status_text}")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)

    plt.tight_layout()
    plot_path = out_dir / f"{task_name}_{eid}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] plot: {plot_path}")

    # ----- optional mp4 -----
    if args.save_video:
        video_path = out_dir / f"{task_name}_{eid}.mp4"
        print(f"[Video] generating: {video_path}")

        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(3, 2, figure=fig, width_ratios=[1.0, 1.8], height_ratios=[1, 1, 1])

        ax_head = fig.add_subplot(gs[0, 0])
        ax_left = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[2, 0])
        ax_value = fig.add_subplot(gs[:, 1])

        im_head = ax_head.imshow(head_imgs[0]); ax_head.set_title("Head"); ax_head.axis("off")
        im_left = ax_left.imshow(left_imgs[0]); ax_left.set_title("Left wrist"); ax_left.axis("off")
        im_right = ax_right.imshow(right_imgs[0]); ax_right.set_title("Right wrist"); ax_right.axis("off")

        ax_value.set_xlim(0, L - 1)
        ax_value.set_ylim(-1.1, 0.1)
        ax_value.set_xlabel("Frame")
        ax_value.set_ylabel("Value")
        ax_value.grid(True, alpha=0.3)

        line_pred, = ax_value.plot([], [], "r-", label="Pred Value", linewidth=2)
        line_true, = ax_value.plot([], [], "g--", label="True Value", linewidth=2)
        vline = ax_value.axvline(x=0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)

        ax_adv = ax_value.twinx()
        adv_lo = float(np.percentile(adv, 1))
        adv_hi = float(np.percentile(adv, 99))
        if abs(adv_hi - adv_lo) < 1e-6:
            adv_lo -= 1.0
            adv_hi += 1.0
        pad = 0.1 * (adv_hi - adv_lo)
        ax_adv.set_ylim(adv_lo - pad, adv_hi + pad)
        ax_adv.set_ylabel("Advantage")

        line_adv, = ax_adv.plot([], [], "b-", label=f"Pred Advantage (n={N})", linewidth=1.5, alpha=0.9)
        _ = ax_adv.axhline(y=eps, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)

        it_strip = ax_value.scatter(frames, np.full_like(frames, -1.05, dtype=np.float32),
                                    c=it_colors, s=10, marker="s", alpha=0.8)

        scatter_v = ax_value.scatter([], [], c="red", s=80, zorder=5, edgecolors="white", linewidths=1.5)
        scatter_a = ax_adv.scatter([], [], c="blue", s=60, zorder=5, edgecolors="white", linewidths=1.2)

        info_text = ax_value.text(
            0.02, 0.98, "",
            transform=ax_value.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray"),
        )

        h1, l1 = ax_value.get_legend_handles_labels()
        h2, l2 = ax_adv.get_legend_handles_labels()
        ax_value.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)

        title = fig.suptitle(
            f"Task: {prompt0}\nFrame: 0/{L-1} | Status: {status_text}",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        def update(frame):
            im_head.set_array(head_imgs[frame])
            im_left.set_array(left_imgs[frame])
            im_right.set_array(right_imgs[frame])

            line_pred.set_data(frames[:frame+1], pred_values_np[:frame+1])
            line_true.set_data(frames[:frame+1], true_values_np[:frame+1])
            line_adv.set_data(frames[:frame+1], adv[:frame+1])

            vline.set_xdata([frame, frame])
            scatter_v.set_offsets([[frame, float(pred_values_np[frame])]])
            scatter_a.set_offsets([[frame, float(adv[frame])]])

            it_flag = bool(It[frame])
            token = adv_token[frame]
            info_text.set_text(
                f"{token}\n"
                f"I_t={it_flag} | eps={eps:.6f}\n"
                f"V_pred={pred_values_np[frame]:.4f} | V_true={true_values_np[frame]:.4f}\n"
                f"A_pred(n={N})={adv[frame]:.4f}"
            )
            title.set_text(
                f"Task: {prompt0}\nFrame: {frame}/{L-1} | Status: {status_text} | I_t={it_flag}"
            )
            return (im_head, im_left, im_right, line_pred, line_true, vline, scatter_v, info_text, title, line_adv, scatter_a)

        anim = FuncAnimation(fig, update, frames=L, interval=50, blit=False)
        anim.save(str(video_path), writer="ffmpeg", fps=int(args.video_fps), dpi=100, bitrate=int(args.video_bitrate))
        plt.close()
        print(f"[Saved] video: {video_path}")

    # return for potential upstream usage
    return {
        "episode_id": eid,
        "task_name": task_name,
        "success": status_text,
        "len": L,
        "mae": mae,
        "bin_acc": acc,
        "corr": corr,
        "eps": eps,
        "it_rate": it_rate,
        "plot_path": str(plot_path),
        "video_path": str(out_dir / f"{task_name}_{eid}.mp4") if args.save_video else None,
    }


def main():
    p = argparse.ArgumentParser("Eval one LeRobot episode with Advantage/I_t visualization")

    p.add_argument("--data_dir", type=str, required=True, help="LeRobot pack root")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--amp", action="store_true")

    # model args (fallback; will prefer ckpt['args'] if present)
    p.add_argument("--siglip_variant", type=str, default=None)
    p.add_argument("--gemma_variant", type=str, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--freeze_vision", action="store_true")
    p.add_argument("--freeze_llm", action="store_true")

    # cameras
    p.add_argument("--cam_head_key", type=str, default="observation.images.cam_high_rgb")
    p.add_argument("--cam_left_key", type=str, default="observation.images.cam_left_wrist_rgb")
    p.add_argument("--cam_right_key", type=str, default="observation.images.cam_right_wrist_rgb")

    # choose episode
    p.add_argument("--episode_id", type=str, default=None, help="episode id (requires meta)")
    p.add_argument("--episode_index", type=int, default=None, help="episode order index (requires meta)")

    # advantage / It
    p.add_argument("--adv_n", type=int, default=50)
    p.add_argument("--it_percentile", type=float, default=30.0)
    p.add_argument("--c_fail", type=float, default=50.0)
    p.add_argument("--denom", type=float, default=None, help="override denom for reward scaling; else estimated from true_values slope")

    # controls
    p.add_argument("--max_steps", type=int, default=None, help="limit steps for faster visualization")
    p.add_argument("--video_stride", type=int, default=1, help="take every k-th frame")

    # outputs
    p.add_argument("--out_dir", type=str, default="./eval_outputs")

    # video
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--video_bitrate", type=int, default=2000)

    args = p.parse_args()
    eval_one_episode(args)


if __name__ == "__main__":
    main()
