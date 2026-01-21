import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from font.*")

import matplotlib
matplotlib.use("Agg")

import random
import argparse
import h5py

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from config import CAMERA_CONFIGS, NUM_BINS
from valuefunc import SigLIPGemmaValueFunction, value_to_bin
from episode import check_dataset_split,compute_task_max_len_from_path,load_prompt_from_instructions


# =========================
# 字体
# =========================
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

# =========================
# 断点续训
# =========================
def save_checkpoint_atomic(state: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


# ============================================
# 评估
# ============================================
def evaluate(args):
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    state_dict = checkpoint['model_state_dict']

    camera_type = checkpoint.get('camera_type', args.camera_type)
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

    with h5py.File(hdf5_path, 'r') as f:
        episode_len = int(f['size'][()])
        left_cam_paths = [f[f'camera/color/{left_camera}'][i].decode('utf-8') for i in range(episode_len)]
        right_cam_paths = [f[f'camera/color/{right_camera}'][i].decode('utf-8') for i in range(episode_len)]

    instr = load_prompt_from_instructions(episode_dir)
    if not isinstance(instr, dict):
        raise ValueError(f"评估失败：{episode_dir/'instructions.json'} 解析失败")
    prompt = (instr.get("prompt") or "").strip()
    is_success = bool(instr.get("success", True))
    task_name = instr.get("task_name", Path(episode_dir).parent.name)

    C_FAIL = 50.0
    is_split, split_info = check_dataset_split(args.data_dir)
    train_episodes = split_info["train"]
    val_episodes = split_info["val"]
    test_episodes = split_info["test"]
    all_episodes = train_episodes + val_episodes + test_episodes
    T_task_map = compute_task_max_len_from_path(all_episodes)  # dict: task_name -> max_len
    # for ep in T_task_map:
    #     print(f"[TaskMaxLen] 任务 {ep} 最大步数: {T_task_map[ep]}")
    

    if task_name in T_task_map:
        T_task = int(T_task_map[task_name])
    else:
        # 兜底：用全局最大，避免KeyError
        T_task = int(max(T_task_map.values())) if len(T_task_map) > 0 else int(episode_len)
        print(f"[Eval] 任务 {task_name} 未指定最大步数，使用全局最大值 T_task={T_task}")
    denom = max(1, T_task - 1)
    T = episode_len - 1
    # print(f"[Eval] 任务 {task_name} 的最大步数: {T_task}, 本 episode 长度: {episode_len}, denom={denom}")

    true_values = []
    for t in range(episode_len):
        remaining = T - t
        v = -float(remaining)
        if (t == T) and (not is_success):
            v -= float(C_FAIL)
        v = v / float(denom)
        v = max(-1.0, min(0.0, v))
        true_values.append(v)
    true_values = np.asarray(true_values, dtype=np.float32)

    print(f"轨迹结果: {'成功' if is_success else '失败'}")
    print(f"轨迹长度: {episode_len}")
    print(f"任务描述: {prompt}")
    print(f"任务名称: {task_name} | 任务最大长度 T_task: {T_task} | 归一化分母 denom: {denom}")

    pred_values = []
    pred_bins = []
    left_images = []
    right_images = []

    kept_ts = []

    with torch.no_grad():
        for t in tqdm(range(episode_len), desc="预测"):
            left_img_path = episode_dir / left_cam_paths[t]
            right_img_path = episode_dir / right_cam_paths[t]

            try:
                left_pil = Image.open(left_img_path).convert("RGB")
                right_pil = Image.open(right_img_path).convert("RGB")
            except Exception as e:
                print(f"警告: 跳过帧 {t}（读图失败）: {e}")
                continue

            left_images.append(np.asarray(left_pil))
            right_images.append(np.asarray(right_pil))
            kept_ts.append(t)

            left_tensor = model.image_processor(images=left_pil, return_tensors="pt")["pixel_values"].to(device)
            right_tensor = model.image_processor(images=right_pil, return_tensors="pt")["pixel_values"].to(device)

            logits, value = model(left_tensor, right_tensor, [prompt])
            pred_values.append(float(value.item()))
            pred_bins.append(int(logits.argmax(dim=-1).item()))

    pred_values = np.array(pred_values)
    true_values = np.array(true_values[:len(pred_values)])
    true_bins = np.array([value_to_bin(v) for v in true_values])
    pred_bins = np.array(pred_bins)

    # ===== 计算 Advantage & I_t =====

    N = int(getattr(args, "adv_n", 1))
    N = max(1, N)

    L = len(pred_values)
    r_norm = np.full((L,), -1.0 / float(denom), dtype=np.float32)

    if kept_ts[-1] == T:
        r_norm[-1] = 0.0 if is_success else (-C_FAIL / float(denom))
    else:

        r_norm[-1] = 0.0 if is_success else (-C_FAIL / float(denom))

    # n-step advantage: sum r + V_{t+N} - V_t；超出末尾时，用 0 作为 V_terminal
    V_pad = np.concatenate([pred_values, np.asarray([0.0], dtype=np.float32)], axis=0)
    adv = np.zeros((L,), dtype=np.float32)
    for i in range(L):
        n = min(N, L - i)
        adv[i] = float(r_norm[i:i+n].sum() + V_pad[i+n] - V_pad[i])

    # ===== 阈值 eps 与 I_t =====
    # 论文里 eps_l 是 task 上 value 分布的 30% 分位；
    it_percentile = float(getattr(args, "it_percentile", 30.0))
    it_percentile = max(0.0, min(100.0, it_percentile))

    eps = float(np.percentile(pred_values, it_percentile))

    It = adv > eps  # bool array
    adv_token = np.where(It, "Advantage: positive", "Advantage: negative")

    # ===== 指标输出 =====
    mae = np.mean(np.abs(pred_values - true_values))
    acc = np.mean(pred_bins == true_bins)
    corr = np.corrcoef(pred_values, true_values)[0, 1] if len(pred_values) > 1 else 0.0
    it_rate = float(It.mean())

    print(f"\n评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Bin Accuracy: {acc:.4f}")
    print(f"  Correlation: {corr:.4f}")
    print(f"  Advantage(n={N}) eps(p{it_percentile:.0f})={eps:.6f}, I_t True rate={it_rate:.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    frames = np.arange(len(pred_values))
    ax.plot(frames, pred_values, 'r-', label='Predicted Value', linewidth=2)
    ax.fill_between(frames, pred_values, -1, alpha=0.2, color='red')
    ax.plot(frames, true_values, 'g--', label='True Value', linewidth=2)
    ax.fill_between(frames, true_values, -1, alpha=0.2, color='green')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Value')
    ax.set_ylim(-1.1, 0.1)
    status_text = "Success" if is_success else "Failure"
    ax.set_title(f'Value Function Prediction | Status: {status_text}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========advantage on twin axis==========
    ax2 = ax.twinx()
    ax2.plot(frames, adv, "b-", label=f"Pred Advantage (n={N})", linewidth=1.5, alpha=0.9)
    ax2.axhline(y=eps, color="gray", linestyle=":", linewidth=1.2, alpha=0.8, label=f"eps(p{it_percentile:.0f})")
    ax2.set_ylabel("Advantage")

    # I_t strip (colored squares near bottom of value axis)
    it_colors = ["green" if b else "red" for b in It.tolist()]
    ax.scatter(frames, np.full_like(frames, -1.05, dtype=np.float32), c=it_colors, s=10, marker="s", alpha=0.8, label="I_t")

    status_text = "Success" if is_success else "Failure"
    ax.set_title(f"Value+Advantage+I_t | Status: {status_text} | eps(p{it_percentile:.0f})={eps:.4f}")

    # 合并 legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 , l1 , loc="lower right", fontsize=9)
    # ============================================

    plt.tight_layout()
    output_path = Path(args.checkpoint).parent / f"{task_name}_{episode_dir.name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"评估图保存至: {output_path}")

    if args.save_video:
        print("\n生成评估视频...")
        video_path = Path(args.checkpoint).parent / f"{task_name}_{episode_dir.name}.mp4"

        camera_desc = CAMERA_CONFIGS[camera_type]["description"]
        left_title = f"Left Camera ({camera_desc})"
        right_title = f"Right Camera ({camera_desc})"

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.6], height_ratios=[1, 1])

        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[1, 0])
        ax_value = fig.add_subplot(gs[:, 1])

        im_left = ax_left.imshow(left_images[0])
        ax_left.set_title(left_title, fontsize=12, fontweight="bold")
        ax_left.axis("off")

        im_right = ax_right.imshow(right_images[0])
        ax_right.set_title(right_title, fontsize=12, fontweight="bold")
        ax_right.axis("off")

        # value axis
        ax_value.set_xlim(0, L - 1)
        ax_value.set_ylim(-1.1, 0.1)
        ax_value.set_xlabel("Frame", fontsize=11)
        ax_value.set_ylabel("Value", fontsize=11)
        ax_value.grid(True, alpha=0.3)

        line_pred, = ax_value.plot([], [], "r-", label="Pred Value", linewidth=2)
        line_true, = ax_value.plot([], [], "g--", label="True Value", linewidth=2)
        vline = ax_value.axvline(x=0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)

        # advantage axis (twin)
        ax_adv = ax_value.twinx()
        # 给 advantage 一个合理 y-range（避免极小范围导致看不清）
        adv_lo = float(np.percentile(adv, 1))
        adv_hi = float(np.percentile(adv, 99))
        if abs(adv_hi - adv_lo) < 1e-6:
            adv_lo -= 1.0
            adv_hi += 1.0
        pad = 0.1 * (adv_hi - adv_lo)
        ax_adv.set_ylim(adv_lo - pad, adv_hi + pad)
        ax_adv.set_ylabel("Advantage", fontsize=11)

        line_adv, = ax_adv.plot([], [], "b-", label=f"Pred Advantage (n={N})", linewidth=1.5, alpha=0.9)
        eps_line = ax_adv.axhline(y=eps, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)

        # I_t strip
        it_colors = ["green" if b else "red" for b in It.tolist()]
        it_strip = ax_value.scatter(frames, np.full_like(frames, -1.05, dtype=np.float32),
                                    c=it_colors, s=10, marker="s", alpha=0.8)

        # 当前点
        scatter_v = ax_value.scatter([], [], c="red", s=80, zorder=5, edgecolors="white", linewidths=1.5)
        scatter_a = ax_adv.scatter([], [], c="blue", s=60, zorder=5, edgecolors="white", linewidths=1.2)

        # 文本框（动态显示 token / V / A / I_t）
        info_text = ax_value.text(
            0.02, 0.98, "",
            transform=ax_value.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray")
        )

        # 合并 legend
        h1, l1 = ax_value.get_legend_handles_labels()
        h2, l2 = ax_adv.get_legend_handles_labels()
        ax_value.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)

        status_text = "Success" if is_success else "Failure"
        title = fig.suptitle(
            f"Task: {prompt}\nFrame: 0/{L-1} | Status: {status_text}",
            fontsize=11, fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        def update(frame):
            # images
            im_left.set_array(left_images[frame])
            im_right.set_array(right_images[frame])

            # curves
            line_pred.set_data(frames[:frame+1], pred_values[:frame+1])
            line_true.set_data(frames[:frame+1], true_values[:frame+1])
            line_adv.set_data(frames[:frame+1], adv[:frame+1])

            # cursors / points
            vline.set_xdata([frame, frame])
            scatter_v.set_offsets([[frame, float(pred_values[frame])]])
            scatter_a.set_offsets([[frame, float(adv[frame])]])

            # text
            it_flag = bool(It[frame])
            token = adv_token[frame]
            info_text.set_text(
                f"{token}\n"
                f"I_t={it_flag} | eps={eps:.6f}\n"
                f"V_pred={pred_values[frame]:.4f} | V_true={true_values[frame]:.4f}\n"
                f"A_pred(n={N})={adv[frame]:.4f}"
            )

            title.set_text(
                f"Task: {prompt}\nFrame: {frame}/{L-1} | Status: {status_text} | I_t={it_flag}"
            )
            return im_left, im_right, line_pred, line_true, vline, scatter_v, info_text, title, line_adv, scatter_a

        anim = FuncAnimation(fig, update, frames=L, interval=50, blit=False)
        anim.save(str(video_path), writer="ffmpeg", fps=20, dpi=100, bitrate=2000)
        plt.close()
        print(f"视频保存至: {video_path}")

# ============================================
# 主函数
# ============================================
def main():
    parser = argparse.ArgumentParser(description="SigLIP + Gemma Value Function Training (Pika HDF5)")

    parser.add_argument("--data_dir", type=str, default="./data",
                        help="数据目录（包含 episode* 子目录）")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/value_pika",
                        help="输出目录")

    # 模型配置
    parser.add_argument("--siglip_variant", type=str, default="so400m")
    parser.add_argument("--gemma_variant", type=str, default="gemma3-270m")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hidden_dim", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--seed", type=int, default=42)
    # 相机配置
    parser.add_argument("--camera_type", type=str, default="fisheye",
                        choices=["fisheye", "depth"])

    parser.add_argument("--tb_log_interval", type=int, default=50,
                        help="每隔多少 step 写一次 TensorBoard(train step)")

    # 评估配置
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/run_20260113_205201/best_model.pt", help="模型 checkpoint 路径")
    parser.add_argument("--episode_path", type=str, default="./data/clean_bowl/episode_44", help="评估 episode 目录路径")   # desktop_object_sorting/episode_38 明天跑这个
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--adv_n", type=int, default=50)
    parser.add_argument("--it_percentile", type=float, default=30.0)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    if args.checkpoint is None or args.episode_path is None:
        print("评估模式需要 --checkpoint 和 --episode_path")
        return
    evaluate(args)

if __name__ == "__main__":
    main()