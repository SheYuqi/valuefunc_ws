# dataset.py
# ============================================================
# Official LeRobotDataset wrapper:
# - uses lerobot.datasets.lerobot_dataset.LeRobotDataset for:
#   metadata, path templates, episode selection, indexing, video decoding
# - returns: 3 images + prompt (+ optional value_target/value_bin)
# ============================================================

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import SiglipImageProcessor

from valuefunc import value_to_bin

# ----------------------------
# Import official LeRobotDataset (compatible with different lerobot versions)
# ----------------------------
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # newer
except Exception:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # older/fallback


# ----------------------------
# Helpers
# ----------------------------
def _read_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(p: Path) -> List[dict]:
    out: List[dict] = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _is_lerobot_dataset_root(root: Path) -> bool:
    # v2.1 typical: meta/episodes.jsonl + data/ (+ videos/)
    return (root / "meta" / "episodes.jsonl").exists() and (root / "data").exists()


def _find_dataset_roots(data_dir: str) -> List[Path]:
    root = Path(data_dir).resolve()
    if not root.exists():
        raise ValueError(f"data_dir 不存在: {root}")

    if _is_lerobot_dataset_root(root):
        return [root]

    ds: List[Path] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and _is_lerobot_dataset_root(d):
            ds.append(d)

    if not ds:
        raise ValueError(f"在 {root} 下未找到 LeRobot 数据集根目录（需要 meta/episodes.jsonl + data/）")
    return ds


def _parse_episode_id(x: Any) -> int:
    """
    splits.json 里可能是:
      - 12
      - "12"
      - "episode_000012"
      - "R1_Lite_make_tea/12"
    """
    if isinstance(x, int):
        return int(x)
    s = str(x)
    if "/" in s:
        s = s.split("/", 1)[1]
    m = re.findall(r"\d+", s)
    if not m:
        raise ValueError(f"无法解析 episode id: {x}")
    return int(m[-1])


def _load_prompt_len_map(dataset_root: Path) -> Dict[int, Tuple[str, int, bool]]:
    """
    你的 episodes.jsonl schema 示例:
      {"episode_index": 0, "tasks": [...], "length": 636}
    这里我们只用它来提供 prompt/length/success（不用于路径/视频解码）
    """
    ep_file = dataset_root / "meta" / "episodes.jsonl"
    recs = _read_jsonl(ep_file)
    mp: Dict[int, Tuple[str, int, bool]] = {}
    for r in recs:
        try:
            ep_idx = int(r.get("episode_index", r.get("episode_id", r.get("id"))))
        except Exception:
            continue

        # prompt
        prompt = ""
        tasks = r.get("tasks", None)
        if isinstance(tasks, list) and len(tasks) > 0:
            prompt = str(tasks[0]).strip()
        elif isinstance(tasks, str):
            prompt = tasks.strip()
        if not prompt:
            for k in ["prompt", "instruction", "language_instruction", "text", "description"]:
                v = r.get(k, None)
                if isinstance(v, str) and v.strip():
                    prompt = v.strip()
                    break
        if not prompt:
            prompt = dataset_root.name.replace("_", " ")

        # length
        try:
            length = int(r.get("length", 0))
        except Exception:
            length = 0

        success = True
        for k in ["success", "is_success", "succeeded", "done"]:
            if k in r:
                try:
                    success = bool(r[k])
                except Exception:
                    pass
                break

        mp[ep_idx] = (prompt, length, success)
    return mp


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    LeRobotDataset 通常返回 torch.Tensor:
      - [C,H,W] 或 [T,C,H,W]
      - dtype 可能是 uint8 或 float
    这里统一转为 PIL(RGB)
    """
    if img.ndim == 4:
        # delta_timestamps 可能返回 [T,C,H,W]，我们取当前帧（通常 T=1）
        img = img[-1]

    if img.ndim != 3:
        raise ValueError(f"期望 image tensor 维度为 3 或 4，但收到 shape={tuple(img.shape)}")

    x = img.detach().cpu()

    # [C,H,W] -> [H,W,C]
    if x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    else:
        raise ValueError(f"无法识别的通道维: shape={tuple(img.shape)}")

    arr = x.numpy()

    if arr.dtype != np.uint8:
        # 常见情况：float32 in [0,1] 或 [0,255]
        mx = float(arr.max()) if arr.size > 0 else 1.0
        if mx <= 1.5:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return Image.fromarray(arr, mode="RGB")


def _compute_value_target(t: int, episode_len: int, denom: int, is_success: bool, c_fail: float) -> float:

    T = max(1, episode_len - 1)
    t = int(max(0, min(t, T)))
    remaining = T - t
    v_raw = -float(remaining)
    if (t == T) and (not is_success):
        v_raw -= float(c_fail)
    v = v_raw / float(max(1, denom))
    v = max(-1.0, min(0.0, v))
    return float(v)


# ----------------------------
# Pack wrapper (supports multi-dataset under a folder)
# ----------------------------
@dataclass
class _SubDS:
    name: str
    root: Path
    ds: Any  # LeRobotDataset
    prompt_len_map: Dict[int, Tuple[str, int, bool]]
    cum_len: int


class LeRobotV21SigLIPDataset(Dataset):
    """
    A wrapper that:
      - builds official LeRobotDataset(s) from local folder(s)
      - selects episodes by splits.json (single-ds or pack)
      - returns 3 images + prompt (+ optional value_target/value_bin)
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        camera_keys: Sequence[str],
        image_processor: SiglipImageProcessor,
        task_max_len: Optional[Dict[str, int]] = None,
        c_fail: float = 50.0,
        return_value: bool = True,
        return_meta: bool = False,
        video_backend: str = "pyav",
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        strict_split: bool = True,
    ):
        if len(camera_keys) != 3:
            raise ValueError(f"camera_keys 必须正好 3 个，但收到: {len(camera_keys)}")
        if image_processor is None:
            raise ValueError("image_processor 不能为空")

        self.data_dir = str(Path(data_dir).resolve())
        self.split = str(split)
        self.camera_keys = list(camera_keys)
        self.image_processor = image_processor

        self.task_max_len = task_max_len or {}
        self.c_fail = float(c_fail)
        self.return_value = bool(return_value)
        self.return_meta = bool(return_meta)
        self.video_backend = str(video_backend)

        # delta_timestamps:
        # - None: LeRobotDataset returns [C,H,W]
        # - dict: LeRobotDataset returns [T,C,H,W] per key
        # For "3 cameras at current time", you can pass:
        #   {k: [0.0] for k in camera_keys}
        self.delta_timestamps = delta_timestamps

        # load splits.json
        split_info = _read_json(Path(self.data_dir) / "splits.json")
        if not split_info:
            raise FileNotFoundError(f"未找到 splits.json: {Path(self.data_dir) / 'splits.json'}")
        if self.split not in split_info:
            raise KeyError(f"splits.json 缺少 split={self.split}")

        # parse split ids
        raw_ids = split_info[self.split]
        if not isinstance(raw_ids, list) or len(raw_ids) == 0:
            raise ValueError(f"splits.json[{self.split}] 为空或不是 list")

        # discover dataset roots
        roots = _find_dataset_roots(self.data_dir)
        multi = len(roots) > 1

        # group episode indices by dataset (pack supports "ds_name/ep")
        by_ds: Dict[str, List[int]] = {}
        for sid in raw_ids:
            s = str(sid)
            if "/" in s:
                ds_name, ep = s.split("/", 1)
                by_ds.setdefault(ds_name, []).append(_parse_episode_id(ep))
            else:
                if multi:
                    if strict_split:
                        raise ValueError(
                            "检测到多数据集 pack，但 splits.json 中的 episode id 未带 dataset 前缀（例如 'R1_Lite_make_tea/32'）。"
                        )
                    # fallback: assign to first dataset
                    ds_name = roots[0].name
                    by_ds.setdefault(ds_name, []).append(_parse_episode_id(s))
                else:
                    ds_name = roots[0].name
                    by_ds.setdefault(ds_name, []).append(_parse_episode_id(s))

        # build sub datasets
        subs: List[_SubDS] = []
        cum = 0
        for r in roots:
            ds_name = r.name
            eps = by_ds.get(ds_name, [])
            if len(eps) == 0:
                continue

            prompt_len_map = _load_prompt_len_map(r)

            # Create official LeRobotDataset (local root + episodes + backend)
            kwargs = dict(
                root=r,
                episodes=eps,
                video_backend=self.video_backend,
            )
            if self.delta_timestamps is not None:
                kwargs["delta_timestamps"] = self.delta_timestamps

            # local-only: different versions may or may not support it
            try:
                kwargs["local_only"] = True
                ds_obj = LeRobotDataset(ds_name, **kwargs)
            except TypeError:
                kwargs.pop("local_only", None)
                ds_obj = LeRobotDataset(ds_name, **kwargs)

            cum += len(ds_obj)
            subs.append(_SubDS(name=ds_name, root=r, ds=ds_obj, prompt_len_map=prompt_len_map, cum_len=cum))

        if not subs:
            raise ValueError(f"split={self.split} 没有匹配到任何 dataset episodes（检查 splits.json 前缀和目录名是否一致）")

        self._subs = subs
        self._total = subs[-1].cum_len

        print(
            f"[LeRobotV21SigLIPDataset] split={self.split} sub_datasets={len(self._subs)} total_frames={self._total} "
            f"camera_keys={self.camera_keys} video_backend={self.video_backend}"
        )

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int) -> Tuple[_SubDS, int]:
        if idx < 0:
            idx = self._total + idx
        if idx < 0 or idx >= self._total:
            raise IndexError(f"idx out of range: {idx} (len={self._total})")
        for sub in self._subs:
            start = 0 if sub is self._subs[0] else self._subs[self._subs.index(sub) - 1].cum_len
            if idx < sub.cum_len:
                return sub, idx - start
        # should never reach
        return self._subs[-1], idx - (self._subs[-2].cum_len if len(self._subs) > 1 else 0)

    def _encode_siglip(self, pil_im: Image.Image) -> torch.Tensor:
        out = self.image_processor(images=pil_im.convert("RGB"), return_tensors="pt")
        return out["pixel_values"][0]  # [3,H,W]

    def __getitem__(self, idx: int) -> Dict:
        sub, j = self._locate(idx)
        sample = sub.ds[j]  # official LeRobotDataset provides tensors + metadata

        # 3 images
        imgs: List[torch.Tensor] = []
        for k in self.camera_keys:
            if k not in sample:
                raise KeyError(
                    f"LeRobotDataset sample 缺少相机 key='{k}'. "
                    f"实际 keys={list(sample.keys())[:20]}..."
                )
            pil = _tensor_to_pil(sample[k])
            imgs.append(self._encode_siglip(pil))

        # prompt: prefer episodes.jsonl mapping (stable), fallback to sample fields
        ep_idx = None
        for kk in ["episode_index", "episode_id"]:
            if kk in sample:
                try:
                    ep_idx = int(sample[kk])
                    break
                except Exception:
                    pass

        prompt = ""
        ep_len = 0
        is_success = True
        if ep_idx is not None and ep_idx in sub.prompt_len_map:
            prompt, ep_len, is_success = sub.prompt_len_map[ep_idx]
        else:
            # fallback if dataset provides language/task string in-row
            for kk in ["task", "language_instruction", "instruction", "prompt", "text"]:
                v = sample.get(kk, None)
                if isinstance(v, str) and v.strip():
                    prompt = v.strip()
                    break
            if not prompt:
                prompt = sub.name.replace("_", " ")

        # frame index (within episode)
        t = 0
        for kk in ["frame_index", "frame_idx", "timestep", "step_index"]:
            if kk in sample:
                try:
                    t = int(sample[kk])
                    break
                except Exception:
                    pass

        out = {
            "image": imgs[0],
            "wrist_image": imgs[1],
            "third_image": imgs[2],
            "prompt": prompt,
        }

        if self.return_value:
            # denom
            # task_max_len keyed by your own task_name; here我们没有稳定 task_name，
            # 所以最稳妥：用 episode_len 自回退
            denom = max(1, max(2, ep_len) - 1) if ep_len > 0 else 1
            v = _compute_value_target(t=t, episode_len=max(2, ep_len), denom=denom, is_success=is_success, c_fail=self.c_fail)
            out["value_target"] = torch.tensor(v, dtype=torch.float32)
            out["value_bin"] = torch.tensor(value_to_bin(v), dtype=torch.long)

        if self.return_meta:
            meta = {
                "dataset_name": sub.name,
                "local_index": j,
                "episode_index": ep_idx if ep_idx is not None else -1,
                "frame_idx": t,
                "episode_len": ep_len,
            }
            # expose original keys for debugging
            for kk in ["timestamp", "task_index"]:
                if kk in sample:
                    meta[kk] = sample[kk]
            out.update(meta)

        return out
