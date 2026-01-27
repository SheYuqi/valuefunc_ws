import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict


# ==========================================================
# JSONL helpers
# ==========================================================
def _read_jsonl(p: Path) -> List[dict]:
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _safe_read_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_root_dir(root: Path) -> Path:
    # splits.json 要写在 dataset 根目录，不需要 meta/
    root.mkdir(parents=True, exist_ok=True)
    return root



# ==========================================================
# Detect dataset roots
# ==========================================================
def _is_lerobot_dataset_root(root: Path) -> bool:

    return (root / "meta" / "episodes.jsonl").exists() and (root / "data").exists()


def _find_dataset_roots(data_dir: str) -> List[Path]:

    root = Path(data_dir)
    if not root.exists():
        raise ValueError(f"data_dir 不存在: {root}")

    if _is_lerobot_dataset_root(root):
        return [root]

    # 聚合目录：扫描一级子目录
    ds = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and _is_lerobot_dataset_root(d):
            ds.append(d)

    if not ds:
        raise ValueError(
            f"在 {root} 下未找到任何 LeRobot 数据集根目录（需要 episodes.jsonl + data/）"
        )
    return ds


# ==========================================================
# Field extraction (robust)
# ==========================================================
def _task_id_to_name(x: Any) -> str:
    # 兼容 task_id = "task-0027" / 27 / "27" / "pour_coffee_beans" 等
    if x is None:
        return "task-unknown"
    s = str(x)
    if s.startswith("task-"):
        return s
    try:
        i = int(s)
        return f"task-{i:04d}"
    except Exception:
        return s


def _extract_task_prompt_map(dataset_root: Path) -> Dict[str, str]:

    meta = dataset_root / "meta"
    task_file = meta / "tasks.jsonl"
    task_prompt_map: Dict[str, str] = {}

    if not task_file.exists():
        return task_prompt_map

    for r in _read_jsonl(task_file):
        tid = r.get("task_name", None)
        if tid is None:
            tid = r.get("task_id", r.get("task_index", r.get("id", r.get("name"))))
        tname = _task_id_to_name(tid)

        txt = r.get(
            "prompt",
            r.get("instruction", r.get("description", r.get("language_instruction", r.get("text", "")))),
        )
        if isinstance(txt, str) and txt.strip():
            task_prompt_map[tname] = txt.strip()

    return task_prompt_map


def _extract_prompt(ep_rec: dict, task_prompt_map: Dict[str, str], default_prompt: str) -> str:
    # 1) episode 记录里直接有
    for k in ["prompt", "instruction", "language_instruction", "text", "task_description", "description"]:
        v = ep_rec.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 2) episode 里有 task_id/task_index -> tasks.jsonl 映射
    tid = ep_rec.get("task_name", None)
    if tid is None:
        tid = ep_rec.get("task_id", ep_rec.get("task_index", ep_rec.get("task")))
    tname = _task_id_to_name(tid)
    if tname in task_prompt_map and task_prompt_map[tname].strip():
        return task_prompt_map[tname].strip()

    return default_prompt


def _extract_success(ep_rec: dict, default: bool = True) -> bool:
    for k in ["success", "is_success", "succeeded", "done"]:
        if k in ep_rec:
            try:
                return bool(ep_rec[k])
            except Exception:
                pass
    return default


def _extract_episode_id(ep_rec: dict, fallback_idx: int) -> str:
    eid = ep_rec.get("episode_id", ep_rec.get("episode_index", ep_rec.get("id", fallback_idx)))
    return str(eid)


def _extract_episode_relpath(ep_rec: dict) -> Optional[str]:
    # 有些数据集会提供具体 parquet 路径（相对 root）
    for k in ["episode_path", "path", "file_path", "data_path", "parquet_path", "filename"]:
        v = ep_rec.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _load_dataset_episodes(dataset_root: Path) -> List[Dict]:

    meta = dataset_root / "meta"
    ep_file = meta / "episodes.jsonl"
    ep_recs = _read_jsonl(ep_file)
    if not ep_recs:
        raise ValueError(f"{dataset_root} 缺少或空的 episodes.jsonl: {ep_file}")

    task_prompt_map = _extract_task_prompt_map(dataset_root)

    dataset_name = dataset_root.name
    default_prompt = dataset_name.replace("_", " ")

    episodes: List[Dict] = []
    for i, r in enumerate(ep_recs):
        eid = _extract_episode_id(r, i)

        # task_name：优先 episode 内字段；否则单任务数据集就用 dataset_name
        tid = r.get("task_name", None)
        if tid is None:
            tid = r.get("task_id", r.get("task_index", r.get("task")))
        task_name = _task_id_to_name(tid) if tid is not None else dataset_name

        prompt = _extract_prompt(r, task_prompt_map, default_prompt=default_prompt)
        success = _extract_success(r, default=True)

        relpath = _extract_episode_relpath(r)
        abspath = None
        if relpath is not None:
            p = dataset_root / relpath
            if p.exists():
                abspath = str(p.resolve())

        episodes.append(
            {
                "dataset_name": dataset_name,
                "dataset_root": str(dataset_root.resolve()),
                "episode_id": eid,
                "task_name": task_name,
                "prompt": prompt,
                "success": success,
                "episode_relpath": relpath,
                "episode_abspath": abspath,
                "raw": r,
            }
        )

    return episodes


# ==========================================================
# Split logic (by task within each dataset)
# ==========================================================
def _split_by_task(
    episodes: List[Dict], train_ratio=0.8, val_ratio=0.1, seed=42
) -> Dict[str, List[str]]:
    """
    分层：每个 task 内 shuffle + split
    返回 episode_id 列表（最稳：split 单位为 episode）
    """
    rng = random.Random(seed)
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for e in episodes:
        by_task[e["task_name"]].append(e)

    train_ids, val_ids, test_ids = [], [], []
    for task_name, eps in by_task.items():
        rng.shuffle(eps)
        n = len(eps)

        if n <= 2:
            # 太少：全部进 train，避免 val 为空导致训练端报错
            train_ids.extend([e["episode_id"] for e in eps])
            continue

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        # 保证不越界
        n_train = min(n_train, n - 1)
        n_val = min(n_val, n - n_train)

        train_ids.extend([e["episode_id"] for e in eps[:n_train]])
        val_ids.extend([e["episode_id"] for e in eps[n_train : n_train + n_val]])
        test_ids.extend([e["episode_id"] for e in eps[n_train + n_val :]])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _write_dataset_splits_json(
    dataset_root: Path,
    split: Dict[str, List[str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Path:
    meta = _ensure_root_dir(dataset_root)
    out = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "train": split["train"],
        "val": split["val"],
        "test": split.get("test", []),
    }
    p = meta / "splits.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return p


def _combine_splits_for_pack(
    pack_root: Path,
    per_dataset_splits: Dict[str, Dict[str, List[str]]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict:
    """
    把多个 dataset 的 splits 合并到 pack_root/meta/splits.json
    合并后的 episode key 采用："<dataset_name>/<episode_id>"
    """
    def _prefix(ds: str, ids: List[str]) -> List[str]:
        return [f"{ds}/{eid}" for eid in ids]

    train, val, test = [], [], []
    for ds_name, sp in per_dataset_splits.items():
        train.extend(_prefix(ds_name, sp.get("train", [])))
        val.extend(_prefix(ds_name, sp.get("val", [])))
        test.extend(_prefix(ds_name, sp.get("test", [])))

    out = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "train": train,
        "val": val,
        "test": test,
        "datasets": sorted(list(per_dataset_splits.keys())),
    }

    meta = _ensure_root_dir(pack_root)
    (meta / "splits.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ==========================================================
# Public APIs (match your training/split.py expectation)
# ==========================================================
def check_dataset_split(data_dir: str) -> Tuple[bool, Optional[Dict]]:

    roots = _find_dataset_roots(data_dir)

    # 单 dataset
    if len(roots) == 1 and Path(data_dir).resolve() == roots[0].resolve():
        split_json = roots[0] / "splits.json"
        split = _safe_read_json(split_json)
        if not split:
            return False, None
        ok = (len(split.get("train", [])) > 0) and (len(split.get("val", [])) > 0)
        if ok:
            print(
                f"[LeRobot] 数据集已划分: train={len(split['train'])}, val={len(split['val'])}, test={len(split.get('test', []))}"
            )
            return True, split
        return False, None

    # 聚合 root：要求所有 dataset 都已划分
    per_ds = {}
    for r in roots:
        split_json = r / "meta" / "splits.json"
        split = _safe_read_json(split_json)
        if not split or len(split.get("train", [])) == 0 or len(split.get("val", [])) == 0:
            return False, None
        per_ds[r.name] = {"train": split["train"], "val": split["val"], "test": split.get("test", [])}

    # 合并返回（不强依赖 pack_root/meta/splits.json 存在）
    pack_root = Path(data_dir)
    combined = _combine_splits_for_pack(
        pack_root=pack_root,
        per_dataset_splits=per_ds,
        seed=int(per_ds[next(iter(per_ds))].get("seed", 42)) if isinstance(next(iter(per_ds.values())), dict) else 42,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    print(
        f"[LeRobot-Pack] 已划分: datasets={len(per_ds)}, train={len(combined['train'])}, val={len(combined['val'])}, test={len(combined.get('test', []))}"
    )
    return True, combined


def split_dataset_episodes(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    force: bool = False,
) -> Dict:

    roots = _find_dataset_roots(data_dir)
    data_path = Path(data_dir)

    # 单 dataset root
    if len(roots) == 1 and data_path.resolve() == roots[0].resolve():
        split_json = roots[0] / "splits.json"
        if (not force) and split_json.exists():
            split = _safe_read_json(split_json)
            if split and len(split.get("train", [])) > 0 and len(split.get("val", [])) > 0:
                print("[LeRobot] splits.json 已存在，直接使用")
                return split

        episodes = _load_dataset_episodes(roots[0])
        print(f"[LeRobot] {roots[0].name}: episodes={len(episodes)}")

        split = _split_by_task(episodes, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        _write_dataset_splits_json(roots[0], split, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)

        print(
            f"[LeRobot] 写入 splits.json: {roots[0]/'splits.json'} | "
            f"train={len(split['train'])}, val={len(split['val'])}, test={len(split.get('test', []))}"
        )
        return {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "train": split["train"],
            "val": split["val"],
            "test": split.get("test", []),
        }

    # 聚合 root：对每个 dataset 分别 split + 写入；再写合并 splits
    per_ds_splits: Dict[str, Dict[str, List[str]]] = {}
    for r in roots:
        split_json = r / "splits.json"
        if (not force) and split_json.exists():
            split = _safe_read_json(split_json)
            if split and len(split.get("train", [])) > 0 and len(split.get("val", [])) > 0:
                per_ds_splits[r.name] = {
                    "train": split["train"],
                    "val": split["val"],
                    "test": split.get("test", []),
                }
                print(f"[LeRobot] {r.name}: reuse existing splits.json")
                continue

        episodes = _load_dataset_episodes(r)
        print(f"[LeRobot] {r.name}: episodes={len(episodes)}")
        sp = _split_by_task(episodes, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        _write_dataset_splits_json(r, sp, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
        per_ds_splits[r.name] = sp
        print(
            f"[LeRobot] {r.name}: train={len(sp['train'])}, val={len(sp['val'])}, test={len(sp.get('test', []))}"
        )

    combined = _combine_splits_for_pack(
        pack_root=data_path,
        per_dataset_splits=per_ds_splits,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    print(
        f"[LeRobot-Pack] 写入合并 splits: {data_path/'splits.json'} | "
        f"datasets={len(per_ds_splits)}, train={len(combined['train'])}, val={len(combined['val'])}, test={len(combined.get('test', []))}"
    )
    return combined


def split_dataset_to_folders(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    force: bool = False,
):
    """
    兼容你旧 split.py 里调用的函数名。
    对 LeRobot：不做任何 parquet 复制/搬运，只生成 splits.json。
    """
    print(f"[Split] Creating split under: {data_dir} (seed={seed})")
    split_dataset_episodes(
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        force=force,
    )
