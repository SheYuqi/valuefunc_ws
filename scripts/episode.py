import random

# ============================================
# 加载任务和 episode 工具函数
# ============================================
import json
from pathlib import Path
from typing import List, Dict, Optional,Tuple

def load_prompt_from_instructions(ep_dir: Path) -> Optional[Dict]:
    p = ep_dir / "instructions.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def norm_relpath(root: Path, p: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")

def scan_episodes(data_dir: str, strict_prompt: bool = True) -> List[Dict]:
    """
    扫描 data/*/episode*/data.hdf5
    返回 episodes: [{episode_dir, prompt, task_name, is_success}]
    """
    tasks_root = Path(data_dir)
    if not tasks_root.exists():
        raise ValueError(f"找不到 tasks 目录: {tasks_root}（期望 data/...）")

    episodes: List[Dict] = []
    h5_list = sorted(tasks_root.glob("*/episode*/data.hdf5"))
    if not h5_list:
        raise ValueError(f"未找到任何 episode HDF5: {tasks_root}/<task>/episode*/data.hdf5")

    for h5 in h5_list:
        ep_dir = h5.parent
        task_dir = ep_dir.parent
        task_name = task_dir.name

        instr = load_prompt_from_instructions(ep_dir)
        if instr is None:
            if strict_prompt:
                raise ValueError(f"缺少或无法解析 instructions.json: {ep_dir/'instructions.json'}")
            else:
                continue

        prompt = instr.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            if strict_prompt:
                raise ValueError(f"instructions.json 缺少有效 prompt: {ep_dir/'instructions.json'}")
            else:
                continue
        prompt = prompt.strip()


        is_success = instr.get("success")

        episodes.append({
            "episode_dir": str(ep_dir),
            "prompt": prompt,
            "task_name": task_name,
            "success": is_success,
        })

    return episodes




# ============================================
# 数据集划分
# ============================================
def relpath_under(root: Path, p: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")

def split_episodes_by_task(all_eps: List[Dict], train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by_task = {}
    for e in all_eps:
        by_task.setdefault(e["task_name"], []).append(e)

    train, test, val = [], [], []
    for task_name, eps in by_task.items():
        rng.shuffle(eps)
        if len(eps) <= 1:
            train += eps
            continue
        n_train = max(1, int(len(eps) * train_ratio))
        n_val = max(1, int(len(eps) * val_ratio))
        train += eps[:n_train]
        val += eps[n_train:n_train + n_val]
        test += eps[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(test)
    rng.shuffle(val)
    return {"train": train, "test": test, "val": val}



def find_all_episodes(data_dir: str) -> List[str]:
    """查找所有 episode 目录"""
    data_path = Path(data_dir)
    episode_dirs = []
    
    # 查找所有包含 data.hdf5 的 episode* 目录
    for d in sorted(data_path.iterdir()):
        if d.is_dir() and d.name.startswith("episode"):
            hdf5_path = d / "data.hdf5"
            if hdf5_path.exists():
                episode_dirs.append(str(d))
    
    return episode_dirs


def check_dataset_split(data_dir: str) -> Tuple[bool, Optional[Dict]]:
    data_path = Path(data_dir)
    split_file = data_path / "split_info.txt"
    if not split_file.exists():
        return False, None

    split_rel = {"train": [], "test": [],"val": []}
    current = None
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line in ["[train]", "[test]", "[val]"]:
            current = line[1:-1]
            continue
        if not line or current is None:
            continue
        split_rel[current].append(line)

    if len(split_rel["train"]) == 0:
        return False, None

    all_eps = scan_episodes(str(data_path))  # data_dir
    index = {relpath_under(data_path, Path(e["episode_dir"])): e for e in all_eps}

    split = {"train": [], "test": [], "val": []}
    for sp in ["train", "test", "val"]:
        for rel in split_rel[sp]:
            if rel in index:
                split[sp].append(index[rel])
            else:
                print(f"警告: split 中的 {rel} 不在扫描结果里，已跳过")

    print(f"数据集已划分: train={len(split['train'])}, test={len(split['test'])}, val={len(split['val'])}")
    return True, split




def split_dataset_episodes(
    data_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    force: bool = False,
) -> Dict[str, List[Dict]]:
    data_path = Path(data_dir)
    is_split, split_info = check_dataset_split(data_dir)
    if not force and is_split:
        print("数据集已经划分好，使用现有划分")
        return split_info

    all_eps = scan_episodes(str(data_path))

    print(f"共找到 {len(all_eps)} 个 episode (跨任务)")

    split = split_episodes_by_task(all_eps, train_ratio=train_ratio, seed=seed)

    split_file = data_path / "split_info.txt"
    with split_file.open("w", encoding="utf-8") as f:
        for sp in ["train", "test", "val"]:
            f.write(f"[{sp}]\n")
            for e in split[sp]:
                rel = relpath_under(data_path, Path(e["episode_dir"]))
                f.write(rel + "\n")
            f.write("\n")

    print(f"划分结果: train={len(split['train'])}, test={len(split['test'])}, val={len(split['val'])}")
    print(f"划分信息已保存到: {split_file}")
    return split

