#!/usr/bin/env python3

import argparse
from episode import check_dataset_split, split_dataset_episodes


def _print_split(split_info: dict):
    train = split_info.get("train", [])
    val = split_info.get("val", [])
    test = split_info.get("test", [])
    print("[Split] done/loaded:")
    print(f"train: {len(train)} episodes")
    print(f"val  : {len(val)} episodes")
    print(f"test : {len(test)} episodes")


def main():
    parser = argparse.ArgumentParser(description="Split Lerobot episodes into train/val/test")
    parser.add_argument("--data_dir", type=str, default="data", help="数据根目录（包含各任务子目录）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--force", action="store_true", help="即使已经存在 split 也强制重新划分")
    args = parser.parse_args()

    is_split, split_info = check_dataset_split(args.data_dir)

    if is_split and not args.force:
        print(f"[Split] Found existing split under: {args.data_dir} (use --force to recreate)")
        _print_split(split_info)
        return

    print(f"[Split] Creating split under: {args.data_dir} (seed={args.seed})")
    split_info = split_dataset_episodes(args.data_dir, seed=args.seed)
    _print_split(split_info)


if __name__ == "__main__":
    main()
