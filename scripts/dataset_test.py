#!/usr/bin/env python3
import argparse
import torch

from transformers import SiglipImageProcessor

from lerobot_dataset import LeRobotV21SigLIPDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--siglip", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--video_backend", type=str, default="pyav", choices=["pyav", "decord"])
    args = parser.parse_args()

    # LeRobot 相机 keys（你之前打印出来的）
    camera_keys = [
        "observation.images.cam_high_rgb",
        "observation.images.cam_left_wrist_rgb",
        "observation.images.cam_right_wrist_rgb",
    ]

    image_processor = SiglipImageProcessor.from_pretrained(args.siglip)

    # 如果你希望 LeRobotDataset 输出 [T,C,H,W]，可以启用 delta_timestamps
    # 这里只取当前帧：T=1
    delta_timestamps = {k: [0.0] for k in camera_keys}

    ds = LeRobotV21SigLIPDataset(
        data_dir=args.data_dir,
        split=args.split,
        camera_keys=camera_keys,
        image_processor=image_processor,
        return_value=True,
        return_meta=True,
        video_backend=args.video_backend,  # 关键：AV1 建议用 pyav
        delta_timestamps=delta_timestamps,
    )

    print(f"len(ds)={len(ds)}")

    for i in range(min(args.n, len(ds))):
        x = ds[i]
        print(
            f"[{i}] image={tuple(x['image'].shape)} wrist={tuple(x['wrist_image'].shape)} third={tuple(x['third_image'].shape)} "
            f"value={x['value_target'].item():.4f} bin={int(x['value_bin'])} "
            f"episode_index={x.get('episode_index')} frame={x.get('frame_idx')} prompt='{x['prompt'][:80]}'"
        )


if __name__ == "__main__":
    main()
