
# SigLIP + Gemma Value Function 

本目录提供 HDF5 episode 数据的 **划分 / 训练 / 评估** 脚本。

## 文件说明
- `config.py`：全局配置（相机/预处理/默认超参等）
- `episode.py`：Episode 结构与读写
- `dataset.py`：Dataset/DataLoader 封装
- `valuefunc.py`：VF 模型与 value/bin 等工具
- `split.py`：划分 train/val/test
- `train.py`：训练入口
- `eval.py`：评估入口（可导出视频/计算 I_t 相关统计）

---

## 数据格式（目录结构 + instructions.json）

**数据集参考https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData/tree/main/Organize_Clutter**

**数据处理参考https://github.com/genrobot-ai/das-datakit**

数据按 **task → episode** 组织；每个 episode 包含 `data.hdf5` 与 `instructions.json`以及 `camera/` 目录(之后可继续添加其他传感器数据)：

```text
data_dir/
└── <task_name>/
    └── episode_xx/
        ├── camera/              # 相机相关资源（图片格式）
        ├── data.hdf5            # 主数据（HDF5：图像/状态/动作等，key 以代码为准）
        └── instructions.json    # 指令/元信息
````

`instructions.json` 示例（字段说明）：

```json
{
  "episode_dir": "/abs/path/to/.../<task_name>/episode_xx",
  "task_name": "clean_bowl",
  "prompt": "Clean the bowl.",
  "success": true
}
```

* `episode_dir`：该 episode 的目录路径（用于定位数据文件）
* `task_name`：任务名（对应 `data_dir/<task_name>/`）
* `prompt`：语言指令（VF 的文本输入/条件）
* `success`：该 episode 是否成功（用于监督/评估；失败轨迹可能额外施加 `--c_fail` 惩罚）


## 快速开始

### 1) 数据集划分
```bash
python scripts/split.py --data_dir /path/to/data 

# 若已存在 split 仍要重划分
python scripts/split.py --data_dir /path/to/data --force
````

### 2) 训练

```bash
python scripts/train.py --data_dir data --camera_type fisheye 
```

常用可选项：

* 冻结 backbone：

```bash
python scripts/train.py ... --freeze_vision
python scripts/train.py ... --freeze_llm
```

* 从断点继续：

```bash
python scripts/train.py ... --resume /path/to/last.pt
```

* 失败惩罚：

```bash
python scripts/train.py ... --c_fail 100.0
```
---

### 3）评估（eval.py）

#### 评估

```bash
python scripts/eval.py \
  --episode_path /path/to/data/<task>/episode_xxxxx \
  --output_dir ./eval_out \
  --checkpoint /path/to/ckpt_best.pt \
  --save_video 
#可选是否导出视频
```


#### I_t / advantage 相关参数

```bash
python scripts/eval.py ... --adv_n 5 --it_percentile 30

python eval.py \
  --data_dir /home/ubuntu/Sheyuqi/data/data \
  --ckpt checkpoints/run_20260127_112753/best_model.pt \
  --batch_size 8 \
  --num_workers 4 \
  --amp \
  --no_lm_head

```

---

## 参数速查

* `--data_dir`：数据根目录（包含各任务子目录/episode*）
* `--output_dir`：输出目录
* `--checkpoint`：模型权重路径
* `--episode_path`：单 episode 评估路径
* `--camera_type {fisheye,depth}`：相机类型
* `--save_video`：生成评估视频

---

python eval_lerobot.py \
  --data_dir /home/ubuntu/Sheyuqi/data/data \
  --ckpt checkpoints/run_20260127_112753/best_model.pt \
  --split val \
  --out_dir /path/to/checkpoints/eval_vis \
  --save_video \
  --video_fps 20 \
  --adv_n 50 \
  --it_percentile 30 \
  --no_lm_head \
  --amp

python eval_single_lerobot.py \
  --data_dir /home/ubuntu/Sheyuqi/data/data \
  --ckpt checkpoints/run_20260127_112753/best_model.pt \
  --split val \
  --episode_id 000000 \
  --out_dir checkpoints/run_20260127_112753/ \
  --adv_n 50 \
  --it_percentile 30 \
  --save_video \
  --video_fps 20 \
  --amp
