
# SigLIP + Gemma Value Function 

本目录提供 HDF5 episode 数据的 **划分 / 训练 / 评估** 脚本。

## 文件说明
- `config.py`：全局配置（相机/预处理/默认超参等）
- `episode.py`：Episode 结构与读写（元信息/帧索引等）
- `dataset.py`：Dataset/DataLoader 封装
- `valuefunc.py`：VF 模型与 value/bin 等工具
- `split.py`：划分 train/val/test
- `train.py`：训练入口
- `eval.py`：评估入口（可导出视频/计算 I_t 相关统计）

---

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
