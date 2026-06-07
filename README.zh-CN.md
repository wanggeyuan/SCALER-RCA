# SCALER-RCA

[English](README.md) | [简体中文](README.zh-CN.md)

这是 ICWS 论文 `SCALER: LLM-based Cross-Modal Alignment for Microservice Root Cause Analysis` 的开源仓库。

仓库内容只保留与终稿论文直接对应的实现：

- metrics、logs、traces 三种模态的编码器
- 基于冻结文本锚点的语义对齐模块
- 用于服务级排序的动态融合模块
- 复杂度感知的课程学习模块
- 主实验与消融实验脚本

基线复现代码、论文草稿、画图草稿、无关实验残留等内容均不包含在本仓库中。

## 最终复现实验

下面的命令用于复现本仓库最终 SCALER 实验结果。建议使用 CUDA GPU；CPU/MPS 可以跑通代码，但完整实验会慢很多。

1. 先准备 RCAEval 数据。

如果你本机已经有 RCAEval 数据：

```bash
./run_scaler.sh prepare-data --source-dir /path/to/RCAEval/data
```

如果你本机还没有 RCAEval 数据：

```bash
./download_rcaeval.sh --target-dir ./data/rcaeval
```

2. 运行本地测试：

```bash
./run_scaler.sh smoke
```

3. 运行最终 full 模型：

```bash
./run_scaler.sh train \
  --config configs/experiments/final_full.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/final_ablation/full \
  --device cuda
```

最终配置使用 seed 42、70/15/15 训练/验证/测试划分、batch size 16、最多 100 个 epoch、基于验证集 early stopping，并启用语义对齐、动态融合和课程学习。

4. 运行规定的三组消融：

```bash
./run_scaler.sh train \
  --config configs/experiments/final_no_semantic_alignment.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/final_ablation/no_semantic_alignment \
  --device cuda

./run_scaler.sh train \
  --config configs/experiments/final_no_dynamic_fusion.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/final_ablation/no_dynamic_fusion \
  --device cuda

./run_scaler.sh train \
  --config configs/experiments/final_no_curriculum_learning.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/final_ablation/no_curriculum_learning \
  --device cuda
```

5. 汇总最终指标：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/final_ablation")
variants = [
    "full",
    "no_semantic_alignment",
    "no_dynamic_fusion",
    "no_curriculum_learning",
]
print("variant\tbest_epoch\tPR@1\tPR@3\tPR@5\tMRR\tMAP@3\tMAP@5")
for variant in variants:
    payload = json.loads((root / variant / "metrics.json").read_text())
    metrics = payload["test_metrics"]
    print(
        "\t".join(
            [
                variant,
                str(payload["best_epoch"]),
                f"{metrics['PR@1']:.4f}",
                f"{metrics['PR@3']:.4f}",
                f"{metrics['PR@5']:.4f}",
                f"{metrics['MRR']:.4f}",
                f"{metrics['MAP@3']:.4f}",
                f"{metrics['MAP@5']:.4f}",
            ]
        )
    )
PY
```

6. 重新评估保存的 checkpoint：

```bash
./run_scaler.sh evaluate \
  --checkpoint outputs/final_ablation/full/scaler.pt \
  --data-root ./data/rcaeval \
  --output-dir outputs/final_eval/full \
  --max-eval-batches 9999 \
  --device cuda
```

seed 42 的 full 模型预期结果约为 `PR@1 = 43.12%`、`MRR = 62.76%`。不同硬件、PyTorch 和 CUDA 版本可能带来很小的浮点差异。

## 数据目录要求

RCAEval 数据目录应满足以下结构：

```text
<root>/
  RE1/
  RE2/
  RE3/
```

你可以通过两种方式准备数据：

1. 复用你已有的 RCAEval 数据目录，并接入当前仓库：

```bash
./run_scaler.sh prepare-data --source-dir /path/to/RCAEval/data
```

2. 直接在当前仓库里下载官方 RCAEval 数据：

```bash
./download_rcaeval.sh --target-dir ./data/rcaeval
```

下载脚本使用的是与官方 RCAEval 项目一致的 RE1/RE2/RE3 Zenodo 数据链接。完整下载需要较长时间，并需要数 GB 以上磁盘空间。

## 云服务器运行方式

完整实验建议在服务器上 clone 仓库、准备 RCAEval 数据，然后用后台方式运行上面的最终配置：

```bash
mkdir -p outputs/final_ablation/full
nohup ./run_scaler.sh train \
  --config configs/experiments/final_full.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/final_ablation/full \
  --device cuda > outputs/final_ablation/full/nohup.log 2>&1 &
```

训练日志也会写入：

```bash
outputs/final_ablation/full/train.log
```

可以用下面的命令实时查看进度：

```bash
tail -f outputs/final_ablation/full/train.log
```

## 仓库范围

当前版本只关注：

- SCALER 主实验
- SCALER 消融实验
- 与论文指标对应的自动化结果汇总

不包含对比基线实验代码。

## 说明

- `outputs/` 用于本地运行产物，不会提交到 GitHub。
- `run_scaler.sh` 会自动创建并使用本地 `.venv` 虚拟环境。
- 默认文本语义锚点使用公开 Hugging Face 模型 `bert-base-uncased`；如果本地无法加载 transformer 权重，会自动回退到仓库内置的哈希文本编码器。
