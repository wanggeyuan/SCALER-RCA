# SCALER-RCA

[English](README.md) | [简体中文](README.zh-CN.md)

SCALER-RCA 是一个面向 RCAEval 数据集的微服务根因定位项目。

给定故障场景下的 metrics、logs 和 traces，SCALER 会对候选服务进行排序，输出最可能的根因服务。这个仓库包含从 RCAEval 数据准备、训练、评估到模块对比的完整运行流程。

主要组件包括：

- metrics、logs、traces 三种模态的编码器
- 基于冻结文本锚点的语义对齐模块
- 用于服务级排序的动态融合模块
- 复杂度感知的课程学习模块
- 训练、评估和模块对比脚本

第三方基线、草稿文件、画图草稿和无关实验残留不包含在本仓库中。

## 快速开始

下面的命令可以跑完整的 SCALER 实验流程。建议使用 CUDA GPU；CPU/MPS 可以跑通代码，但完整实验会慢很多。

先 clone 仓库：

```bash
git clone https://github.com/wanggeyuan/SCALER-RCA.git
cd SCALER-RCA
```

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

3. 训练完整模型：

```bash
./run_scaler.sh train \
  --config configs/experiments/scaler_full.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/scaler_run/full \
  --device cuda
```

该配置使用 seed 42、70/15/15 训练/验证/测试划分、batch size 16、最多 100 个 epoch、基于验证集 early stopping，并启用语义对齐、动态融合和课程学习。

4. 运行模块对比实验：

```bash
./run_scaler.sh train \
  --config configs/experiments/scaler_no_semantic_alignment.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/scaler_run/no_semantic_alignment \
  --device cuda

./run_scaler.sh train \
  --config configs/experiments/scaler_no_dynamic_fusion.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/scaler_run/no_dynamic_fusion \
  --device cuda

./run_scaler.sh train \
  --config configs/experiments/scaler_no_curriculum_learning.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/scaler_run/no_curriculum_learning \
  --device cuda
```

5. 汇总指标：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/scaler_run")
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
  --checkpoint outputs/scaler_run/full/scaler.pt \
  --data-root ./data/rcaeval \
  --output-dir outputs/scaler_eval/full \
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

## 环境说明

建议使用 Python 3.10、3.11 或 3.12。GPU 流程已在 PyTorch 2.5.x 和兼容 CUDA 12.4 的驱动环境下验证。

默认流程使用 `requirements.txt`，以 editable 模式安装当前项目：

```bash
python -m pip install -r requirements.txt
```

如果希望使用更接近 Linux/CUDA 验证环境的锁定依赖版本：

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

## 云服务器运行方式

完整实验建议在服务器上 clone 仓库、准备 RCAEval 数据，然后用后台方式运行上面的训练命令：

```bash
mkdir -p outputs/scaler_run/full
nohup ./run_scaler.sh train \
  --config configs/experiments/scaler_full.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/scaler_run/full \
  --device cuda > outputs/scaler_run/full/nohup.log 2>&1 &
```

训练日志也会写入：

```bash
outputs/scaler_run/full/train.log
```

可以用下面的命令实时查看进度：

```bash
tail -f outputs/scaler_run/full/train.log
```

## 仓库范围

当前版本只关注：

- SCALER 完整模型训练
- 模块对比实验
- 自动化结果汇总

不包含第三方基线对比代码。

## 说明

- `outputs/` 用于本地运行产物，不会提交到 GitHub。
- `run_scaler.sh` 会自动创建并使用本地 `.venv` 虚拟环境。
- 默认文本语义锚点使用公开 Hugging Face 模型 `bert-base-uncased`；如果本地无法加载 transformer 权重，会自动回退到仓库内置的哈希文本编码器。

## 项目文件

- 贡献说明：[CONTRIBUTING.md](CONTRIBUTING.md)
- 安全说明：[SECURITY.md](SECURITY.md)
- 更新日志：[CHANGELOG.md](CHANGELOG.md)
- 发布检查清单：[docs/release-checklist.md](docs/release-checklist.md)
- 引用元数据：[CITATION.cff](CITATION.cff)
- 许可证：[LICENSE](LICENSE)
