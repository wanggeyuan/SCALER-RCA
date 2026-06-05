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

## 快速开始

1. 先准备数据。

如果你本机已经有 RCAEval 数据：

```bash
./run_scaler.sh prepare-data --source-dir /path/to/RCAEval/data
```

如果你本机还没有 RCAEval 数据：

```bash
./download_rcaeval.sh --target-dir ./data/rcaeval
```

2. 运行基础自检：

```bash
./run_scaler.sh smoke
```

3. 运行 SCALER 主实验：

```bash
./run_scaler.sh train --data-root ./data/rcaeval --epochs 10
```

训练脚本会自动选择 `cuda`、`mps` 或 `cpu`。如果是在租用的 GPU 服务器上运行，也可以显式指定：

```bash
./run_scaler.sh train --data-root ./data/rcaeval --epochs 10 --device cuda
```

4. 运行消融实验：

```bash
./run_scaler.sh ablation --data-root ./data/rcaeval --epochs 10
```

5. 汇总结果：

```bash
./run_scaler.sh summarize
```

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

完整实验建议在服务器上 clone 仓库、准备 RCAEval 数据，然后用后台方式启动训练：

```bash
nohup ./run_scaler.sh train --data-root ./data/rcaeval --epochs 10 --device cuda > outputs/main/nohup.log 2>&1 &
```

训练日志也会写入：

```bash
outputs/main/train.log
```

可以用下面的命令实时查看进度：

```bash
tail -f outputs/main/train.log
```

消融实验和评估脚本同样支持设备参数：

```bash
./run_scaler.sh ablation --data-root ./data/rcaeval --epochs 10 --device cuda
./run_scaler.sh evaluate --checkpoint outputs/main/scaler.pt --data-root ./data/rcaeval --device cuda
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
