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

1. 准备数据：

```bash
./run_scaler.sh prepare-data --source-dir /path/to/RCAEval/data
```

2. 运行基础自检：

```bash
./run_scaler.sh smoke
```

3. 运行 SCALER 主实验：

```bash
./run_scaler.sh train --data-root ./data/rcaeval --epochs 10
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

你可以直接把已有的 RCAEval 数据根目录传给脚本，也可以将其复制或软链接到 `data/rcaeval`。

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

