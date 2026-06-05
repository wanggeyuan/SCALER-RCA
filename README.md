# SCALER-RCA

[English](README.md) | [简体中文](README.zh-CN.md)

Official repository for the ICWS paper `SCALER: LLM-based Cross-Modal Alignment for Microservice Root Cause Analysis`.

This repository only keeps the code paths that map directly to the camera-ready paper:

- multi-modal encoders for metrics, logs, and traces
- semantic alignment with a frozen text anchor
- dynamic fusion for service ranking
- complexity-aware curriculum learning
- main experiments and ablation experiments

Baseline re-implementations, paper drafts, plotting scratch files, and unrelated experiment artifacts are intentionally excluded.

## Quick Start

1. Prepare data:

```bash
./run_scaler.sh prepare-data --source-dir /path/to/RCAEval/data
```

2. Run a smoke check:

```bash
./run_scaler.sh smoke
```

3. Train the main SCALER model:

```bash
./run_scaler.sh train --data-root ./data/rcaeval --epochs 10
```

4. Run the ablations:

```bash
./run_scaler.sh ablation --data-root ./data/rcaeval --epochs 10
```

5. Summarize results:

```bash
./run_scaler.sh summarize
```

## Data

The expected RCAEval layout is:

```text
<root>/
  RE1/
  RE2/
  RE3/
```

You can either point to an existing dataset root or copy/symlink it into `data/rcaeval`.

## Scope

This release focuses on:

- SCALER main experiment
- SCALER ablation experiments
- automatic result summarization aligned with the paper metrics

It does not include baseline comparison code.

## Notes

- `outputs/` is for local run artifacts and is not committed to the repository.
- `run_scaler.sh` creates and uses a local `.venv` automatically.
- The default text anchor is a public Hugging Face model (`bert-base-uncased`), with a hashed fallback if transformer weights are unavailable.

