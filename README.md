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

1. Prepare data.

If you already have RCAEval data:

```bash
./run_scaler.sh prepare-data --source-dir /path/to/RCAEval/data
```

If you do not have RCAEval data yet:

```bash
./download_rcaeval.sh --target-dir ./data/rcaeval
```

2. Run a smoke check:

```bash
./run_scaler.sh smoke
```

3. Train the main SCALER model:

```bash
./run_scaler.sh train --data-root ./data/rcaeval --epochs 10
```

The training script selects `cuda`, `mps`, or `cpu` automatically. On a rented GPU server, you can make the choice explicit:

```bash
./run_scaler.sh train --data-root ./data/rcaeval --epochs 10 --device cuda
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

You can prepare data in two ways:

1. Reuse an existing RCAEval root and link it into this repository:

```bash
./run_scaler.sh prepare-data --source-dir /path/to/RCAEval/data
```

2. Download the official RCAEval datasets into this repository:

```bash
./download_rcaeval.sh --target-dir ./data/rcaeval
```

The download script follows the same RE1/RE2/RE3 Zenodo assets used by the official RCAEval project. Full download takes time and requires several gigabytes of free disk space.

## Running on a Cloud GPU

For a full run, clone the repository on the server, prepare the RCAEval data, and run training in the background:

```bash
nohup ./run_scaler.sh train --data-root ./data/rcaeval --epochs 10 --device cuda > outputs/main/nohup.log 2>&1 &
```

Training logs are also written to:

```bash
outputs/main/train.log
```

You can monitor progress with:

```bash
tail -f outputs/main/train.log
```

The same device option is available for ablation and evaluation runs:

```bash
./run_scaler.sh ablation --data-root ./data/rcaeval --epochs 10 --device cuda
./run_scaler.sh evaluate --checkpoint outputs/main/scaler.pt --data-root ./data/rcaeval --device cuda
```

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
