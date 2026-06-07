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

## Final Reproduction

The commands below reproduce the final SCALER experiment used by this repository. A CUDA GPU is recommended; CPU/MPS can run the code but will be much slower.

1. Prepare RCAEval data.

If you already have RCAEval data:

```bash
./run_scaler.sh prepare-data --source-dir /path/to/RCAEval/data
```

If you do not have RCAEval data yet:

```bash
./download_rcaeval.sh --target-dir ./data/rcaeval
```

2. Run the local test suite:

```bash
./run_scaler.sh smoke
```

3. Run the final full model:

```bash
./run_scaler.sh train \
  --config configs/experiments/final_full.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/final_ablation/full \
  --device cuda
```

The final configuration uses seed 42, 70/15/15 train/validation/test split, batch size 16, up to 100 epochs, validation-based early stopping, semantic alignment, dynamic fusion, and curriculum learning.

4. Run the required ablations:

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

5. Summarize the final metrics:

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

6. Re-evaluate a saved checkpoint:

```bash
./run_scaler.sh evaluate \
  --checkpoint outputs/final_ablation/full/scaler.pt \
  --data-root ./data/rcaeval \
  --output-dir outputs/final_eval/full \
  --max-eval-batches 9999 \
  --device cuda
```

Expected seed-42 full-model metrics are approximately `PR@1 = 43.12%` and `MRR = 62.76%`. Small floating-point differences can occur across hardware, PyTorch, and CUDA versions.

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

For a full run, clone the repository on the server, prepare RCAEval data, and run the same final commands in the background:

```bash
mkdir -p outputs/final_ablation/full
nohup ./run_scaler.sh train \
  --config configs/experiments/final_full.yaml \
  --data-root ./data/rcaeval \
  --output-dir outputs/final_ablation/full \
  --device cuda > outputs/final_ablation/full/nohup.log 2>&1 &
```

Training logs are also written to:

```bash
outputs/final_ablation/full/train.log
```

You can monitor progress with:

```bash
tail -f outputs/final_ablation/full/train.log
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
