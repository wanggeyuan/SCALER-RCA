#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaler.config import SCALERExperimentConfig
from scripts.train_scaler import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCALER ablations.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaler.yaml"))
    parser.add_argument("--ablation-config", type=Path, default=Path("configs/ablation.yaml"))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-cases-per-system", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = SCALERExperimentConfig.from_yaml(args.config)
    if args.epochs is not None:
        base_config.epochs = args.epochs
    variants = json.loads(json.dumps(__import__("yaml").safe_load(args.ablation_config.read_text())["variants"]))
    summary = {}
    for name, overrides in variants.items():
        config = SCALERExperimentConfig.from_yaml(args.config)
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.max_cases_per_system is not None:
            config.max_cases_per_system = args.max_cases_per_system
        if args.max_train_batches is not None:
            config.max_train_batches = args.max_train_batches
        if args.max_eval_batches is not None:
            config.max_eval_batches = args.max_eval_batches
        for key, value in overrides.items():
            setattr(config, key, value)
        config.output_dir = str(args.output_dir / name)
        metrics = run_training(config, args.data_root, Path(config.output_dir), "scaler.pt")
        summary[name] = metrics["test_metrics"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
