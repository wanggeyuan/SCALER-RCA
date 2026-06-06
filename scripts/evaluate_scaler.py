#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set HF endpoint early, before any imports that might trigger huggingface_hub
try:
    import yaml as _yaml
    _cfg = _yaml.safe_load((ROOT / "configs" / "scaler.yaml").read_text()) or {}
    _hf = (_cfg.get("text_encoder") or {}).get("hf_endpoint", "")
    if _hf:
        os.environ["HF_ENDPOINT"] = _hf
except Exception:
    pass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scaler.config import SCALERExperimentConfig
from scaler.data.rcaeval import RCAEvalDataset, build_service_candidate_sets, collate_rca_batch, create_service_candidate_mask, create_splits
from scaler.evaluation.metrics import compute_ranking_metrics
from scaler.model import SCALERModel
from scaler.utils.device import describe_device, select_device
from scaler.utils.logging import configure_logging
from scaler.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SCALER checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--max-eval-batches", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.output_dir, "evaluation.log")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_dict = checkpoint["config"]
    config = SCALERExperimentConfig.from_yaml(Path("configs/scaler.yaml"))
    for key, value in config_dict.items():
        if key in {"text_encoder", "curriculum"}:
            continue
        setattr(config, key, value)
    set_seed(config.seed)
    dataset = RCAEvalDataset(
        data_root=args.data_root,
        stages=config.stages,
        include_modalities=config.include_modalities,
        systems=config.systems,
        max_cases_per_system=config.max_cases_per_system,
    )
    train_idx, _, test_idx = create_splits(
        dataset,
        config.train_fraction,
        config.val_fraction,
        config.seed,
        stratify_by=config.split_stratify_by,
    )
    loader = DataLoader(Subset(dataset, test_idx), batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_rca_batch)
    candidate_sets = checkpoint.get("service_candidate_sets") or build_service_candidate_sets(dataset, train_idx)
    service_class_weights = torch.tensor(checkpoint.get("service_class_weights", [1.0] * len(checkpoint["service_classes"])))
    model = SCALERModel(dataset.input_dims, len(dataset.service_encoder.classes_), len(dataset.fault_encoder.classes_), config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = select_device(args.device)
    logger.info("Using device: %s", describe_device(device))
    model.to(device)
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_eval_batches is not None and batch_idx >= args.max_eval_batches:
                break
            batch["service_candidate_mask"] = create_service_candidate_mask(
                batch["systems"],
                candidate_sets,
                len(checkpoint["service_classes"]),
            )
            batch["service_class_weights"] = service_class_weights
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(batch)
            scores.append(outputs["service_probs"].cpu().numpy())
            targets.append(batch["service_labels"].cpu().numpy())
    metrics = compute_ranking_metrics(score_matrix=np.concatenate(scores, axis=0), targets=np.concatenate(targets, axis=0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Saved evaluation metrics to %s", args.output_dir / "evaluation.json")


if __name__ == "__main__":
    main()
