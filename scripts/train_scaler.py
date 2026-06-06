#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaler.config import SCALERExperimentConfig
from scaler.data.rcaeval import RCAEvalDataset, collate_rca_batch, create_splits
from scaler.evaluation.metrics import compute_ranking_metrics
from scaler.model import SCALERModel
from scaler.training.curriculum import ComplexityScheduler, compute_batch_complexity
from scaler.utils.device import describe_device, select_device
from scaler.utils.logging import configure_logging
from scaler.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SCALER model.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaler.yaml"))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--max-cases-per-system", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--save-name", type=str, default="scaler.pt")
    return parser.parse_args()


def _evaluate(model: SCALERModel, loader: DataLoader, device: torch.device, max_batches: int | None) -> dict:
    model.eval()
    scores, targets = [], []
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(batch)
            loss_dict = model.compute_losses(outputs, batch)
            losses.append(float(loss_dict["total_loss"].item()))
            scores.append(outputs["service_probs"].cpu().numpy())
            targets.append(batch["service_labels"].cpu().numpy())
    if not scores:
        raise RuntimeError("No evaluation batches were processed.")
    metrics = compute_ranking_metrics(score_matrix=np.concatenate(scores, axis=0), targets=np.concatenate(targets, axis=0))
    metrics["loss"] = float(sum(losses) / len(losses))
    return metrics


def run_training(
    config: SCALERExperimentConfig,
    data_root: str | None,
    output_dir: Path,
    checkpoint_name: str,
    preferred_device: str = "auto",
) -> dict:
    set_seed(config.seed)
    logger = configure_logging(output_dir, "train.log")
    dataset = RCAEvalDataset(
        data_root=data_root,
        stages=config.stages,
        include_modalities=config.include_modalities,
        systems=config.systems,
        max_cases_per_system=config.max_cases_per_system,
    )
    train_idx, val_idx, test_idx = create_splits(dataset, config.train_fraction, config.val_fraction, config.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True, collate_fn=collate_rca_batch)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_rca_batch)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_rca_batch)

    model = SCALERModel(
        input_dims=dataset.input_dims,
        num_services=len(dataset.service_encoder.classes_),
        num_fault_types=len(dataset.fault_encoder.classes_),
        config=config,
    )
    device = select_device(preferred_device)
    logger.info("Using device: %s", describe_device(device))
    logger.info("Loaded %d RCAEval cases", len(dataset))
    # Log text encoder backend and cache status
    anchor_encoder = model.semantic_alignment.anchor_encoder
    logger.info(
        "Text encoder backend: %s (model=%s, allow_download=%s, fallback_reason=%s)",
        anchor_encoder.backend,
        config.text_encoder.model_name,
        config.text_encoder.allow_download,
        getattr(anchor_encoder, "_fallback_reason", None) or "N/A",
    )
    # Log modality availability stats
    modality_counts = {"metrics": 0, "logs": 0, "traces": 0}
    for case in dataset.cases:
        for name in modality_counts:
            if getattr(case, name) is not None:
                modality_counts[name] += 1
    logger.info("Modality coverage: metrics=%d/%d, logs=%d/%d, traces=%d/%d",
                modality_counts["metrics"], len(dataset),
                modality_counts["logs"], len(dataset),
                modality_counts["traces"], len(dataset))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ComplexityScheduler(
        threshold=config.curriculum.initial_threshold,
        min_threshold=config.curriculum.min_threshold,
        max_threshold=config.curriculum.max_threshold,
        step_size=config.curriculum.step_size,
        target_score=config.curriculum.target_score,
        patience=config.curriculum.patience,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        start = perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                break
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch)
            sample_weights = None
            if config.curriculum_enabled and config.curriculum.enabled:
                complexity = compute_batch_complexity(batch).to(device)
                sample_weights = scheduler.weights(complexity)
            loss_dict = model.compute_losses(outputs, batch, sample_weights=sample_weights)
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss_dict["total_loss"].item()))
        val_metrics = _evaluate(model, val_loader, device, config.max_eval_batches)
        scheduler.update(val_metrics["PR@1"])
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(sum(epoch_losses) / max(len(epoch_losses), 1)),
            "val": val_metrics,
            "seconds": perf_counter() - start,
            "curriculum_threshold": scheduler.threshold,
        }
        history.append(epoch_record)
        logger.info(
            "Epoch %d/%d - train_loss %.4f - val_PR@1 %.4f - val_MRR %.4f - %.1fs",
            epoch + 1,
            config.epochs,
            epoch_record["train_loss"],
            val_metrics["PR@1"],
            val_metrics["MRR"],
            epoch_record["seconds"],
        )

    test_metrics = _evaluate(model, test_loader, device, config.max_eval_batches)
    checkpoint_path = output_dir / checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "input_dims": dataset.input_dims,
            "service_classes": dataset.service_encoder.classes_.tolist(),
            "fault_classes": dataset.fault_encoder.classes_.tolist(),
            "history": history,
        },
        checkpoint_path,
    )
    payload = {
        "config": config.to_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    logger.info("Saved checkpoint to %s", checkpoint_path)
    logger.info("Saved metrics to %s", output_dir / "metrics.json")
    return payload


def main() -> None:
    args = parse_args()
    config = SCALERExperimentConfig.from_yaml(args.config)
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.max_train_batches is not None:
        config.max_train_batches = args.max_train_batches
    if args.max_eval_batches is not None:
        config.max_eval_batches = args.max_eval_batches
    if args.max_cases_per_system is not None:
        config.max_cases_per_system = args.max_cases_per_system
    output_dir = Path(config.output_dir)
    run_training(config, args.data_root, output_dir, args.save_name, preferred_device=args.device)


if __name__ == "__main__":
    main()
