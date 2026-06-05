from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class ComplexityScheduler:
    threshold: float
    min_threshold: float
    max_threshold: float
    step_size: float
    target_score: float
    patience: int
    wait: int = 0

    def update(self, score: float) -> None:
        if score >= self.target_score:
            self.threshold = min(self.threshold + self.step_size, self.max_threshold)
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.threshold = max(self.threshold - self.step_size, self.min_threshold)
            self.wait = 0

    def weights(self, scores: torch.Tensor) -> torch.Tensor:
        below = scores <= self.threshold
        weights = torch.ones_like(scores)
        weights[~below] = torch.exp(-(scores[~below] - self.threshold) * 5.0)
        return weights


def compute_batch_complexity(batch: Dict[str, object]) -> torch.Tensor:
    fault_weights: List[float] = []
    for fault_type in batch["fault_types"]:
        fault_weights.append(
            {
                "cpu": 0.25,
                "mem": 0.25,
                "delay": 0.5,
                "loss": 0.5,
                "disk": 0.65,
                "socket": 0.65,
            }.get(str(fault_type), 0.75)
        )

    complexity = []
    for index in range(len(batch["fault_types"])):
        modality_count = 0
        seq_lengths = []
        scope_sizes = []
        for modality in ("metrics", "logs", "traces"):
            mask = batch.get(f"{modality}_mask")
            tensor = batch.get(modality)
            if mask is not None and tensor is not None and int(mask[index].item()) == 1:
                modality_count += 1
                seq_lengths.append(int(batch[f"{modality}_lengths"][index].item()))
                scope_sizes.append(int(tensor.size(-1)))
        modal_weight = modality_count / 3.0
        temporal_weight = min((max(seq_lengths) if seq_lengths else 1) / 1000.0, 1.0)
        scope_weight = min((max(scope_sizes) if scope_sizes else 1) / 100.0, 1.0)
        complexity.append((fault_weights[index] + modal_weight + temporal_weight + scope_weight) / 4.0)

    return torch.tensor(complexity, dtype=torch.float32)

