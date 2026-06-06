from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaler.config import SCALERExperimentConfig
from scaler.models.encoders import LogsEncoder, MetricsEncoder, TracesEncoder
from scaler.models.fusion import DynamicFusion
from scaler.models.semantic_alignment import SemanticAlignmentModule


def _temporal_statistics(x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    if lengths is None:
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    else:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        safe_lengths = lengths.to(x.device).clamp(min=1, max=x.size(1))
        valid = positions >= (x.size(1) - safe_lengths).unsqueeze(1)
    mask = valid.unsqueeze(-1)
    count = mask.sum(dim=1).clamp_min(1)
    mean = (x * mask).sum(dim=1) / count
    variance = (((x - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / count
    maximum = x.masked_fill(~mask, torch.finfo(x.dtype).min).max(dim=1).values
    last = x[:, -1, :]
    return torch.cat([mean, variance.sqrt(), maximum, last], dim=-1)


def _pairwise_cosine_loss(vectors: List[torch.Tensor], masks: List[torch.Tensor] | None = None) -> torch.Tensor:
    if len(vectors) < 2:
        return vectors[0].new_tensor(0.0)
    losses = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            similarities = F.cosine_similarity(vectors[i], vectors[j], dim=-1)
            if masks is not None:
                valid = masks[i].bool() & masks[j].bool()
                if not valid.any():
                    continue
                similarities = similarities[valid]
            losses.append(1.0 - similarities.mean())
    if not losses:
        return vectors[0].new_tensor(0.0)
    return torch.stack(losses).mean()


def _contrastive_loss(vectors: List[torch.Tensor], masks: List[torch.Tensor] | None = None, temperature: float = 0.07) -> torch.Tensor:
    if len(vectors) < 2:
        return vectors[0].new_tensor(0.0)
    normalized = [F.normalize(vec, dim=-1) for vec in vectors]
    losses = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            left, right = normalized[i], normalized[j]
            if masks is not None:
                valid = masks[i].bool() & masks[j].bool()
                if not valid.any():
                    continue
                left, right = left[valid], right[valid]
            logits = left @ right.T / temperature
            targets = torch.arange(logits.size(0), device=logits.device)
            losses.append(F.cross_entropy(logits, targets))
    if not losses:
        return vectors[0].new_tensor(0.0)
    return torch.stack(losses).mean()


class SCALERModel(nn.Module):
    def __init__(self, input_dims: Dict[str, int], num_services: int, num_fault_types: int, config: SCALERExperimentConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.encoders = nn.ModuleDict(
            {
                "metrics": MetricsEncoder(input_dims["metrics"], hidden_dim, dropout=config.dropout),
                "logs": LogsEncoder(input_dims["logs"], hidden_dim, dropout=config.dropout),
                "traces": TracesEncoder(input_dims["traces"], hidden_dim, dropout=config.dropout),
            }
        )
        self.projection = nn.Linear(hidden_dim, config.projection_dim)
        self.semantic_alignment = SemanticAlignmentModule(config.projection_dim, config.text_encoder, dropout=config.dropout)
        self.dynamic_fusion = DynamicFusion(config.projection_dim, dropout=config.dropout)
        self.service_head = nn.Sequential(
            nn.Linear(config.projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, num_services),
        )
        nn.init.zeros_(self.service_head[-1].weight)
        nn.init.zeros_(self.service_head[-1].bias)
        self.fault_head = nn.Sequential(
            nn.Linear(config.projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, num_fault_types),
        )
        self.metrics_statistics_head = (
            nn.Linear(input_dims["metrics"] * 4, num_services)
            if config.metrics_statistics_head_enabled
            else None
        )

    def encode_modalities(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        encoded = {}
        for name, encoder in self.encoders.items():
            if name not in batch:
                continue
            mask = batch[f"{name}_mask"].to(batch[name].device).unsqueeze(-1)
            encoded[name] = encoder(batch[name], batch.get(f"{name}_lengths")) * mask
        return encoded

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(batch)
        modality_masks = {name: batch[f"{name}_mask"].to(tensor.device) for name, tensor in encoded.items()}
        projected = {
            name: F.normalize(self.projection(tensor), dim=-1) * modality_masks[name].unsqueeze(-1)
            for name, tensor in encoded.items()
        }

        if self.config.semantic_alignment_enabled:
            semantic = self.semantic_alignment(projected, batch["fault_texts"], modality_masks)
            aligned = semantic["aligned"]
            consistency_score = semantic["consistency_score"]
        else:
            aligned = projected
            consistency_score = torch.ones(next(iter(aligned.values())).size(0), device=next(iter(aligned.values())).device)

        if self.config.dynamic_fusion_enabled:
            fusion = self.dynamic_fusion(aligned, modality_masks)
            fused = fusion["fused"]
            strategy_weights = fusion["strategy_weights"]
        else:
            ordered = [aligned[name] for name in ("metrics", "logs", "traces") if name in aligned]
            ordered_masks = [modality_masks[name].unsqueeze(-1) for name in ("metrics", "logs", "traces") if name in aligned]
            fused = torch.stack(ordered, dim=0).sum(dim=0) / torch.stack(ordered_masks, dim=0).sum(dim=0).clamp_min(1)
            strategy_weights = torch.ones(fused.size(0), 1, device=fused.device)

        service_logits = self.service_head(fused)
        statistics_logits = None
        if self.metrics_statistics_head is not None and "metrics" in batch:
            statistics = _temporal_statistics(batch["metrics"], batch.get("metrics_lengths"))
            statistics_logits = self.metrics_statistics_head(statistics)
            metrics_mask = batch["metrics_mask"].to(statistics_logits.device).unsqueeze(-1)
            service_logits = service_logits + self.config.metrics_statistics_head_weight * statistics_logits * metrics_mask
        if "service_candidate_mask" in batch:
            candidate_mask = batch["service_candidate_mask"].to(service_logits.device).bool()
            service_logits = service_logits.masked_fill(~candidate_mask, torch.finfo(service_logits.dtype).min)
        fault_logits = self.fault_head(fused)

        return {
            "service_logits": service_logits,
            "fault_logits": fault_logits,
            "service_probs": torch.softmax(service_logits, dim=-1),
            "metrics_statistics_logits": statistics_logits,
            "fault_probs": torch.softmax(fault_logits, dim=-1),
            "projected": projected,
            "aligned": aligned,
            "consistency_score": consistency_score,
            "strategy_weights": strategy_weights,
        }

    def compute_losses(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, object], sample_weights: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        service_weights = batch.get("service_class_weights")
        if service_weights is not None:
            service_weights = service_weights.to(outputs["service_logits"].device)
        service_loss = F.cross_entropy(outputs["service_logits"], batch["service_labels"], weight=service_weights, reduction="none")
        fault_loss = F.cross_entropy(outputs["fault_logits"], batch["fault_labels"], reduction="none")
        if sample_weights is not None:
            sample_weights = sample_weights.to(service_loss.device)
            service_loss = (service_loss * sample_weights).mean()
            fault_loss = (fault_loss * sample_weights).mean()
        else:
            service_loss = service_loss.mean()
            fault_loss = fault_loss.mean()

        aligned_values = list(outputs["aligned"].values())
        aligned_masks = [batch[f"{name}_mask"].to(aligned_values[0].device) for name in outputs["aligned"]]
        projected_values = list(outputs["projected"].values())
        projected_masks = [batch[f"{name}_mask"].to(projected_values[0].device) for name in outputs["projected"]]
        contrastive_loss = _contrastive_loss(projected_values, projected_masks)
        alignment_loss = _pairwise_cosine_loss(aligned_values, aligned_masks)
        consistency_loss = 1.0 - outputs["consistency_score"].mean()
        fusion_loss = (outputs["strategy_weights"] ** 2).sum(dim=-1).mean()

        weights = self.config.loss_weights
        total_loss = (
            weights["task"] * service_loss
            + weights["fault"] * fault_loss
            + weights["contrastive"] * contrastive_loss
            + weights["alignment"] * alignment_loss
            + weights["consistency"] * consistency_loss
            + weights["fusion"] * fusion_loss
        )
        return {
            "task_loss": service_loss,
            "fault_loss": fault_loss,
            "contrastive_loss": contrastive_loss,
            "alignment_loss": alignment_loss,
            "consistency_loss": consistency_loss,
            "fusion_loss": fusion_loss,
            "total_loss": total_loss,
        }
