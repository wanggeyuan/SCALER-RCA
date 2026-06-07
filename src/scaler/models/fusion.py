from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class DynamicFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, num_experts: int = 8) -> None:
        super().__init__()
        self.num_strategies = 4
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_strategies),
        )
        self.modality_importance = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.attention_pool = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=dropout)
        self.gated_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_experts)]
        )
        self.expert_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)
        self.residual_logit = nn.Parameter(torch.tensor(-4.0))

    def _ordered_stack(self, aligned: Dict[str, torch.Tensor]) -> torch.Tensor:
        first = next(iter(aligned.values()))
        tensors = [aligned.get(name, torch.zeros_like(first)) for name in ("metrics", "logs", "traces")]
        return torch.stack(tensors, dim=1)

    def forward(self, aligned: Dict[str, torch.Tensor], modality_masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        modal_stack = self._ordered_stack(aligned)
        first = next(iter(aligned.values()))
        mask = torch.stack(
            [modality_masks.get(name, torch.zeros(first.size(0), device=first.device)) for name in ("metrics", "logs", "traces")],
            dim=1,
        )
        modal_stack = modal_stack * mask.unsqueeze(-1)
        context = modal_stack.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        strategy_weights = torch.softmax(self.context_gate(context), dim=-1)
        modality_logits = self.modality_importance(modal_stack.reshape(modal_stack.size(0), -1))
        modality_logits = modality_logits.masked_fill(~mask.bool(), torch.finfo(modality_logits.dtype).min)
        modality_weights = torch.softmax(modality_logits, dim=-1)

        mean_fused = (modal_stack * modality_weights.unsqueeze(-1)).sum(dim=1)
        attention_fused, _ = self.attention_pool(context.unsqueeze(1), modal_stack, modal_stack, key_padding_mask=~mask.bool())
        attention_fused = attention_fused.squeeze(1)
        gated_fused = self.gated_fusion(modal_stack.reshape(modal_stack.size(0), -1))

        expert_input = modal_stack.reshape(modal_stack.size(0), -1)
        expert_scores = torch.softmax(self.expert_selector(context), dim=-1)
        expert_outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=1)
        expert_fused = (expert_outputs * expert_scores.unsqueeze(-1)).sum(dim=1)

        strategies = torch.stack([mean_fused, attention_fused, gated_fused, expert_fused], dim=1)
        dynamic_fused = (strategies * strategy_weights.unsqueeze(-1)).sum(dim=1)
        fused = context + torch.sigmoid(self.residual_logit) * self.output_projection(dynamic_fused)
        return {
            "fused": fused,
            "strategy_weights": strategy_weights,
            "modality_weights": modality_weights,
        }
