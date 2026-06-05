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
        )

    def _ordered_stack(self, aligned: Dict[str, torch.Tensor]) -> torch.Tensor:
        first = next(iter(aligned.values()))
        tensors = [aligned.get(name, torch.zeros_like(first)) for name in ("metrics", "logs", "traces")]
        return torch.stack(tensors, dim=1)

    def forward(self, aligned: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        modal_stack = self._ordered_stack(aligned)
        context = modal_stack.mean(dim=1)
        strategy_weights = torch.softmax(self.context_gate(context), dim=-1)
        modality_logits = self.modality_importance(modal_stack.reshape(modal_stack.size(0), -1))
        modality_weights = torch.softmax(modality_logits, dim=-1)

        mean_fused = (modal_stack * modality_weights.unsqueeze(-1)).sum(dim=1)
        attention_fused, _ = self.attention_pool(context.unsqueeze(1), modal_stack, modal_stack)
        attention_fused = attention_fused.squeeze(1)
        gated_fused = self.gated_fusion(modal_stack.reshape(modal_stack.size(0), -1))

        expert_input = modal_stack.reshape(modal_stack.size(0), -1)
        expert_scores = torch.softmax(self.expert_selector(context), dim=-1)
        expert_outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=1)
        expert_fused = (expert_outputs * expert_scores.unsqueeze(-1)).sum(dim=1)

        strategies = torch.stack([mean_fused, attention_fused, gated_fused, expert_fused], dim=1)
        fused = (strategies * strategy_weights.unsqueeze(-1)).sum(dim=1)
        return {
            "fused": self.output_projection(fused),
            "strategy_weights": strategy_weights,
            "modality_weights": modality_weights,
        }

