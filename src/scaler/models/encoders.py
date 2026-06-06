from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def _valid_time_mask(x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    if lengths is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    safe_lengths = lengths.to(x.device).clamp(min=1, max=x.size(1))
    positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
    return positions >= (x.size(1) - safe_lengths).unsqueeze(1)


class MetricsEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.stats_proj = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.stats_residual_logit = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        valid = _valid_time_mask(x, lengths)
        x = x * valid.unsqueeze(-1)
        temporal = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.lstm(temporal)
        temporal_embedding = self.dropout(self.proj(encoded[:, -1, :]))

        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        mean = x.sum(dim=1) / denominator
        variance = (((x - mean.unsqueeze(1)) ** 2) * valid.unsqueeze(-1)).sum(dim=1) / denominator
        std = torch.sqrt(variance + 1e-6)
        maximum = x.masked_fill(~valid.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values
        last = x[:, -1, :]
        stats_embedding = self.stats_proj(torch.cat([mean, std, maximum, last], dim=-1))

        return temporal_embedding + torch.sigmoid(self.stats_residual_logit) * stats_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class LogsEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = TransformerEncoder(layer, num_layers=4)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        valid = _valid_time_mask(x, lengths)
        x = x * valid.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.transformer(x, src_key_padding_mask=~valid)
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        return self.dropout(self.output_proj(pooled))


class TracesEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        valid = _valid_time_mask(x, lengths)
        x = x * valid.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.lstm(x)
        attn_logits = self.attn(encoded)
        attn_logits = attn_logits.masked_fill(~valid.unsqueeze(-1), torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (encoded * attn_weights).sum(dim=1)
        return self.dropout(self.output_proj(pooled))
