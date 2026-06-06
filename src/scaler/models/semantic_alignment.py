from __future__ import annotations

import hashlib
import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaler.config import TextEncoderConfig

logger = logging.getLogger("scaler")


class HashedTextEncoder(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int = 50000, max_tokens: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position = nn.Embedding(max_tokens, embedding_dim)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=embedding_dim * 4, batch_first=True) for _ in range(2)]
        )
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size

    def _tokenize(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tokens = torch.zeros((len(texts), self.max_tokens), dtype=torch.long, device=device)
        for row, text in enumerate(texts):
            pieces = text.lower().split()[: self.max_tokens]
            for col, piece in enumerate(pieces):
                digest = hashlib.md5(piece.encode("utf-8")).hexdigest()
                tokens[row, col] = int(digest[:8], 16) % self.vocab_size
        return tokens

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tokens = self._tokenize(texts, device)
        positions = torch.arange(tokens.size(1), device=device).unsqueeze(0).expand(tokens.size(0), -1)
        hidden = self.embedding(tokens) + self.position(positions)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden.mean(dim=1)


class TextAnchorEncoder(nn.Module):
    def __init__(self, config: TextEncoderConfig, output_dim: int) -> None:
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.backend = "hashed"
        self.hashed = HashedTextEncoder(output_dim, max_tokens=config.max_tokens)
        self.transformer_model = None
        self.transformer_tokenizer = None
        self._fallback_reason = None
        if config.backend in {"auto", "transformers"}:
            try:
                import os
                from transformers import AutoModel, AutoTokenizer

                local_files_only = not config.allow_download
                if config.hf_endpoint:
                    os.environ["HF_ENDPOINT"] = config.hf_endpoint
                logger.info(
                    "Loading text encoder '%s' (local_files_only=%s, allow_download=%s, HF_ENDPOINT=%s)...",
                    config.model_name,
                    local_files_only,
                    config.allow_download,
                    os.environ.get("HF_ENDPOINT", "(default)"),
                )
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=local_files_only)
                self.transformer_model = AutoModel.from_pretrained(config.model_name, local_files_only=local_files_only)
                hidden = self.transformer_model.config.hidden_size
                self.transformer_proj = nn.Linear(hidden, output_dim)
                self.backend = "transformers"
                logger.info("Text encoder backend: transformers (model=%s, hidden=%d, output=%d)", config.model_name, hidden, output_dim)
            except Exception as exc:
                self.transformer_model = None
                self.transformer_tokenizer = None
                self._fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Failed to load transformers backend for '%s': %s. Falling back to hashed encoder.",
                    config.model_name,
                    self._fallback_reason,
                )
        if self.backend == "hashed":
            logger.info("Text encoder backend: hashed (embedding_dim=%d, max_tokens=%d)", output_dim, config.max_tokens)

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if self.backend == "transformers" and self.transformer_model is not None and self.transformer_tokenizer is not None:
            self.transformer_model = self.transformer_model.to(device)
            encoded = self.transformer_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_tokens,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.transformer_model(**encoded)
            pooled = outputs.last_hidden_state.mean(dim=1)
            return self.transformer_proj(pooled)
        return self.hashed(texts, device)


class SemanticAlignmentModule(nn.Module):
    def __init__(self, hidden_dim: int, anchor_config: TextEncoderConfig, dropout: float = 0.1) -> None:
        super().__init__()
        self.anchor_encoder = TextAnchorEncoder(anchor_config, hidden_dim)
        self.modal_projections = nn.ModuleDict({name: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)) for name in ("metrics", "logs", "traces")})
        self.anchor_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=dropout)
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, modality_embeddings: Dict[str, torch.Tensor], fault_texts: List[str]) -> Dict[str, torch.Tensor]:
        if not modality_embeddings:
            raise ValueError("modality_embeddings must not be empty")
        device = next(iter(modality_embeddings.values())).device
        anchor = self.anchor_encoder(fault_texts, device=device)

        aligned: Dict[str, torch.Tensor] = {}
        ordered_names = [name for name in ("metrics", "logs", "traces") if name in modality_embeddings]
        anchor_token = anchor.unsqueeze(1)
        projected_tokens = []
        for name in ordered_names:
            projected = self.modal_projections[name](modality_embeddings[name])
            attended, _ = self.anchor_attention(projected.unsqueeze(1), anchor_token, anchor_token)
            aligned[name] = projected + attended.squeeze(1)
            projected_tokens.append(aligned[name])

        stacked = torch.stack(projected_tokens, dim=1)
        cross_modal, _ = self.cross_modal_attention(stacked, stacked, stacked)
        for idx, name in enumerate(ordered_names):
            aligned[name] = cross_modal[:, idx, :]

        concat = []
        for name in ("metrics", "logs", "traces"):
            concat.append(aligned.get(name, torch.zeros_like(anchor)))
        consistency_score = torch.sigmoid(self.consistency_head(torch.cat(concat, dim=-1))).squeeze(-1)

        return {
            "anchor": anchor,
            "aligned": aligned,
            "consistency_score": consistency_score,
            "normalized": {name: F.normalize(value, dim=-1) for name, value in aligned.items()},
        }

