from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class TextEncoderConfig:
    model_name: str = "bert-base-uncased"
    backend: str = "auto"
    max_tokens: int = 256
    allow_download: bool = True
    hf_endpoint: str = ""


@dataclass
class CurriculumConfig:
    enabled: bool = True
    initial_threshold: float = 0.45
    min_threshold: float = 0.2
    max_threshold: float = 1.0
    step_size: float = 0.05
    target_score: float = 0.6
    patience: int = 2


@dataclass
class SCALERExperimentConfig:
    seed: int = 42
    batch_size: int = 8
    eval_batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    projection_dim: int = 128
    dropout: float = 0.1
    warmup_epochs: int = 0
    early_stopping_patience: int = 15
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    min_learning_rate: float = 1e-6
    max_sequence_length: int = 256
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    include_modalities: List[str] = field(default_factory=lambda: ["metrics", "logs", "traces"])
    stages: List[str] = field(default_factory=lambda: ["RE1", "RE2", "RE3"])
    systems: List[str] = field(default_factory=list)
    max_cases_per_system: int | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    output_dir: str = "outputs/main"
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "task": 1.0,
            "fault": 0.5,
            "contrastive": 0.1,
            "alignment": 0.1,
            "consistency": 0.05,
            "fusion": 0.01,
        }
    )
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    semantic_alignment_enabled: bool = True
    dynamic_fusion_enabled: bool = True
    curriculum_enabled: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SCALERExperimentConfig":
        payload = yaml.safe_load(Path(path).read_text()) or {}
        text_encoder = TextEncoderConfig(**payload.pop("text_encoder", {}))
        curriculum = CurriculumConfig(**payload.pop("curriculum", {}))
        config = cls(**payload)
        config.text_encoder = text_encoder
        config.curriculum = curriculum
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.__dict__,
            "text_encoder": self.text_encoder.__dict__,
            "curriculum": self.curriculum.__dict__,
        }
