import torch

from scaler.config import SCALERExperimentConfig, TextEncoderConfig
from scaler.model import SCALERModel


def test_missing_modality_padding_does_not_change_predictions():
    config = SCALERExperimentConfig(
        hidden_dim=16,
        projection_dim=16,
        dropout=0.0,
        text_encoder=TextEncoderConfig(backend="hashed", max_tokens=8),
    )
    model = SCALERModel(
        input_dims={"metrics": 3, "logs": 2, "traces": 2},
        num_services=4,
        num_fault_types=2,
        config=config,
    )
    model.eval()

    batch = {
        "metrics": torch.randn(2, 5, 3),
        "metrics_mask": torch.ones(2, dtype=torch.long),
        "logs": torch.randn(2, 5, 2),
        "logs_mask": torch.zeros(2, dtype=torch.long),
        "traces": torch.randn(2, 5, 2),
        "traces_mask": torch.zeros(2, dtype=torch.long),
        "fault_texts": ["RE1-SS: cpu fault", "RE1-OB: delay fault"],
    }
    changed_padding = {
        **batch,
        "logs": torch.randn(2, 5, 2) * 1000,
        "traces": torch.randn(2, 5, 2) * 1000,
    }

    with torch.no_grad():
        original = model(batch)["service_logits"]
        changed = model(changed_padding)["service_logits"]

    torch.testing.assert_close(original, changed)


def test_contrastive_loss_uses_projected_embeddings():
    config = SCALERExperimentConfig(
        hidden_dim=16,
        projection_dim=16,
        dropout=0.0,
        text_encoder=TextEncoderConfig(backend="hashed", max_tokens=8),
    )
    model = SCALERModel(
        input_dims={"metrics": 2, "logs": 2, "traces": 2},
        num_services=2,
        num_fault_types=2,
        config=config,
    )
    batch = {
        "service_labels": torch.tensor([0, 1]),
        "fault_labels": torch.tensor([0, 1]),
        "metrics_mask": torch.ones(2, dtype=torch.long),
        "logs_mask": torch.ones(2, dtype=torch.long),
    }
    common = {
        "service_logits": torch.zeros(2, 2),
        "fault_logits": torch.zeros(2, 2),
        "aligned": {"metrics": torch.eye(2, 16), "logs": torch.eye(2, 16)},
        "consistency_score": torch.ones(2),
        "strategy_weights": torch.full((2, 4), 0.25),
    }
    matching = {**common, "projected": {"metrics": torch.eye(2, 16), "logs": torch.eye(2, 16)}}
    mismatching = {**common, "projected": {"metrics": torch.eye(2, 16), "logs": torch.flip(torch.eye(2, 16), dims=(0,))}}

    matching_loss = model.compute_losses(matching, batch)["contrastive_loss"]
    mismatching_loss = model.compute_losses(mismatching, batch)["contrastive_loss"]

    assert matching_loss < mismatching_loss
