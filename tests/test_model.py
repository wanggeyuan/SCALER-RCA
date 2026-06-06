import torch

from scaler.config import SCALERExperimentConfig, TextEncoderConfig
from scaler.model import SCALERModel
from scaler.models.encoders import MetricsEncoder
from scaler.models.semantic_alignment import SemanticAlignmentModule


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
        "metrics_lengths": torch.full((2,), 5, dtype=torch.long),
        "logs": torch.randn(2, 5, 2),
        "logs_mask": torch.zeros(2, dtype=torch.long),
        "logs_lengths": torch.zeros(2, dtype=torch.long),
        "traces": torch.randn(2, 5, 2),
        "traces_mask": torch.zeros(2, dtype=torch.long),
        "traces_lengths": torch.zeros(2, dtype=torch.long),
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


def test_temporal_padding_does_not_change_predictions():
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

    logs = torch.randn(2, 5, 2)
    traces = torch.randn(2, 5, 2)
    batch = {
        "metrics": torch.randn(2, 5, 3),
        "metrics_mask": torch.ones(2, dtype=torch.long),
        "metrics_lengths": torch.full((2,), 5, dtype=torch.long),
        "logs": logs,
        "logs_mask": torch.ones(2, dtype=torch.long),
        "logs_lengths": torch.full((2,), 2, dtype=torch.long),
        "traces": traces,
        "traces_mask": torch.ones(2, dtype=torch.long),
        "traces_lengths": torch.full((2,), 2, dtype=torch.long),
        "fault_texts": ["RE2-SS: cpu fault", "RE2-OB: delay fault"],
    }
    changed_padding = {
        **batch,
        "logs": torch.cat([torch.randn(2, 3, 2) * 1000, logs[:, -2:]], dim=1),
        "traces": torch.cat([torch.randn(2, 3, 2) * 1000, traces[:, -2:]], dim=1),
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


def test_semantic_alignment_preserves_modality_residual():
    module = SemanticAlignmentModule(
        hidden_dim=16,
        anchor_config=TextEncoderConfig(backend="hashed", max_tokens=8),
        dropout=0.0,
    )
    for parameter in module.parameters():
        torch.nn.init.zeros_(parameter)
    embeddings = {"metrics": torch.randn(2, 16), "logs": torch.randn(2, 16)}
    masks = {"metrics": torch.ones(2, dtype=torch.long), "logs": torch.ones(2, dtype=torch.long)}

    aligned = module(embeddings, ["cpu fault", "delay fault"], masks)["aligned"]

    torch.testing.assert_close(aligned["metrics"], embeddings["metrics"])
    torch.testing.assert_close(aligned["logs"], embeddings["logs"])


def test_metrics_encoder_preserves_per_channel_statistics():
    encoder = MetricsEncoder(input_dim=3, hidden_dim=16, dropout=0.0)
    for name, parameter in encoder.named_parameters():
        if name.startswith(("cnn.", "lstm.", "proj.")):
            torch.nn.init.zeros_(parameter)
    baseline = torch.zeros(2, 5, 3)
    anomalous = baseline.clone()
    anomalous[:, 1:4, 1] = 10.0
    lengths = torch.full((2,), 5, dtype=torch.long)

    with torch.no_grad():
        baseline_embedding = encoder(baseline, lengths)
        anomalous_embedding = encoder(anomalous, lengths)

    assert not torch.allclose(baseline_embedding, anomalous_embedding)
