import torch
import torch.nn as nn
from types import SimpleNamespace

from scaler.config import SCALERExperimentConfig, TextEncoderConfig
from scaler.model import SCALERModel
from scaler.models.fusion import DynamicFusion
from scaler.models.semantic_alignment import SemanticAlignmentModule, TextAnchorEncoder


def test_dynamic_fusion_preserves_mean_context_when_dynamic_branch_is_zero():
    fusion = DynamicFusion(hidden_dim=16, dropout=0.0)
    for parameter in fusion.parameters():
        torch.nn.init.zeros_(parameter)
    aligned = {
        "metrics": torch.full((2, 16), 1.0),
        "logs": torch.full((2, 16), 3.0),
    }
    masks = {
        "metrics": torch.ones(2, dtype=torch.long),
        "logs": torch.ones(2, dtype=torch.long),
    }

    output = fusion(aligned, masks)["fused"]

    torch.testing.assert_close(output, torch.full((2, 16), 2.0))


def test_transformer_text_anchor_ignores_padding_and_stays_frozen():
    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            }

    class FakeTransformer(nn.Module):
        def forward(self, input_ids, attention_mask):
            hidden = torch.tensor([[[1.0] * 8, [3.0] * 8, [100.0] * 8], [[7.0] * 8, [100.0] * 8, [100.0] * 8]])
            return SimpleNamespace(last_hidden_state=hidden)

    encoder = TextAnchorEncoder(TextEncoderConfig(backend="hashed"), output_dim=8)
    encoder.backend = "transformers"
    encoder.transformer_tokenizer = FakeTokenizer()
    encoder.transformer_model = FakeTransformer()
    encoder.transformer_proj = nn.Identity()
    encoder.train()

    anchors = encoder(["first", "second"], device=torch.device("cpu"))

    torch.testing.assert_close(anchors, torch.tensor([[2.0] * 8, [7.0] * 8]))
    assert not encoder.transformer_model.training


def test_metrics_statistics_head_ignores_temporal_padding():
    config = SCALERExperimentConfig(
        hidden_dim=16,
        projection_dim=16,
        dropout=0.0,
        metrics_statistics_head_enabled=True,
        text_encoder=TextEncoderConfig(backend="hashed", max_tokens=8),
    )
    model = SCALERModel(
        input_dims={"metrics": 3, "logs": 2, "traces": 2},
        num_services=4,
        num_fault_types=2,
        config=config,
    )
    model.eval()
    valid = torch.randn(2, 3, 3)
    batch = {
        "metrics": torch.cat([torch.zeros(2, 2, 3), valid], dim=1),
        "metrics_mask": torch.ones(2, dtype=torch.long),
        "metrics_lengths": torch.full((2,), 3, dtype=torch.long),
        "fault_texts": ["RE1-SS: cpu fault", "RE1-OB: delay fault"],
    }
    changed_padding = {
        **batch,
        "metrics": torch.cat([torch.randn(2, 2, 3) * 1000, valid], dim=1),
    }

    with torch.no_grad():
        original = model(batch)["service_logits"]
        changed = model(changed_padding)["service_logits"]

    torch.testing.assert_close(original, changed)


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


def test_service_candidate_mask_removes_invalid_classes():
    config = SCALERExperimentConfig(
        hidden_dim=16,
        projection_dim=16,
        dropout=0.0,
        semantic_alignment_enabled=False,
        dynamic_fusion_enabled=False,
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
        "fault_texts": ["RE2-SS: cpu fault", "RE2-OB: delay fault"],
        "service_candidate_mask": torch.tensor([[True, True, False, False], [False, False, True, True]]),
    }

    with torch.no_grad():
        probabilities = model(batch)["service_probs"]

    torch.testing.assert_close(probabilities[0, 2:], torch.zeros(2))
    torch.testing.assert_close(probabilities[1, :2], torch.zeros(2))


def test_service_loss_accepts_training_class_weights():
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
    outputs = {
        "service_logits": torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
        "fault_logits": torch.zeros(2, 2),
        "projected": {"metrics": torch.eye(2, 16)},
        "aligned": {"metrics": torch.eye(2, 16)},
        "consistency_score": torch.ones(2),
        "strategy_weights": torch.full((2, 4), 0.25),
    }
    batch = {
        "service_labels": torch.tensor([0, 1]),
        "fault_labels": torch.tensor([0, 1]),
        "metrics_mask": torch.ones(2, dtype=torch.long),
    }

    unweighted = model.compute_losses(outputs, batch)["task_loss"]
    weighted = model.compute_losses(
        outputs,
        {**batch, "service_class_weights": torch.tensor([1.0, 3.0])},
    )["task_loss"]

    assert weighted > unweighted
