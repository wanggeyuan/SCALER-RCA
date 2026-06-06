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
