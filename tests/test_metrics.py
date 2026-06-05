import numpy as np

from scaler.evaluation.metrics import compute_ranking_metrics


def test_compute_ranking_metrics_prefers_high_scores():
    scores = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.3],
            [0.2, 0.4, 0.7],
        ]
    )
    targets = np.array([0, 1, 2])
    metrics = compute_ranking_metrics(scores, targets)
    assert metrics["PR@1"] == 1.0
    assert metrics["MRR"] == 1.0

