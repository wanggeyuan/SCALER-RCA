from __future__ import annotations

from typing import Dict

import numpy as np


def _precision_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def _mean_reciprocal_rank(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / ranks))


def _average_precision_for_rank(rank: int, k: int) -> float:
    return 1.0 / rank if rank <= k else 0.0


def _ndcg_for_rank(rank: int, k: int) -> float:
    return (1.0 / np.log2(rank + 1)) if rank <= k else 0.0


def compute_ranking_metrics(score_matrix: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    ranks = []
    for scores, target in zip(score_matrix, targets):
        order = np.argsort(scores)[::-1]
        rank = int(np.where(order == target)[0][0]) + 1
        ranks.append(rank)
    ranks_array = np.asarray(ranks, dtype=np.int64)
    metrics = {
        "PR@1": _precision_at_k(ranks_array, 1),
        "PR@3": _precision_at_k(ranks_array, 3),
        "PR@5": _precision_at_k(ranks_array, 5),
        "PR@10": _precision_at_k(ranks_array, 10),
        "MAP@3": float(np.mean([_average_precision_for_rank(rank, 3) for rank in ranks])),
        "MAP@5": float(np.mean([_average_precision_for_rank(rank, 5) for rank in ranks])),
        "MAP@10": float(np.mean([_average_precision_for_rank(rank, 10) for rank in ranks])),
        "MRR": _mean_reciprocal_rank(ranks_array),
        "NDCG@5": float(np.mean([_ndcg_for_rank(rank, 5) for rank in ranks])),
        "NDCG@10": float(np.mean([_ndcg_for_rank(rank, 10) for rank in ranks])),
    }
    return metrics

