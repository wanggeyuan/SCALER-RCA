from scripts.train_scaler import is_better_ranking_score


def test_checkpoint_selection_uses_mrr_to_break_pr1_ties():
    best = {"PR@1": 0.25, "MRR": 0.45}

    assert is_better_ranking_score({"PR@1": 0.26, "MRR": 0.40}, best)
    assert is_better_ranking_score({"PR@1": 0.25, "MRR": 0.46}, best)
    assert not is_better_ranking_score({"PR@1": 0.25, "MRR": 0.44}, best)
