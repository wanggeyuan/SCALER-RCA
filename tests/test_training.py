from scripts.train_scaler import is_better_ranking_score
from scaler.training.curriculum import ComplexityScheduler


def test_checkpoint_selection_uses_mrr_to_break_pr1_ties():
    best = {"PR@1": 0.25, "MRR": 0.45}

    assert is_better_ranking_score({"PR@1": 0.26, "MRR": 0.40}, best)
    assert is_better_ranking_score({"PR@1": 0.25, "MRR": 0.46}, best)
    assert not is_better_ranking_score({"PR@1": 0.25, "MRR": 0.44}, best)


def test_curriculum_threshold_only_expands_to_include_harder_samples():
    scheduler = ComplexityScheduler(
        threshold=0.45,
        min_threshold=0.2,
        max_threshold=1.0,
        step_size=0.05,
        target_score=0.6,
        patience=2,
    )

    scheduler.update(0.3)
    assert scheduler.threshold == 0.45
    scheduler.update(0.3)
    assert scheduler.threshold == 0.5
    scheduler.update(0.7)
    assert scheduler.threshold == 0.55
