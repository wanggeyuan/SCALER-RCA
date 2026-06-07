import json
from pathlib import Path

import pytest

from scripts.summarize_multiseed import render_markdown, summarize_runs


def test_summarize_runs_reports_full_delta(tmp_path: Path):
    variants = ("full", "no_dynamic_fusion")
    for seed, full, ablated in ((42, 0.5, 0.4), (43, 0.7, 0.6)):
        for variant, score in (("full", full), ("no_dynamic_fusion", ablated)):
            output = tmp_path / f"seed_{seed}" / variant
            output.mkdir(parents=True)
            metrics = {key: score for key in ("PR@1", "PR@3", "PR@5", "MRR")}
            (output / "metrics.json").write_text(json.dumps({"test_metrics": metrics}))

    summary = summarize_runs(tmp_path, [42, 43], variants=variants)

    assert summary["full"]["aggregate"]["PR@1"]["mean"] == 0.6
    assert summary["no_dynamic_fusion"]["full_delta"]["PR@1"] == pytest.approx(0.1)
    assert "no_dynamic_fusion" in render_markdown(summary)
