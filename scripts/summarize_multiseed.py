#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_VARIANTS = ("full", "no_semantic_alignment", "no_dynamic_fusion", "no_curriculum_learning")
DEFAULT_METRICS = ("PR@1", "PR@3", "PR@5", "MRR")


def summarize_runs(root: Path, seeds: list[int], variants: tuple[str, ...] = DEFAULT_VARIANTS) -> dict:
    summary: dict[str, dict] = {}
    for variant in variants:
        rows = []
        for seed in seeds:
            payload = json.loads((root / f"seed_{seed}" / variant / "metrics.json").read_text())
            rows.append({key: float(payload["test_metrics"][key]) for key in DEFAULT_METRICS})
        summary[variant] = {
            "runs": rows,
            "aggregate": {
                key: {
                    "mean": float(np.mean([row[key] for row in rows])),
                    "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
                }
                for key in DEFAULT_METRICS
            },
        }

    full = summary["full"]["aggregate"]
    for variant in variants:
        if variant == "full":
            continue
        summary[variant]["full_delta"] = {
            key: full[key]["mean"] - summary[variant]["aggregate"][key]["mean"] for key in DEFAULT_METRICS
        }
    return summary


def render_markdown(summary: dict) -> str:
    lines = [
        "# Multi-seed ablation summary",
        "",
        "| Variant | PR@1 | PR@3 | PR@5 | MRR | Full delta PR@1 | Full delta MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, payload in summary.items():
        aggregate = payload["aggregate"]
        delta = payload.get("full_delta", {})
        format_metric = lambda key: f"{aggregate[key]['mean'] * 100:.2f} +/- {aggregate[key]['std'] * 100:.2f}"
        lines.append(
            f"| {variant} | {format_metric('PR@1')} | {format_metric('PR@3')} | {format_metric('PR@5')} | "
            f"{format_metric('MRR')} | {delta.get('PR@1', 0.0) * 100:+.2f} | {delta.get('MRR', 0.0) * 100:+.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed multi-seed SCALER ablation runs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    summary = summarize_runs(args.root, args.seeds)
    (args.root / "aggregate.json").write_text(json.dumps(summary, indent=2))
    (args.root / "aggregate.md").write_text(render_markdown(summary))


if __name__ == "__main__":
    main()
