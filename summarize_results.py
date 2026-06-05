#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tabulate import tabulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SCALER outputs.")
    parser.add_argument("--main-dir", type=Path, default=Path("outputs/main"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/summary"))
    return parser.parse_args()


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("test_metrics", payload)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    main_metrics = _load_metrics(args.main_dir / "metrics.json")
    if main_metrics:
        rows.append(["main", *[main_metrics.get(name) for name in ("PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5", "MRR", "NDCG@5")]])
    ablation_summary = args.ablation_dir / "summary.json"
    if ablation_summary.exists():
        payload = json.loads(ablation_summary.read_text())
        for name, metrics in payload.items():
            rows.append([name, *[metrics.get(metric) for metric in ("PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5", "MRR", "NDCG@5")]])
    headers = ["variant", "PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5", "MRR", "NDCG@5"]
    table = tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f")
    print(table)
    (args.output_dir / "summary.md").write_text(table + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps({"rows": rows, "headers": headers}, indent=2))
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    main()

