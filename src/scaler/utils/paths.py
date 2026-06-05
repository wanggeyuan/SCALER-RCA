from __future__ import annotations

import os
from pathlib import Path


RE2_SYSTEMS = ("RE2-SS", "RE2-OB", "RE2-TT")


def is_valid_rcaeval_root(root: Path) -> bool:
    root = root.expanduser().resolve()
    return root.is_dir() and any((root / "RE2" / name).is_dir() for name in RE2_SYSTEMS)


def resolve_data_root(data_root: str | None = None) -> Path:
    candidates = []
    if data_root:
        candidates.append(Path(data_root))

    env_root = os.environ.get("RCAEVAL_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            repo_root / "data" / "rcaeval",
            repo_root.parent / "RCA_RCAEval" / "rcaeval_workspace" / "RCAEval" / "data",
            repo_root.parent / "RCA_RCAEval_GH" / "rcaeval_workspace" / "RCAEval" / "data",
        ]
    )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if is_valid_rcaeval_root(resolved):
            return resolved

    tried = "\n".join(f"- {path.expanduser().resolve()}" for path in candidates)
    raise FileNotFoundError(
        "RCAEval data root not found. Set RCAEVAL_DATA_ROOT or use --data-root.\n"
        f"Tried:\n{tried}"
    )

