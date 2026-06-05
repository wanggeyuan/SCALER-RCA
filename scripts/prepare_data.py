#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaler.utils.paths import is_valid_rcaeval_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RCAEval data for SCALER-RCA.")
    parser.add_argument("--source-dir", type=Path, default=None, help="Existing RCAEval data root.")
    parser.add_argument("--target-dir", type=Path, default=Path("data/rcaeval"), help="Local target data directory.")
    parser.add_argument("--copy", action="store_true", help="Copy data instead of creating a symlink.")
    parser.add_argument("--download", action="store_true", help="Try RCAEval.utility download helpers if available.")
    return parser.parse_args()


def prepare_from_existing(source_dir: Path, target_dir: Path, copy_data: bool) -> None:
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    if not is_valid_rcaeval_root(source_dir):
        raise FileNotFoundError(f"Invalid RCAEval root: {source_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() or target_dir.is_symlink():
        if target_dir.is_symlink() or target_dir.is_file():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)
    if copy_data:
        shutil.copytree(source_dir, target_dir)
    else:
        target_dir.symlink_to(source_dir, target_is_directory=True)
    print(f"Prepared data at {target_dir}")


def download_with_rcaeval(target_dir: Path) -> None:
    from RCAEval.utility import download_re1_dataset, download_re2_dataset, download_re3_dataset

    target_dir.mkdir(parents=True, exist_ok=True)
    current = Path.cwd()
    try:
        import os

        os.chdir(target_dir)
        download_re1_dataset()
        download_re2_dataset()
        download_re3_dataset()
    finally:
        os.chdir(current)


def main() -> None:
    args = parse_args()
    if args.source_dir:
        prepare_from_existing(args.source_dir, args.target_dir, args.copy)
        return
    if args.download:
        download_with_rcaeval(args.target_dir)
        return
    raise SystemExit("Provide --source-dir or --download.")


if __name__ == "__main__":
    main()
