#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


DATASET_URLS = {
    "re1": {
        "RE1-OB": "https://zenodo.org/records/14590730/files/RE1-OB.zip?download=1",
        "RE1-SS": "https://zenodo.org/records/14590730/files/RE1-SS.zip?download=1",
        "RE1-TT": "https://zenodo.org/records/14590730/files/RE1-TT.zip?download=1",
    },
    "re2": {
        "RE2-OB": "https://zenodo.org/records/14590730/files/RE2-OB.zip?download=1",
        "RE2-SS": "https://zenodo.org/records/14590730/files/RE2-SS.zip?download=1",
        "RE2-TT": "https://zenodo.org/records/14590730/files/RE2-TT.zip?download=1",
    },
    "re3": {
        "RE3-OB": "https://zenodo.org/records/14590730/files/RE3-OB.zip?download=1",
        "RE3-SS": "https://zenodo.org/records/14590730/files/RE3-SS.zip?download=1",
        "RE3-TT": "https://zenodo.org/records/14590730/files/RE3-TT.zip?download=1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download RCAEval datasets into a local directory.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/rcaeval"),
        help="Target RCAEval root directory. Final layout will contain RE1/ RE2/ RE3/.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="re1,re2,re3",
        help="Comma-separated dataset suites to download: re1,re2,re3",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload suites even if target subdirectories already exist.",
    )
    return parser.parse_args()


def _download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        with destination.open("wb") as handle, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))


def _extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)


def _normalize_requested_datasets(raw_value: str) -> list[str]:
    items = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    invalid = [item for item in items if item not in DATASET_URLS]
    if invalid:
        raise ValueError(f"Unsupported dataset suites: {', '.join(invalid)}")
    return items


def _download_suite(suite: str, target_dir: Path, force: bool) -> None:
    suite_root = target_dir / suite.upper()
    if suite_root.exists() and not force:
        print(f"Skip {suite.upper()}: {suite_root} already exists")
        return

    if suite_root.exists() and force:
        shutil.rmtree(suite_root)
    suite_root.mkdir(parents=True, exist_ok=True)

    for dataset_name, url in DATASET_URLS[suite].items():
        dataset_dir = suite_root / dataset_name
        if dataset_dir.exists() and not force:
            print(f"Skip {dataset_name}: already exists")
            continue

        zip_path = target_dir / f"{dataset_name}.zip"
        _download_file(url, zip_path)
        _extract_zip(zip_path, suite_root)
        zip_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    suites = _normalize_requested_datasets(args.datasets)
    for suite in suites:
        _download_suite(suite, target_dir, args.force)
    print(f"RCAEval data is available at {target_dir}")


if __name__ == "__main__":
    main()

