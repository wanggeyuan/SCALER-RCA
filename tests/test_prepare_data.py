from pathlib import Path

from scripts.download_rcaeval import _normalize_requested_datasets
from scripts.prepare_data import prepare_from_existing


def test_prepare_from_existing_creates_symlink(tmp_path: Path):
    source = tmp_path / "source"
    for system in ("RE2-SS", "RE2-OB", "RE2-TT"):
        (source / "RE2" / system).mkdir(parents=True, exist_ok=True)
    target = tmp_path / "target"
    prepare_from_existing(source, target, copy_data=False)
    assert target.exists()


def test_normalize_requested_datasets_accepts_known_suites():
    assert _normalize_requested_datasets("re1,re2,re3") == ["re1", "re2", "re3"]
