from pathlib import Path

from scripts.prepare_data import prepare_from_existing


def test_prepare_from_existing_creates_symlink(tmp_path: Path):
    source = tmp_path / "source"
    for system in ("RE2-SS", "RE2-OB", "RE2-TT"):
        (source / "RE2" / system).mkdir(parents=True, exist_ok=True)
    target = tmp_path / "target"
    prepare_from_existing(source, target, copy_data=False)
    assert target.exists()

