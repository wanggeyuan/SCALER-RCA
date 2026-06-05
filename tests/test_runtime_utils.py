from pathlib import Path

import torch

from scaler.utils.device import select_device
from scaler.utils.logging import configure_logging


def test_select_device_accepts_explicit_cpu():
    device = select_device("cpu")
    assert device.type == "cpu"


def test_select_device_uses_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device = select_device("auto")
    assert device.type == "mps"


def test_configure_logging_writes_log_file(tmp_path: Path):
    logger = configure_logging(tmp_path, "train.log")
    logger.info("runtime logging check")
    log_file = tmp_path / "train.log"
    assert log_file.exists()
    assert "runtime logging check" in log_file.read_text()
