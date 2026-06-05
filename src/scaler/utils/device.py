from __future__ import annotations

import torch


def select_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower().strip()
    if preferred not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("preferred device must be one of: auto, cuda, mps, cpu")

    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if preferred == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"cuda ({name})"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"

