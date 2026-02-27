"""Utility functions for Boltz."""

import torch


def boltz_device_type() -> str:
    """Return the appropriate device type string for torch.autocast.

    Returns "cuda" if CUDA is available, "mps" if Apple Silicon MPS
    is available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
