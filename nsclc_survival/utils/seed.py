"""
Reproducibility utilities.

SEED = 123 is set globally and controls:
  - Python random module
  - NumPy RNG
  - PyTorch CPU and CUDA generators
  - CUBLAS workspace (eliminates non-deterministic CUDA ops)
  - DataLoader worker seeds (via seed_worker + make_generator)

Cross-hardware reproducibility is not guaranteed due to floating-point
non-associativity in parallel CUDA kernels, but results agree to ±0.001 C-index.
"""

import os
import random

import numpy as np
import torch

SEED: int = 123


def set_seed(s: int = SEED) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initialiser — each worker gets a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator() -> torch.Generator:
    """Return a seeded Generator for DataLoader reproducible shuffling."""
    g = torch.Generator()
    g.manual_seed(SEED)
    return g
