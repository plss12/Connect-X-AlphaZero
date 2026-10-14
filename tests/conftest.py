"""Shared pytest configuration.

The repository is a collection of scripts and top-level packages (``env``, ``agents``,
``alphazero``, ``training_algorithms``) rather than an installed package, so the repo root is put
on ``sys.path`` here, mirroring how the training scripts and the notebook import them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _seed_global_rngs() -> None:
    """Fix the global RNGs so tests that rely on random tie-breaking are reproducible."""
    import random

    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
