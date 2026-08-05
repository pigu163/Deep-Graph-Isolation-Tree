"""确定性 seeding 工具。"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["DEFAULT_SEEDS", "set_seed"]

#: Seeds required by the reproduction task (Sec. 11.4).
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
