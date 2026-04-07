"""Random seed utility for reproducibility."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and any downstream libraries."""
    random.seed(seed)
    np.random.seed(seed)
