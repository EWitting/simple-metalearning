"""Greedy (pure exploitation) acquisition function."""

from typing import Optional

import numpy as np

from .base import AcquisitionFunction


class GreedyAcquisition(AcquisitionFunction):
    """Pure-exploitation acquisition: scores equal the predicted mean.

    This is the default behaviour when no acquisition function is specified.
    It ignores uncertainty entirely and simply suggests the flow with the
    highest predicted performance.
    """

    def compute(
        self,
        mu: np.ndarray,
        sigma: Optional[np.ndarray],
        f_star: float,
        known_mask: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return mu.copy()
