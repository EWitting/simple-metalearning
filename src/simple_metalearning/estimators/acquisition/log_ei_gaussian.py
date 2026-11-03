"""Numerically stable Log Expected Improvement acquisition function."""

from typing import Optional

import numpy as np
from scipy.stats import norm

from .base import AcquisitionFunction


class LogEIGaussianAcquisition(AcquisitionFunction):
    """Log Expected Improvement under a Gaussian predictive distribution.

    Computes log(EI(x) + floor) to avoid numerical underflow when many
    candidates have near-zero EI. Preserves the same ranking as standard EI
    but is more discriminative in the tail.

    Args:
        xi: Exploration bonus added to f_star. Defaults to 0.
        floor: Small constant preventing log(0). Defaults to 1e-300.
    """

    def __init__(self, xi: float = 0.0, floor: float = 1e-300) -> None:
        self.xi = xi
        self.floor = floor

    def compute(
        self,
        mu: np.ndarray,
        sigma: Optional[np.ndarray],
        f_star: float,
        known_mask: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if sigma is None:
            return mu.copy()

        improvement_threshold = f_star + self.xi
        ei = np.zeros_like(mu)

        pos_std = sigma > 0
        diff = mu[pos_std] - improvement_threshold
        Z = diff / sigma[pos_std]
        ei[pos_std] = diff * norm.cdf(Z) + sigma[pos_std] * norm.pdf(Z)
        ei[~pos_std] = np.maximum(0.0, mu[~pos_std] - improvement_threshold)

        return np.log(np.maximum(ei, self.floor))
