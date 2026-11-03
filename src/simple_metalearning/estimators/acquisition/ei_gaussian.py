"""Standard Gaussian Expected Improvement acquisition function."""

from typing import Optional

import numpy as np
from scipy.stats import norm

from .base import AcquisitionFunction


class EIGaussianAcquisition(AcquisitionFunction):
    """Expected Improvement under a Gaussian predictive distribution.

    Assumes the surrogate model provides a Gaussian posterior
    ``p(y | x) = N(μ(x), σ²(x))``.  The closed-form EI is:

        EI(x) = (μ(x) - f*) · Φ(Z) + σ(x) · φ(Z)
        where Z = (μ(x) - f*) / σ(x)

    For flows with ``σ = 0``, EI degenerates to ``max(0, μ - f*)``.

    If ``sigma`` is ``None`` (model provides no uncertainty), falls back to
    ``GreedyAcquisition`` behaviour (returns ``mu``).

    Args:
        xi: Exploration bonus added to ``f_star`` before computing EI.
            Positive values increase exploration.  Defaults to 0.
    """

    def __init__(self, xi: float = 0.0) -> None:
        self.xi = xi

    def compute(
        self,
        mu: np.ndarray,
        sigma: Optional[np.ndarray],
        f_star: float,
        known_mask: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if sigma is None:
            # No uncertainty estimate — fall back to greedy
            return mu.copy()

        improvement_threshold = f_star + self.xi
        ei = np.zeros_like(mu)

        pos_std = sigma > 0
        diff = mu[pos_std] - improvement_threshold
        Z = diff / sigma[pos_std]
        ei[pos_std] = diff * norm.cdf(Z) + sigma[pos_std] * norm.pdf(Z)
        # Zero-variance points: deterministic improvement (no uncertainty bonus)
        ei[~pos_std] = np.maximum(0.0, mu[~pos_std] - improvement_threshold)

        return ei
