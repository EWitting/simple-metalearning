"""Upper Confidence Bound (UCB) acquisition function."""

from typing import Optional

import numpy as np

from .base import AcquisitionFunction


class UCBAcquisition(AcquisitionFunction):
    """Upper Confidence Bound: score = μ(x) + κ · σ(x).

    Balances exploitation (high predicted mean) and exploration (high
    uncertainty).  Larger ``kappa`` increases exploration.

    If ``sigma`` is ``None`` (model provides no uncertainty), falls back to
    ``GreedyAcquisition`` behaviour (returns ``mu``).

    Args:
        kappa: Exploration weight.  ``kappa=0`` is equivalent to greedy.
            Typical values: 1–3.  Defaults to 2.0.
    """

    def __init__(self, kappa: float = 2.0) -> None:
        self.kappa = kappa

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
        return mu + self.kappa * sigma
