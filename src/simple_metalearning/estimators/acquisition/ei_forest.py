"""SMAC-style Random Forest Expected Improvement via Gaussian closed-form.

The forest posterior is approximated as a Gaussian using the empirical mean
and variance of individual tree predictions:

    mu(x)  = mean_over_trees( tree_pred(x) )
    var(x) = var_over_trees( tree_pred(x) )   [ddof=0, matching SMAC]

EI is then the standard closed-form Gaussian EI:

    Z      = (mu(x) - f*) / sigma(x)
    EI(x)  = (mu(x) - f*) · Φ(Z) + sigma(x) · φ(Z)

This matches SMAC3's EPMRandomForest.predict() + EI._compute() exactly.
A variance floor (``var_threshold``) prevents division by zero when all trees
agree on a prediction (typically for observed flows).
"""

from typing import Optional

import numpy as np
from scipy.stats import norm

from .base import AcquisitionFunction


class EIForestAcquisition(AcquisitionFunction):
    """SMAC-style EI for Random Forest surrogates (Gaussian closed-form).

    Args:
        xi: Exploration bonus subtracted from ``f_star`` threshold. Defaults to 0.
        var_threshold: Minimum variance floor to avoid zero-sigma divisions.
    """

    def __init__(self, xi: float = 0.0, var_threshold: float = 1e-10) -> None:
        self.xi = xi
        self.var_threshold = var_threshold

    def compute(
        self,
        mu: np.ndarray,
        sigma: Optional[np.ndarray],
        f_star: float,
        known_mask: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Gaussian EI from pre-computed mu/sigma (fallback when no tree preds)."""
        if sigma is None:
            return mu.copy()
        threshold = f_star + self.xi
        ei = np.zeros_like(mu)
        pos = sigma > 0
        diff = mu[pos] - threshold
        Z = diff / sigma[pos]
        ei[pos] = diff * norm.cdf(Z) + sigma[pos] * norm.pdf(Z)
        ei[~pos] = np.maximum(0.0, mu[~pos] - threshold)
        return ei

    def compute_from_trees(
        self,
        tree_predictions: np.ndarray,
        f_star: float,
    ) -> np.ndarray:
        """Compute SMAC-style Gaussian EI from per-tree predictions.

        Args:
            tree_predictions: Shape ``(n_estimators, n_flows)`` — each tree's
                prediction for every flow.
            f_star: Best observed performance so far.

        Returns:
            EI score array of shape ``(n_flows,)``.
        """
        mu = tree_predictions.mean(axis=0)
        var = tree_predictions.var(axis=0)           # ddof=0, matches SMAC
        var = np.maximum(var, self.var_threshold)    # floor to avoid zero sigma
        sigma = np.sqrt(var)

        threshold = f_star + self.xi
        diff = mu - threshold
        Z = diff / sigma
        return diff * norm.cdf(Z) + sigma * norm.pdf(Z)
