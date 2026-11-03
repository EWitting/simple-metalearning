"""Cost-cooled Expected Improvement acquisition function.

Motivation
----------
Standard EI treats all flows as equally cheap to evaluate.  In AutoML
benchmarks (and in practice), model training costs vary by orders of
magnitude — a tuned neural network may take 1000× longer than a decision
tree.  Cost-cooling divides the EI score by the evaluation cost so that the
optimizer prefers cheap improvements early in the search and only commits to
expensive flows when the expected gain is large enough to justify the cost.

Formula
-------
    CostCooledEI(x) = EI(x) / cost(x)^alpha

where ``EI`` is any base acquisition function (defaults to Gaussian EI),
``cost(x)`` is the wall-clock evaluation cost for flow ``x`` (e.g.
``time_train_s + time_infer_s`` for TabArena), and ``alpha`` controls how
strongly cost is penalised:

- ``alpha = 1.0``  — linear penalty; recommended default (Snoek et al. 2012)
- ``alpha = 0.5``  — soft penalty; useful when cost varies less than one OOM
- ``alpha = 0.0``  — no cost penalty; equivalent to the base acquisition

Missing / zero costs
--------------------
If ``cost`` is ``None`` (the dataset has no cost matrix), the function falls
back transparently to the base acquisition.  Flows with NaN or non-positive
costs receive the median cost of the remaining flows so they are not
penalised unfairly.

References
----------
Snoek, J., Larochelle, H., & Adams, R. P. (2012).  Practical Bayesian
optimization of machine learning algorithms.  NeurIPS.
"""

from typing import Optional

import numpy as np

from .base import AcquisitionFunction
from .ei_gaussian import EIGaussianAcquisition


class CostCooledEIAcquisition(AcquisitionFunction):
    """EI divided by evaluation cost raised to a power.

    Prefers cheap flows when expected improvement is similar, and only
    recommends expensive flows when the EI gap is large enough to compensate.

    Args:
        alpha: Cost penalty exponent.  ``1.0`` = linear penalty (Snoek 2012
            default).  ``0.5`` = square-root penalty.  ``0.0`` = no penalty.
        xi: Exploration bonus forwarded to the base acquisition function.
        base_acq: Underlying acquisition function whose raw score is divided
            by ``cost^alpha``.  Defaults to :class:`EIGaussianAcquisition`.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        xi: float = 0.0,
        base_acq: Optional[AcquisitionFunction] = None,
    ) -> None:
        self.alpha = alpha
        self.xi = xi
        self.base_acq: AcquisitionFunction = base_acq or EIGaussianAcquisition(xi=xi)

    def compute(
        self,
        mu: np.ndarray,
        sigma: Optional[np.ndarray],
        f_star: float,
        known_mask: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute cost-cooled acquisition scores.

        Falls back to the base acquisition when ``cost`` is ``None`` or when
        ``alpha == 0``.

        Args:
            mu: Predicted mean for all flows, shape ``(n_flows,)``.
            sigma: Predicted std for all flows, or ``None``.
            f_star: Best observed performance so far.
            known_mask: Boolean mask of already-evaluated flows.
            cost: Per-flow evaluation cost, shape ``(n_flows,)``.  Should be
                in consistent units (e.g. seconds).  ``None`` → no cooling.

        Returns:
            Score array of shape ``(n_flows,)``.
        """
        base_scores = self.base_acq.compute(mu, sigma, f_star, known_mask, cost=None)

        if cost is None or self.alpha == 0.0:
            return base_scores

        cost = np.asarray(cost, dtype=float)

        # Replace non-positive and NaN costs with the median of valid costs so
        # those flows are not given an artificially inflated score.
        valid = np.isfinite(cost) & (cost > 0)
        if not np.any(valid):
            # No usable cost information at all — fall back to base EI.
            return base_scores
        median_cost = float(np.median(cost[valid]))
        cost_safe = np.where(valid, cost, median_cost)

        return base_scores / (cost_safe ** self.alpha)
