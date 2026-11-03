"""Acquisition function optimisers for continuous-space BO.

Two strategies:
- ``LocalAndRandomSearchOptimizer``: SMAC-style local-and-random search over
  the XGBoost ConfigurationSpace (reviewer-defensible for mixed spaces).
  Samples a large random pool, scores all candidates, then runs local search
  from the top-k starting points via one-exchange neighbourhood enumeration.
- ``RandomSearchOptimizer``: pure random pool, argmax acquisition.  Faster but
  coarser; good for small budgets or debugging.

Both expose the same interface: ``ask(estimator, f_star) -> Configuration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import Configuration

from simple_metalearning.search_spaces.xgboost_space import (
    ConfigFeaturizer,
    get_configspace,
    neighbors,
    sample,
)

if TYPE_CHECKING:
    from simple_metalearning.estimators.continuous.base import ContinuousEstimator


class RandomSearchOptimizer:
    """Maximise acquisition over a random pool of candidates.

    Args:
        n_random: Number of random candidates to score per call.
        featurizer: :class:`ConfigFeaturizer` used by the estimator.
    """

    def __init__(self, n_random: int = 1000, featurizer: ConfigFeaturizer | None = None):
        self.n_random = n_random
        self.featurizer = featurizer

    def ask(
        self,
        estimator: "ContinuousEstimator",
        f_star: float,
        rng: np.random.Generator | None = None,
    ) -> Configuration:
        """Return the best candidate from a random pool.

        Args:
            estimator: Fitted continuous estimator with ``acquisition_scores``.
            f_star: Best observed performance so far.
            rng: Optional random generator for reproducibility.

        Returns:
            The Configuration with the highest acquisition score.
        """
        candidates = sample(self.n_random, rng=rng)
        X_cand = self.featurizer.transform(candidates)
        scores = estimator.acquisition_scores(X_cand, f_star)
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]


class LocalAndRandomSearchOptimizer:
    """SMAC-style acquisition optimiser for mixed continuous/integer/categorical spaces.

    Algorithm:
    1. Sample ``n_random`` candidates uniformly from the ConfigSpace.
    2. Score all via the estimator's acquisition function.
    3. Take the top-``n_local_starts`` candidates as local-search starting points
       (plus the current incumbent if provided).
    4. For each starting point, enumerate ``get_one_exchange_neighbourhood`` and
       update the start if a neighbour improves the acquisition score.  Repeat
       for ``max_local_steps`` steps or until no improvement.
    5. Return the globally best candidate across random pool + local search.

    This is the core acquisition optimisation strategy used by SMAC3 (Lindauer
    et al., 2022) and is standard for RF-based BO over mixed search spaces.

    Args:
        n_random: Random pool size (default 1000).
        n_local_starts: Number of local-search chains (default 10).
        max_local_steps: Max iterations per local-search chain (default 20).
        featurizer: :class:`ConfigFeaturizer` aligned to the estimator.
    """

    def __init__(
        self,
        n_random: int = 1000,
        n_local_starts: int = 10,
        max_local_steps: int = 20,
        featurizer: ConfigFeaturizer | None = None,
    ):
        self.n_random = n_random
        self.n_local_starts = n_local_starts
        self.max_local_steps = max_local_steps
        self.featurizer = featurizer

    def _score_single(
        self,
        config: Configuration,
        estimator: "ContinuousEstimator",
        f_star: float,
    ) -> float:
        X = self.featurizer.transform([config])
        scores = estimator.acquisition_scores(X, f_star)
        return float(scores[0])

    def ask(
        self,
        estimator: "ContinuousEstimator",
        f_star: float,
        incumbent: Configuration | None = None,
        rng: np.random.Generator | None = None,
    ) -> Configuration:
        """Return the best configuration found by local-and-random search.

        Args:
            estimator: Fitted continuous estimator.
            f_star: Best observed performance so far.
            incumbent: Current best-observed configuration (seeded as an extra
                local-search start if provided).
            rng: Optional random generator.

        Returns:
            The Configuration with the highest acquisition score.
        """
        # 1. Random pool
        candidates = sample(self.n_random, rng=rng)
        X_cand = self.featurizer.transform(candidates)
        scores = estimator.acquisition_scores(X_cand, f_star)

        best_random_idx = int(np.argmax(scores))
        best_config = candidates[best_random_idx]
        best_score = float(scores[best_random_idx])

        # 2. Local-search starting points: top-k from random pool + incumbent
        top_k_idx = np.argsort(scores)[-self.n_local_starts:][::-1]
        local_starts = [candidates[i] for i in top_k_idx]
        if incumbent is not None:
            local_starts.append(incumbent)

        # 3. Local search from each start
        for start in local_starts:
            current = start
            current_score = self._score_single(current, estimator, f_star)

            for _ in range(self.max_local_steps):
                # ConfigSpace's neighbour generation can raise when a config
                # sits at a hyperparameter boundary ("Failed to find enough
                # neighbors").  Treat that as "no neighbours" and stop the local
                # search from this start rather than aborting the whole task.
                try:
                    nbrs = list(neighbors(current, rng=rng))
                except Exception:
                    break
                if not nbrs:
                    break
                X_nbr = self.featurizer.transform(nbrs)
                nbr_scores = estimator.acquisition_scores(X_nbr, f_star)
                best_nbr_idx = int(np.argmax(nbr_scores))
                best_nbr_score = float(nbr_scores[best_nbr_idx])

                if best_nbr_score > current_score:
                    current = nbrs[best_nbr_idx]
                    current_score = best_nbr_score
                else:
                    break  # local optimum reached

            if current_score > best_score:
                best_score = current_score
                best_config = current

        return best_config
