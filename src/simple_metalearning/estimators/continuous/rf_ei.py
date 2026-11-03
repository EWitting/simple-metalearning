"""Random Forest EI surrogate for continuous-space BO.

Trains one RandomForestRegressor per task on observed (config_features, perf)
pairs.  Uses SMAC-style per-tree EI via
:class:`~simple_metalearning.estimators.acquisition.EIForestAcquisition`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from simple_metalearning.estimators.acquisition import EIForestAcquisition
from .base import ContinuousEstimator


class ContinuousRFEIEstimator(ContinuousEstimator):
    """Single-task Random Forest surrogate for continuous BO.

    Args:
        dataset: Not used (single-task; accepted for factory compatibility).
        rf_model: Prototype RandomForestRegressor.  Will be cloned each time
            the surrogate is refit.
        acquisition_fn: Override the default ``EIForestAcquisition``.
    """

    def __init__(
        self,
        dataset=None,
        rf_model: RandomForestRegressor | None = None,
        acquisition_fn=None,
    ):
        super().__init__(dataset=dataset)
        if rf_model is None:
            rf_model = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=3, random_state=42
            )
        self._rf_proto = rf_model
        self._acq_fn_obj = acquisition_fn or EIForestAcquisition()
        self._model: RandomForestRegressor | None = None

    @property
    def _acq_fn(self):
        return self._acq_fn_obj

    def _refit(self) -> None:
        if self.n_obs < 2:
            return
        model = clone(self._rf_proto)
        model.fit(self._X, self._y)
        self._model = model

    def predict_mu_sigma(
        self, X_cand: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self._model is None:
            mu = np.full(len(X_cand), float(np.mean(self._y)) if self._obs_y else 0.0)
            return mu, np.zeros(len(X_cand))
        per_tree = np.array([t.predict(X_cand) for t in self._model.estimators_])
        mu = per_tree.mean(axis=0)
        sigma = per_tree.std(axis=0)
        return mu, sigma

    def acquisition_scores(
        self,
        X_cand: np.ndarray,
        f_star: float,
        known_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(X_cand))
        per_tree = np.array([t.predict(X_cand) for t in self._model.estimators_])
        return self._acq_fn_obj.compute_from_trees(per_tree, f_star)
