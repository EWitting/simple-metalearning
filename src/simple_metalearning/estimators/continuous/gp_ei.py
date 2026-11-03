"""Gaussian Process EI surrogate for continuous-space BO."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler

from simple_metalearning.estimators.acquisition import EIGaussianAcquisition
from .base import ContinuousEstimator


class ContinuousGPEIEstimator(ContinuousEstimator):
    """Single-task GP surrogate with ARD Matern-5/2 kernel.

    Mirrors the kernel choice of ``task_only_botorch_gp_ei.py`` but uses
    sklearn's GP for simplicity.  Input features are standardised before
    fitting (important for ARD to work correctly).

    Args:
        dataset: Not used (single-task; accepted for factory compatibility).
        gp_model: Prototype GaussianProcessRegressor.  Defaults to
            Matern-5/2 with a ConstantKernel amplitude.
        acquisition_fn: Override the default ``EIGaussianAcquisition``.
        normalize_y: Standardise targets before fitting (default True).
    """

    def __init__(
        self,
        dataset=None,
        gp_model: GaussianProcessRegressor | None = None,
        acquisition_fn=None,
        normalize_y: bool = True,
    ):
        super().__init__(dataset=dataset)
        if gp_model is None:
            kernel = ConstantKernel(1.0) * Matern(nu=2.5)
            gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=normalize_y,
                n_restarts_optimizer=5,
            )
        self._gp_proto = gp_model
        self._acq_fn_obj = acquisition_fn or EIGaussianAcquisition()
        self._model: GaussianProcessRegressor | None = None
        self._scaler = StandardScaler()

    @property
    def _acq_fn(self):
        return self._acq_fn_obj

    def _refit(self) -> None:
        if self.n_obs < 2:
            return
        X = self._scaler.fit_transform(self._X)
        model = clone(self._gp_proto)
        model.fit(X, self._y)
        self._model = model

    def predict_mu_sigma(
        self, X_cand: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self._model is None:
            mu = np.full(len(X_cand), float(np.mean(self._y)) if self._obs_y else 0.0)
            return mu, np.ones(len(X_cand))
        X_scaled = self._scaler.transform(X_cand)
        mu, sigma = self._model.predict(X_scaled, return_std=True)
        return mu, sigma

    def acquisition_scores(
        self,
        X_cand: np.ndarray,
        f_star: float,
        known_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mu, sigma = self.predict_mu_sigma(X_cand)
        dummy_mask = np.zeros(len(X_cand), dtype=bool)
        return self._acq_fn_obj.compute(mu, sigma, f_star, dummy_mask)
