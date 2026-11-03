"""Abstract base class for continuous-space BO surrogates.

Unlike the discrete :class:`~simple_metalearning.estimators.estimator.Estimator`, this
interface does not assume a fixed flow set.  It operates directly on feature
matrices (produced by :class:`~simple_metalearning.search_spaces.xgboost_space.ConfigFeaturizer`)
and accumulates (X, y) observations as BO proceeds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ContinuousEstimator(ABC):
    """Surrogate model for continuous-space Bayesian optimisation.

    Subclasses implement :meth:`fit` (meta-training; no-op for single-task
    surrogates), :meth:`observe` (incorporate new live evaluation), and
    :meth:`predict_mu_sigma` (posterior mean and std for candidate configs).

    The :meth:`acquisition_scores` method is concrete and delegates to an
    :class:`~simple_metalearning.estimators.acquisition.AcquisitionFunction`; subclasses
    may override it for non-standard acquisition (e.g. per-tree EI for RF).
    """

    def __init__(self, dataset=None):
        # dataset is the masked MetaDataset used for meta-training (e.g.
        # OOFRidge); single-task surrogates may leave it None.
        self.dataset = dataset
        self._obs_X: list[np.ndarray] = []
        self._obs_y: list[float] = []

    # ------------------------------------------------------------------
    # Meta-training hook (optional; single-task surrogates no-op)
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Meta-training.  Override for surrogates that learn across tasks."""
        pass

    # ------------------------------------------------------------------
    # Per-task observations
    # ------------------------------------------------------------------

    def observe(self, X: np.ndarray, y: float | list[float]) -> None:
        """Incorporate one or more new observations for the current task.

        Args:
            X: Feature matrix of shape ``(n, n_features)`` for the new configs.
            y: Observed performance value(s).  Scalar or list of length *n*.
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]
        ys = [y] if np.isscalar(y) else list(y)
        for i, yi in enumerate(ys):
            self._obs_X.append(X[i])
            self._obs_y.append(float(yi))
        self._refit()

    def _refit(self) -> None:
        """Refit the surrogate on all accumulated observations."""
        pass

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @abstractmethod
    def predict_mu_sigma(
        self, X_cand: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict (mean, std) for candidate feature matrix.

        Args:
            X_cand: Shape ``(n_cand, n_features)``.

        Returns:
            Tuple ``(mu, sigma)`` where sigma may be ``None`` if the surrogate
            does not provide calibrated uncertainty.
        """

    # ------------------------------------------------------------------
    # Acquisition scores
    # ------------------------------------------------------------------

    def acquisition_scores(
        self,
        X_cand: np.ndarray,
        f_star: float,
        known_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Score candidates via the configured acquisition function.

        Subclasses may override (e.g. RF per-tree EI).

        Args:
            X_cand: Shape ``(n_cand, n_features)``.
            f_star: Best observed performance so far.
            known_mask: Not used for continuous estimators (all candidates are
                unevaluated); accepted for API compatibility.

        Returns:
            Score array of shape ``(n_cand,)`` — higher = better next candidate.
        """
        mu, sigma = self.predict_mu_sigma(X_cand)
        # Default known_mask: nothing observed in the candidate set
        dummy_mask = np.zeros(len(X_cand), dtype=bool)
        return self._acq_fn.compute(mu, sigma, f_star, dummy_mask)

    @property
    def _acq_fn(self):
        raise NotImplementedError(
            "Subclasses must expose self._acq_fn or override acquisition_scores()"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _X(self) -> np.ndarray:
        """Stacked observation feature matrix, shape (n_obs, n_features)."""
        if not self._obs_X:
            return np.empty((0, 0))
        return np.vstack(self._obs_X)

    @property
    def _y(self) -> np.ndarray:
        """Observation targets, shape (n_obs,)."""
        return np.array(self._obs_y, dtype=float)

    @property
    def n_obs(self) -> int:
        return len(self._obs_y)
