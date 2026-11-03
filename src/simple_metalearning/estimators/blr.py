"""Fixed-precision Bayesian Linear Regression head for Linear-Perf (ablation).

Linear-Perf currently uses sklearn ``BayesianRidge`` (BRR), which re-estimates the
noise and weight-prior precisions by evidence (marginal-likelihood) maximisation on
*every* per-task fit.  This module provides the same estimator but with a
**fixed-precision** Bayesian linear regression (BLR) head, using the exact closed
form of the ABLR baseline (``ablr.ABLREstimator._blr``):

    posterior precision   A = lam * I + alpha * XᵀX      (lam = weight precision,
    posterior mean        m = alpha * A⁻¹ Xᵀ y            alpha = noise precision)
    predictive mean       x m
    predictive variance   1/alpha + x A⁻¹ xᵀ

This matches ABLR's BLR head (ABLR learns alpha/lam offline by marginal likelihood
and then *fixes* them at BO time; here they are fixed too).  Everything else --
the performance-vector feature construction, cold-start behaviour, EI acquisition --
is inherited from ``BayesianRidgeEIEstimator`` unchanged, so the only variable
between Linear-Perf-BRR and Linear-Perf-BLR is per-fit precision adaptation.

The fixed precisions default to a one-shot evidence estimate: a single sklearn
``BayesianRidge`` fit on the first usable (X, y) at fit time provides reference
``alpha`` and ``lambda`` that are then frozen.  Pass ``blr_alpha`` / ``blr_lambda``
to override.
"""

from typing import Optional

import numpy as np
from sklearn.linear_model import BayesianRidge

from ..datasets.dataset import MetaDataset
from .bayesian_ridge_ei import BayesianRidgeEIEstimator


class _FixedBLR:
    """Closed-form Bayesian linear regression with fixed precisions.

    Mimics the subset of the sklearn ``BayesianRidge`` predict API used by
    ``BayesianRidgeEIEstimator`` (``predict(X, return_std=...)``).
    """

    def __init__(self, alpha: float, lam: float):
        self.alpha = float(alpha)   # noise precision
        self.lam = float(lam)       # weight-prior precision
        self.m_ = None
        self.Sigma_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FixedBLR":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        d = X.shape[1]
        A = self.lam * np.eye(d) + self.alpha * (X.T @ X)
        self.Sigma_ = np.linalg.inv(A)
        self.m_ = self.alpha * (self.Sigma_ @ (X.T @ y))
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        X = np.asarray(X, dtype=float)
        mean = X @ self.m_
        if not return_std:
            return mean
        var = 1.0 / self.alpha + np.einsum("ij,jk,ik->i", X, self.Sigma_, X)
        return mean, np.sqrt(np.clip(var, 1e-12, None))


class BLRFixedEIEstimator(BayesianRidgeEIEstimator):
    """Linear-Perf with a fixed-precision BLR head (ablation vs BRR)."""

    def __init__(
        self,
        dataset: MetaDataset,
        blr_alpha: Optional[float] = None,
        blr_lambda: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self.blr_alpha = blr_alpha
        self.blr_lambda = blr_lambda

    # -- reference precisions from one sklearn BayesianRidge fit, once ----------
    def _ensure_precisions(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.blr_alpha is not None and self.blr_lambda is not None:
            return
        try:
            br = BayesianRidge(fit_intercept=self.fit_intercept)
            br.fit(X, y)
            self.blr_alpha = float(br.alpha_)
            self.blr_lambda = float(br.lambda_)
        except Exception:
            self.blr_alpha = 1.0
            self.blr_lambda = 1.0

    # -- replace the per-task head: same feature construction as the parent -----
    def _train_model(self, task_index: int) -> None:
        df = self.dataset.performance
        y_full = df.iloc[:, task_index]
        known_flow_mask = y_full.notna()
        if known_flow_mask.sum() < 2:
            self.models[task_index] = (None, None)
            return
        y = y_full[known_flow_mask].values.astype(float)

        if self.use_metadata:
            if self.dataset.flow_metadata is None:
                raise ValueError("use_metadata=True but dataset has no flow_metadata")
            X_full = self.dataset.flow_metadata.values.astype(float)
            X = self._impute(X_full[known_flow_mask.values], fill=0.0)
            feature_indices = None
        else:
            other_task_indices = [i for i in range(df.shape[1]) if i != task_index]
            if not other_task_indices:
                self.models[task_index] = (None, None)
                return
            X_candidates = df.iloc[known_flow_mask.values, other_task_indices]
            valid_feature_mask = X_candidates.notna().all(axis=0)
            feature_indices = np.array(other_task_indices)[valid_feature_mask.values]
            if len(feature_indices) == 0:
                self.models[task_index] = (None, None)
                return
            X = df.iloc[known_flow_mask.values, feature_indices].values.astype(float)

        self._ensure_precisions(X, y)
        try:
            model = _FixedBLR(self.blr_alpha, self.blr_lambda).fit(X, y)
            self.models[task_index] = (model, feature_indices)
        except Exception:
            self.models[task_index] = (None, None)
