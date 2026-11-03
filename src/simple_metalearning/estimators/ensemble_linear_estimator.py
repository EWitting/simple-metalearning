"""Bootstrap ensemble of OLS/Ridge with uncertainty estimates for EI/LogEI."""

from typing import Optional, Union, List

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from ..datasets.dataset import MetaDataset
from .estimator import Estimator
from .acquisition import AcquisitionFunction, EIGaussianAcquisition


class EnsembleLinearEstimator(Estimator):
    """Bootstrap ensemble of OLS or Ridge for uncertainty-aware acquisition.

    Fits n_estimators models on bootstrap resamples (with replacement) of the
    observed (flow, performance) pairs. Ensemble mean and std serve as mu/sigma
    for EI or LogEI acquisition.

    Args:
        model_type: "ols" for ordinary least squares, "ridge" for Ridge.
        alpha: Ridge regularization strength (ignored for OLS).
        n_estimators: Number of bootstrap models. Defaults to 50.
        fit_intercept: Whether to include an intercept term.
        acquisition_fn: Defaults to EIGaussianAcquisition.
        random_state: Seed for reproducible bootstrap sampling.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        model_type: str = "ols",
        alpha: float = 1.0,
        n_estimators: int = 50,
        fit_intercept: bool = False,
        acquisition_fn: Optional[AcquisitionFunction] = None,
        random_state: int = 42,
    ):
        super().__init__(dataset, acquisition_fn or EIGaussianAcquisition())
        self.model_type = model_type
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.models = {}  # task_index -> (list[fitted_model], feature_indices | None)

        perf_values = self.dataset.performance.values.astype(float)
        self.global_mean = np.nanmean(perf_values)
        if np.isnan(self.global_mean):
            self.global_mean = 0.0

    def _make_model(self):
        if self.model_type == "ols":
            return LinearRegression(fit_intercept=self.fit_intercept)
        return Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)

    def fit(self) -> None:
        pass

    def partial_fit(
        self,
        flow_index: Union[int, List[int]],
        task_index: int,
        performance: Union[float, List[float]],
    ) -> None:
        self.dataset.performance.iloc[flow_index, task_index] = performance
        self._train_models(task_index)

    def predict(self, task_index: int) -> np.ndarray:
        if task_index not in self.models:
            self._train_models(task_index)
        fitted, feature_indices = self.models[task_index]
        if not fitted:
            return self._fallback_mean()
        X = self._build_predict_features(feature_indices)
        preds = np.stack([m.predict(X) for m in fitted], axis=0)
        return preds.mean(axis=0)

    def predict_uncertainty(self, task_index: int) -> Optional[np.ndarray]:
        if task_index not in self.models:
            self._train_models(task_index)
        fitted, feature_indices = self.models[task_index]
        if len(fitted) < 2:
            return None
        X = self._build_predict_features(feature_indices)
        preds = np.stack([m.predict(X) for m in fitted], axis=0)
        return preds.std(axis=0)

    def _fallback_mean(self) -> np.ndarray:
        row_means = np.nanmean(self.dataset.performance.values, axis=1)
        nan_mask = np.isnan(row_means)
        if nan_mask.any():
            row_means[nan_mask] = self.global_mean
        return row_means

    def _build_predict_features(self, feature_indices) -> np.ndarray:
        X = self.dataset.performance.iloc[:, feature_indices].values.astype(float)
        col_means = np.nanmean(X, axis=0)
        col_means[np.isnan(col_means)] = self.global_mean
        inds = np.where(np.isnan(X))
        X = X.copy()
        X[inds] = np.take(col_means, inds[1])
        return X

    def _train_models(self, task_index: int) -> None:
        df = self.dataset.performance
        y_full = df.iloc[:, task_index]
        known_flow_mask = y_full.notna()

        if known_flow_mask.sum() < 2:
            self.models[task_index] = ([], None)
            return

        y = y_full[known_flow_mask].values.astype(float)
        other_task_indices = [i for i in range(df.shape[1]) if i != task_index]

        if not other_task_indices:
            self.models[task_index] = ([], None)
            return

        X_candidates = df.iloc[known_flow_mask.values, other_task_indices]
        valid_feature_mask = X_candidates.notna().all(axis=0)
        feature_indices = np.array(other_task_indices)[valid_feature_mask.values]

        if len(feature_indices) == 0:
            self.models[task_index] = ([], None)
            return

        X = df.iloc[known_flow_mask.values, feature_indices].values.astype(float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        fitted = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            model = self._make_model()
            try:
                model.fit(X[idx], y[idx])
                fitted.append(model)
            except Exception:
                pass

        self.models[task_index] = (fitted, feature_indices)
