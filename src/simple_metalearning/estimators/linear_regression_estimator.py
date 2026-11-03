"""OLS and plain Ridge per-task surrogates on performance-vector features."""

from typing import Optional, Union, List

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from ..datasets.dataset import MetaDataset
from .estimator import Estimator
from .acquisition import AcquisitionFunction, GreedyAcquisition


class LinearRegressionEstimator(Estimator):
    """Per-task OLS or Ridge regression using performance-vector features.

    Other tasks' performance columns serve as features (collaborative filtering).
    No native predictive uncertainty — pairs naturally with Greedy or MMR
    acquisition. For uncertainty-aware acquisition (EI/LogEI) use
    EnsembleLinearEstimator instead.

    Args:
        model_type: "ols" for ordinary least squares, "ridge" for L2-penalized.
        alpha: Regularization strength (Ridge only).
        fit_intercept: Whether to include an intercept term.
        acquisition_fn: Defaults to GreedyAcquisition.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        model_type: str = "ols",
        alpha: float = 1.0,
        fit_intercept: bool = False,
        acquisition_fn: Optional[AcquisitionFunction] = None,
    ):
        super().__init__(dataset, acquisition_fn or GreedyAcquisition())
        self.model_type = model_type
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.models = {}  # task_index -> (model | None, feature_indices | None)

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
        self._train_model(task_index)

    def predict(self, task_index: int) -> np.ndarray:
        if task_index not in self.models:
            self._train_model(task_index)

        model, feature_indices = self.models[task_index]

        if model is None:
            row_means = np.nanmean(self.dataset.performance.values, axis=1)
            nan_mask = np.isnan(row_means)
            if nan_mask.any():
                row_means[nan_mask] = self.global_mean
            return row_means

        X = self._build_predict_features(feature_indices)
        try:
            return model.predict(X).astype(float)
        except Exception:
            return np.full(self.dataset.performance.shape[0], self.global_mean)

    @staticmethod
    def _impute(X: np.ndarray, fill: float) -> np.ndarray:
        col_means = np.nanmean(X, axis=0)
        col_means[np.isnan(col_means)] = fill
        inds = np.where(np.isnan(X))
        X = X.copy()
        X[inds] = np.take(col_means, inds[1])
        return X

    def _build_predict_features(self, feature_indices) -> np.ndarray:
        X = self.dataset.performance.iloc[:, feature_indices].values.astype(float)
        return self._impute(X, fill=self.global_mean)

    def _train_model(self, task_index: int) -> None:
        df = self.dataset.performance
        y_full = df.iloc[:, task_index]
        known_flow_mask = y_full.notna()

        if known_flow_mask.sum() < 2:
            self.models[task_index] = (None, None)
            return

        y = y_full[known_flow_mask].values.astype(float)
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
        model = self._make_model()
        try:
            model.fit(X, y)
            self.models[task_index] = (model, feature_indices)
        except Exception:
            self.models[task_index] = (None, None)
