from typing import Optional, Union, List

import numpy as np
from sklearn.linear_model import BayesianRidge

from ..datasets.dataset import MetaDataset
from .estimator import Estimator
from .acquisition import AcquisitionFunction, EIGaussianAcquisition


class BayesianRidgeEIEstimator(Estimator):
    """Bayesian Ridge Regression estimator with Expected Improvement.

    Two feature modes:
    - ``use_metadata=False`` (default): uses other tasks' performance columns as
      features — collaborative filtering via linear regression on the perf matrix.
    - ``use_metadata=True``: uses ``flow_metadata`` (hyperparameter features) as
      features — per-task content-based linear model.

    In both modes, BayesianRidge provides calibrated predictive uncertainty which
    is fed into ``EIGaussianAcquisition``.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        fit_intercept: bool = True,
        prior_equal_weight: bool = True,
        prior_shape: float = 2.0,
        f_star_from_predictions: bool = False,
        use_metadata: bool = False,
        warm_start_mean: bool = False,
        acquisition_fn: Optional[AcquisitionFunction] = None,
    ):
        super().__init__(dataset, acquisition_fn or EIGaussianAcquisition())
        self.fit_intercept = fit_intercept
        self.prior_equal_weight = prior_equal_weight
        self.prior_shape = prior_shape
        self.f_star_from_predictions = f_star_from_predictions
        self.use_metadata = use_metadata
        # Cold-start behaviour while <2 target observations exist (model unfit).
        # warm_start_mean=False (default): return a constant score so the BO loop
        # picks the first flows at random -- matching RF-EI / GP-EI, which return
        # zeros at cold start. Setting True restores the legacy mean-best warm
        # start (each flow scored by its mean performance over prior tasks), which
        # gave the linear estimators a free strong first pick that the other
        # surrogates never had. See scripts/coldstart_ablation.py.
        self.warm_start_mean = warm_start_mean

        if self.prior_equal_weight and self.prior_shape <= 1.0:
            raise ValueError("prior_shape must be > 1.0 when prior_equal_weight is True.")

        self.models = {}  # task_index -> (model, feature_indices_or_None)

        perf_values = self.dataset.performance.values.astype(float)
        self.global_mean = np.nanmean(perf_values)
        if np.isnan(self.global_mean):
            self.global_mean = 0.0

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

    def predict(self, task_index: int, return_std: bool = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        if task_index not in self.models:
            self._train_model(task_index)

        model, feature_indices = self.models[task_index]
        n_flows = self.dataset.performance.shape[0]

        if model is None:
            if not self.warm_start_mean:
                # Cold start: uninformative constant -> random first picks.
                preds = np.zeros(n_flows)
                if return_std:
                    return preds, np.zeros_like(preds)
                return preds
            row_means = np.nanmean(self.dataset.performance.values, axis=1)
            nan_mask = np.isnan(row_means)
            if nan_mask.any():
                row_means[nan_mask] = self.global_mean
            if return_std:
                return row_means, np.zeros_like(row_means)
            return row_means

        X_predict = self._build_predict_features(feature_indices)

        try:
            if return_std:
                return model.predict(X_predict, return_std=True)
            return model.predict(X_predict).astype(float)
        except Exception:
            preds = np.full(n_flows, self.global_mean)
            if return_std:
                return preds, np.zeros_like(preds)
            return preds

    def predict_uncertainty(self, task_index: int) -> Optional[np.ndarray]:
        if task_index not in self.models:
            self._train_model(task_index)

        model, feature_indices = self.models[task_index]
        if model is None:
            return None

        X_predict = self._build_predict_features(feature_indices)

        try:
            _, sigma = model.predict(X_predict, return_std=True)
            return sigma
        except Exception:
            return np.zeros(self.dataset.performance.shape[0])

    def suggest(
        self, task_index: int, predictions: Optional[Union[np.ndarray, List[float]]] = None
    ) -> np.ndarray:
        if task_index not in self.models:
            self._train_model(task_index)

        # Cold start (model unfit): with no warm start, return a constant so the
        # experiment loop selects the first flows uniformly at random.
        if self.models[task_index][0] is None and not self.warm_start_mean:
            return np.zeros(self.dataset.performance.shape[0])

        if not self.f_star_from_predictions:
            return super().suggest(task_index, predictions)

        perf_col = self.dataset.performance.iloc[:, task_index].values
        known_mask = ~np.isnan(perf_col)

        if predictions is None:
            mu = np.asarray(self.predict(task_index))
        else:
            mu = np.asarray(predictions)

        sigma = self.predict_uncertainty(task_index)

        model, feature_indices = self.models[task_index]
        if np.any(known_mask) and model is not None:
            X_known = self._build_known_features(known_mask, feature_indices)
            f_star = float(np.max(model.predict(X_known)))
        elif np.any(known_mask):
            f_star = float(np.max(perf_col[known_mask]))
        else:
            f_star = float("-inf")

        return self.acquisition_fn.compute(mu, sigma, f_star, known_mask)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _build_predict_features(self, feature_indices) -> np.ndarray:
        """Build (n_flows, n_features) matrix for prediction over all flows."""
        if self.use_metadata:
            X = self.dataset.flow_metadata.values.astype(float)
            return self._impute(X, fill=0.0)
        else:
            X = self.dataset.performance.iloc[:, feature_indices].values.astype(float)
            return self._impute(X, fill=self.global_mean)

    def _build_known_features(self, known_mask: np.ndarray, feature_indices) -> np.ndarray:
        """Build feature matrix restricted to flows in ``known_mask``."""
        if self.use_metadata:
            X = self.dataset.flow_metadata.values[known_mask].astype(float)
            return self._impute(X, fill=0.0)
        else:
            X = self.dataset.performance.iloc[known_mask, feature_indices].values.astype(float)
            return self._impute(X, fill=self.global_mean)

    @staticmethod
    def _impute(X: np.ndarray, fill: float) -> np.ndarray:
        col_means = np.nanmean(X, axis=0)
        col_means[np.isnan(col_means)] = fill
        inds = np.where(np.isnan(X))
        X = X.copy()
        X[inds] = np.take(col_means, inds[1])
        return X

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

        N = X.shape[1]
        if self.prior_equal_weight:
            l1 = self.prior_shape
            l2 = float(N) * (l1 - 1.0)
            model = BayesianRidge(
                lambda_1=l1,
                lambda_2=l2,
                lambda_init=1.0 / float(N),
                fit_intercept=self.fit_intercept,
            )
        else:
            model = BayesianRidge(fit_intercept=self.fit_intercept)

        try:
            model.fit(X, y)
            self.models[task_index] = (model, feature_indices)
        except Exception:
            self.models[task_index] = (None, None)
