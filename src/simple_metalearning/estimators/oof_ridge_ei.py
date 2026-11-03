"""OOF-Ridge EI estimator: BayesianRidge on out-of-fold RF predictions.

Instead of using raw performance-matrix columns as features, this estimator
first trains per-task Random Forests on hyperparameter features using k-fold
cross-validation to produce out-of-fold (OOF) predictions. These OOF predictions
form the feature matrix for the BayesianRidge model, replacing the observed
performance values from prior tasks.

During BO on the target task, each newly observed true performance value is
used as a training label (y) for the BayesianRidge model; the OOF predictions
serve as fixed features (X) throughout.
"""

from typing import Optional, Union, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold

from ..datasets.dataset import MetaDataset
from .bayesian_ridge_ei import BayesianRidgeEIEstimator
from .acquisition import AcquisitionFunction, EIGaussianAcquisition, MMRAcquisition


def build_oof_rf_matrix(
    performance: pd.DataFrame,
    flow_features: np.ndarray,
    n_splits: int = 5,
    rf_n_estimators: int = 100,
    rf_min_samples_leaf: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Build an (n_flows, n_tasks) matrix of OOF Random Forest predictions.

    For each task column:
    - Flows with known performance: k-fold cross-validation OOF predictions
      (or in-sample if fewer than n_splits known flows).
    - Flows with unknown performance (NaN): predictions from an RF trained on
      all known flows for that task.
    - Tasks with fewer than 2 known flows: filled with the global mean.

    Args:
        performance: (n_flows, n_tasks) DataFrame with NaN for unevaluated pairs.
        flow_features: (n_flows, n_hyper_features) array of flow content features
            (hyperparameter encodings).
        n_splits: Number of folds for cross-validation.
        rf_n_estimators: Number of trees in each Random Forest.
        rf_min_samples_leaf: Minimum leaf size (prevents over-exploration; SMAC default=3).
        random_state: Random seed for reproducibility.

    Returns:
        Dense (n_flows, n_tasks) float array with no NaN values.
    """
    n_flows, n_tasks = performance.shape
    global_mean = float(np.nanmean(performance.values))
    if np.isnan(global_mean):
        global_mean = 0.0

    oof_matrix = np.full((n_flows, n_tasks), global_mean)

    rf_proto = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        min_samples_leaf=rf_min_samples_leaf,
        random_state=random_state,
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for task_idx in range(n_tasks):
        y = performance.iloc[:, task_idx].values
        known = ~np.isnan(y)
        n_known = int(known.sum())

        if n_known < 2:
            continue  # leave column as global_mean

        known_indices = np.where(known)[0]
        unknown_indices = np.where(~known)[0]
        X_known = flow_features[known]
        y_known = y[known]

        oof_preds = np.empty(n_known)

        if n_known >= n_splits:
            for fold_train, fold_val in kf.split(X_known):
                rf = clone(rf_proto)
                rf.fit(X_known[fold_train], y_known[fold_train])
                oof_preds[fold_val] = rf.predict(X_known[fold_val])
        else:
            # Too few samples for k-fold: in-sample predictions
            rf = clone(rf_proto)
            rf.fit(X_known, y_known)
            oof_preds = rf.predict(X_known)

        oof_matrix[known_indices, task_idx] = oof_preds

        if len(unknown_indices) > 0:
            rf_full = clone(rf_proto)
            rf_full.fit(X_known, y_known)
            oof_matrix[unknown_indices, task_idx] = rf_full.predict(flow_features[unknown_indices])

    return oof_matrix


class OOFRidgeEIEstimator(BayesianRidgeEIEstimator):
    """BayesianRidge EI using OOF RF predictions as the feature matrix.

    Replaces the raw performance-vector columns with out-of-fold Random Forest
    predictions (trained on hyperparameter features per task). This decouples the
    feature representation from the observed performances on prior tasks, letting
    the BayesianRidge model learn a mapping from predicted-performance-vectors to
    target-task performance.

    Fixed settings: fit_intercept=False, prior_equal_weight=False (sklearn default
    priors), matching perf_vector_ridge_ei_no_intercept semantics but on OOF features.

    Args:
        dataset: MetaDataset with performance matrix and flow_metadata (required).
        rf_n_estimators: Number of trees per per-task Random Forest.
        rf_min_samples_leaf: Minimum leaf size for all RFs.
        n_splits: Number of folds used to build OOF predictions in fit().
        acquisition_fn: Acquisition function; defaults to EIGaussianAcquisition.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        rf_n_estimators: int = 100,
        rf_min_samples_leaf: int = 3,
        n_splits: int = 5,
        acquisition_fn: Optional[AcquisitionFunction] = None,
    ):
        super().__init__(
            dataset=dataset,
            fit_intercept=False,
            prior_equal_weight=False,
            acquisition_fn=acquisition_fn or EIGaussianAcquisition(),
        )
        self.rf_n_estimators = rf_n_estimators
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.n_splits = n_splits
        self.oof_matrix_: Optional[np.ndarray] = None

    def fit(self) -> None:
        if self.dataset.flow_metadata is None:
            raise ValueError("OOFRidgeEIEstimator requires flow_metadata (hyperparameter features)")
        self.oof_matrix_ = build_oof_rf_matrix(
            performance=self.dataset.performance,
            flow_features=self.dataset.flow_metadata.values.astype(float),
            n_splits=self.n_splits,
            rf_n_estimators=self.rf_n_estimators,
            rf_min_samples_leaf=self.rf_min_samples_leaf,
        )

    def _build_predict_features(self, feature_indices) -> np.ndarray:
        return self.oof_matrix_[:, feature_indices]

    def _build_known_features(self, known_mask: np.ndarray, feature_indices) -> np.ndarray:
        return self.oof_matrix_[np.ix_(known_mask, feature_indices)]

    def _train_model(self, task_index: int) -> None:
        if self.oof_matrix_ is None:
            self.models[task_index] = (None, None)
            return

        df = self.dataset.performance
        y_full = df.iloc[:, task_index]
        known_flow_mask = y_full.notna()

        if known_flow_mask.sum() < 2:
            self.models[task_index] = (None, None)
            return

        y = y_full[known_flow_mask].values.astype(float)

        feature_indices = np.array([i for i in range(df.shape[1]) if i != task_index])
        if len(feature_indices) == 0:
            self.models[task_index] = (None, None)
            return

        X = self.oof_matrix_[np.ix_(known_flow_mask.values, feature_indices)]

        model = BayesianRidge(fit_intercept=False)
        try:
            model.fit(X, y)
            self.models[task_index] = (model, feature_indices)
        except Exception:
            self.models[task_index] = (None, None)


class OOFRidgeMMREstimator(OOFRidgeEIEstimator):
    """OOFRidgeEI variant that uses the OOF matrix for MMR flow similarity.

    MMRAcquisition normally builds flow-flow cosine similarity from the raw
    (sparse) performance matrix. This subclass overrides suggest() to pass the
    dense OOF RF prediction matrix to prepare() instead, so the similarity
    structure reflects predicted rather than observed performance.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        rf_n_estimators: int = 100,
        rf_min_samples_leaf: int = 3,
        n_splits: int = 5,
        acquisition_fn: Optional[AcquisitionFunction] = None,
    ):
        super().__init__(
            dataset=dataset,
            rf_n_estimators=rf_n_estimators,
            rf_min_samples_leaf=rf_min_samples_leaf,
            n_splits=n_splits,
            acquisition_fn=acquisition_fn or MMRAcquisition(),
        )

    def suggest(
        self, task_index: int, predictions: Optional[Union[np.ndarray, List]] = None
    ) -> np.ndarray:
        if task_index not in self.models:
            self._train_model(task_index)

        mu = np.asarray(self.predict(task_index) if predictions is None else predictions)
        sigma = self.predict_uncertainty(task_index)

        perf_col = self.dataset.require_matrix().iloc[:, task_index].values
        known_mask = ~np.isnan(perf_col)
        f_star = float(np.max(perf_col[known_mask])) if np.any(known_mask) else float("-inf")

        # Use OOF matrix for MMR similarity; fall back to raw performance if not built yet.
        sim_source = self.oof_matrix_ if self.oof_matrix_ is not None else self.dataset.performance.values
        self.acquisition_fn.prepare(sim_source, known_mask)

        cost = self.dataset.cost.iloc[:, task_index].values if self.dataset.cost is not None else None
        return self.acquisition_fn.compute(mu, sigma, f_star, known_mask, cost=cost)
