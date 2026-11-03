"""Task-only Random Forest estimator with Expected Improvement.

This estimator trains a separate Random Forest model for each task using only the
task's metafeatures. It implements the suggest method using Expected Improvement (EI)
calculated across the individual decision trees in the ensemble.
"""

from typing import Optional, Union, List

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from ..datasets.dataset import MetaDataset
from ..datasets.metafeature_transforms import PerformanceVectorTransform
from .estimator import Estimator
from .acquisition import AcquisitionFunction, EIForestAcquisition


class TaskOnlyRandomForestEIEstimator(Estimator):
    """Random Forest estimator that trains per-task models using task metafeatures only.

    Predicts performance for flows on a task by learning from observed flow
    performances on that task. Suggests next flows using Expected Improvement
    computed across individual decision trees (SMAC-style).

    Args:
        dataset: MetaDataset with performance matrix and flow metadata.
        rf_model: A RandomForestRegressor instance to use (will be cloned per task).
        performance_transform: Optional :class:`PerformanceVectorTransform`.
            When supplied, per-flow performance vectors are appended to
            ``flow_metadata``.  The transform is fitted once in :meth:`fit`
            on the (already test-task-masked) performance matrix so there is
            no leakage.  At least one of ``flow_metadata`` or
            ``performance_transform`` must be provided.
        f_star_from_predictions: If False (default), ``f_star`` is the best
            *observed* performance value — correct for deterministic evaluations.
            Set to True to use the best *predicted* mean over known flows, which
            is more appropriate when observations are noisy.
        acquisition_fn: Acquisition function.  Defaults to
            :class:`~.acquisition.EIForestAcquisition` (SMAC-style per-tree EI).
        use_metadata: If True (default), ``flow_metadata`` is included in the
            flow feature matrix.  Set to False to use only ``performance_transform``
            features (pure collaborative filtering with no content signal).
    """

    def __init__(
        self,
        dataset: MetaDataset,
        rf_model: RandomForestRegressor,
        performance_transform: Optional[PerformanceVectorTransform] = None,
        f_star_from_predictions: bool = False,
        acquisition_fn: Optional[AcquisitionFunction] = None,
        use_metadata: bool = True,
    ):
        # Default to EIForestAcquisition — uses per-tree predictions (SMAC-style)
        super().__init__(dataset, acquisition_fn or EIForestAcquisition())
        self.rf_model = rf_model
        self.performance_transform = performance_transform
        self.f_star_from_predictions = f_star_from_predictions
        self.use_metadata = use_metadata

        self.n_flows = len(dataset.performance)
        self.n_tasks = len(dataset.performance.columns)

        # Store one model per task
        self.task_models_: dict[int, RandomForestRegressor] = {}

        if (not use_metadata or dataset.flow_metadata is None) and performance_transform is None:
            raise ValueError(
                "TaskOnlyRandomForestEIEstimator requires at least one of "
                "flow_metadata (with use_metadata=True) or performance_transform."
            )

        # Initial fit (fits performance_transform if provided)
        self.fit()

    def fit(self) -> None:
        """Fit the performance transform (if provided); per-task models are lazy."""
        if self.performance_transform is not None:
            self.performance_transform.fit(self.dataset.performance)

    def _get_flow_features(self) -> np.ndarray:
        """Return the flow feature matrix, optionally augmented with perf vectors.

        Combines ``flow_metadata`` (content features) and/or the output of
        ``performance_transform`` (collaborative features from the performance
        matrix rows).  The transform was fitted on the already-masked dataset
        so test-task columns are all-NaN and do not leak into the features.
        """
        parts = []
        if self.use_metadata and self.dataset.flow_metadata is not None:
            parts.append(self.dataset.flow_metadata.values.astype(float))
        if self.performance_transform is not None:
            parts.append(self.performance_transform.transform(self.dataset.performance))
        return np.concatenate(parts, axis=1)

    def _fit_task(self, task_idx: int) -> None:
        """Fit a Random Forest model for a specific task.

        Args:
            task_idx: Index of the task to fit
        """
        # Get performance for this task
        task_performance = self.dataset.performance.iloc[:, task_idx].values

        # Find non-NaN observations
        observed_mask = ~np.isnan(task_performance)

        # Skip if less than two observations
        if observed_mask.sum() < 2:
            return

        X = self._get_flow_features()
        y = task_performance

        # Filter to observed only
        X_observed = X[observed_mask]
        y_observed = y[observed_mask]

        # Clone and fit Random Forest
        model = clone(self.rf_model)
        model.fit(X_observed, y_observed)

        # Store the fitted model
        self.task_models_[task_idx] = model

    def predict(self, task_index: int) -> np.ndarray:
        """Return predicted scores for all flows on the target task.

        Args:
            task_index: Index of the target task

        Returns:
            Array of predicted scores for each flow (higher is better)
        """
        self._fit_task(task_index)
        
        # In case fit not successful, return zeros
        if task_index not in self.task_models_:
            return np.zeros(self.n_flows)

        X = self._get_flow_features()
        predictions = self.task_models_[task_index].predict(X)

        return predictions

    def suggest(
        self, task_index: int, predictions: Optional[Union[np.ndarray, List[float]]] = None
    ) -> np.ndarray:
        """Suggest the next flow using SMAC-style per-tree Expected Improvement.

        Computes f_star from observed values (or predicted means when
        ``f_star_from_predictions=True``), then uses
        :class:`~.acquisition.EIForestAcquisition` to score all unknown flows.
        Falls back to :meth:`~.estimator.Estimator.suggest` (greedy) when the
        task model has not been fitted successfully.
        """
        # Fit model if not yet done
        if task_index not in self.task_models_:
            self._fit_task(task_index)

        perf_col = self.dataset.performance.iloc[:, task_index].values
        known_mask = ~np.isnan(perf_col)

        # If fitting failed, delegate to base class (greedy fallback)
        if task_index not in self.task_models_:
            return super().suggest(task_index, predictions)

        model = self.task_models_[task_index]
        X_all = self._get_flow_features()

        # Determine f_star
        if self.f_star_from_predictions:
            f_star = float(np.max(model.predict(X_all[known_mask]))) if np.any(known_mask) else float("-inf")
        else:
            f_star = float(np.max(perf_col[known_mask])) if np.any(known_mask) else float("-inf")

        # Per-tree predictions for ALL flows — shape (n_estimators, n_flows)
        per_tree_preds = np.array([tree.predict(X_all) for tree in model.estimators_])

        # Allow acquisition function to pre-compute task-specific structures
        # (e.g. MMRAcquisition builds its similarity matrix here).
        self.acquisition_fn.prepare(
            self.dataset.performance.values, known_mask
        )

        # Use EIForestAcquisition (or whatever acquisition_fn was configured)
        if isinstance(self.acquisition_fn, EIForestAcquisition):
            scores = self.acquisition_fn.compute_from_trees(per_tree_preds, f_star)
        else:
            # Non-forest acquisition: fall back to mean prediction + configured acq
            mu = np.mean(per_tree_preds, axis=0)
            sigma = np.std(per_tree_preds, axis=0)
            scores = self.acquisition_fn.compute(mu, sigma, f_star, known_mask)

        return scores

    def partial_fit(
        self,
        flow_index: Union[int, List[int]],
        task_index: int,
        performance: Union[float, List[float]],
    ) -> None:
        """Update the dataset with new observation and refit the task model.

        Args:
            flow_index: Index of the flow
            task_index: Index of the task
            performance: Observed performance value
        """
        # Update the dataset
        self.dataset.performance.iloc[flow_index, task_index] = performance

        # Refit only the affected task model
        self._fit_task(task_index)
