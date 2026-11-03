"""Task-only Gaussian Process estimator with Expected Improvement.

This estimator trains a separate Gaussian Process Regressor for each task using only the
task's metafeatures. It implements the suggest method using Expected Improvement (EI)
calculated from the predictive mean and standard deviation of the Gaussian Process.
"""

import warnings
from typing import Optional, Union, List

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from ..datasets.dataset import MetaDataset
from ..datasets.metafeature_transforms import PerformanceVectorTransform
from .estimator import Estimator
from .acquisition import AcquisitionFunction, EIGaussianAcquisition


class TaskOnlyGPEIEstimator(Estimator):
    """Gaussian Process estimator that trains per-task models using flow metafeatures only.

    Predicts performance for flows on a task by learning from observed flow
    performances on that task. Suggests next flows using Expected Improvement
    with the exact GP posterior mean and standard deviation.

    Args:
        dataset: MetaDataset with performance matrix and flow metadata
        gp_model: A GaussianProcessRegressor instance to use (will be cloned per task)
        ard: Whether to enable Automatic Relevance Determination (ARD). If True,
            a separate length scale will be learned for each input dimension.
        f_star_from_predictions: If False (default), ``f_star`` is the best
            *observed* performance value — correct for deterministic evaluations.
            Set to True to use the best *predicted* GP mean over known flows,
            which can be preferable when observations are noisy.
        performance_transform: Optional :class:`PerformanceVectorTransform`.
            When supplied, per-flow performance vectors are appended to (or
            replace) ``flow_metadata`` as GP input features.  Fitted once in
            :meth:`fit` on the masked dataset to avoid test-task leakage.
        use_metadata: If True (default), ``flow_metadata`` is included in the
            GP input.  Set to False to use only ``performance_transform``
            features (pure collaborative GP).
    """

    def __init__(
        self,
        dataset: MetaDataset,
        gp_model: GaussianProcessRegressor,
        ard: bool = False,
        f_star_from_predictions: bool = False,
        acquisition_fn: Optional[AcquisitionFunction] = None,
        performance_transform: Optional[PerformanceVectorTransform] = None,
        use_metadata: bool = True,
    ):
        # Default to Gaussian EI — the GP provides calibrated uncertainty estimates
        super().__init__(dataset, acquisition_fn or EIGaussianAcquisition())
        self.gp_model = gp_model
        self.f_star_from_predictions = f_star_from_predictions
        self.performance_transform = performance_transform
        self.use_metadata = use_metadata

        self.n_flows = len(dataset.performance)
        self.n_tasks = len(dataset.performance.columns)

        if (not use_metadata or dataset.flow_metadata is None) and performance_transform is None:
            raise ValueError(
                "TaskOnlyGPEIEstimator requires either flow_metadata "
                "(with use_metadata=True) or a performance_transform (or both)."
            )

        if ard:
            n_dims = self._get_flow_features().shape[1]
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=[1.0] * n_dims,
                length_scale_bounds=(1e-2, 1e4)
            )
            self.gp_model.set_params(kernel=kernel, n_restarts_optimizer=2)

        # Store one model per task
        self.task_models_: dict[int, GaussianProcessRegressor] = {}
        # Store optimized kernels for warm starting per task
        self.task_kernels_: dict[int, any] = {}

        self.fit()

    def fit(self) -> None:
        """Fit the performance transform (if any); per-task GPs are fitted lazily."""
        if self.performance_transform is not None:
            self.performance_transform.fit(self.dataset.performance)

    def _get_flow_features(self) -> np.ndarray:
        """Return the (n_flows, d) feature matrix used as GP inputs.

        Combines ``flow_metadata`` (content) and/or the output of
        ``performance_transform`` (collaborative performance vectors).
        The transform was fitted on the already test-task-masked matrix, so
        there is no leakage from test tasks into the GP inputs.
        """
        parts = []
        if self.use_metadata and self.dataset.flow_metadata is not None:
            parts.append(self.dataset.flow_metadata.values.astype(float))
        if self.performance_transform is not None:
            parts.append(self.performance_transform.transform(self.dataset.performance))
        return np.concatenate(parts, axis=1)

    def _fit_task(self, task_idx: int) -> None:
        """Fit a Gaussian Process model for a specific task.

        Args:
            task_idx: Index of the task to fit
        """
        # Get performance for this task
        task_performance = self.dataset.performance.iloc[:, task_idx].values

        # Find non-NaN observations
        observed_mask = ~np.isnan(task_performance)

        # Skip if no observations
        if not observed_mask.any():
            return

        X = self._get_flow_features()
        y = task_performance

        # Filter to observed only
        X_observed = X[observed_mask]
        y_observed = y[observed_mask]

        # Clone and fit GP
        model = clone(self.gp_model)
        
        # Warm start: if we have a previous kernel for this task, use its parameters
        if task_idx in self.task_kernels_:
            model.set_params(kernel=self.task_kernels_[task_idx])
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_observed, y_observed)

        # Store the fitted model and the optimized kernel for warm starting
        self.task_models_[task_idx] = model
        self.task_kernels_[task_idx] = model.kernel_

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

    def predict_uncertainty(self, task_index: int) -> Optional[np.ndarray]:
        """Return GP posterior standard deviation for all flows.

        Args:
            task_index: Index of the target task.

        Returns:
            Array of shape ``(n_flows,)`` with predictive std, or ``None`` if
            the model has not been fitted yet.
        """
        if task_index not in self.task_models_:
            self._fit_task(task_index)

        if task_index not in self.task_models_:
            return None

        X = self._get_flow_features()
        _, sigma = self.task_models_[task_index].predict(X, return_std=True)
        return sigma

    def suggest(
        self, task_index: int, predictions: Optional[Union[np.ndarray, List[float]]] = None
    ) -> np.ndarray:
        """Suggest the next flow using the configured acquisition function.

        Fits the task model if not already done, then delegates to the base
        class implementation which calls :meth:`predict_uncertainty` and passes
        results to :attr:`acquisition_fn`.

        The ``f_star_from_predictions`` flag is handled here: when enabled,
        ``f_star`` is overridden with the best GP predicted mean over known
        flows instead of the best observed value.
        """
        # Ensure model is ready
        if task_index not in self.task_models_:
            self._fit_task(task_index)

        if not self.f_star_from_predictions:
            # Standard path: base class uses best observed value
            return super().suggest(task_index, predictions)

        # f_star_from_predictions=True: override f_star with best predicted mean
        perf_col = self.dataset.performance.iloc[:, task_index].values
        known_mask = ~np.isnan(perf_col)

        if predictions is None:
            mu = np.asarray(self.predict(task_index))
        else:
            mu = np.asarray(predictions)

        sigma = self.predict_uncertainty(task_index)

        if np.any(known_mask) and task_index in self.task_models_:
            X_known = self._get_flow_features()[known_mask]
            f_star = float(np.max(self.task_models_[task_index].predict(X_known)))
        elif np.any(known_mask):
            f_star = float(np.max(perf_col[known_mask]))
        else:
            f_star = float("-inf")

        return self.acquisition_fn.compute(mu, sigma, f_star, known_mask)

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
