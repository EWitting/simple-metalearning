from abc import ABC, abstractmethod
from typing import Optional, Union, List

import numpy as np

from ..datasets.dataset import MetaDataset
from .acquisition import AcquisitionFunction, GreedyAcquisition


class Estimator(ABC):
    """Abstract base class for performance estimation using meta-learning.

    Given a partial matrix of flows (model/pipeline + hyperparameter
    configurations) versus tasks (datasets), predicts performance of flows on
    tasks and suggests which flow to evaluate next.

    The :meth:`suggest` method drives the Bayesian Optimisation loop.  By
    default it calls :meth:`predict` (and optionally :meth:`predict_uncertainty`)
    and passes the results to the :attr:`acquisition_fn`.  Subclasses may
    override ``suggest`` when they need special handling (e.g. per-tree EI for
    Random Forests).
    """

    def __init__(
        self,
        dataset: MetaDataset,
        acquisition_fn: Optional[AcquisitionFunction] = None,
    ) -> None:
        self.dataset = dataset
        self.acquisition_fn: AcquisitionFunction = (
            acquisition_fn if acquisition_fn is not None else GreedyAcquisition()
        )

    @abstractmethod
    def fit(self) -> None:
        """Initial training on the dataset."""
        pass

    @abstractmethod
    def partial_fit(
        self,
        flow_index: int | list[int],
        task_index: int,
        performance: float | list[float],
    ) -> None:
        """Incorporate a new evaluation result to improve future predictions."""
        raise NotImplementedError()

    @abstractmethod
    def predict(self, task_index: int) -> np.ndarray:
        """Predict the performance of all flows on a specific task.

        Includes both evaluated and unevaluated flows.  Scores may be on an
        arbitrary scale; only the relative ordering matters.

        Returns:
            Array of shape ``(n_flows,)`` with a predicted score per flow.
        """
        raise NotImplementedError()

    def predict_uncertainty(self, task_index: int) -> Optional[np.ndarray]:
        """Return predictive uncertainty (std) for all flows on a task.

        Returns ``None`` by default.  Subclasses that provide calibrated
        uncertainty estimates (e.g. GP, Bayesian Ridge) should override this
        and return an array of shape ``(n_flows,)``.
        """
        return None

    def suggest(
        self,
        task_index: int,
        predictions: Optional[np.ndarray | list[float]] = None,
    ) -> np.ndarray:
        """Suggest the next flow to evaluate using the configured acquisition function.

        Passes the per-flow cost vector to the acquisition function when the
        dataset has a cost matrix (e.g. ``time_train_s + time_infer_s`` from
        TabArena).  Acquisition functions that are cost-aware (e.g.
        :class:`~simple_metalearning.estimators.acquisition.CostCooledEIAcquisition`)
        will use this to penalise expensive flows; all other acquisition
        functions ignore it.

        Args:
            task_index: Index of the target task.
            predictions: Optional precomputed predictions for all flows.  If
                not provided, :meth:`predict` is called automatically.

        Returns:
            Score array of shape ``(n_flows,)``.  The experiment loop will
            select the unevaluated flow with the highest score.
        """
        if predictions is None:
            mu = np.asarray(self.predict(task_index))
        else:
            mu = np.asarray(predictions)

        sigma = self.predict_uncertainty(task_index)

        perf_col = self.dataset.require_matrix().iloc[:, task_index].values
        known_mask = ~np.isnan(perf_col)

        if np.any(known_mask):
            f_star = float(np.max(perf_col[known_mask]))
        else:
            f_star = float("-inf")

        # Allow the acquisition function to pre-compute any task-specific
        # data structures (e.g. MMRAcquisition builds a similarity matrix here).
        if self.dataset.has_matrix:
            self.acquisition_fn.prepare(
                self.dataset.performance.values, known_mask
            )

        # Pass per-flow cost to the acquisition function if available.
        # CostCooledEIAcquisition uses this; all other functions ignore it.
        cost: Optional[np.ndarray] = None
        if self.dataset.cost is not None:
            cost = self.dataset.cost.iloc[:, task_index].values

        return self.acquisition_fn.compute(mu, sigma, f_star, known_mask, cost=cost)