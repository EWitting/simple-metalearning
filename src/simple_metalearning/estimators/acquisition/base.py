"""Abstract base class for acquisition functions.

An acquisition function maps the surrogate model's output (predicted mean and
optional uncertainty) to a score for each candidate flow.  The experiment loop
then picks the highest-scoring unevaluated flow.

Design
------
``compute()`` receives *all* flows (including already-evaluated ones).  The
caller is responsible for zeroing-out or masking already-known flows; this keeps
the interface simple and lets acquisition functions access information about
known flows (e.g. for diversity/MMR).
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AcquisitionFunction(ABC):
    """Base class for acquisition functions.

    All acquisition functions must implement ``compute()``, which returns a
    score array for every flow.  Higher score = recommend evaluating this flow
    next.  Already-evaluated flows should be given a score of ``-inf`` or 0 by
    the caller so they are never selected.
    """

    def prepare(self, performance: np.ndarray, known_mask: np.ndarray) -> None:
        """Optional hook called before ``compute()`` to allow pre-computation.

        The default implementation is a no-op.  Subclasses that need to build
        data structures from the current performance matrix (e.g.
        :class:`MMRAcquisition` building a flow similarity matrix) should
        override this.

        Args:
            performance: Full performance matrix, shape ``(n_flows, n_tasks)``.
                NaN values are still present; implementations should impute as
                needed.
            known_mask: Boolean array, shape ``(n_flows,)`` — ``True`` for
                flows already evaluated on the target task.
        """
        pass

    @abstractmethod
    def compute(
        self,
        mu: np.ndarray,
        sigma: Optional[np.ndarray],
        f_star: float,
        known_mask: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute an acquisition score for each flow.

        Args:
            mu: Predicted performance for every flow, shape ``(n_flows,)``.
            sigma: Predicted uncertainty (std) for every flow, shape
                ``(n_flows,)``, or ``None`` if the model does not provide
                uncertainty estimates.
            f_star: Best observed performance so far on the target task.
                ``-inf`` when no flows have been evaluated yet.
            known_mask: Boolean array, shape ``(n_flows,)``.  ``True`` for
                flows that have already been evaluated on the target task.
            cost: Optional evaluation cost for every flow, shape
                ``(n_flows,)``.  Provided when the dataset has a cost matrix
                (e.g. ``time_train_s + time_infer_s`` for TabArena).  Most
                acquisition functions ignore this; :class:`CostCooledEI`
                divides EI by ``cost^alpha``.

        Returns:
            Score array of shape ``(n_flows,)``.  Unknown flows with the
            highest score will be suggested next.  Already-known flows may
            receive any score — the experiment loop will not select them.
        """
        raise NotImplementedError
