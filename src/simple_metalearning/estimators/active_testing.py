"""Active Testing for discrete algorithm/configuration selection.

Surrogate-free meta-learning selection of

    Leite, Brazdil, Vanschoren. "Selecting Classification Algorithms with
    Active Testing." MLDM 2012.

Idea
----
No surrogate model is trained.  Selection is driven by *relative landmarks*
``RL(a, b, j) = P(a, j) - P(b, j)`` — the signed performance difference between
two configurations ``a, b`` on a past task ``j`` — held in the performance
matrix.  The loop is tournament-style:

1.  Warm-start the current best ``a_best`` with the globally best configuration
    (best mean performance across training tasks).
2.  Each round, pick the *competitor* ``a_c`` estimated to most improve over
    ``a_best`` on the new task, by summing its relative landmark over the
    training tasks weighted by task similarity::

        score(a) = sum_j  sim_j * (P(a, j) - P(a_best, j)).

3.  Evaluate ``a_c`` on the new task, update ``a_best`` if it is better, and
    update the task similarities with the new relative landmarks.

Task similarity (AT1; the recommended variant in the paper) deems a training
task similar to the new task when the relative landmarks of the *already
evaluated* configurations agree: ``sim_j`` is the fraction of evaluated
configuration pairs whose performance ordering matches between the new task and
task ``j``.  Before any configuration is evaluated, all tasks are weighted
equally (AT0 behaviour), so the first pick is the globally best configuration.

Integration
-----------
Fits the :class:`Estimator` interface with a greedy acquisition: :meth:`predict`
returns ``score(a)`` for every flow and the experiment selects the argmax.  Only
flows present in the matrix can be scored, so this is discrete-only — the same
constraint as the perf-vector baselines.

Deviations from the paper (documented)
--------------------------------------
* The base columns are *z-scored per task* before forming relative landmarks, so
  tasks on very different scales contribute comparably (the paper used accuracy
  in ``[0, 1]``; our matrix spans ~100x in per-task spread).  This matches the
  standardisation used by the other discrete transfer baselines here.
* Continuous / cost-sensitive extensions of active testing are not implemented.
"""

from typing import List, Optional, Union

import numpy as np

from ..datasets.dataset import MetaDataset
from .acquisition import AcquisitionFunction, GreedyAcquisition
from .estimator import Estimator


class ActiveTestingEstimator(Estimator):
    """Active-testing (AT1) discrete selector over the performance matrix.

    Args:
        dataset: MetaDataset with a performance matrix.
        sim_sharpness: Exponent applied to the agreement-fraction similarity
            (1.0 = raw AT1; higher concentrates weight on the most similar tasks).
        max_base_tasks: Optional cap on training tasks (random subset if exceeded).
        acquisition_fn: Acquisition; defaults to greedy argmax over scores.
        random_state: Seed for optional base-task subsampling.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        sim_sharpness: float = 1.0,
        max_base_tasks: Optional[int] = None,
        acquisition_fn: Optional[AcquisitionFunction] = None,
        random_state: int = 42,
    ):
        super().__init__(dataset, acquisition_fn or GreedyAcquisition())
        self.sim_sharpness = sim_sharpness
        self.max_base_tasks = max_base_tasks
        self.random_state = random_state

        self.n_flows = len(dataset.performance)
        self.n_tasks = len(dataset.performance.columns)

        self.base_task_indices_: List[int] = []
        self._base_z: Optional[np.ndarray] = None  # (n_base, n_flows) z-scored columns
        self._fitted = False
        self.fit()

    def fit(self) -> None:
        """Z-score every fully-observed training column (once)."""
        if self._fitted:
            return
        perf = self.dataset.require_matrix().values
        base_idx = [
            j for j in range(self.n_tasks)
            if np.sum(~np.isnan(perf[:, j])) >= 2 and np.nanstd(perf[:, j]) > 1e-10
        ]
        if self.max_base_tasks is not None and len(base_idx) > self.max_base_tasks:
            rng = np.random.default_rng(self.random_state)
            base_idx = sorted(rng.choice(base_idx, size=self.max_base_tasks, replace=False).tolist())

        z = []
        for j in base_idx:
            col = perf[:, j]
            obs = ~np.isnan(col)
            m, s = float(np.mean(col[obs])), float(np.std(col[obs])) or 1.0
            z.append(np.nan_to_num((col - m) / s, nan=0.0))
        self.base_task_indices_ = base_idx
        self._base_z = np.array(z) if z else np.zeros((0, self.n_flows))
        self._fitted = True

    def _similarities(self, revealed_idx: np.ndarray, y_rev: np.ndarray) -> np.ndarray:
        """AT1 task similarity from relative-landmark agreement over revealed flows.

        sim_j = fraction of revealed configuration pairs whose ordering on task j
        matches the ordering on the new task.  Uniform when < 2 revealed.
        """
        n_base = self._base_z.shape[0]
        if len(revealed_idx) < 2:
            return np.ones(n_base)
        # Target ordering of revealed flows (strictly-greater matrix over pairs).
        y_gt = y_rev[:, None] > y_rev[None, :]                      # (r, r)
        zr = self._base_z[:, revealed_idx]                          # (n_base, r)
        base_gt = zr[:, :, None] > zr[:, None, :]                   # (n_base, r, r)
        agree = (base_gt == y_gt[None, :, :])                       # (n_base, r, r)
        # Exclude the diagonal (a pair with itself); average over off-diagonal pairs.
        r = len(revealed_idx)
        off = r * (r - 1)
        sim = (agree.sum(axis=(1, 2)) - r) / off  # subtract r diagonal True's
        return np.clip(sim, 0.0, 1.0) ** self.sim_sharpness

    def predict(self, task_index: int) -> np.ndarray:
        """Estimated improvement score over the current best for every flow."""
        col = self.dataset.performance.iloc[:, task_index].values
        revealed_idx = np.where(~np.isnan(col))[0]
        n_base = self._base_z.shape[0]
        if n_base == 0:
            return np.zeros(self.n_flows)

        sim = self._similarities(revealed_idx, col[revealed_idx])
        total = sim.sum()
        if total <= 0:
            sim = np.ones(n_base)
            total = float(n_base)

        # Weighted mean of the (z-scored) base columns = estimated target perf.
        est = (sim @ self._base_z) / total                         # (n_flows,)

        if len(revealed_idx) == 0:
            # Cold start: no incumbent yet -> score by estimated performance,
            # so the argmax is the globally best-on-average configuration.
            return est

        # Relative-landmark score: estimated improvement over the current best.
        best_flow = revealed_idx[int(np.argmax(col[revealed_idx]))]
        return est - est[best_flow]

    def partial_fit(
        self,
        flow_index: Union[int, List[int]],
        task_index: int,
        performance: Union[float, List[float]],
    ) -> None:
        # Reveal the observation; similarities/scores are recomputed lazily.
        self.dataset.performance.iloc[flow_index, task_index] = performance
