"""Multi-Task Gaussian Process estimator (intrinsic coregionalization) + EI.

Implements the multi-task Bayesian optimization surrogate of

    Swersky, Snoek, Adams. "Multi-Task Bayesian Optimization." NeurIPS 2013.

using BoTorch's :class:`~botorch.models.MultiTaskGP`, whose kernel *is* the
intrinsic model of coregionalization from that paper:

    K((x, t), (x', t')) = K_t(t, t') * K_x(x, x'),

with ``K_x`` a Matern-5/2 data kernel and ``K_t`` an ``IndexKernel`` (a low-rank
+ diagonal task covariance, parameterised by a Cholesky factor) — exactly Eq. 3
of the paper.  Transfer to a new task is "restrict future observations to the
task of interest and proceed as normal; the other tasks act as additional
observations" (paper Sec. 3.1).  We select from the discrete flow set of one
target task at a time, so no multi-task acquisition is needed — standard EI over
the predicted target column (paper Eqs. 4-5).

Deviations from the paper (documented)
--------------------------------------
* Kernel / task-covariance hyperparameters are fit by MAP (marginal likelihood)
  rather than slice sampling — the standard modern BoTorch approach.
* Exact MTGP over all ~40 training tasks is infeasible here (cubic in total
  observations).  We therefore use three tractability levers:
    1. ``n_aux_tasks`` auxiliary tasks selected for **performance diversity**
       (max-min on column correlation) — no task metafeatures are used, matching
       the rest of this repo.
    2. ``n_context_flows`` optional sub-sampling of the auxiliary observations
       (needed for CASH's 3623 flows); the target column is still predicted at
       *all* flows.
    3. Kernel hyperparameters are re-fit only on a geometric schedule
       (``refit_factor``) as target observations accumulate; in between, the
       model is rebuilt with the cached hyperparameters and the new target
       points conditioned in (no MLL re-optimisation).
"""

import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from ..datasets.dataset import MetaDataset
from ..datasets.metafeature_transforms import PerformanceVectorTransform
from .acquisition import AcquisitionFunction, EIGaussianAcquisition
from .estimator import Estimator


def select_diverse_tasks(perf: np.ndarray, task_idx: List[int], n: int,
                         seed: int = 42) -> List[int]:
    """Greedy max-min selection of ``n`` performance-diverse task columns.

    Distance between two task columns is ``1 - |pearson corr|`` over the flows
    both evaluate.  Starts from the highest-variance column and repeatedly adds
    the column whose minimum distance to the chosen set is largest, so the subset
    spans a variety of task "types" without using any task metafeatures.
    """
    if len(task_idx) <= n:
        return list(task_idx)
    cols = perf[:, task_idx]                                  # (n_flows, n_cand)
    # Correlation matrix over candidate columns (pairwise complete).
    corr = np.corrcoef(np.nan_to_num(cols.T, nan=0.0))
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - np.abs(corr)
    chosen = [int(np.argmax(np.nanstd(cols, axis=0)))]       # seed: most-varied
    while len(chosen) < n:
        mind = dist[:, chosen].min(axis=1)
        mind[chosen] = -np.inf
        chosen.append(int(np.argmax(mind)))
    return [task_idx[c] for c in chosen]


class MultiTaskGPEIEstimator(Estimator):
    """BoTorch MultiTaskGP (ICM) surrogate with discrete EI over one target task.

    Args:
        dataset: MetaDataset with performance matrix and flow metadata.
        n_aux_tasks: Number of performance-diverse auxiliary tasks to model.
        n_context_flows: Sub-sample of flows used as auxiliary observations
            (None = all flows; set for CASH).  The target is always predicted at
            all flows.
        rank: Rank of the ICM task covariance (None = BoTorch default).
        refit_factor: Re-fit kernel hyperparameters when the number of target
            observations has grown by this factor since the last fit.
        use_metadata: Include flow_metadata (hyperparameters) in K_x inputs.
        performance_transform: Optional collaborative features for K_x.
        standardize_outputs: Apply BoTorch Standardize to the pooled targets.
        acquisition_fn: Acquisition (defaults to Gaussian EI).
        random_state: Seed for diversity selection / context subsampling.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        n_aux_tasks: int = 8,
        n_context_flows: Optional[int] = 64,
        rank: Optional[int] = None,
        refit_factor: float = 3.0,
        max_refit_n: int = 20,
        fit_maxiter: int = 75,
        use_metadata: bool = True,
        performance_transform: Optional[PerformanceVectorTransform] = None,
        standardize_outputs: bool = True,
        acquisition_fn: Optional[AcquisitionFunction] = None,
        random_state: int = 42,
    ):
        super().__init__(dataset, acquisition_fn or EIGaussianAcquisition())
        self.n_aux_tasks = n_aux_tasks
        self.n_context_flows = n_context_flows
        self.rank = rank
        self.refit_factor = refit_factor
        # MLL refits dominate cost and blow up with data size, so cap how late we
        # refit (hyperparameters stabilise early; later steps reuse them and only
        # condition on the new points) and cap the optimiser iterations.
        self.max_refit_n = max_refit_n
        self.fit_maxiter = fit_maxiter
        self.use_metadata = use_metadata
        self.performance_transform = performance_transform
        self.standardize_outputs = standardize_outputs
        self.random_state = random_state

        if (not use_metadata or dataset.flow_metadata is None) and performance_transform is None:
            raise ValueError("MultiTaskGPEIEstimator needs flow_metadata or a performance_transform.")

        self.n_flows = len(dataset.performance)
        self.n_tasks = len(dataset.performance.columns)
        self._target_task_id = n_aux_tasks  # task index reserved for the target

        # Populated by fit():
        self.aux_task_indices_: List[int] = []
        self._feats: Optional[np.ndarray] = None          # (n_flows, d) normalised
        self._aux_X: Optional[torch.Tensor] = None        # (M*S, d+1)
        self._aux_Y: Optional[torch.Tensor] = None        # (M*S, 1)

        # Per-target state.
        self._model = None
        self._task_idx: Optional[int] = None
        self._last_fit_n: int = 0
        self._state_dict = None
        self._pred_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._fitted = False
        self.fit()

    def _flow_features(self) -> np.ndarray:
        parts = []
        if self.use_metadata and self.dataset.flow_metadata is not None:
            import pandas as pd
            parts.append(
                self.dataset.flow_metadata.apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype="float64", na_value=np.nan)
            )
        if self.performance_transform is not None:
            parts.append(self.performance_transform.transform(self.dataset.performance))
        X = np.concatenate(parts, axis=1)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self) -> None:
        """Select aux tasks, build the (normalised) auxiliary training block once."""
        if self._fitted:
            return
        if self.performance_transform is not None:
            self.performance_transform.fit(self.dataset.performance)

        perf = self.dataset.require_matrix().values
        train_idx = [j for j in range(self.n_tasks)
                     if np.sum(~np.isnan(perf[:, j])) >= 2 and np.nanstd(perf[:, j]) > 1e-10]
        self.aux_task_indices_ = select_diverse_tasks(
            perf, train_idx, self.n_aux_tasks, self.random_state
        )

        feats = self._flow_features()
        lo, hi = feats.min(axis=0), feats.max(axis=0)
        self._feats = (feats - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)  # min-max -> [0,1]

        rng = np.random.default_rng(self.random_state)
        all_flows = np.arange(self.n_flows)
        if self.n_context_flows is not None and self.n_context_flows < self.n_flows:
            context = np.sort(rng.choice(all_flows, size=self.n_context_flows, replace=False))
        else:
            context = all_flows

        Xs, Ys = [], []
        for t, j in enumerate(self.aux_task_indices_):
            col = perf[:, j]
            obs = context[~np.isnan(col[context])]
            if len(obs) == 0:
                continue
            xt = np.column_stack([self._feats[obs], np.full(len(obs), float(t))])
            Xs.append(xt)
            Ys.append(col[obs])
        self._aux_X = torch.tensor(np.concatenate(Xs), dtype=torch.float64)
        self._aux_Y = torch.tensor(np.concatenate(Ys), dtype=torch.float64).unsqueeze(-1)

        self._model = None
        self._task_idx = None
        self._last_fit_n = 0
        self._state_dict = None
        self._pred_cache.clear()
        self._fitted = True

    def _build_model(self, rev_idx: np.ndarray, rev_y: np.ndarray, refit: bool):
        from botorch.models import MultiTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        tid = float(self._target_task_id)
        X_tgt = np.column_stack([self._feats[rev_idx], np.full(len(rev_idx), tid)])
        train_X = torch.cat([self._aux_X, torch.tensor(X_tgt, dtype=torch.float64)])
        train_Y = torch.cat([self._aux_Y, torch.tensor(rev_y, dtype=torch.float64).unsqueeze(-1)])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                model = MultiTaskGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    task_feature=-1,
                    output_tasks=[self._target_task_id],
                    rank=self.rank,
                    outcome_transform=Standardize(m=1) if self.standardize_outputs else None,
                )
                if refit or self._state_dict is None:
                    fit_gpytorch_mll(
                        ExactMarginalLogLikelihood(model.likelihood, model),
                        optimizer_kwargs={"options": {"maxiter": self.fit_maxiter}},
                    )
                    self._state_dict = model.state_dict()
                    self._last_fit_n = len(rev_idx)
                else:
                    model.load_state_dict(self._state_dict, strict=False)
            except Exception:
                return None
        model.eval()
        return model

    def _ensure_model(self, task_index: int) -> None:
        col = self.dataset.performance.iloc[:, task_index].values
        rev_idx = np.where(~np.isnan(col))[0]
        n_rev = len(rev_idx)
        if n_rev < 1:
            self._model = None
            self._task_idx = task_index
            return
        # Rebuild the model structure every step (data changed), but only re-run
        # the expensive MLL fit on the geometric schedule — otherwise reuse the
        # cached kernel hyperparameters (state_dict).
        refit = (
            self._state_dict is None
            or self._task_idx != task_index
            or (n_rev >= self.refit_factor * max(self._last_fit_n, 1)
                and n_rev <= self.max_refit_n)
        )
        if self._task_idx != task_index:
            self._state_dict = None  # new target: refit from scratch
        self._model = self._build_model(rev_idx, col[rev_idx], refit=refit)
        self._task_idx = task_index

    def _aux_prior(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean of the auxiliary columns — prior used at cold start / on failure."""
        perf = self.dataset.performance.values[:, self.aux_task_indices_]
        mu = np.nan_to_num(np.nanmean(perf, axis=1), nan=0.0)
        sigma = np.full(self.n_flows, float(np.nanstd(perf)) or 1.0)
        return mu, sigma

    def _predict(self, task_index: int) -> tuple[np.ndarray, np.ndarray]:
        if task_index in self._pred_cache:
            return self._pred_cache[task_index]
        if self._task_idx != task_index or self._model is None:
            self._ensure_model(task_index)

        result = None
        if self._model is not None:
            # Any numerical failure in posterior inference must not propagate:
            # a single raised task would wipe out the whole parallel run.
            try:
                tid = float(self._target_task_id)
                test_X = torch.tensor(
                    np.column_stack([self._feats, np.full(self.n_flows, tid)]),
                    dtype=torch.float64,
                )
                with torch.no_grad():
                    post = self._model.posterior(test_X)
                    mu = post.mean.squeeze(-1).numpy()
                    sigma = post.variance.clamp(min=1e-12).sqrt().squeeze(-1).numpy()
                if np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)):
                    result = (mu, sigma)
            except Exception:
                result = None

        if result is None:
            # Cold start (no target obs) or any fit/predict failure -> aux prior.
            result = self._aux_prior()
        self._pred_cache[task_index] = result
        return result

    def predict(self, task_index: int) -> np.ndarray:
        return self._predict(task_index)[0]

    def predict_uncertainty(self, task_index: int) -> Optional[np.ndarray]:
        return self._predict(task_index)[1]

    def partial_fit(
        self,
        flow_index: Union[int, List[int]],
        task_index: int,
        performance: Union[float, List[float]],
    ) -> None:
        self.dataset.performance.iloc[flow_index, task_index] = performance
        # Just invalidate the cache; the model is rebuilt lazily on the next
        # predict (avoids rebuilding twice per BO step).
        self._pred_cache.pop(task_index, None)
        self._model = None
