"""Adaptive Bayesian Linear Regression (ABLR) transfer surrogate + EI.

Implements the multi-task surrogate of

    Perrone, Jenatton, Seeger, Archambeau.
    "Scalable Hyperparameter Transfer Learning." NeurIPS 2018.

A single neural network feature map ``phi_theta(x)`` is shared across all tasks;
each task gets its own Bayesian linear regression (BLR) head on top of those
features.  The shared network is meta-trained by maximising the sum of the
per-task BLR log marginal likelihoods, so the learned representation transfers
across tasks.  At BO time the target task only needs a *closed-form* BLR fit on
its revealed points (no network retraining), which is linear in the number of
observations — the method's selling point over a joint GP.

This mirrors the spirit of the two-tower model here but with a calibrated BLR
head that yields the mean/variance needed for Expected Improvement.

Standardisation / deviations (documented)
-----------------------------------------
* Each task's targets are z-scored by that task's own statistics (the paper
  standardises per task), so the shared head sees comparable scales.  The target
  is predicted in this z-space and EI's ``f_star`` is standardised by the target
  task's observed statistics (handled in :meth:`suggest`).
* Shared scalar prior/noise precisions ``alpha, beta`` (learned in log-space)
  rather than per-task; sufficient given the per-task standardisation.
* At zero target observations the BLR posterior equals the prior (zero mean), so
  the first pick is uninformed (cold start) — ABLR transfers via the shared
  representation, which only helps once >=1 point is seen.
"""

import warnings
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ..datasets.dataset import MetaDataset
from ..datasets.metafeature_transforms import PerformanceVectorTransform
from .acquisition import AcquisitionFunction, EIGaussianAcquisition
from .estimator import Estimator


class _FeatureNet(nn.Module):
    def __init__(self, d_in: int, hidden: tuple, d_out: int):
        super().__init__()
        layers: list = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ABLREstimator(Estimator):
    """ABLR transfer surrogate with discrete EI over one target task.

    Args:
        dataset: MetaDataset with performance matrix and flow metadata.
        feature_dim: Output dimension D of the shared feature map.
        hidden: Hidden layer widths of the shared network.
        n_epochs: Adam steps for meta-training the shared network.
        lr: Adam learning rate.
        use_metadata / performance_transform: inputs to the feature map.
        max_train_tasks: optional cap on meta-training tasks.
        acquisition_fn: defaults to Gaussian EI.
        random_state: torch / numpy seed.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        feature_dim: int = 50,
        hidden: tuple = (50, 50),
        n_epochs: int = 300,
        lr: float = 1e-2,
        use_metadata: bool = True,
        performance_transform: Optional[PerformanceVectorTransform] = None,
        max_train_tasks: Optional[int] = None,
        acquisition_fn: Optional[AcquisitionFunction] = None,
        random_state: int = 42,
    ):
        super().__init__(dataset, acquisition_fn or EIGaussianAcquisition())
        self.feature_dim = feature_dim
        self.hidden = tuple(hidden)
        self.n_epochs = n_epochs
        self.lr = lr
        self.use_metadata = use_metadata
        self.performance_transform = performance_transform
        self.max_train_tasks = max_train_tasks
        self.random_state = random_state

        if (not use_metadata or dataset.flow_metadata is None) and performance_transform is None:
            raise ValueError("ABLREstimator needs flow_metadata or a performance_transform.")

        self.n_flows = len(dataset.performance)
        self.n_tasks = len(dataset.performance.columns)
        self._net: Optional[_FeatureNet] = None
        self._log_alpha = None
        self._log_beta = None
        self._phi_all: Optional[np.ndarray] = None  # cached features at all flows
        self._pred_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._fitted = False
        self.fit()

    def _flow_features(self) -> np.ndarray:
        import pandas as pd
        parts = []
        if self.use_metadata and self.dataset.flow_metadata is not None:
            parts.append(
                self.dataset.flow_metadata.apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype="float64", na_value=np.nan)
            )
        if self.performance_transform is not None:
            parts.append(self.performance_transform.transform(self.dataset.performance))
        X = np.concatenate(parts, axis=1)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _blr(Phi: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        """Return (posterior mean m, lower-Cholesky L of A) for a BLR head."""
        D = Phi.shape[1]
        A = beta * (Phi.t() @ Phi) + alpha * torch.eye(D, dtype=Phi.dtype)
        L = torch.linalg.cholesky(A)
        rhs = beta * (Phi.t() @ y)
        m = torch.cholesky_solve(rhs, L)
        return m, L

    def fit(self) -> None:
        """Meta-train the shared feature map across training tasks (once)."""
        if self._fitted:
            return
        torch.manual_seed(self.random_state)
        if self.performance_transform is not None:
            self.performance_transform.fit(self.dataset.performance)

        perf = self.dataset.require_matrix().values
        feats = self._flow_features()
        lo, hi = feats.min(0), feats.max(0)
        feats = (feats - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
        self._feats_norm = feats

        train_idx = [j for j in range(self.n_tasks)
                     if np.sum(~np.isnan(perf[:, j])) >= 5 and np.nanstd(perf[:, j]) > 1e-10]
        if self.max_train_tasks is not None and len(train_idx) > self.max_train_tasks:
            rng = np.random.default_rng(self.random_state)
            train_idx = sorted(rng.choice(train_idx, size=self.max_train_tasks, replace=False).tolist())

        # Per-task standardised (X, y) tensors.
        tasks = []
        for j in train_idx:
            col = perf[:, j]
            obs = ~np.isnan(col)
            y = col[obs]
            yz = (y - y.mean()) / (y.std() or 1.0)
            X = torch.tensor(feats[obs], dtype=torch.float64)
            tasks.append((X, torch.tensor(yz, dtype=torch.float64).unsqueeze(-1)))

        d = feats.shape[1]
        net = _FeatureNet(d, self.hidden, self.feature_dim).double()
        log_alpha = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        log_beta = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam(list(net.parameters()) + [log_alpha, log_beta], lr=self.lr)

        const = 0.5 * np.log(2 * np.pi)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for _ in range(self.n_epochs):
                opt.zero_grad()
                alpha, beta = log_alpha.exp(), log_beta.exp()
                neg_ll = torch.zeros((), dtype=torch.float64)
                for X, y in tasks:
                    Phi = net(X)
                    try:
                        m, L = self._blr(Phi, y, alpha, beta)
                    except Exception:
                        continue
                    n, D = Phi.shape
                    resid = y - Phi @ m
                    logdetA = 2.0 * torch.log(torch.diag(L)).sum()
                    ll = (0.5 * D * log_alpha + 0.5 * n * log_beta
                          - 0.5 * beta * resid.pow(2).sum() - 0.5 * alpha * m.pow(2).sum()
                          - 0.5 * logdetA - n * const)
                    neg_ll = neg_ll - ll.squeeze()
                if neg_ll.requires_grad:
                    neg_ll.backward()
                    opt.step()

        net.eval()
        self._net = net
        self._log_alpha = log_alpha.detach()
        self._log_beta = log_beta.detach()
        with torch.no_grad():
            self._phi_all = net(torch.tensor(feats, dtype=torch.float64)).numpy()
        self._pred_cache.clear()
        self._fitted = True

    def _target_stats(self, task_index: int):
        col = self.dataset.performance.iloc[:, task_index].values
        y = col[~np.isnan(col)]
        if len(y) < 2:
            return None
        s = float(np.std(y))
        return (float(np.mean(y)), s) if s > 1e-12 else None

    def _predict(self, task_index: int) -> tuple[np.ndarray, np.ndarray]:
        if task_index in self._pred_cache:
            return self._pred_cache[task_index]
        col = self.dataset.performance.iloc[:, task_index].values
        obs = ~np.isnan(col)
        alpha = self._log_alpha.exp()
        beta = self._log_beta.exp()
        Phi_all = torch.tensor(self._phi_all, dtype=torch.float64)
        D = self.feature_dim

        def _prior():
            mu = np.zeros(self.n_flows)
            var = (1.0 / beta).item() + (self._phi_all**2).sum(1) / alpha.item()
            return mu, np.sqrt(np.clip(var, 1e-12, None))

        stats = self._target_stats(task_index)
        if obs.sum() < 2 or stats is None:
            res = _prior()  # cold start: uninformed prior predictive
        else:
            # Guard the BLR solve: a numerical failure must not propagate, or a
            # single task would wipe the whole parallel run (empty metrics).
            try:
                m_, s_ = stats
                yz = (col[obs] - m_) / s_
                Phi_obs = torch.tensor(self._phi_all[obs], dtype=torch.float64)
                y = torch.tensor(yz, dtype=torch.float64).unsqueeze(-1)
                m, L = self._blr(Phi_obs, y, alpha, beta)
                with torch.no_grad():
                    mu = (Phi_all @ m).squeeze(-1).numpy()
                    v = torch.cholesky_solve(Phi_all.t(), L)      # A^{-1} Phi^T
                    quad = (Phi_all * v.t()).sum(1)
                    var = (1.0 / beta).item() + quad.numpy()
                res = (mu, np.sqrt(np.clip(var, 1e-12, None)))
                if not (np.all(np.isfinite(res[0])) and np.all(np.isfinite(res[1]))):
                    res = _prior()
            except Exception:
                res = _prior()
        self._pred_cache[task_index] = res
        return res

    def predict(self, task_index: int) -> np.ndarray:
        return self._predict(task_index)[0]

    def predict_uncertainty(self, task_index: int) -> Optional[np.ndarray]:
        return self._predict(task_index)[1]

    def suggest(self, task_index, predictions=None):
        """EI with ``f_star`` standardised into the target's z-space (BLR space)."""
        mu = np.asarray(predictions) if predictions is not None else self.predict(task_index)
        sigma = self.predict_uncertainty(task_index)
        col = self.dataset.require_matrix().iloc[:, task_index].values
        known_mask = ~np.isnan(col)
        stats = self._target_stats(task_index)
        if stats is not None and np.any(known_mask):
            m_, s_ = stats
            f_star = (float(np.max(col[known_mask])) - m_) / s_
        else:
            f_star = float("-inf")
        if self.dataset.has_matrix:
            self.acquisition_fn.prepare(self.dataset.performance.values, known_mask)
        cost = self.dataset.cost.iloc[:, task_index].values if self.dataset.cost is not None else None
        return self.acquisition_fn.compute(mu, sigma, f_star, known_mask, cost=cost)

    def partial_fit(
        self,
        flow_index: Union[int, List[int]],
        task_index: int,
        performance: Union[float, List[float]],
    ) -> None:
        # Shared net is fixed; only the closed-form BLR head changes -> just
        # invalidate the cache so the next predict recomputes it.
        self.dataset.performance.iloc[flow_index, task_index] = performance
        self._pred_cache.pop(task_index, None)
