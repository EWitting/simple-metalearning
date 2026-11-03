"""Ranking-Weighted Gaussian Process Ensemble (RGPE) estimator.

Faithful implementation of the transfer-learning surrogate of

    Feurer, Letham, Hutter, Bakshy.
    "Practical Transfer Learning for Bayesian Optimization." (arXiv:1802.02219)

and cross-checked against the official BoTorch tutorial
(``meta_learning_with_rgpe``).

Idea
----
One *base model* (a GP) is fit on every previously-seen task (here: every
fully-observed training column of the performance matrix).  A *target model*
(a GP) is fit on the observations revealed so far for the task currently being
optimised.  The ensemble prediction is a weighted sum of all models, and the
ensemble is itself a GP::

    f_bar(x) ~ N( sum_i w_i mu_i(x),  sum_i w_i^2 sigma_i^2(x) )        (Eq. 6)

so it drops straight into the existing Gaussian-EI acquisition machinery.

Weights
-------
Each model is weighted by the (Monte-Carlo estimated) probability that it is
the ensemble member with the lowest *ranking loss* on the target observations.

* Base-model loss (Eq. 3): number of misranked pairs between the model's mean
  predictions and the observed targets on the target task.
* Target-model loss (Eq. 4): the same, but predictions are *out-of-fold* so the
  in-sample fit is not rewarded.  The paper uses leave-one-out; here we use
  K-fold (default 5) for speed, reusing the target model's fitted kernel
  hyperparameters per fold (no per-fold MLL re-optimisation), mirroring the
  no-refit LOOCV trick in the BoTorch tutorial.
* The probability is estimated by bootstrap-resampling the target observations
  ``n_bootstrap_samples`` times and counting how often each model attains the
  minimum loss (Eq. 5 / Algorithm 1), splitting ties uniformly.
* Cold start: with fewer than 3 target observations all models get uniform
  weight (Algorithm 1, lines 2-3).

Deviations from the paper (by design)
-------------------------------------
* K-fold instead of leave-one-out for the target loss (speed; ``target_cv_folds``).
* Base-model inputs are the flow *hyperparameter* features (``flow_metadata``)
  only.  Performance-vector / collaborative features are deliberately excluded
  because a base task's own column would leak into its features and conflate the
  ensemble-weighting transfer mechanism with feature-level transfer.
* Standard (noiseless) EI is used rather than RGPE(NEI); our table-lookup
  evaluations are deterministic, matching every other estimator in this repo.
"""

import warnings
from typing import List, Optional, Union

import numpy as np
import torch
from sklearn.model_selection import KFold

from ..datasets.dataset import MetaDataset
from .acquisition import AcquisitionFunction, EIGaussianAcquisition
from .estimator import Estimator


def _ranking_loss_disagreement(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise rank-disagreement tensor between predictions and targets.

    Args:
        pred: ``(n_models, n)`` predicted values at the ``n`` target points.
        y: ``(n,)`` observed targets.

    Returns:
        ``(n_models, n, n)`` boolean array ``D`` with
        ``D[m, k, l] = (pred[m,k] < pred[m,l]) XOR (y[k] < y[l])`` — i.e. True
        where model ``m`` orders the pair ``(k, l)`` differently from ground truth.

    The full ranking loss (Eq. 3) for model ``m`` is ``D[m].sum()``; weighting
    pairs by bootstrap multiplicities gives the per-bootstrap losses in Eq. 5.
    """
    pred_order = pred[:, :, None] < pred[:, None, :]        # (M, n, n)
    y_order = (y[:, None] < y[None, :])[None, :, :]          # (1, n, n)
    return pred_order ^ y_order


class RGPEEstimator(Estimator):
    """Ranking-Weighted GP Ensemble surrogate for discrete flow selection.

    Args:
        dataset: MetaDataset with a performance matrix and ``flow_metadata``.
        n_bootstrap_samples: Number of bootstrap samples ``S`` used to estimate
            the ranking weights (Algorithm 1).  Lower = faster, noisier weights.
        target_cv_folds: Number of folds for the out-of-fold target-model loss.
            Clamped to the number of observations (i.e. LOO when very few obs).
        use_base_variance: If True (default, RGPE(EI)) the ensemble variance is
            ``sum_i w_i^2 sigma_i^2`` (Eq. 6).  If False (RGPE(mean)) only the
            target model's variance is used, base variances dropped (TST-R(EI)).
        prevent_weight_dilution: Enable the Sec. 4.3 / Eq. 9 base-model dropping
            (recommended with many base tasks; decays to pure target BO at H).
        horizon: Optimisation horizon ``H`` for the dilution decay; set equal to
            ``experiment.max_visible``.
        oracle_base: Ablation — *no GPs at all*.  Base models are replaced by the
            exact known matrix columns and the target GP is dropped entirely, so
            the model is a purely rank-weighted collaborative ensemble of the
            ground-truth base-task columns (discrete-only, like the perf-vector
            baselines).  The ranking-loss weights are unchanged; weight dilution is
            not applicable (it is defined relative to a target model) and is
            skipped.  Predictive uncertainty for EI comes from *base-task
            disagreement* (weighted variance of the base predictions) rather than a
            GP posterior.
        weighting: ``"rgpe"`` (ABE bootstrap probability-best weights, default) or
            ``"tstr"`` (TST-R Nadaraya-Watson Epanechnikov kernel over the
            discordant-pairs distance; Wistuba et al. 2016).  With
            ``oracle_base=True`` the latter gives the GP-free TST-R(discrete).
        tstr_bandwidth: Epanechnikov bandwidth for ``weighting="tstr"``; base
            tasks with a discordant-pair fraction >= this value get zero weight.
        max_base_tasks: Optional cap on the number of base tasks; when exceeded a
            random subset is used (keeps CASH tractable).  None = use all.
        normalize_inputs / standardize_outputs: BoTorch input/output transforms
            applied to every GP (Standardize handles cross-task scale, per paper).
        acquisition_fn: Acquisition function (defaults to Gaussian EI).
        random_state: Seed for bootstrap resampling and base-task subsampling.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        n_bootstrap_samples: int = 256,
        target_cv_folds: int = 5,
        use_base_variance: bool = True,
        prevent_weight_dilution: bool = True,
        horizon: int = 200,
        oracle_base: bool = False,
        weighting: str = "rgpe",
        tstr_bandwidth: float = 0.5,
        max_base_tasks: Optional[int] = None,
        normalize_inputs: bool = True,
        standardize_outputs: bool = True,
        acquisition_fn: Optional[AcquisitionFunction] = None,
        random_state: int = 42,
    ):
        super().__init__(dataset, acquisition_fn or EIGaussianAcquisition())
        self.n_bootstrap_samples = n_bootstrap_samples
        self.target_cv_folds = target_cv_folds
        self.use_base_variance = use_base_variance
        self.prevent_weight_dilution = prevent_weight_dilution
        self.horizon = horizon
        self.oracle_base = oracle_base
        if weighting not in ("rgpe", "tstr"):
            raise ValueError(f"weighting must be 'rgpe' or 'tstr', got {weighting!r}")
        self.weighting = weighting
        self.tstr_bandwidth = tstr_bandwidth
        self.max_base_tasks = max_base_tasks
        self.normalize_inputs = normalize_inputs
        self.standardize_outputs = standardize_outputs
        self.random_state = random_state

        if dataset.flow_metadata is None:
            raise ValueError("RGPEEstimator requires flow_metadata (hyperparameter features).")

        self.n_flows = len(dataset.performance)
        self.n_tasks = len(dataset.performance.columns)

        # Fixed flow feature matrix (hyperparameters); BoTorch rejects NaN/inf.
        # Coerce defensively: in normal runs a metafeature transform has already
        # made flow_metadata numeric, but raw nullable/object columns may remain.
        import pandas as pd

        feats = (
            dataset.flow_metadata.apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype="float64", na_value=np.nan)
        )
        self._flow_features = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Populated by fit() — base-model predictions at *all* flows (fixed run).
        self.base_task_indices_: List[int] = []
        self._base_mu: Optional[np.ndarray] = None       # raw, for weight ranking
        self._base_zmu: Optional[np.ndarray] = None      # z-scored, for prediction
        self._base_zsigma: Optional[np.ndarray] = None   # z-scored std

        # Per-target-task state (target index -> fitted model / caches).
        self._target_model = None
        self._target_task_idx: Optional[int] = None
        self._weight_cache: dict[int, tuple[np.ndarray, float]] = {}
        self._pred_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._fitted = False

        self.fit()

    # ------------------------------------------------------------------
    # GP construction helper
    # ------------------------------------------------------------------
    def _make_gp(self, X: np.ndarray, y: np.ndarray, state_dict=None):
        """Build a SingleTaskGP; fit via MLL unless a state_dict is supplied.

        When ``state_dict`` is given the kernel hyperparameters are loaded
        directly (no MLL optimisation) — used for the out-of-fold target models.
        Returns the fitted model in eval mode, or None on failure.
        """
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        if len(y) < 2 or float(np.std(y)) < 1e-10:
            return None

        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64).unsqueeze(-1)
        d = X_t.shape[1]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                model = SingleTaskGP(
                    train_X=X_t,
                    train_Y=y_t,
                    input_transform=Normalize(d=d) if self.normalize_inputs else None,
                    outcome_transform=Standardize(m=1) if self.standardize_outputs else None,
                )
                if state_dict is not None:
                    model.load_state_dict(state_dict, strict=False)
                else:
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            except Exception:
                return None

        model.eval()
        return model

    def _predict_all_flows(self, model) -> tuple[np.ndarray, np.ndarray]:
        """Posterior (mu, sigma) for a model at every flow."""
        X_t = torch.tensor(self._flow_features, dtype=torch.float64)
        with torch.no_grad():
            post = model.posterior(X_t)
            mu = post.mean.squeeze(-1).numpy()
            sigma = post.variance.clamp(min=1e-12).sqrt().squeeze(-1).numpy()
        return mu, sigma

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Fit one base GP per fully-observed training task (once).

        Idempotent: the experiment calls ``fit()`` after construction, which
        already fits in ``__init__``; the base GPs are expensive, so a redundant
        call is a no-op.  The base columns are fixed for the run, so refitting
        would reproduce the same models.
        """
        if self._fitted:
            return
        perf = self.dataset.require_matrix().values  # (n_flows, n_tasks)

        # Base tasks = columns observed at fit time (training tasks). Test/target
        # columns are all-NaN here and are excluded.
        base_idx = [
            j for j in range(self.n_tasks)
            if np.sum(~np.isnan(perf[:, j])) >= 2
            and np.nanstd(perf[:, j]) > 1e-10
        ]

        if self.max_base_tasks is not None and len(base_idx) > self.max_base_tasks:
            rng = np.random.default_rng(self.random_state)
            base_idx = sorted(rng.choice(base_idx, size=self.max_base_tasks, replace=False).tolist())

        base_mu, base_zmu, base_zsigma, kept = [], [], [], []
        for j in base_idx:
            col = perf[:, j]
            obs = ~np.isnan(col)
            if self.oracle_base:
                # Ablation: skip GP training and use the known matrix column as the
                # base model's prediction for every flow (exact, zero uncertainty).
                # Only defined on the discrete flow set -> discrete-only, like the
                # perf-vector baselines.  The target side keeps its GP so that the
                # ranking / dilution machinery is unchanged.
                mu = np.nan_to_num(col, nan=0.0)
                sigma = np.zeros_like(mu)
            else:
                model = self._make_gp(self._flow_features[obs], col[obs])
                if model is None:
                    continue
                mu, sigma = self._predict_all_flows(model)
            # Per-task standardisation for ensembling (paper Sec. 4.2): each task
            # lives on a different scale, so z-score every model's predictions by
            # its own column stats before they are combined.  Ranking-loss weights
            # are scale-invariant and use the raw means below.
            m_j = float(np.mean(col[obs]))
            s_j = float(np.std(col[obs])) or 1.0
            base_mu.append(mu)                       # raw — for weight ranking
            base_zmu.append((mu - m_j) / s_j)        # standardised — for prediction
            base_zsigma.append(sigma / s_j)
            kept.append(j)

        self.base_task_indices_ = kept
        self._base_mu = np.array(base_mu) if base_mu else np.zeros((0, self.n_flows))
        self._base_zmu = np.array(base_zmu) if base_zmu else np.zeros((0, self.n_flows))
        self._base_zsigma = np.array(base_zsigma) if base_zsigma else np.zeros((0, self.n_flows))

        # Reset target state.
        self._target_model = None
        self._target_task_idx = None
        self._weight_cache.clear()
        self._pred_cache.clear()
        self._fitted = True

    # ------------------------------------------------------------------
    # Target model + weights
    # ------------------------------------------------------------------
    def _fit_target(self, task_idx: int) -> None:
        self._target_task_idx = task_idx
        if self.oracle_base:
            self._target_model = None  # fully GP-free ablation: no target model
            return
        col = self.dataset.performance.iloc[:, task_idx].values
        obs = ~np.isnan(col)
        self._target_model = (
            self._make_gp(self._flow_features[obs], col[obs]) if obs.sum() >= 2 else None
        )

    def _target_oof_predictions(self, obs_idx: np.ndarray, y_obs: np.ndarray) -> Optional[np.ndarray]:
        """Out-of-fold target predictions reusing the target model's hyperparameters."""
        if self._target_model is None:
            return None
        n = len(obs_idx)
        n_splits = int(min(self.target_cv_folds, n))
        if n_splits < 2:
            return None

        X = self._flow_features[obs_idx]
        state_dict = self._target_model.state_dict()
        oof = np.full(n, np.nan)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(np.arange(n)):
            if len(tr) < 2 or float(np.std(y_obs[tr])) < 1e-10:
                continue
            m = self._make_gp(X[tr], y_obs[tr], state_dict=state_dict)
            if m is None:
                continue
            with torch.no_grad():
                pred = m.posterior(torch.tensor(X[te], dtype=torch.float64)).mean.squeeze(-1).numpy()
            oof[te] = pred
        if np.isnan(oof).any():
            return None
        return oof

    def _compute_weights(self, task_idx: int) -> tuple[np.ndarray, float]:
        """Return (base_weights[n_base], target_weight) via Algorithm 1."""
        col = self.dataset.performance.iloc[:, task_idx].values
        obs_idx = np.where(~np.isnan(col))[0]
        n_obs = len(obs_idx)
        n_base = len(self.base_task_indices_)
        target_ok = self._target_model is not None and n_obs >= 2

        # Cold start (Eq./Alg. lines 2-3): uniform over all usable models.
        if n_obs < 3 or n_base == 0:
            n_models = n_base + (1 if target_ok else 0)
            if n_models == 0:
                return np.zeros(0), 0.0
            w = 1.0 / n_models
            return np.full(n_base, w), (w if target_ok else 0.0)

        y_obs = col[obs_idx]

        # Predictions at observed target points: base (precomputed) + target OOF.
        preds = [self._base_mu[:, obs_idx]]  # (n_base, n_obs)
        target_oof = self._target_oof_predictions(obs_idx, y_obs) if target_ok else None
        if target_oof is not None:
            preds.append(target_oof[None, :])
        pred = np.concatenate(preds, axis=0)          # (M, n_obs)
        n_models = pred.shape[0]
        has_target = target_oof is not None

        # Pairwise disagreement tensor D[m, k, l]; loss = c^T D[m] c per bootstrap.
        D = _ranking_loss_disagreement(pred, y_obs).astype(np.float64)

        # TST-R weighting (Wistuba et al. 2016): Nadaraya-Watson kernel over the
        # fraction-of-discordant-pairs distance, instead of the RGPE bootstrap.
        #   d_i = (# discordant unordered pairs) / (n(n-1)/2)
        #   w_i ∝ Epanechnikov(d_i / bandwidth)
        # Base variances are dropped (handled by the oracle disagreement path).
        if self.weighting == "tstr":
            n_pairs = n_obs * (n_obs - 1)  # ordered; D is symmetric so this halves out
            dist = D.sum(axis=(1, 2)) / n_pairs            # (n_models,) in [0, 1]
            r = dist / max(self.tstr_bandwidth, 1e-9)
            kern = np.where(r <= 1.0, 0.75 * (1.0 - r**2), 0.0)
            total = kern.sum()
            weights = (kern / total) if total > 0 else np.full(n_models, 1.0 / n_models)
            if has_target:
                return weights[:n_base], float(weights[n_base])
            return weights, 0.0

        # Bootstrap resamples of the target observations -> multiplicity vectors.
        rng = np.random.default_rng(self.random_state + task_idx * 1000 + n_obs)
        S = self.n_bootstrap_samples
        boot = rng.integers(0, n_obs, size=(S, n_obs))
        counts = np.zeros((S, n_obs))
        for s in range(S):
            np.add.at(counts[s], boot[s], 1.0)

        # losses[s, m] = counts[s] @ D[m] @ counts[s]
        losses = np.einsum("sk,mkl,sl->sm", counts, D, counts)

        # Weight-dilution prevention (Sec. 4.3, Eq. 9). With many base tasks the
        # bootstrap gives spurious weight to poor models, so each base model is
        # dropped *before* weighting with probability
        #   p_drop(i) = 1 - (1 - n_t/H) * P(base_i out-ranks target),
        # which both removes base tasks that rarely beat the target and decays
        # all base weight to zero as the budget H is spent (-> pure target BO).
        model_keep = np.ones(n_models, dtype=bool)
        if self.prevent_weight_dilution and has_target:
            target_losses = losses[:, n_base]                       # (S,)
            frac_better = (losses[:, :n_base] < target_losses[:, None]).mean(axis=0)
            keep_prob = np.clip((1.0 - n_obs / max(self.horizon, 1)) * frac_better, 0.0, 1.0)
            model_keep[:n_base] = rng.random(n_base) < keep_prob    # keep target always

        # Distribute each sample's weight uniformly across argmin ties (Eq. 5),
        # competing only among surviving models.
        kept_losses = losses[:, model_keep]
        min_loss = kept_losses.min(axis=1, keepdims=True)
        is_min = kept_losses <= min_loss + 1e-9
        contrib = is_min / is_min.sum(axis=1, keepdims=True)
        weights = np.zeros(n_models)
        weights[model_keep] = contrib.mean(axis=0)

        if has_target:
            return weights[:n_base], float(weights[n_base])
        return weights, 0.0

    def _weights(self, task_idx: int) -> tuple[np.ndarray, float]:
        if task_idx not in self._weight_cache:
            if self._target_task_idx != task_idx or self._target_model is None:
                self._fit_target(task_idx)
            self._weight_cache[task_idx] = self._compute_weights(task_idx)
        return self._weight_cache[task_idx]

    # ------------------------------------------------------------------
    # Estimator API
    # ------------------------------------------------------------------
    def _target_stats(self, task_index: int) -> Optional[tuple[float, float]]:
        """(mean, std) of the target task's observed values, or None if < 2 / flat.

        These define the z-score space the target model (and ``f_star``) live in,
        consistent with the per-task standardisation used for the base models.
        """
        col = self.dataset.performance.iloc[:, task_index].values
        y = col[~np.isnan(col)]
        if len(y) < 2:
            return None
        s = float(np.std(y))
        return (float(np.mean(y)), s) if s > 1e-12 else None

    def _target_z(self, task_index: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Standardised target-model (mu, sigma) over all flows, or None."""
        if self._target_model is None:
            return None
        stats = self._target_stats(task_index)
        if stats is None:
            return None
        m, s = stats
        t_mu, t_sigma = self._predict_all_flows(self._target_model)
        return (t_mu - m) / s, t_sigma / s

    def _ensemble(self, task_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Rank-weighted ensemble (mu, sigma) in z-score space (Eq. 6), cached."""
        if task_index in self._pred_cache:
            return self._pred_cache[task_index]

        w_base, w_target = self._weights(task_index)
        mu = np.zeros(self.n_flows)
        var = np.zeros(self.n_flows)
        if len(w_base) > 0:
            mu += w_base @ self._base_zmu
        if w_target > 0:
            tz = self._target_z(task_index)
            if tz is not None:
                mu += w_target * tz[0]

        if self.oracle_base:
            # GP-free uncertainty: weighted disagreement of the base columns about
            # each flow (high where related tasks disagree -> worth exploring).
            if len(w_base) > 0:
                diff = self._base_zmu - mu[None, :]
                var = (w_base[:, None] * diff**2).sum(axis=0)
        else:
            if self.use_base_variance and len(w_base) > 0:
                var += (w_base**2) @ (self._base_zsigma**2)
            if w_target > 0 and tz is not None:
                var += (w_target**2) * (tz[1] ** 2)
            # Fallback so EI is not degenerate when no usable variance accumulated.
            if not np.any(var) and len(w_base) > 0:
                var = (self._base_zsigma**2).mean(axis=0)

        sigma = np.sqrt(np.clip(var, 1e-12, None))
        self._pred_cache[task_index] = (mu, sigma)
        return mu, sigma

    def predict(self, task_index: int) -> np.ndarray:
        return self._ensemble(task_index)[0]

    def predict_uncertainty(self, task_index: int) -> Optional[np.ndarray]:
        return self._ensemble(task_index)[1]

    def suggest(self, task_index, predictions=None):
        """EI over the discrete flows, with ``f_star`` in the ensemble z-score space.

        Overrides the base ``suggest`` because the ensemble predicts standardised
        scores, so the incumbent must be standardised by the same target stats.
        """
        mu = np.asarray(predictions) if predictions is not None else self.predict(task_index)
        sigma = self.predict_uncertainty(task_index)

        col = self.dataset.require_matrix().iloc[:, task_index].values
        known_mask = ~np.isnan(col)
        stats = self._target_stats(task_index)
        if stats is not None and np.any(known_mask):
            m, s = stats
            f_star = (float(np.max(col[known_mask])) - m) / s
        else:
            f_star = float("-inf")  # cold start: everything is an improvement

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
        # Write the observation into the matrix *before* re-fitting (per CLAUDE.md).
        self.dataset.performance.iloc[flow_index, task_index] = performance
        self._fit_target(task_index)
        self._weight_cache.pop(task_index, None)
        self._pred_cache.pop(task_index, None)
