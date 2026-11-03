"""Meta-learned OOFRidge surrogate for continuous-space BO.

During :meth:`fit` (called once per CV fold on the masked MetaDataset), one
RandomForestRegressor is trained per meta-train task using the discrete
flow_metadata features.  These per-task RFs define a feature map

    φ(x) = [RF_j.predict(featurize(x))]_{j=1..J}

that embeds any XGBoost configuration into a cross-task predicted-performance
space.  At BO time, a BayesianRidge is fit on observed (φ(x_i), y_i) pairs to
provide calibrated EI.

The MMR variant (ContinuousOOFRidgeMMREstimator) uses cosine similarity in φ-
space for the diversity term, mirroring the discrete OOFRidgeMMREstimator.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from simple_metalearning.estimators.acquisition import EIGaussianAcquisition, MMRAcquisition
from .base import ContinuousEstimator

log = logging.getLogger(__name__)


class ContinuousOOFRidgeEstimator(ContinuousEstimator):
    """Meta-learned OOFRidge surrogate.

    The per-task RF feature map is trained on the masked MetaDataset (train
    tasks only) in :meth:`fit`.  At BO time a BayesianRidge is fit on the φ-
    embedded observations.

    Args:
        dataset: Masked MetaDataset (test-task columns are NaN; flow_metadata
            must be present).
        rf_n_estimators: Trees per per-task RF.
        rf_min_samples_leaf: Leaf size (SMAC default = 3).
        acquisition_fn: Defaults to EIGaussianAcquisition.
    """

    def __init__(
        self,
        dataset=None,
        rf_n_estimators: int = 100,
        rf_min_samples_leaf: int = 3,
        acquisition_fn=None,
    ):
        super().__init__(dataset=dataset)
        self.rf_n_estimators = rf_n_estimators
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self._acq_fn_obj = acquisition_fn or EIGaussianAcquisition()
        self._task_rfs: dict[int, RandomForestRegressor] = {}
        self._ridge: Optional[BayesianRidge] = None
        self._featurizer = None  # set lazily from dataset.flow_metadata.columns
        self._phi_mean: Optional[float] = None  # fallback prediction

    @property
    def _acq_fn(self):
        return self._acq_fn_obj

    # ------------------------------------------------------------------
    # Meta-training
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Train one RF per meta-train task on discrete flow_metadata features."""
        if self.dataset is None or self.dataset.flow_metadata is None:
            raise ValueError(
                "ContinuousOOFRidgeEstimator requires dataset with flow_metadata."
            )
        perf = self.dataset.performance  # (n_flows, n_tasks), NaN for test tasks
        fm = self.dataset.flow_metadata

        # Build featurizer if not yet set
        from simple_metalearning.search_spaces.xgboost_space import ConfigFeaturizer
        self._featurizer = ConfigFeaturizer(columns=list(fm.columns))

        flow_features = fm.to_numpy(dtype=float, na_value=np.nan)
        # Impute NaN in flow_features with column means
        col_means = np.nanmean(flow_features, axis=0)
        nan_mask_cols = np.isnan(col_means)
        col_means[nan_mask_cols] = 0.0
        inds = np.where(np.isnan(flow_features))
        flow_features = flow_features.copy()
        flow_features[inds] = np.take(col_means, inds[1])

        rf_proto = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            min_samples_leaf=self.rf_min_samples_leaf,
            random_state=42,
        )

        self._task_rfs = {}
        for task_idx in range(perf.shape[1]):
            y = perf.iloc[:, task_idx].values
            known = ~np.isnan(y)
            if known.sum() < 2:
                continue
            rf = clone(rf_proto)
            rf.fit(flow_features[known], y[known])
            self._task_rfs[task_idx] = rf

        self._phi_mean = float(np.nanmean(perf.values)) if perf is not None else 0.0

    # ------------------------------------------------------------------
    # Feature map φ(x)
    # ------------------------------------------------------------------

    def _phi(self, X_feat: np.ndarray) -> np.ndarray:
        """Map feature matrix to cross-task RF prediction space.

        Args:
            X_feat: Shape ``(n, n_flow_features)`` — featurized configs.

        Returns:
            Shape ``(n, n_meta_tasks)`` — predicted performance per meta-task.
        """
        if not self._task_rfs:
            return np.full((len(X_feat), 1), self._phi_mean or 0.0)

        task_preds = []
        for task_idx in sorted(self._task_rfs):
            rf = self._task_rfs[task_idx]
            task_preds.append(rf.predict(X_feat))
        return np.column_stack(task_preds)  # (n, n_meta_tasks)

    def _featurize_configs(self, configs_or_X) -> np.ndarray:
        """Accept either a feature matrix or Configuration list."""
        if isinstance(configs_or_X, np.ndarray):
            return configs_or_X
        # list of Configuration
        return self._featurizer.transform(configs_or_X)

    # ------------------------------------------------------------------
    # Surrogate refit on observations
    # ------------------------------------------------------------------

    def _refit(self) -> None:
        if self.n_obs < 2:
            return
        X_obs = self._X
        y_obs = self._y
        Phi_obs = self._phi(X_obs)
        model = BayesianRidge(fit_intercept=False)
        try:
            model.fit(Phi_obs, y_obs)
            self._ridge = model
        except Exception as exc:
            # Degrade to the prior-mean fallback rather than crash, but make the
            # degradation visible — a silently dead surrogate looks like a bad
            # method, not a numerical failure.
            log.warning(
                "OOFRidge BayesianRidge.fit failed at n_obs=%d (%s); "
                "falling back to prior mean for this step.", self.n_obs, exc,
            )
            self._ridge = None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_mu_sigma(
        self, X_cand: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        Phi_cand = self._phi(X_cand)
        if self._ridge is None:
            mu = np.full(len(X_cand), self._phi_mean or 0.0)
            return mu, np.ones(len(X_cand))
        mu, sigma = self._ridge.predict(Phi_cand, return_std=True)
        return mu, sigma

    def acquisition_scores(
        self,
        X_cand: np.ndarray,
        f_star: float,
        known_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mu, sigma = self.predict_mu_sigma(X_cand)
        dummy_mask = np.zeros(len(X_cand), dtype=bool)
        return self._acq_fn_obj.compute(mu, sigma, f_star, dummy_mask)


class ContinuousOOFRidgeMMREstimator(ContinuousOOFRidgeEstimator):
    """OOFRidge variant that scores acquisition with MMR in φ-space.

    Similarity between configs is computed via cosine similarity on φ(x)
    (the cross-task RF prediction vectors), mirroring the discrete
    OOFRidgeMMREstimator which uses the OOF matrix for similarity.
    """

    def __init__(
        self,
        dataset=None,
        rf_n_estimators: int = 100,
        rf_min_samples_leaf: int = 3,
        acquisition_fn=None,
    ):
        super().__init__(
            dataset=dataset,
            rf_n_estimators=rf_n_estimators,
            rf_min_samples_leaf=rf_min_samples_leaf,
            acquisition_fn=acquisition_fn or MMRAcquisition(),
        )

    def acquisition_scores(
        self,
        X_cand: np.ndarray,
        f_star: float,
        known_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mu, sigma = self.predict_mu_sigma(X_cand)

        # Build φ-space similarity for all candidates + observed configs
        # Stack observed and candidate phi vectors; similarity matrix over all.
        Phi_cand = self._phi(X_cand)
        dummy_mask_cand = np.zeros(len(X_cand), dtype=bool)

        # Encode observed configs in φ-space to build similarity
        if self.n_obs > 0:
            Phi_obs = self._phi(self._X)
            # Full φ matrix: observed first, then candidates
            Phi_all = np.vstack([Phi_obs, Phi_cand])
            known_mask_all = np.array(
                [True] * self.n_obs + [False] * len(X_cand), dtype=bool
            )
        else:
            Phi_all = Phi_cand
            known_mask_all = dummy_mask_cand

        # Pass the phi matrix to MMRAcquisition.prepare() for similarity.
        # prepare() builds cosine similarity over the ROWS of what it is given,
        # and compute() indexes that matrix by entity (config) index, so the
        # rows must be the configs themselves: Phi_all is already (n_entities,
        # n_meta_tasks).  Transposing here would build a (meta_task, meta_task)
        # matrix and make compute() index out of bounds.
        self._acq_fn_obj.prepare(Phi_all, known_mask_all)

        # Compute MMR scores for candidates only
        # mu and sigma are for candidates; known_mask_all covers observed+cand
        mu_all = np.concatenate([
            self._ridge.predict(Phi_obs) if (self.n_obs > 0 and self._ridge is not None)
            else np.full(self.n_obs, self._phi_mean or 0.0),
            mu,
        ])
        sigma_all = np.concatenate([
            np.zeros(self.n_obs),
            sigma if sigma is not None else np.zeros(len(X_cand)),
        ])

        scores_all = self._acq_fn_obj.compute(
            mu_all, sigma_all, f_star, known_mask_all
        )
        # Return scores for candidate portion only
        return scores_all[self.n_obs:]
