"""Maximal Marginal Relevance (MMR) acquisition function.

MMR promotes diversity by penalising candidates that are too similar to flows
that have already been evaluated.  The score is:

    MMR(x) = λ · relevance(x) - (1 - λ) · max_{x' ∈ known} sim(x, x')

where ``relevance`` is the predicted mean (or any other score) and ``sim`` is
cosine similarity between flow performance vectors.

This is extracted from the existing ``SVD``/``MeanMMR`` logic and generalised
so it can be composed with any base acquisition function.
"""

import warnings
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import AcquisitionFunction


class MMRAcquisition(AcquisitionFunction):
    """Diversity-promoting acquisition via Maximal Marginal Relevance.

    Requires a flow similarity matrix (cosine similarity of performance
    vectors) which is built lazily from the performance matrix on first use.

    Args:
        mmr_lambda: Trade-off weight in [0, 1].  ``1.0`` → pure relevance
            (greedy), ``0.0`` → pure diversity.  Defaults to 0.5.
        negative_sim_filter: If True, treats negative cosine similarities as
            zero (flows that are anti-correlated are not penalised for
            diversity).  Defaults to True.
    """

    def __init__(
        self,
        mmr_lambda: float = 0.5,
        negative_sim_filter: bool = True,
    ) -> None:
        self.mmr_lambda = mmr_lambda
        self.negative_sim_filter = negative_sim_filter
        self._similarity_matrix: Optional[np.ndarray] = None
        self._performance_matrix_id: Optional[int] = None

    def set_similarity_matrix(self, similarity_matrix: np.ndarray) -> None:
        """Directly set the flow similarity matrix (shape n_flows × n_flows)."""
        self._similarity_matrix = similarity_matrix

    def build_similarity_from_performance(self, performance: np.ndarray) -> None:
        """Build flow-flow cosine similarity from an imputed performance matrix.

        Args:
            performance: Shape ``(n_flows, n_tasks)``, NaN values should be
                imputed before calling this (e.g. with column means).
        """
        sim = cosine_similarity(performance)
        if self.negative_sim_filter:
            sim = np.clip(sim, 0.0, None)
        self._similarity_matrix = sim

    def prepare(self, performance: np.ndarray, known_mask: np.ndarray) -> None:
        """Build the flow similarity matrix from the current performance matrix.

        Called automatically by :meth:`~.estimator.Estimator.suggest` before
        each ``compute()`` call so the similarity matrix always reflects the
        latest observations.  NaN entries (unobserved cells) are imputed with
        column means before computing cosine similarity.

        Args:
            performance: Shape ``(n_flows, n_tasks)``, may contain NaN.
            known_mask: Unused by MMR but required by the base interface.
        """
        perf = performance.astype(float).copy()
        # All-NaN test-task columns (masked by k-fold) produce a harmless
        # "Mean of empty slice" RuntimeWarning from nanmean — suppress it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            col_means = np.nanmean(perf, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_idx = np.where(np.isnan(perf))
        perf[nan_idx] = np.take(col_means, nan_idx[1])
        self.build_similarity_from_performance(perf)

    def compute(
        self,
        mu: np.ndarray,
        sigma: Optional[np.ndarray],
        f_star: float,
        known_mask: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute MMR scores.

        Falls back to ``mu`` (greedy) if the similarity matrix has not been
        set or if no flows are known yet.
        """
        if self._similarity_matrix is None or not np.any(known_mask):
            return mu.copy()

        n_flows = len(mu)
        unknown_indices = np.where(~known_mask)[0]
        known_indices = np.where(known_mask)[0]

        scores = np.full(n_flows, -np.inf)
        sim_matrix = self._similarity_matrix

        for i in unknown_indices:
            relevance = mu[i]
            max_sim = np.max(sim_matrix[i, known_indices])
            scores[i] = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * max_sim

        return scores


class EmbeddingMMRAcquisition(MMRAcquisition):
    """MMR acquisition that measures flow similarity in a learned embedding space.

    Instead of building the similarity matrix from the raw performance matrix,
    this class uses cosine similarity between flow tower embeddings provided by
    the estimator.  The embeddings must be set via :meth:`set_flow_embeddings`
    before each ``compute()`` call (done automatically by
    :class:`~simple_metalearning.estimators.two_tower.TwoTowerEstimator` when this
    acquisition function is active).

    Falls back to performance-matrix similarity if no embeddings have been set.

    Args:
        mmr_lambda: Trade-off weight — same semantics as :class:`MMRAcquisition`.
        negative_sim_filter: Clip negative similarities to zero (default True).
    """

    def __init__(
        self,
        mmr_lambda: float = 0.5,
        negative_sim_filter: bool = True,
    ) -> None:
        super().__init__(mmr_lambda=mmr_lambda, negative_sim_filter=negative_sim_filter)
        self._use_embedding_similarity = False

    def set_flow_embeddings(self, embeddings: np.ndarray) -> None:
        """Compute and cache the flow similarity matrix from tower embeddings.

        Args:
            embeddings: Shape ``(n_flows, embedding_dim)``.
        """
        self.build_similarity_from_performance(embeddings)
        self._use_embedding_similarity = True

    def prepare(self, performance: np.ndarray, known_mask: np.ndarray) -> None:
        """Skip performance-based similarity rebuild when embeddings are available."""
        if not self._use_embedding_similarity:
            super().prepare(performance, known_mask)
