from typing import Optional, Union, List
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .estimator import Estimator
from ..datasets.dataset import MetaDataset


class MeanMMR(Estimator):
    """Estimates performance based on mean performance across tasks, with MMR for diversity in suggest."""

    def __init__(
        self,
        dataset: MetaDataset,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
    ):
        """Initialize the MeanMMR estimator.

        Args:
            dataset: The meta-learning dataset
            use_mmr: Whether to use maximal marginal relevance in suggest.
            mmr_lambda: Weight for relevance vs diversity in suggest. 1.0 is pure relevance.
        """
        super().__init__(dataset)
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        
        self.row_means = None
        self.global_mean = None
        self.flow_similarity_matrix = None
        
        self.fit()

    def fit(self) -> None:
        """Compute global mean, row means, and flow similarity matrix."""
        self._update_means()
        self._compute_similarity()  # computed once per task start; not updated in partial_fit

    def _update_means(self) -> None:
        """Recompute global mean and row means from the current performance matrix."""
        perf_matrix = self.dataset.performance.values.astype(float)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            self.global_mean = np.nanmean(perf_matrix)

        if np.isnan(self.global_mean):
            self.global_mean = 0.0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            self.row_means = np.nanmean(perf_matrix, axis=1)

        nan_rows = np.isnan(self.row_means)
        if nan_rows.any():
            self.row_means[nan_rows] = self.global_mean

    def _compute_similarity(self) -> None:
        """Compute cosine flow similarity matrix from the current performance matrix.
        
        Flows with no observations get zero similarity so they carry no diversity penalty.
        """
        X = self.dataset.performance.values.astype(float).copy()

        all_nan_rows = np.all(np.isnan(X), axis=1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            col_means = np.nanmean(X, axis=0)

        col_means[np.isnan(col_means)] = self.global_mean

        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

        sim = cosine_similarity(X)
        sim[all_nan_rows, :] = 0.0
        sim[:, all_nan_rows] = 0.0
        self.flow_similarity_matrix = sim

    def predict(self, task_index: int) -> np.ndarray:
        """Return row means as scores."""
        return self.row_means

    def partial_fit(
        self,
        flow_index: Union[int, List[int]],
        task_index: int,
        performance: Union[float, List[float]],
    ) -> None:
        """Update means with the new observation. Similarity is not recomputed here."""
        self.dataset.performance.iloc[flow_index, task_index] = performance
        self._update_means()

    def suggest(
        self, task_index: int, predictions: Optional[Union[np.ndarray, List[float]]] = None
    ) -> np.ndarray:
        """Suggest the next flow to evaluate using Maximum Marginal Relevance (MMR).

        Args:
            task_index: Index of the target task.
            predictions: Optional precomputed performance predictions for all flows.

        Returns:
            A binary mask (np.ndarray) where 1 indicates the suggested flow.
        """
        if predictions is None:
            predictions = self.predict(task_index)
        else:
            predictions = np.asarray(predictions)

        if not self.use_mmr:
            # greedy selection, return the prediction array
            return predictions

        perf_col = self.dataset.performance.iloc[:, task_index].values
        known_mask = ~np.isnan(perf_col)
        unknown_indices = np.where(np.isnan(perf_col))[0]

        if len(unknown_indices) == 0:
            # All flows are already known
            return np.zeros(len(perf_col))

        if self.flow_similarity_matrix is None:
            self.fit()

        # S: indices of flows already selected (known)
        selected_indices = np.where(known_mask)[0]

        # If no flows are selected yet, just pick the best predicted one
        if len(selected_indices) == 0:
            best_idx = unknown_indices[np.argmax(predictions[unknown_indices])]
            suggestion = np.zeros(len(perf_col))
            suggestion[best_idx] = 1.0
            return suggestion

        # MMR Calculation for each unknown flow i:
        # Score_i = lambda * relevance_i - (1 - lambda) * max_{j in S} similarity(i, j)
        best_mmr_score = -np.inf
        best_idx = unknown_indices[0]

        for i in unknown_indices:
            relevance = predictions[i]
            # Max similarity to any already selected flow
            max_sim = np.max(self.flow_similarity_matrix[i, selected_indices])

            mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i

        suggestion = np.zeros(len(perf_col))
        suggestion[best_idx] = 1.0
        return suggestion
