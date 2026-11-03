from typing import Optional

import numpy as np

from .estimator import Estimator
from ..datasets.dataset import MetaDataset


class Mean(Estimator):
    """Estimates performance based on mean performance across tasks."""

    def __init__(self, dataset: MetaDataset):
        super().__init__(dataset)
        self.row_means = None
        self.global_mean = None
        self.fit()

    def fit(self):
        """Compute row means and global mean from the dataset."""
        perf_matrix = self.dataset.performance.values.astype(float)
        self.global_mean = np.nanmean(perf_matrix)
        self.row_means = np.nanmean(perf_matrix, axis=1)

        # Replace NaN row means with global mean
        nan_rows = np.isnan(self.row_means)
        if nan_rows.any():
            self.row_means[nan_rows] = self.global_mean if not np.isnan(self.global_mean) else 0.0

    def predict(self, task_index: int) -> np.ndarray:
        """Return row means as scores."""
        return self.row_means

    def partial_fit(
        self,
        flow_index: int,
        task_index: int,
        performance: float,
    ) -> None:
        """Update the internal means with the new observation."""
        perf_matrix = self.dataset.performance.values
        row = perf_matrix[flow_index]

        new_row_mean = np.nanmean(row)

        if np.isnan(new_row_mean):
             self.row_means[flow_index] = self.global_mean
        else:
             self.row_means[flow_index] = new_row_mean

        # Update global mean
        self.global_mean = np.nanmean(perf_matrix)
