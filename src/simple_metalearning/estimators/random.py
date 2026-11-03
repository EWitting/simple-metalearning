from typing import Optional

import numpy as np

from .estimator import Estimator


class Random(Estimator):
    """Random baseline estimator."""

    def __init__(self, dataset, random_state: Optional[int] = None):
        super().__init__(dataset)
        self.rng = np.random.default_rng(random_state)

    def fit(self):
        pass

    def predict(self, task_index: int) -> np.ndarray:
        """Return random scores for each flow."""
        n_flows = self.dataset.performance.shape[0]
        return self.rng.uniform(0, 1, size=n_flows)

    def partial_fit(
        self,
        flow_index: int,
        task_index: int,
        performance: float,
    ) -> None:
        """No update needed for random prediction."""
        pass
