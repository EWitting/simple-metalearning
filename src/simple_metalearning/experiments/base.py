from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from omegaconf import DictConfig

from ..datasets.dataset import MetaDataset


class Experiment(ABC):
    """Base class for experiments."""

    @abstractmethod
    def run(self, dataset: MetaDataset, estimator_factory: Callable) -> Any:
        """
        Execute the experiment.

        Returns:
            Experiment results, dictionary or other
        """
        raise NotImplementedError()
