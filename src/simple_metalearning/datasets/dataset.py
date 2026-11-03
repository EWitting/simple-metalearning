import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd


@dataclass
class MetaDataset:
    """Stores the results of evaluating a set of flows (model/pipeline configurations)
    across a set of tasks (datasets), together with associated metadata.

    The central ``performance`` matrix has shape ``(n_flows, n_tasks)`` with
    ``NaN`` for unevaluated combinations.  It is **optional**: datasets for
    continuous search-space settings (where evaluations are performed live) may
    not have a pre-computed matrix.  Estimators that require the matrix should
    call :meth:`require_matrix` to get a clear error at initialisation time.

    Performance orientation
    -----------------------
    ``performance`` values are always interpreted as **higher is better**.  If
    the raw metric is a loss (lower is better), negate it before constructing
    the dataset and set ``direction="minimize"`` to record the original intent.

    The ``direction`` field is metadata only — it does **not** flip the sign
    automatically.  All estimators, acquisition functions, and regret
    computations assume higher-is-better values in ``performance``.  Use
    :meth:`prepare_performance` to obtain a correctly signed matrix.
    """

    dataset_name: str

    # Matrix of performance with shape (n_flows, n_tasks).
    # NaN for unknown/unevaluated combinations.
    # Values must be higher-is-better (negate loss metrics before storing).
    # Optional: continuous-search-space datasets may omit this.
    performance: Optional[pd.DataFrame] = None

    # Optimisation direction of the *original* metric.
    # "maximize"  (default) — raw metric is higher-is-better.
    # "minimize"  — raw metric is lower-is-better; performance stores -metric.
    direction: str = "maximize"

    # Optional matrix with the same shape as performance, representing
    # computational cost (e.g. wall-clock training time in seconds).
    cost: Optional[pd.DataFrame] = None

    # Optional held-out test performance matrix (n_flows, n_tasks).
    # When present (produced by val_test_split=True in TabArena), the
    # ``performance`` matrix contains validation-fold averages and this
    # matrix contains test-fold averages.  Used to compute test regret:
    # select the best flow according to val, then evaluate it on test.
    test_performance: Optional[pd.DataFrame] = None

    # Boolean mask (n_flows, n_tasks) — True where a (flow, task) pair has a
    # genuinely evaluated result.  False means the cell was imputed (e.g. a
    # classification-only model on a regression task, or a crashed run) and
    # only contains the dummy-baseline fill value.
    # When present, BOSelection can optionally exclude imputed flows from BO
    # selection while still using them for surrogate training (configurable
    # via ``exclude_invalid`` in the experiment config).
    valid_mask: Optional[pd.DataFrame] = None

    # Human-readable names for display; do not have to be unique.
    flow_names: Optional[list[str]] = None
    task_names: Optional[list[str]] = None

    # Metadata available for flows and tasks.
    # Flows: e.g. hyperparameter values (pd.DataFrame with one row per flow).
    # Tasks: e.g. dataset metafeatures (pd.DataFrame with one row per task).
    flow_metadata: Optional[pd.DataFrame] = None
    task_metadata: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(
                f"MetaDataset.direction must be 'maximize' or 'minimize', "
                f"got '{self.direction}'."
            )

    # ------------------------------------------------------------------
    # Helpers for matrix-requiring code
    # ------------------------------------------------------------------

    def prepare_performance(self) -> "pd.DataFrame":
        """Return the performance matrix in higher-is-better orientation.

        When ``direction == "maximize"`` this is identical to
        :attr:`performance`.  When ``direction == "minimize"`` the matrix is
        returned *negated* so that higher values still mean better
        configurations.

        This is a convenience method for visualisation and reporting.
        Estimators should always work with :attr:`performance` directly (which
        must already be in higher-is-better form).
        """
        df = self.require_matrix()
        if self.direction == "minimize":
            return -df
        return df

    @property
    def has_matrix(self) -> bool:
        """True when a pre-computed performance matrix is available."""
        return self.performance is not None

    def require_matrix(self) -> pd.DataFrame:
        """Return the performance matrix, raising a clear error if absent.

        Call this at the top of any estimator ``__init__`` or ``fit()`` that
        needs the matrix.  This gives an actionable error message instead of a
        confusing ``AttributeError`` deep in the code.

        Returns:
            The performance ``pd.DataFrame``.

        Raises:
            ValueError: If ``performance`` is ``None``.
        """
        if self.performance is None:
            raise ValueError(
                "This estimator requires a pre-computed performance matrix "
                "(MetaDataset.performance), but none was provided.  "
                "Ensure the dataset factory populates the performance matrix, "
                "or use a ContinuousEstimator that does not require one."
            )
        return self.performance


@dataclass
class MetaDatasetFactory(ABC):
    """Class used to generate, load, download, or scrape meta datasets.
    Should be deterministic and include a hash function for caching the result."""

    cache_dir: str = "./data/processed"

    @abstractmethod
    def _create(self) -> MetaDataset:
        """Create a dataset"""
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> str:
        """Returns a string hash/identifier for this factory's parameters (for caching)."""
        raise NotImplementedError

    def cache_path(self) -> str:
        """Returns the complete file path used for storing the cached dataset."""
        identifier = self.hash()
        return f"{self.cache_dir}/{identifier}.pkl"

    def load(self) -> MetaDataset:
        """Creates dataset or loads from cache if available"""

        # Load cache if exists
        cache_file_path = self.cache_path()
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "rb") as f:
                dataset = pickle.load(f)
                print(f"Loaded dataset from cache: {cache_file_path}")
            return dataset

        # Create dataset if necessary
        dataset = self._create()

        # Ensure the cache directory exists, and save result
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file_path, "wb") as f:
            pickle.dump(dataset, f)

        return dataset
