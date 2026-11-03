"""Transformations for flow and task metafeatures.

This module provides a flexible pipeline for transforming metafeatures before
they are used by estimators. Transformations can be composed in a list to create
complex feature engineering pipelines.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from simple_metalearning.datasets.dataset import MetaDataset


class MetafeatureTransform(ABC):
    """Base class for metafeature transformations."""

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "MetafeatureTransform":
        """Fit the transform on the data.

        Args:
            X: DataFrame of shape (n_samples, n_features)

        Returns:
            self for chaining
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Args:
            X: DataFrame of shape (n_samples, n_features)

        Returns:
            Transformed DataFrame
        """
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data.

        Args:
            X: DataFrame of shape (n_samples, n_features)

        Returns:
            Transformed DataFrame
        """
        return self.fit(X).transform(X)


class IdentityTransform(MetafeatureTransform):
    """Identity transform that returns the input unchanged."""

    def fit(self, X: pd.DataFrame) -> "IdentityTransform":
        """Fit is a no-op for identity transform."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the input unchanged."""
        return X.copy()


class OneHotTransform(MetafeatureTransform):
    """One-hot encoding based on the index (flow or task ID).

    This creates a one-hot encoded representation of the sample indices,
    which can be useful for capturing flow/task-specific effects.
    """

    def __init__(self):
        self.n_samples_: Optional[int] = None
        self.index_: Optional[pd.Index] = None

    def fit(self, X: pd.DataFrame) -> "OneHotTransform":
        """Fit by storing the number of samples and index."""
        self.n_samples_ = len(X)
        self.index_ = X.index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform to one-hot encoded index.

        Returns a DataFrame where each row is a one-hot vector indicating
        which sample (by index position) it represents.
        """
        if self.n_samples_ is None:
            raise ValueError("Transform not fitted. Call fit() first.")

        # Create one-hot encoded matrix based on index positions
        onehot = np.zeros((len(X), self.n_samples_))

        # Map each index in X to its position in the fitted index
        for i, idx in enumerate(X.index):
            if idx in self.index_:
                pos = self.index_.get_loc(idx)
                onehot[i, pos] = 1
            # If index not found, leave as zeros (could also raise error)

        # Create column names
        columns = [f"onehot_{i}" for i in range(self.n_samples_)]

        return pd.DataFrame(onehot, index=X.index, columns=columns)


class StandardScalerTransform(MetafeatureTransform):
    """Standardize features by removing the mean and scaling to unit variance.

    Uses sklearn's StandardScaler under the hood.
    """

    def __init__(self, fill_na: bool = False):
        self.fill_na = fill_na
        self.scaler_ = None

    def fit(self, X: pd.DataFrame) -> "StandardScalerTransform":
        """Fit the scaler."""
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X.astype("float64").values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardize the features."""
        if self.scaler_ is None:
            raise ValueError("Transform not fitted. Call fit() first.")

        X_transformed = self.scaler_.transform(X.astype("float64").values)

        if self.fill_na:
            X_transformed[np.isnan(X_transformed)] = 0

        # Keep original column names
        return pd.DataFrame(X_transformed, index=X.index, columns=X.columns)


class PCATransform(MetafeatureTransform):
    """PCA dimensionality reduction.

    By default, standardizes features before applying PCA (recommended for most cases).
    This ensures features on different scales contribute equally to the principal components.

    Args:
        n_components: Number of components to keep. If None, keeps all.
        standardize: Whether to standardize before PCA. Default True (recommended).
    """

    def __init__(self, n_components: Optional[int] = None, standardize: bool = True):
        self.n_components = n_components
        self.standardize = standardize
        self.scaler_ = StandardScaler() if standardize else None
        self.pca_ = None

    def fit(self, X: pd.DataFrame) -> "PCATransform":
        """Fit the PCA transform."""
        X_arr = X.values.astype(float)

        # Standardize if requested (default)
        if self.standardize:
            X_arr = self.scaler_.fit_transform(X_arr)

        # Determine number of components
        n_components = self.n_components
        if n_components is None:
            n_components = min(X_arr.shape)

        # Fit PCA
        self.pca_ = PCA(n_components=n_components)
        self.pca_.fit(X_arr)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted PCA."""
        if self.pca_ is None:
            raise ValueError("Transform not fitted. Call fit() first.")

        X_arr = X.values.astype(float)

        # Standardize if requested
        if self.standardize:
            X_arr = self.scaler_.transform(X_arr)

        # Apply PCA
        X_transformed = self.pca_.transform(X_arr)

        # Create column names
        columns = [f"pca_{i}" for i in range(X_transformed.shape[1])]

        return pd.DataFrame(X_transformed, index=X.index, columns=columns)


class PipelineTransform(MetafeatureTransform):
    """Chain transforms sequentially: output of each step becomes input of the next.

    Unlike :class:`CompositeTransform` (which applies each transform to the
    *original* input and concatenates outputs), ``PipelineTransform`` threads
    the data through the steps in order.  Use this when later steps depend on
    earlier ones, e.g. DropConstant → StandardScaler.

    Steps may be already-instantiated :class:`MetafeatureTransform` objects or
    Hydra DictConfig dicts (they will be instantiated lazily).
    """

    def __init__(self, steps: list):
        from hydra.utils import instantiate as _instantiate
        from omegaconf import DictConfig
        self.steps = [
            _instantiate(s) if isinstance(s, DictConfig) else s for s in steps
        ]

    def fit(self, X: pd.DataFrame) -> "PipelineTransform":
        X_curr = X
        for step in self.steps:
            step.fit(X_curr)
            X_curr = step.transform(X_curr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_curr = X
        for step in self.steps:
            X_curr = step.transform(X_curr)
        return X_curr


class DropConstantTransform(MetafeatureTransform):
    """Drop columns that are constant (≤1 unique non-NaN value) across all samples.

    Fitted once on the full dataset; the same set of columns is dropped at
    transform time.  Useful for removing features that are uninformative when
    a single method is selected (e.g. ``model_type_XGB`` in XGBoost-only HPO,
    or ``enable_categorical`` which is always 1 for XGBoost configs).
    """

    def __init__(self):
        self.const_cols_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "DropConstantTransform":
        self.const_cols_ = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.const_cols_, errors="ignore")


class QuantileTransform(MetafeatureTransform):
    """Map each feature to a standard normal distribution via rank-based normalization.

    Uses sklearn's QuantileTransformer, which is distribution-agnostic: it
    handles log-uniform, uniform, or any other sampling distribution correctly
    without needing to know which hyperparameters were log-sampled.  NaN values
    are optionally filled with 0 (the median of the output distribution) after
    transformation.

    Args:
        n_quantiles: Number of quantiles used for estimation.  Defaults to the
            number of training samples (i.e. one quantile per sample).
        fill_na: If True, replace any remaining NaN after transformation with 0.
    """

    def __init__(self, n_quantiles: Optional[int] = None, fill_na: bool = True):
        self.n_quantiles = n_quantiles
        self.fill_na = fill_na
        self.transformer_: Optional[QuantileTransformer] = None

    def fit(self, X: pd.DataFrame) -> "QuantileTransform":
        X_arr = X.astype("float64").values
        n_quantiles = self.n_quantiles if self.n_quantiles is not None else X_arr.shape[0]
        self.transformer_ = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=42,
        )
        # Fit ignoring NaN rows per column by masking; sklearn raises on NaN so
        # we fit per-column on the non-NaN subset.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Fit on a NaN-filled copy (column means) so sklearn doesn't error.
            col_means = np.nanmean(X_arr, axis=0)
            X_filled = np.where(np.isnan(X_arr), col_means, X_arr)
            self.transformer_.fit(X_filled)
        self._col_means = col_means
        self._columns = list(X.columns)
        self._index = X.index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.transformer_ is None:
            raise ValueError("QuantileTransform not fitted. Call fit() first.")
        X_arr = X.astype("float64").values
        nan_mask = np.isnan(X_arr)
        X_filled = np.where(nan_mask, self._col_means, X_arr)
        X_transformed = self.transformer_.transform(X_filled)
        if self.fill_na:
            X_transformed[nan_mask] = 0.0
        else:
            X_transformed[nan_mask] = np.nan
        return pd.DataFrame(X_transformed, index=X.index, columns=X.columns)


class CompositeTransform(MetafeatureTransform):
    """Applies multiple transforms and concatenates their outputs.

    Args:
        transforms: List of transforms to apply. Outputs are concatenated horizontally.
    """

    def __init__(self, transforms: list[MetafeatureTransform]):
        self.transforms = transforms

    def fit(self, X: pd.DataFrame) -> "CompositeTransform":
        """Fit all transforms."""
        for transform in self.transforms:
            transform.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transforms and concatenate results."""
        if not self.transforms:
            return X.copy()

        transformed_parts = []
        for transform in self.transforms:
            transformed_parts.append(transform.transform(X))

        # Concatenate horizontally
        return pd.concat(transformed_parts, axis=1)


class PerformanceVectorTransform:
    """Convert the performance matrix rows into per-flow feature vectors.

    Applies per-column mean imputation (so NaN entries, i.e. unevaluated
    flow-task pairs, are filled with the observed column mean), then
    optionally reduces dimensionality with PCA.

    **Must be applied inside estimator ``fit()``**, after the experiment has
    already masked test-task columns to NaN.  Applying it before the k-fold
    split would leak test-task performance into the flow representation.

    Args:
        n_components: Number of PCA components to retain.  ``None`` (default)
            skips PCA and keeps all task dimensions.
        impute_strategy: How to fill NaN entries.  One of ``"mean"``
            (default), ``"median"``, or ``"zero"``.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        impute_strategy: str = "mean",
    ) -> None:
        self.n_components = n_components
        self.impute_strategy = impute_strategy
        self._col_fill: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None

    def fit(self, performance: pd.DataFrame) -> "PerformanceVectorTransform":
        """Fit imputation statistics (and optional PCA) on *performance*.

        Args:
            performance: DataFrame of shape ``(n_flows, n_tasks)`` with NaN
                for unevaluated entries.  Test-task columns should already be
                all-NaN at this point.

        Returns:
            self
        """
        arr = performance.values.astype(float)

        # np.nanmean/nanmedian emit a Python-level RuntimeWarning ("Mean of empty
        # slice") for all-NaN columns (test tasks after k-fold masking).  Use
        # warnings.catch_warnings — np.errstate only suppresses C-level FP flags,
        # not the warnings.warn call inside nanfunctions.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.impute_strategy == "mean":
                col_fill = np.nanmean(arr, axis=0)
            elif self.impute_strategy == "median":
                col_fill = np.nanmedian(arr, axis=0)
            elif self.impute_strategy == "zero":
                col_fill = np.zeros(arr.shape[1])
            else:
                raise ValueError(
                    f"Unknown impute_strategy '{self.impute_strategy}'. "
                    "Choose 'mean', 'median', or 'zero'."
                )

        # Columns that are entirely NaN (e.g. test-task columns) get filled
        # with 0 so they don't propagate NaN into the feature matrix.
        self._col_fill = np.nan_to_num(col_fill, nan=0.0)

        arr_imputed = self._impute(arr)

        if self.n_components is not None:
            n_comp = min(self.n_components, min(arr_imputed.shape) - 1)
            self._pca = PCA(n_components=n_comp)
            self._pca.fit(arr_imputed)

        return self

    def transform(self, performance: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation (and optional PCA) to *performance*.

        Args:
            performance: DataFrame of shape ``(n_flows, n_tasks)``.

        Returns:
            DataFrame of shape ``(n_flows, n_tasks)`` or
            ``(n_flows, n_components)`` if PCA is enabled.
        """
        if self._col_fill is None:
            raise ValueError("Not fitted yet. Call fit() first.")

        arr = performance.values.astype(float)
        arr_out = self._impute(arr)

        if self._pca is not None:
            arr_out = self._pca.transform(arr_out)
            columns = [f"perf_pca_{i}" for i in range(arr_out.shape[1])]
        else:
            columns = [f"perf_{c}" for c in performance.columns]

        return pd.DataFrame(arr_out, index=performance.index, columns=columns)

    def fit_transform(self, performance: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(performance).transform(performance)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _impute(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        nan_rows, nan_cols = np.where(np.isnan(out))
        out[nan_rows, nan_cols] = self._col_fill[nan_cols]
        return out


def apply_metafeature_transforms(
    flow_metadata: pd.DataFrame,
    task_metadata: pd.DataFrame,
    flow_transforms: list[MetafeatureTransform],
    task_transforms: list[MetafeatureTransform],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply transformation pipelines to flow and task metadata.

    Args:
        flow_metadata: Original flow metadata (n_flows, n_flow_features)
        task_metadata: Original task metadata (n_tasks, n_task_features)
        flow_transforms: List of transforms to apply to flow metadata
        task_transforms: List of transforms to apply to task metadata

    Returns:
        Tuple of (transformed_flow_metadata, transformed_task_metadata)
    """
    # Create composite transforms
    flow_composite = CompositeTransform(flow_transforms)
    task_composite = CompositeTransform(task_transforms)

    # Fit and transform
    transformed_flow = flow_composite.fit_transform(flow_metadata)
    transformed_task = task_composite.fit_transform(task_metadata)

    return transformed_flow, transformed_task


def apply_metafeature_transforms_from_config(
    full_dataset: MetaDataset, cfg: DictConfig
):

    # Instantiate flow transforms
    flow_transforms = []
    if cfg.get("flow", None) is not None:
        for transform_cfg in cfg.flow:
            flow_transforms.append(instantiate(transform_cfg))

    # Instantiate task transforms
    task_transforms = []
    if cfg.get("task", None) is not None:
        for transform_cfg in cfg.task:
            task_transforms.append(instantiate(transform_cfg))

    # Apply transformations
    if flow_transforms or task_transforms:
        transformed_flow, transformed_task = apply_metafeature_transforms(
            flow_metadata=full_dataset.flow_metadata,
            task_metadata=full_dataset.task_metadata,
            flow_transforms=flow_transforms,
            task_transforms=task_transforms,
        )

        # Update dataset with transformed metadata
        full_dataset.flow_metadata = transformed_flow
        full_dataset.task_metadata = transformed_task

    return full_dataset
