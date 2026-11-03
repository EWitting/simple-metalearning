import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from tabarena.nips2025_utils.artifacts import (
    tabarena_method_metadata_collection,  # type: ignore[import]
)
from tabarena.nips2025_utils.fetch_metadata import (
    load_task_metadata,  # type: ignore[import]
)

from .dataset import MetaDataset, MetaDatasetFactory

# Methods that store raw config results in processed/configs.parquet with a 'framework' column
# instead of results/model_results.parquet with a 'method' column
_PROCESSED_CONFIGS_METHODS = {"LinearModel"}

# Methods to skip entirely (portfolio ensembles, not individual model configs)
_SKIP_METHODS = {"Portfolio-N200-4h"}

NormalizationMode = Literal["none", "rmse", "dummy"]

# Available problem types and their metrics (lower error is better in all cases):
#   "binary"     -> roc_auc error (= 1 - AUC),  30 datasets
#   "multiclass" -> log_loss,                     8 datasets
#   "regression" -> rmse,                        13 datasets
ProblemType = Literal["binary", "multiclass", "regression"]


_AG_ARGS_KEYS = {"ag_args", "ag_args_ensemble", "ag_args_fit"}
_META_KEYS = {"model_cls", "model_type", "ag_key", "name_prefix", "_method"}


class TabArenaDatasetFactory(MetaDatasetFactory):
    ALL_METHODS = [
        "CatBoost",
        "Dummy",
        "ExtraTrees",
        "LightGBM",
        "NeuralNetFastAI",
        "NeuralNetTorch",
        "RandomForest",
        "XGBoost",
        "ModernNCA_GPU",
        "TabICL_GPU",
        "TabM_GPU",
        "TabPFNv2_GPU",
        "ExplainableBM",
        "LimiX_GPU",
        "Mitra_GPU",
        "RealMLP_GPU",
        "xRFM_GPU",
        "BetaTabPFN_GPU",
        "TabFlex_GPU",
        "LinearModel",
        "KNeighbors",
        "Portfolio-N200-4h",
        "AutoGluon_v140_bq_4h8c",
        "AutoGluon_v140_eq_4h8c",
        "TabDPT_GPU",
        "RealTabPFN-v2.5",
        "LightGBM_aio_0808",
    ]

    BASE_METHODS = (
        [  # All of the above, excluding the Portfolio ensemble, AutoGluon, or dummy
            "CatBoost",
            "ExtraTrees",
            "LightGBM",
            "NeuralNetFastAI",
            "NeuralNetTorch",
            "RandomForest",
            "XGBoost",
            "ModernNCA_GPU",
            "TabICL_GPU",
            "TabM_GPU",
            "TabPFNv2_GPU",
            "ExplainableBM",
            "LimiX_GPU",
            "Mitra_GPU",
            "RealMLP_GPU",
            "xRFM_GPU",
            "BetaTabPFN_GPU",
            "TabFlex_GPU",
            "LinearModel",
            "KNeighbors",
            "TabDPT_GPU",
            "RealTabPFN-v2.5",
            "LightGBM_aio_0808",
        ]
    )

    def __init__(
        self,
        methods: Union[str, List[str]] = "all",
        version: Literal["full", "light"] = "full",
        problem_types: Optional[Union[ProblemType, List[ProblemType]]] = None,
        normalization: NormalizationMode = "none",
        cache_dir: str = "./data/processed",
        flip_sign: bool = False,
        val_test_split: bool = False,
        drop_default_configs: bool = False,
        fold_subset_n_val: Optional[int] = None,
        fold_subset_n_test: Optional[int] = None,
    ):
        if isinstance(methods, str):
            if methods.lower() == "all":
                self.methods = self.ALL_METHODS.copy()
            elif methods.lower() == "base":
                self.methods = self.BASE_METHODS.copy()
            else:
                self.methods = [methods]
        else:
            if any(isinstance(m, str) and m.lower() == "all" for m in methods):
                self.methods = self.ALL_METHODS.copy()
            else:
                self.methods = methods
        self.version = version
        if problem_types is None:
            self.problem_types = None
        elif isinstance(problem_types, str):
            self.problem_types = [problem_types]
        else:
            self.problem_types = list(problem_types)
        self.normalization = normalization
        self.flip_sign = flip_sign
        self.val_test_split = val_test_split
        self.drop_default_configs = drop_default_configs
        # Optional fold subsampling: when set (with val_test_split=True), keep
        # only the first N sorted folds of each dataset's val / test half before
        # averaging.  This produces a *noisier* matrix built from the same data
        # as the continuous-BO objective's ``fold_mode=subsample`` evaluation,
        # making the two directly comparable.  None = use all folds in each half.
        self.fold_subset_n_val = fold_subset_n_val
        self.fold_subset_n_test = fold_subset_n_test
        super().__init__(cache_dir=cache_dir)

    def hash(self) -> str:
        HASH_THRESHOLD = 5
        sorted_methods = sorted(self.methods)
        if len(sorted_methods) == len(self.ALL_METHODS):
            methods_str = "all"
        elif len(sorted_methods) > HASH_THRESHOLD:
            import hashlib

            methods_bytes = repr(sorted_methods).encode("utf-8")
            methods_hash = hashlib.sha256(methods_bytes).hexdigest()[:10]
            methods_str = f"hash_{methods_hash}"
        else:
            methods_str = "_".join(sorted_methods)
        parts = ["tabarena", self.version, self.normalization, methods_str]
        if self.problem_types is not None:
            parts.append("_".join(sorted(self.problem_types)))
        if self.flip_sign:
            parts.append("flip_sign_v2")  # v2: bug fixed — sign now applied correctly
        if self.val_test_split:
            parts.append("valsplit")
        if self.fold_subset_n_val is not None or self.fold_subset_n_test is not None:
            parts.append(f"fsub{self.fold_subset_n_val}-{self.fold_subset_n_test}")
        if self.drop_default_configs:
            parts.append("no_default")
        return "_".join(parts)

    def _load_task_metadata(self) -> Dict[str, Dict]:
        task_metadata_df = load_task_metadata(paper=True)
        return {row["dataset"]: row.to_dict() for _, row in task_metadata_df.iterrows()}

    def _load_raw_results(self, method: str, method_metadata) -> Optional[pd.DataFrame]:
        """
        Load the raw per-config results DataFrame for a method.

        For most methods: results/model_results.parquet (config col = 'method')
        For LinearModel: processed/configs.parquet (config col = 'framework')

        Returns None if the method should be skipped or loading fails.
        """
        if method in _SKIP_METHODS:
            print(f"Skipping {method} (ensemble/portfolio method)")
            return None

        if method in _PROCESSED_CONFIGS_METHODS:
            processed_path = Path(method_metadata.path_processed)
            configs_file = processed_path / "configs.parquet"
            if not configs_file.exists():
                if not method_metadata.path_processed_exists:
                    print(f"Downloading processed data for {method}...")
                    method_metadata.method_downloader().download_processed()
                if not configs_file.exists():
                    print(
                        f"Warning: configs.parquet not found for {method} at {configs_file}"
                    )
                    return None
            return pd.read_parquet(configs_file)
        else:
            results_path = Path(method_metadata.path_results)
            model_results_file = results_path / "model_results.parquet"
            if not model_results_file.exists():
                if not method_metadata.path_results_exists:
                    print(f"Downloading results data for {method}...")
                    method_metadata.method_downloader().download_results()
                if not model_results_file.exists():
                    print(
                        f"Warning: model_results.parquet not found for {method} at {model_results_file}"
                    )
                    return None
            return pd.read_parquet(model_results_file)

    def _load_method_data(self, method: str) -> Optional[Tuple]:
        """
        Load raw config results, hyperparameters, and config names for a method.

        Returns:
            Tuple of (df_results, configs_hyperparameters, configs) or None if loading fails.
        """
        print(f"Processing method: {method}")

        try:
            method_metadata = tabarena_method_metadata_collection.get_method_metadata(
                method=method
            )
        except Exception as e:
            print(f"Error getting metadata for method {method}: {e}")
            return None

        df_results = self._load_raw_results(method, method_metadata)
        if df_results is None:
            return None

        try:
            configs_hyperparameters = method_metadata.load_configs_hyperparameters(
                holdout=False, download="auto"
            )
        except (FileNotFoundError, ClientError) as e:
            print(f"Warning: Hyperparameters not available for method {method}: {e}")
            configs_hyperparameters = {}

        config_col = "framework" if "framework" in df_results.columns else "method"
        configs = df_results[config_col].unique()

        return df_results, configs_hyperparameters, configs

    @staticmethod
    def _split_folds_per_dataset(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split rows into validation and test halves based on the fold column.

        For each dataset, the unique fold values are sorted; the first half
        goes to validation and the second half (including any extra fold when
        the count is odd) goes to test.

        Returns:
            (val_df, test_df) — two DataFrames with the same columns as ``df``.
        """
        val_rows: List[pd.DataFrame] = []
        test_rows: List[pd.DataFrame] = []
        for _, group in df.groupby("dataset"):
            folds = sorted(group["fold"].unique())
            n = len(folds)
            split = n // 2
            val_folds = set(folds[:split])
            test_folds = set(folds[split:])
            val_rows.append(group[group["fold"].isin(val_folds)])
            test_rows.append(group[group["fold"].isin(test_folds)])
        return pd.concat(val_rows), pd.concat(test_rows)

    @staticmethod
    def _keep_first_n_folds(df: pd.DataFrame, n_keep: Optional[int]) -> pd.DataFrame:
        """Per dataset, keep only rows from the first ``n_keep`` sorted folds.

        Used to build a noisier matrix from a fold subset that matches the
        continuous-BO objective's ``fold_mode=subsample`` (which uses the first
        ``n_val``/``n_test`` splits of each half).  ``n_keep=None`` is a no-op.
        """
        if n_keep is None:
            return df
        rows: List[pd.DataFrame] = []
        for _, group in df.groupby("dataset"):
            keep = sorted(group["fold"].unique())[:n_keep]
            rows.append(group[group["fold"].isin(keep)])
        return pd.concat(rows)

    def _create_pivot_tables(
        self,
        df_results: pd.DataFrame,
        perf_col: str = "metric_error",
        cost_cols: list = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Create pivot tables (datasets x configs) for performance and cost.

        For 'light' version, only fold 0 is used. For 'full', all folds are averaged.

        Cost column
        -----------
        ``cost_cols`` lists raw time columns to *sum* into a single total-cost
        figure per (dataset, config) cell.  Defaults to
        ``["time_train_s", "time_infer_s"]`` so that cost reflects the full
        wall-clock time a user would wait: fit time plus one inference pass.
        Only columns that actually exist in ``df_results`` are included; if
        none exist, ``pivot_cost`` is ``None``.
        """
        if cost_cols is None:
            cost_cols = ["time_train_s", "time_infer_s"]

        config_col = "framework" if "framework" in df_results.columns else "method"

        if self.problem_types is not None and "problem_type" in df_results.columns:
            df_results = df_results[df_results["problem_type"].isin(self.problem_types)]
            if df_results.empty:
                return pd.DataFrame(), None

        if "fold" in df_results.columns:
            if self.version == "light":
                df_results = df_results[df_results["fold"] == 0]
                groupby_cols = ["dataset", config_col]
            else:
                groupby_cols = ["dataset", config_col, "fold"]
        else:
            groupby_cols = ["dataset", config_col]

        pivot_perf = (
            df_results.groupby(groupby_cols)[perf_col].mean().unstack(level=config_col)
        )
        pivot_perf = pivot_perf.replace([np.inf, -np.inf], np.nan)
        # NaN imputation happens later in _create using the Dummy baseline

        if "fold" in groupby_cols:
            pivot_perf = pivot_perf.groupby(level="dataset").mean()

        # Build total-cost pivot by summing available time columns.
        present_cost_cols = [c for c in cost_cols if c in df_results.columns]
        pivot_cost = None
        if present_cost_cols:
            # Sum all present time columns per row, then aggregate.
            df_results = df_results.copy()
            df_results["_total_cost_s"] = df_results[present_cost_cols].sum(axis=1)
            pivot_cost = (
                df_results.groupby(groupby_cols)["_total_cost_s"]
                .mean()
                .unstack(level=config_col)
            )
            if "fold" in groupby_cols:
                pivot_cost = pivot_cost.groupby(level="dataset").mean()

        return pivot_perf, pivot_cost

    def _load_dummy_baseline(self) -> pd.Series:
        """
        Load Dummy method results and return a Series of metric_error per dataset,
        averaged across folds (or fold 0 for light version).
        """
        print("Loading Dummy baseline for normalization...")
        try:
            dummy_metadata = tabarena_method_metadata_collection.get_method_metadata(
                "Dummy"
            )
        except Exception as e:
            raise RuntimeError(f"Could not load Dummy method metadata: {e}")

        df_dummy = self._load_raw_results("Dummy", dummy_metadata)
        if df_dummy is None:
            raise RuntimeError("Could not load Dummy results for normalization.")

        if "fold" in df_dummy.columns:
            if self.version == "light":
                df_dummy = df_dummy[df_dummy["fold"] == 0]

        if self.problem_types is not None and "problem_type" in df_dummy.columns:
            df_dummy = df_dummy[df_dummy["problem_type"].isin(self.problem_types)]

        dummy_baseline = df_dummy.groupby("dataset")["metric_error"].mean()
        return dummy_baseline

    def _apply_normalization(
        self,
        combined_perf: pd.DataFrame,
        task_metadata_map: Dict[str, Dict],
        dummy_baseline: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Apply normalization to the combined performance matrix (datasets x configs).

        "none":  no-op
        "rmse":  divide regression task rows by the mean RMSE across all configs for that dataset
        "dummy": divide all rows by the Dummy model's error for that dataset
        """
        if self.normalization == "none":
            perf = combined_perf
            if self.flip_sign:
                perf = perf * -1
            return perf

        if self.normalization == "rmse":
            import openml

            perf = combined_perf.copy()
            for dataset, meta in task_metadata_map.items():
                if (
                    meta.get("problem_type") != "regression"
                    or dataset not in perf.index
                ):
                    continue
                did = meta.get("did")
                if did is None:
                    print(
                        f"Warning: No OpenML did for regression dataset {dataset}, skipping RMSE normalization."
                    )
                    continue
                try:
                    oml_dataset = openml.datasets.get_dataset(
                        int(did), download_data=True
                    )
                    _, y, _, _ = oml_dataset.get_data(
                        target=oml_dataset.default_target_attribute
                    )
                    target_std = float(np.std(y))
                    if target_std > 0:
                        perf.loc[dataset] = perf.loc[dataset] / target_std
                    else:
                        print(
                            f"Warning: Target std is 0 for dataset {dataset}, skipping normalization."
                        )
                except Exception as e:
                    print(
                        f"Warning: Could not fetch OpenML dataset {did} for {dataset}: {e}"
                    )
            if self.flip_sign:
                perf = perf * -1
            return perf

        if self.normalization == "dummy":
            if dummy_baseline is None:
                dummy_baseline = self._load_dummy_baseline()
            perf = combined_perf.copy()
            for dataset in perf.index:
                if dataset in dummy_baseline.index:
                    baseline = dummy_baseline[dataset]
                    if baseline > 0:
                        perf.loc[dataset] = perf.loc[dataset] / baseline
                    else:
                        print(
                            f"Warning: Dummy baseline is 0 for dataset {dataset}, skipping normalization."
                        )
            if self.flip_sign:
                perf = perf * -1
            return perf

        raise ValueError(f"Unknown normalization mode: {self.normalization!r}")

    def _extract_flow_metadata(
        self, configs: np.ndarray, configs_hyperparameters: Dict, method: str
    ) -> Tuple[List, List]:
        flow_metadata = []
        flow_names = []

        for config in configs:
            hp = configs_hyperparameters.get(config, {})
            if isinstance(hp, dict):
                hp["_method"] = method
            flow_metadata.append(hp)
            flow_names.append(config)

        return flow_metadata, flow_names

    def _combine_dataframes(
        self, all_perf_dfs: List[pd.DataFrame], all_cost_dfs: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if not all_perf_dfs:
            raise ValueError("No data collected from any method.")

        combined_perf = pd.concat(all_perf_dfs, axis=1, join="outer")

        combined_cost = None
        if all_cost_dfs:
            combined_cost = pd.concat(all_cost_dfs, axis=1, join="outer")

        combined_perf = combined_perf.sort_index()
        if combined_cost is not None:
            combined_cost = combined_cost.sort_index()
            combined_cost = combined_cost.reindex(columns=combined_perf.columns)

        return combined_perf, combined_cost

    def _build_metadata_arrays(
        self,
        task_names: List[str],
        flow_names: List[str],
        task_metadata_map: Dict[str, Dict],
        all_flow_names: List[str],
        all_flow_metadata: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        ordered_task_metadata = [task_metadata_map.get(task, {}) for task in task_names]

        flow_meta_map = dict(zip(all_flow_names, all_flow_metadata))
        ordered_flow_metadata = [flow_meta_map.get(flow, {}) for flow in flow_names]

        return ordered_task_metadata, ordered_flow_metadata

    @staticmethod
    def flow_metadata_to_dataframe(
        flow_names: List[str], flow_metadata: List[Dict]
    ) -> pd.DataFrame:
        """
        Convert flow_metadata list to a DataFrame indexed by flow name.

        Columns:
        - model_type: the 'model_type' field from the metadata (e.g. 'CAT', 'GBM')
        - one column per hyperparameter key found across all configs, excluding
          ag_args / ag_args_ensemble / ag_args_fit and other metadata-only keys
          (_method, model_cls, ag_key, name_prefix).
        When a config does not have a hyperparameter, the cell is NaN.
        The row order matches the order of flow_names.
        """
        rows = []
        for name, meta in zip(flow_names, flow_metadata):
            if not isinstance(meta, dict):
                rows.append({"_flow_name": name, "model_type": None})
                continue
            row: Dict = {"_flow_name": name, "model_type": meta.get("model_type")}
            hp = meta.get("hyperparameters", {}) or {}
            for k, v in hp.items():
                if k not in _AG_ARGS_KEYS:
                    row[k] = v
            # also pick up any top-level keys that are not metadata or ag_args
            for k, v in meta.items():
                if (
                    k not in _META_KEYS
                    and k != "hyperparameters"
                    and k not in _AG_ARGS_KEYS
                ):
                    row.setdefault(k, v)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index("_flow_name")
        df.index.name = None

        # Unfold inference_config/PREPROCESS_TRANSFORMS: list of 1 dict → separate columns
        pt_col = "inference_config/PREPROCESS_TRANSFORMS"
        if pt_col in df.columns:
            expanded = (
                df[pt_col]
                .apply(
                    lambda v: (
                        v[0]
                        if isinstance(v, list)
                        and len(v) == 1
                        and isinstance(v[0], dict)
                        else {}
                    )
                )
                .apply(pd.Series)
                .add_prefix(f"{pt_col}/")
                .replace({None: pd.NA})
            )
            df = df.drop(columns=[pt_col])
            df = pd.concat([df, expanded], axis=1)

        # Classify object columns: booleans vs everything else
        bool_cols = []
        ohe_cols = []
        for col in df.columns:
            if df[col].dtype != object:
                continue
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            if non_null.apply(lambda x: isinstance(x, bool)).all():
                bool_cols.append(col)
            else:
                ohe_cols.append(col)

        # Booleans → 0/1 nullable int
        for col in bool_cols:
            df[col] = (
                df[col]
                .map(lambda x: int(x) if isinstance(x, bool) else x)
                .astype("Float64")
            )

        # Unhashable values (lists, dicts) → string repr so get_dummies can handle them
        for col in ohe_cols:
            df[col] = df[col].map(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )
        if ohe_cols:
            df = pd.get_dummies(df, columns=ohe_cols, dummy_na=False)

        return df

    def _create(self) -> MetaDataset:
        all_perf_dfs = []
        all_test_perf_dfs: List[pd.DataFrame] = []  # only used when val_test_split=True
        all_cost_dfs = []
        all_flow_metadata = []
        all_flow_names = []

        task_metadata_map = self._load_task_metadata()

        # Always load Dummy: used for NaN/inf imputation and optionally for normalization.
        # Load it even if not in the methods list.
        methods_to_load = list(self.methods)
        if "Dummy" not in methods_to_load:
            methods_to_load = ["Dummy"] + methods_to_load

        for method in methods_to_load:
            method_data = self._load_method_data(method)
            if method_data is None:
                continue

            df_results, configs_hyperparameters, configs = method_data

            if self.val_test_split and "fold" in df_results.columns:
                val_df, test_df = self._split_folds_per_dataset(df_results)
                # Optionally keep only the first N folds of each half so the
                # matrix matches the continuous objective's subsample folds.
                val_df = self._keep_first_n_folds(val_df, self.fold_subset_n_val)
                test_df = self._keep_first_n_folds(test_df, self.fold_subset_n_test)
                pivot_perf, pivot_cost = self._create_pivot_tables(val_df)
                pivot_test_perf, _ = self._create_pivot_tables(test_df)
            else:
                pivot_perf, pivot_cost = self._create_pivot_tables(df_results)
                pivot_test_perf = None

            if pivot_perf.empty:
                continue

            all_perf_dfs.append(pivot_perf)
            if pivot_test_perf is not None and not pivot_test_perf.empty:
                all_test_perf_dfs.append(pivot_test_perf)
            if pivot_cost is not None:
                all_cost_dfs.append(pivot_cost)

            flow_metadata, flow_names = self._extract_flow_metadata(
                configs, configs_hyperparameters, method
            )
            all_flow_metadata.extend(flow_metadata)
            all_flow_names.extend(flow_names)

        combined_perf, combined_cost = self._combine_dataframes(
            all_perf_dfs, all_cost_dfs
        )

        # Combine test performance if available
        combined_test_perf: Optional[pd.DataFrame] = None
        if all_test_perf_dfs:
            combined_test_perf, _ = self._combine_dataframes(all_test_perf_dfs, [])

        # Impute NaN cells (crashed/missing models) with the Dummy baseline score.
        # This penalizes failures at the level of a trivial baseline.
        # Extract from the already-loaded combined matrix to avoid a second disk read.
        dummy_cols = [c for c in combined_perf.columns if c.startswith("Dummy")]
        if dummy_cols:
            dummy_baseline = combined_perf[dummy_cols].mean(axis=1)
        else:
            dummy_baseline = self._load_dummy_baseline()

        # Record which cells are genuinely evaluated (not imputed) before filling.
        # Shape (n_tasks, n_flows) — transposed to (n_flows, n_tasks) when stored.
        # False means the cell will be filled with the dummy baseline (model not
        # applicable to the task, or crashed run).
        combined_valid_mask = ~combined_perf.isna()

        def _impute(df: pd.DataFrame) -> pd.DataFrame:
            for dataset in df.index:
                if (
                    dataset in dummy_baseline.index
                    and df.loc[dataset].isna().any()
                ):
                    nan_mask = df.loc[dataset].isna()
                    df.loc[dataset, nan_mask] = dummy_baseline[dataset]
            return df

        combined_perf = _impute(combined_perf)
        if combined_test_perf is not None:
            combined_test_perf = _impute(combined_test_perf)

        combined_perf = self._apply_normalization(
            combined_perf, task_metadata_map, dummy_baseline
        )
        if combined_test_perf is not None:
            combined_test_perf = self._apply_normalization(
                combined_test_perf, task_metadata_map, dummy_baseline
            )

        # Remove Dummy if it wasn't explicitly requested by the user
        if "Dummy" not in self.methods:
            combined_perf = combined_perf.drop(columns=dummy_cols)
            if combined_cost is not None:
                combined_cost = combined_cost.drop(columns=dummy_cols, errors="ignore")
            if combined_test_perf is not None:
                combined_test_perf = combined_test_perf.drop(
                    columns=dummy_cols, errors="ignore"
                )

            # Update all_flow_names and all_flow_metadata to stay in sync
            # This is important for _build_metadata_arrays and flow_metadata_to_dataframe
            keep = [not name.startswith("Dummy") for name in all_flow_names]
            all_flow_names = [name for name, m in zip(all_flow_names, keep) if m]
            all_flow_metadata = [meta for meta, m in zip(all_flow_metadata, keep) if m]

        if self.drop_default_configs:
            # A "default" config has no explicit hyperparameters beyond ag_args metadata.
            # Only drop defaults for methods that also have non-default (HPO) configs;
            # methods with only a single default config are kept as-is.
            def _is_default(meta) -> bool:
                if not isinstance(meta, dict):
                    return True
                hp = meta.get("hyperparameters", {}) or {}
                return not any(k not in _AG_ARGS_KEYS for k in hp)

            is_default = [_is_default(m) for m in all_flow_metadata]

            methods_with_hpo: set = set()
            for meta, default in zip(all_flow_metadata, is_default):
                if not default and isinstance(meta, dict):
                    methods_with_hpo.add(meta.get("_method"))

            drop_names = {
                name
                for name, meta, default in zip(all_flow_names, all_flow_metadata, is_default)
                if default
                and isinstance(meta, dict)
                and meta.get("_method") in methods_with_hpo
            }

            if drop_names:
                combined_perf = combined_perf.drop(columns=list(drop_names), errors="ignore")
                if combined_cost is not None:
                    combined_cost = combined_cost.drop(columns=list(drop_names), errors="ignore")
                if combined_test_perf is not None:
                    combined_test_perf = combined_test_perf.drop(
                        columns=list(drop_names), errors="ignore"
                    )
                keep = [name not in drop_names for name in all_flow_names]
                all_flow_names = [n for n, k in zip(all_flow_names, keep) if k]
                all_flow_metadata = [m for m, k in zip(all_flow_metadata, keep) if k]

        task_names = combined_perf.index.tolist()
        flow_names = combined_perf.columns.tolist()

        # Align test_performance and valid_mask to same row/column order as performance
        if combined_test_perf is not None:
            combined_test_perf = combined_test_perf.reindex(
                index=task_names, columns=flow_names
            )

        combined_valid_mask = combined_valid_mask.reindex(
            index=task_names, columns=flow_names
        ).fillna(False)

        ordered_task_metadata, ordered_flow_metadata = self._build_metadata_arrays(
            task_names, flow_names, task_metadata_map, all_flow_names, all_flow_metadata
        )

        return MetaDataset(
            dataset_name=self.hash(),
            performance=combined_perf.T,
            cost=combined_cost.T if combined_cost is not None else None,
            test_performance=combined_test_perf.T if combined_test_perf is not None else None,
            valid_mask=combined_valid_mask.T,
            flow_names=flow_names,
            task_names=task_names,
            flow_metadata=self.flow_metadata_to_dataframe(
                flow_names, ordered_flow_metadata
            ),
            # TODO: do this in a cleaner way
            task_metadata=pd.DataFrame(ordered_task_metadata)
                .dropna(axis=1)
                .select_dtypes(include=['number'])
                .drop(columns=['tid','did','source_data','n_folds','n_repeats'])
                .astype('float64'),
        )


if __name__ == "__main__":
    dataset = TabArenaDatasetFactory(
        methods="base",
        version="full",
        problem_types=None,
        normalization="rmse",
        cache_dir="data/processed",
    ).load()
    print(dataset.flow_metadata.dtypes.unique())
    print(dataset.flow_metadata.head())
