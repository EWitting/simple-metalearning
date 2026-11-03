"""XGBoost search space utilities for continuous-space Bayesian optimisation.

Wraps the TabArena XGBoost ConfigurationSpace (identical bounds to the discrete
benchmark) and provides a ConfigFeaturizer that encodes any Configuration into
the same numeric feature matrix as the discrete flow_metadata, guaranteeing
that meta-train Random Forests trained on discrete configs can predict on live
candidate configs without distribution shift.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.util import get_one_exchange_neighbourhood


def get_configspace() -> ConfigurationSpace:
    """Return the XGBoost ConfigurationSpace (same bounds as TabArena benchmark)."""
    from tabarena.models.xgboost.generate import generate_configs_xgboost  # noqa: F401

    # Import triggers the module-level ConfigurationSpace definition; we
    # rebuild it here so we own the object (no shared state / seed side-effects).
    from ConfigSpace import Categorical, Float, Integer

    cs = ConfigurationSpace(
        space=[
            Float("learning_rate", (5e-3, 1e-1), log=True),
            Integer("max_depth", (4, 10), log=True),
            Float("min_child_weight", (1e-3, 5.0), log=True),
            Float("subsample", (0.6, 1.0)),
            Float("colsample_bylevel", (0.6, 1.0)),
            Float("colsample_bynode", (0.6, 1.0)),
            Float("reg_alpha", (1e-4, 5.0)),
            Float("reg_lambda", (1e-4, 5.0)),
            Categorical("grow_policy", ["depthwise", "lossguide"]),
            Integer("max_cat_to_onehot", (8, 100), log=True),
            Integer("max_leaves", (8, 1024), log=True),
        ],
    )
    return cs


def sample(n: int, rng: np.random.Generator | None = None) -> list[Configuration]:
    """Sample *n* configurations uniformly from the XGBoost search space."""
    seed = int(rng.integers(0, 2**31)) if rng is not None else None
    cs = get_configspace()
    if seed is not None:
        cs.seed(seed)
    configs = cs.sample_configuration(n)
    if n == 1:
        configs = [configs]
    return list(configs)


def neighbors(
    config: Configuration,
    rng: np.random.Generator | None = None,
    n_steps: int = 1,
) -> Iterable[Configuration]:
    """Yield one-exchange neighbours of *config* (SMAC-style local search)."""
    seed = int(rng.integers(0, 2**31)) if rng is not None else 0
    return get_one_exchange_neighbourhood(
        config, seed=seed, num_neighbors=n_steps
    )


def config_to_xgb_hyperparameters(config: Configuration) -> dict:
    """Convert a Configuration to XGBoost hyperparameters dict for AutoGluon."""
    from tabarena.models.utils import convert_numpy_dtypes

    hp = dict(config)
    hp["enable_categorical"] = True
    return convert_numpy_dtypes(hp)


class ConfigFeaturizer:
    """Encode XGBoost Configurations into a numeric feature matrix.

    The output columns are identical to the discrete ``flow_metadata`` DataFrame
    produced by ``TabArenaDatasetFactory.flow_metadata_to_dataframe``:

        colsample_bylevel, colsample_bynode, enable_categorical,
        learning_rate, max_cat_to_onehot, max_depth, max_leaves,
        min_child_weight, reg_alpha, reg_lambda, subsample,
        model_type_XGB, grow_policy_depthwise, grow_policy_lossguide

    Callers must pass the reference ``columns`` list (from
    ``dataset.flow_metadata.columns``) at construction time so the featurizer
    stays aligned even if the dataset factory column order changes.
    """

    # Hyperparameter keys that are on a log scale in the ConfigSpace; we
    # do NOT apply any manual log transform here — the raw values are stored
    # in flow_metadata as-is, so we mirror that.
    _NUMERIC_KEYS = [
        "colsample_bylevel",
        "colsample_bynode",
        "enable_categorical",
        "learning_rate",
        "max_cat_to_onehot",
        "max_depth",
        "max_leaves",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "subsample",
    ]

    def __init__(self, columns: list[str]):
        """
        Args:
            columns: Ordered list of column names from
                ``dataset.flow_metadata.columns`` (the reference schema).
        """
        self.columns = list(columns)

    def transform(self, configs: list[Configuration] | Configuration) -> np.ndarray:
        """Encode *configs* → float array of shape ``(n, len(self.columns))``.

        Missing columns (e.g. model-type dummies for non-XGBoost flows) are
        filled with 0.  The ``model_type_XGB`` column is always 1 for any
        XGBoost config.
        """
        if isinstance(configs, Configuration):
            configs = [configs]

        rows = []
        for cfg in configs:
            hp = dict(cfg)
            row: dict[str, float] = {}

            # Numeric HP columns
            for key in self._NUMERIC_KEYS:
                if key == "enable_categorical":
                    row[key] = 1.0  # always True for XGBoost
                else:
                    row[key] = float(hp.get(key, np.nan))

            # model_type dummy (always XGB)
            row["model_type_XGB"] = 1.0

            # grow_policy one-hot
            gp = hp.get("grow_policy", "depthwise")
            row["grow_policy_depthwise"] = 1.0 if gp == "depthwise" else 0.0
            row["grow_policy_lossguide"] = 1.0 if gp == "lossguide" else 0.0

            rows.append(row)

        df = pd.DataFrame(rows)
        # Reindex to reference schema, filling any missing columns with 0
        df = df.reindex(columns=self.columns, fill_value=0.0)
        return df.values.astype(float)
