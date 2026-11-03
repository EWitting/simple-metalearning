"""Live XGBoost evaluation objective wrapping TabArena.

Evaluates a single XGBoost Configuration on a given OpenML task across
specified val/test *split indices* and returns normalised performance values on
the same scale as the precomputed TabArena metric matrix.

Split-index convention
----------------------
TabArena stores results per OpenML repeated-CV split in a ``fold`` column whose
integer value is the flattened ``split_idx``:

    split_idx = n_folds * n_samples * repeat + n_samples * fold + sample

So ``val_split_indices=[0]`` means split_idx 0 = (repeat 0, fold 0), and the
matrix's per-task value is the mean ``metric_error`` over the split indices that
fall in that task's val (or test) half — see ``_split_folds_per_dataset`` in the
dataset factory.  This module converts each flat split index back to
``(repeat, fold, sample)`` via the task wrapper so that a live evaluation is
performed on exactly the same data split as the matrix cell it is compared to.

Normalisation note
------------------
The discrete performance matrix stores ``-metric_error / dummy_baseline``
(negated so higher is better, matching ``flip_sign=True``).  ``_normalize``
mirrors that sign so continuous and discrete values are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ConfigSpace import Configuration


@dataclass
class ObjResult:
    """Result of evaluating one config on one task."""

    val_perf: float                  # normalised val performance (matrix scale)
    test_perf: Optional[float]       # normalised test performance (None if no test splits)
    val_metric_error: float          # raw val metric_error (for logging)
    test_metric_error: Optional[float]
    time_train_s: float              # summed training time across val splits
    time_infer_s: float              # summed inference time across val splits


class TabArenaObjective:
    """Evaluate XGBoost configurations live on an OpenML task.

    Constructs one ``AGModelBagExperiment`` per evaluated split and averages
    ``metric_error`` over the requested val / test split indices.

    Args:
        task_name: Human-readable dataset name (matches ``MetaDataset.task_names``).
        tid: OpenML task ID.
        dummy_baseline: Dummy model's ``metric_error`` for this task.
        val_split_indices: Flat split indices (parquet ``fold`` numbering) to
            average for the val score.
        test_split_indices: Flat split indices to average for the test score.
            Pass an empty list to skip test evaluation.
        num_bag_folds: Inner bag folds for each ``AGModelBagExperiment`` run
            (8 matches the precomputed matrix's ``_BAG_L1`` models).
        time_limit: Per-split time limit in seconds for XGBoost training.
    """

    def __init__(
        self,
        task_name: str,
        tid: int,
        dummy_baseline: float,
        val_split_indices: list[int],
        test_split_indices: list[int],
        num_bag_folds: int = 8,
        time_limit: float = 3600.0,
    ):
        self.task_name = task_name
        self.tid = tid
        self.dummy_baseline = dummy_baseline
        self.val_split_indices = list(val_split_indices)
        self.test_split_indices = list(test_split_indices)
        self.num_bag_folds = num_bag_folds
        self.time_limit = time_limit

        self._task = None  # lazy loaded

    def _get_task(self):
        if self._task is None:
            from tabarena.benchmark.task.openml import OpenMLTaskWrapper
            self._task = OpenMLTaskWrapper.from_task_id(task_id=self.tid)
        return self._task

    def _run_split(self, hp: dict, split_idx: int, name: str) -> dict:
        """Run one bagged XGBoost on the (repeat, fold, sample) for *split_idx*."""
        from tabarena.benchmark.experiment.experiment_constructor import AGModelBagExperiment
        from autogluon.tabular.models import XGBoostModel

        task = self._get_task()
        repeat, fold, sample = task.split_vals_from_split_idx(split_idx)

        exp = AGModelBagExperiment(
            name=name,
            model_cls=XGBoostModel,
            model_hyperparameters=hp,
            num_bag_folds=self.num_bag_folds,
            num_bag_sets=1,
            time_limit=self.time_limit,
            experiment_kwargs={"compute_simulation_artifacts": False},
        )
        return exp.run(
            task=task,
            fold=fold,
            repeat=repeat,
            sample=sample,
            task_name=self.task_name,
            debug_mode=True,
            ignore_cache=True,
        )

    def _normalize(self, metric_error: float) -> float:
        """Convert raw metric_error to matrix scale (-error / dummy_baseline)."""
        if self.dummy_baseline > 0:
            return -metric_error / self.dummy_baseline
        return -metric_error

    def evaluate(self, config: Configuration, config_name: str = "cont_r1") -> ObjResult:
        """Evaluate *config* across val and test split indices."""
        from simple_metalearning.search_spaces.xgboost_space import config_to_xgb_hyperparameters

        hp = config_to_xgb_hyperparameters(config)
        # Force SEQUENTIAL bag-fitting (no Ray). AutoGluon's default parallel
        # strategy spins up a Ray cluster + dashboard, which fails on restricted
        # cluster nodes (dashboard MetricsHead timeout) and crashes the worker.
        # Sequential is robust everywhere and avoids the heavy Ray spam.
        hp = dict(hp)
        hp.setdefault("ag_args_ensemble", {})
        hp["ag_args_ensemble"]["fold_fitting_strategy"] = "sequential_local"

        val_errors = []
        total_train_s = 0.0
        total_infer_s = 0.0
        for si in self.val_split_indices:
            result = self._run_split(hp, si, name=f"{config_name}_val_s{si}")
            val_errors.append(result["metric_error"])
            total_train_s += result.get("time_train_s", 0.0)
            total_infer_s += result.get("time_infer_s", 0.0)
        val_me = float(sum(val_errors) / len(val_errors)) if val_errors else float("nan")
        val_perf = self._normalize(val_me)

        test_me = None
        test_perf = None
        if self.test_split_indices:
            test_errors = []
            for si in self.test_split_indices:
                result = self._run_split(hp, si, name=f"{config_name}_test_s{si}")
                test_errors.append(result["metric_error"])
            test_me = float(sum(test_errors) / len(test_errors))
            test_perf = self._normalize(test_me)

        return ObjResult(
            val_perf=val_perf,
            test_perf=test_perf,
            val_metric_error=val_me,
            test_metric_error=test_me,
            time_train_s=total_train_s,
            time_infer_s=total_infer_s,
        )
