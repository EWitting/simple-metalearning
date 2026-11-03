"""Continuous-space Bayesian Optimisation experiment.

Same k-fold cross-validation structure as ``BOSelection``, but instead of
revealing precomputed matrix cells, each BO iteration actually trains an
XGBoost model live on the target OpenML task.

Performance is normalised to the same scale as the discrete matrix
(``metric_error / dummy_baseline``) so regret curves can be directly
overlaid on discrete-experiment plots.

Important: the regret references (``optimal_overall``, ``median``,
``test_optimal``) are read from the *discrete* performance matrix.  This is
the ``np.max``-as-optimal convention used throughout the existing codebase
(see normalisation note in ``tabarena_objective.py``).
"""

from __future__ import annotations

import copy
import logging
import time
from multiprocessing import Manager
from threading import Thread
from typing import Any, Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from rich.progress import Progress

from simple_metalearning.datasets.dataset import MetaDataset
from simple_metalearning.experiments.bo_selection import BOSelection
from simple_metalearning.metrics.metric_store import MetricStore
from simple_metalearning.metrics.tabrepo import compute_improvability, compute_normalized_regret

log = logging.getLogger(__name__)


class ContinuousBOSelection(BOSelection):
    """Bayesian Optimisation in the continuous XGBoost configuration space.

    Subclasses :class:`~simple_metalearning.experiments.bo_selection.BOSelection` and
    inherits the fold-masking, parallel execution, and MetricStore logging
    scaffold.  Overrides :meth:`_run_task` with a live-evaluation BO loop.

    Args:
        max_visible: Maximum BO iterations per test task.
        n_folds: Total CV folds.
        n_folds_used: How many folds to actually run.
        random_state: RNG seed.
        n_jobs: Parallel workers (default 1; live AutoGluon training is heavy).
        task_limit: Limit meta-train tasks per fold (scalability study).
        n_init: Random initialisation iterations before surrogate is used.
        fold_mode: ``"faithful"`` (all val/test splits), ``"subsample"`` (first
            ``n_val_folds``/``n_test_folds`` of each half), or ``"manual"``.
        n_val_folds: Number of val splits to use in subsample mode (None = all).
        n_test_folds: Number of test splits to use in subsample mode (None = all).
        val_split_indices: Explicit flat split indices for ``manual`` mode.
        test_split_indices: Explicit flat split indices for ``manual`` mode.
        num_bag_folds: Inner bag folds for each XGBoost training run.
        time_limit: Per-fold XGBoost time limit in seconds.
        acq_optimizer_type: ``"local_and_random"`` or ``"random"``.
        n_random: Random candidates per acquisition optimisation step.
        n_local_starts: Local-search chains (``local_and_random`` only).
        test_task_limit: Cap number of test tasks per fold (for smoke tests).
    """

    def __init__(
        self,
        max_visible: int = 30,
        n_folds: int = 5,
        n_folds_used: int = 5,
        random_state: int = 42,
        n_jobs: int = 1,
        task_limit: int | None = None,
        test_task_limit: int | None = None,
        n_init: int = 5,
        fold_mode: str = "subsample",
        n_val_folds: int | None = 1,
        n_test_folds: int | None = 1,
        val_split_indices: list[int] | None = None,
        test_split_indices: list[int] | None = None,
        num_bag_folds: int = 8,
        time_limit: float = 3600.0,
        acq_optimizer_type: str = "local_and_random",
        n_random: int = 1000,
        n_local_starts: int = 10,
        single_fold: int | None = None,
        single_task: str | None = None,
        raise_on_task_failure: bool = False,
    ):
        super().__init__(
            max_visible=max_visible,
            n_folds=n_folds,
            n_folds_used=n_folds_used,
            random_state=random_state,
            n_jobs=n_jobs,
            task_limit=task_limit,
        )
        self.test_task_limit = test_task_limit
        self.n_init = n_init
        # Fold selection.  The matrix value for a task is the mean metric_error
        # over the split indices in that task's val (or test) half — see
        # ``_split_folds_per_dataset`` in the dataset factory.
        #   fold_mode="faithful"  → use ALL val / all test splits (exact match,
        #                           expensive: 8 bags × ~4-15 splits per iter).
        #   fold_mode="subsample" → use the first ``n_val_folds`` / ``n_test_folds``
        #                           splits of each half (noisy but unbiased).
        #   fold_mode="manual"    → use the explicit ``val_split_indices`` /
        #                           ``test_split_indices`` for every task.
        self.fold_mode = fold_mode
        self.n_val_folds = n_val_folds
        self.n_test_folds = n_test_folds
        self.val_split_indices = val_split_indices
        self.test_split_indices = test_split_indices
        self.num_bag_folds = num_bag_folds
        self.time_limit = time_limit
        self.acq_optimizer_type = acq_optimizer_type
        self.n_random = n_random
        self.n_local_starts = n_local_starts
        # Cluster sharding: when set, run exactly one (fold, task) — used by the
        # SLURM array launcher so one array index = one independent job.
        # ``single_task`` is a task *name* (matches MetaDataset.task_names).
        self.single_fold = single_fold
        self.single_task = single_task
        # If True, a per-task exception aborts the whole run (use locally to
        # surface bugs).  If False (cluster default), the task is recorded as a
        # failure and the run continues — but the failure is logged loudly with
        # a full traceback and reported in the run summary, never silent.
        self.raise_on_task_failure = raise_on_task_failure

        # Populated by run()
        self._task_name_to_tid: dict[str, int] = {}
        self._dummy_baseline: dict[str, float] = {}
        self._task_splits: dict[str, tuple[list[int], list[int]]] = {}

    @staticmethod
    def _n_repeats_for(n_instances: int) -> int:
        """TabArena's repeat count per dataset size.

        Delegates to tabarena's own ``_get_n_repeats`` so this never drifts from
        how the precomputed matrix was generated.
        """
        from tabarena.nips2025_utils.fetch_metadata import _get_n_repeats
        return _get_n_repeats(n_instances)

    def _derive_split_indices(self, tid: int) -> tuple[list[int], list[int]]:
        """Return (val_split_indices, test_split_indices) for a task.

        Reproduces the matrix's val/test partition: the flat split indices
        ``0 .. T-1`` (T = n_repeats_run × n_folds) are split in half — first
        half → val, second half → test — exactly as ``_split_folds_per_dataset``
        does on the sorted, contiguous fold values present in the parquet.
        """
        if self.fold_mode == "manual":
            return (
                list(self.val_split_indices or [0]),
                list(self.test_split_indices or []),
            )

        from tabarena.benchmark.task.openml import OpenMLTaskWrapper

        task = OpenMLTaskWrapper.from_task_id(task_id=tid)
        n_instances = len(task.X)
        _, n_folds, n_samples = task.get_split_dimensions()
        total = self._n_repeats_for(n_instances) * n_folds * n_samples
        all_idx = list(range(total))
        split = total // 2
        val_all, test_all = all_idx[:split], all_idx[split:]

        if self.fold_mode == "faithful":
            return val_all, test_all

        # subsample: first n_val_folds of val half, first n_test_folds of test half
        nv = self.n_val_folds if self.n_val_folds is not None else len(val_all)
        nt = self.n_test_folds if self.n_test_folds is not None else len(test_all)
        return val_all[:nv], test_all[:nt]

    # ------------------------------------------------------------------
    # run() override: pre-compute task→tid mapping and dummy baselines
    # ------------------------------------------------------------------

    def run(self, dataset: MetaDataset, estimator_factory: Callable):
        self.dataset = dataset
        self.estimator_factory = estimator_factory

        # Resolve task→tid from tabarena metadata
        self._resolve_task_metadata(dataset)

        # Delegate the rest to the inherited run() — it calls _run_task via
        # _run_parallel after building the fold splits.
        num_tasks = dataset.performance.shape[1]
        log.info(f"Number of flows (discrete reference): {dataset.performance.shape[0]}")
        log.info(f"Number of tasks: {num_tasks}")
        log.info(f"Cross-validation: {self.n_folds_used}/{self.n_folds} folds")

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(np.arange(num_tasks)))[: self.n_folds_used]

        # Optional task-name → index for single-task sharding.
        name_to_idx = {n: i for i, n in enumerate(dataset.task_names)}
        single_task_idx = (
            name_to_idx.get(self.single_task) if self.single_task is not None else None
        )

        tasks_to_run: list[tuple[int, int, Any]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # Cluster sharding: skip all folds except the requested one.
            if self.single_fold is not None and fold_idx != self.single_fold:
                continue
            if self.task_limit is not None and len(train_idx) > self.task_limit:
                rng_limit = np.random.default_rng(self.random_state + fold_idx)
                train_idx = np.sort(
                    rng_limit.choice(train_idx, size=self.task_limit, replace=False)
                )
            # Determine which test tasks this fold will run *before* the
            # expensive meta-training fit, so single-task shards skip folds
            # that don't contain the requested task.
            test_idx_use = test_idx
            if single_task_idx is not None:
                if single_task_idx not in set(test_idx):
                    continue  # requested task not a test task in this fold
                test_idx_use = np.array([single_task_idx])
            elif self.test_task_limit is not None and len(test_idx_use) > self.test_task_limit:
                rng_t = np.random.default_rng(self.random_state + fold_idx + 99999)
                test_idx_use = np.sort(
                    rng_t.choice(test_idx_use, size=self.test_task_limit, replace=False)
                )

            log.info(f"Fold {fold_idx + 1}: train={len(train_idx)}, test={len(test_idx_use)}")
            masked_dataset = copy.deepcopy(dataset)
            all_task_idx = np.arange(dataset.performance.shape[1])
            non_train_idx = np.setdiff1d(all_task_idx, train_idx)
            masked_dataset.performance.iloc[:, non_train_idx] = np.nan
            estimator = estimator_factory(masked_dataset)
            estimator.fit()
            for task_idx in test_idx_use:
                tasks_to_run.append((fold_idx, task_idx, estimator))

        # Pre-load OpenML tasks serially to populate the local filesystem cache
        # before workers run.  Workers then read from cache — no API calls needed.
        test_task_indices = sorted({task_idx for (_, task_idx, _) in tasks_to_run})
        self._preload_openml_tasks([dataset.task_names[i] for i in test_task_indices])

        metric_store = MetricStore()
        results = self._run_parallel(tasks_to_run, len(folds))

        n_failed = 0
        failed_tasks: list[tuple[int, int, str]] = []
        for task_logs in results:
            for log_entry in task_logs:
                if log_entry.get("status") == "failed":
                    n_failed += 1
                    failed_tasks.append(
                        (log_entry["fold_idx"], log_entry["task_idx"],
                         log_entry.get("failure_reason", "?"))
                    )
                    continue  # keep failure markers OUT of metric aggregation
                metric_store.log(log_entry)

        # Loud, unmissable summary of failures (offline cluster logs are rarely read).
        n_attempted = len(tasks_to_run)
        if n_failed:
            log.error(
                "%d/%d (fold, task) jobs FAILED and were excluded from metrics:",
                n_failed, n_attempted,
            )
            for f, t, reason in failed_tasks:
                name = dataset.task_names[t] if t < len(dataset.task_names) else t
                log.error("    fold %s task %s (%s): %s", f, t, name, reason)
        else:
            log.info("All %d (fold, task) jobs completed successfully.", n_attempted)

        metric_store.wandb_plot(x_axis="visible_num", exclude_cols=["cumulative_cost_s"])
        metric_store.wandb_plot_interpolated(x_axis="cumulative_cost_s", exclude_cols=["visible_num"])
        metric_store.log_artifact()

    def _resolve_task_metadata(self, dataset: MetaDataset) -> None:
        from simple_metalearning.datasets.tabarena import TabArenaDatasetFactory
        from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

        task_meta_df = load_task_metadata(paper=True)
        task_meta_map = {row["dataset"]: row.to_dict() for _, row in task_meta_df.iterrows()}

        for task_name in dataset.task_names:
            meta = task_meta_map.get(task_name, {})
            tid = meta.get("tid")
            if tid is not None:
                self._task_name_to_tid[task_name] = int(tid)
            else:
                log.warning(f"No tid found for task {task_name!r} — will be skipped.")

        # Dummy baseline: one value per task dataset name.  Construct a real
        # factory (not __new__ hackery) so this stays correct if the factory's
        # __init__ changes; the args match how the discrete matrix was built.
        factory = TabArenaDatasetFactory(
            methods="XGBoost", version="full", normalization="dummy",
        )
        dummy_series = factory._load_dummy_baseline()
        for task_name in dataset.task_names:
            self._dummy_baseline[task_name] = float(dummy_series.get(task_name, 1.0))

    def _preload_openml_tasks(self, task_names: list[str]) -> None:
        """Pre-download OpenML task data serially to populate local cache.

        Workers subsequently read from disk (no API calls).  Retries on
        transient server errors (504, etc.) with exponential backoff.
        """
        import time as _time
        from tabarena.benchmark.task.openml import OpenMLTaskWrapper

        for task_name in task_names:
            tid = self._task_name_to_tid.get(task_name)
            if tid is None:
                continue
            max_retries = 8
            for attempt in range(max_retries):
                try:
                    task_obj = OpenMLTaskWrapper.from_task_id(task_id=tid)
                    task_obj.get_split_dimensions()  # also caches the splits ARFF
                    # Derive + cache the val/test split indices for this task now
                    # (serially, so workers don't re-load the task just for dims).
                    self._task_splits[task_name] = self._derive_split_indices(tid)
                    log.info(
                        f"Pre-loaded OpenML task {task_name!r} (tid={tid}); "
                        f"val_splits={self._task_splits[task_name][0]} "
                        f"test_splits={self._task_splits[task_name][1]}"
                    )
                    break
                except Exception as exc:
                    delay = 2 ** attempt
                    if attempt == max_retries - 1:
                        log.error(
                            f"Failed to load task {task_name!r} (tid={tid}) "
                            f"after {max_retries} retries: {exc}"
                        )
                    else:
                        log.warning(
                            f"OpenML error for {task_name!r} (tid={tid}), "
                            f"retry {attempt+1}/{max_retries} in {delay}s: {exc}"
                        )
                        _time.sleep(delay)

    # ------------------------------------------------------------------
    # Per-task continuous BO loop
    # ------------------------------------------------------------------

    def _run_task(
        self,
        fold_idx: int,
        task_idx: int,
        estimator: Any,
        dataset: MetaDataset,
        status_dict: dict,
        bar_idx: int,
    ) -> list[dict]:
        """Run one (fold, task); never fail silently.

        A failure returns a single marker log entry (``status="failed"``) so the
        task is *visible* in the aggregate as failed rather than vanishing.  The
        full traceback is logged.  With ``raise_on_task_failure=True`` the
        exception propagates instead (use locally to debug).
        """
        import traceback

        task_name = dataset.task_names[task_idx]
        tid = self._task_name_to_tid.get(task_name)
        if tid is None:
            log.warning(f"Skipping task {task_name!r}: no OpenML tid.")
            status_dict[bar_idx] = self.max_visible + 1
            return [self._failure_entry(fold_idx, task_idx, "no OpenML tid")]

        try:
            return self._run_task_inner(
                fold_idx, task_idx, estimator, dataset, status_dict, bar_idx,
                task_name, tid,
            )
        except Exception as exc:
            if self.raise_on_task_failure:
                raise
            log.error(
                "Task %r (fold %d) FAILED — recorded as failure, run continues:\n%s",
                task_name, fold_idx, traceback.format_exc(),
            )
            status_dict[bar_idx] = self.max_visible + 1
            return [self._failure_entry(fold_idx, task_idx, repr(exc))]

    @staticmethod
    def _failure_entry(fold_idx: int, task_idx: int, reason: str) -> dict:
        """A marker log entry so failed tasks are countable, not silent."""
        return {
            "fold_idx": fold_idx,
            "task_idx": task_idx,
            "visible_num": 0,
            "status": "failed",
            "failure_reason": reason,
        }

    def _run_task_inner(
        self,
        fold_idx: int,
        task_idx: int,
        estimator: Any,
        dataset: MetaDataset,
        status_dict: dict,
        bar_idx: int,
        task_name: str,
        tid: int,
    ) -> list[dict]:
        from simple_metalearning.estimators.continuous.base import ContinuousEstimator
        from simple_metalearning.search_spaces.xgboost_space import ConfigFeaturizer, sample
        from simple_metalearning.search_spaces.tabarena_objective import TabArenaObjective
        from simple_metalearning.search_spaces.acquisition_optimizer import (
            LocalAndRandomSearchOptimizer,
            RandomSearchOptimizer,
        )

        rng = np.random.default_rng(self.random_state + fold_idx * 10000 + task_idx)
        dummy_baseline = self._dummy_baseline.get(task_name, 1.0)

        # Discrete references for regret normalisation (same as BOSelection)
        ground_truth = dataset.performance.iloc[:, task_idx].values
        available_mask = ~np.isnan(ground_truth)
        available_gt = ground_truth[available_mask]
        optimal_overall = float(np.max(available_gt))

        test_col = None
        test_available_gt = None
        test_optimal = None
        if dataset.test_performance is not None:
            tc = dataset.test_performance.iloc[:, task_idx].values
            test_avail = ~np.isnan(tc)
            test_available_gt = tc[test_avail]
            if len(test_available_gt) > 0:
                test_optional_val = float(np.max(test_available_gt))
                test_col = tc
                test_optimal = test_optional_val

        # Build live objective using the val/test split indices derived (and
        # cached) during preload; fall back to deriving here if missing.
        val_splits, test_splits = self._task_splits.get(
            task_name, (None, None)
        )
        if val_splits is None:
            val_splits, test_splits = self._derive_split_indices(tid)
        objective = TabArenaObjective(
            task_name=task_name,
            tid=tid,
            dummy_baseline=dummy_baseline,
            val_split_indices=val_splits,
            test_split_indices=test_splits,
            num_bag_folds=self.num_bag_folds,
            time_limit=self.time_limit,
        )

        # Featurizer aligned to discrete flow_metadata columns
        featurizer = ConfigFeaturizer(columns=list(dataset.flow_metadata.columns))

        # Acquisition optimizer
        if self.acq_optimizer_type == "local_and_random":
            optimizer = LocalAndRandomSearchOptimizer(
                n_random=self.n_random,
                n_local_starts=self.n_local_starts,
                featurizer=featurizer,
            )
        else:
            optimizer = RandomSearchOptimizer(
                n_random=self.n_random, featurizer=featurizer
            )

        # Deep-copy the surrogate (OOFRidge already fit; RF/GP start empty)
        est: ContinuousEstimator = copy.deepcopy(estimator)

        task_logs = []
        observed_val_perfs: list[float] = []
        observed_test_perfs: list[float] = []
        cumulative_cost_s = 0.0
        incumbent_config = None

        for num_visible in range(self.max_visible + 1):
            # --- Metrics for current state ---
            if observed_val_perfs:
                best_so_far = float(np.max(observed_val_perfs))
                regret = optimal_overall - best_so_far
            else:
                best_so_far = float("nan")
                regret = float("nan")

            normalized_regret = compute_normalized_regret(available_gt, observed_val_perfs)
            improvability = compute_improvability(available_gt, observed_val_perfs)

            test_regret = float("nan")
            test_normalized_regret = float("nan")
            test_improvability = float("nan")
            if test_col is not None and observed_val_perfs and observed_test_perfs:
                best_val_idx = int(np.argmax(observed_val_perfs))
                test_perf_of_best = observed_test_perfs[best_val_idx]
                test_regret = test_optimal - test_perf_of_best
                test_normalized_regret = compute_normalized_regret(
                    test_available_gt, [test_perf_of_best]
                )
                test_improvability = compute_improvability(
                    test_available_gt, [test_perf_of_best]
                )

            log_entry: dict = {
                "fold_idx": fold_idx,
                "task_idx": task_idx,
                "visible_num": num_visible,
                "cumulative_cost_s": cumulative_cost_s,
                "regret": regret,
                "normalized_regret": normalized_regret,
                "improvability": improvability,
            }
            if test_col is not None:
                log_entry["test_regret"] = test_regret
                log_entry["test_normalized_regret"] = test_normalized_regret
                log_entry["test_improvability"] = test_improvability
            task_logs.append(log_entry)

            if num_visible == self.max_visible:
                status_dict[bar_idx] = num_visible + 1
                break

            # --- Choose next config ---
            if num_visible < self.n_init or est.n_obs < 2:
                # Random initialisation
                config = sample(1, rng=rng)[0]
            else:
                config = optimizer.ask(
                    est, f_star=best_so_far, incumbent=incumbent_config, rng=rng
                )

            # --- Evaluate live ---
            t0 = time.time()
            config_name = f"cont_fold{fold_idx}_task{task_idx}_iter{num_visible}"
            result = objective.evaluate(config, config_name=config_name)
            t_eval = time.time() - t0

            # A non-finite score means the live evaluation silently produced
            # garbage (e.g. NaN metric_error).  Fail loudly rather than poison
            # the surrogate with NaN.
            if not np.isfinite(result.val_perf):
                raise RuntimeError(
                    f"Non-finite val_perf={result.val_perf} for task {task_name!r} "
                    f"at iter {num_visible} (config={dict(config)})"
                )

            log.debug(
                f"  fold={fold_idx} task={task_name} iter={num_visible} "
                f"val_perf={result.val_perf:.5f} t={t_eval:.1f}s"
            )

            # --- Update surrogate ---
            X_new = featurizer.transform([config])
            est.observe(X_new, result.val_perf)

            observed_val_perfs.append(result.val_perf)
            if result.test_perf is not None:
                observed_test_perfs.append(result.test_perf)

            if not np.isfinite(best_so_far) or result.val_perf >= best_so_far:
                incumbent_config = config

            cumulative_cost_s += result.time_train_s + result.time_infer_s
            status_dict[bar_idx] = num_visible + 1

        return task_logs
