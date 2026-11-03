import copy
import logging
import sys
import time
from multiprocessing import Manager
from threading import Thread
from typing import Any, Callable

import numpy as np
from joblib import Parallel, delayed
from rich.progress import Progress
from sklearn.model_selection import KFold

from simple_metalearning.datasets.dataset import MetaDataset
from simple_metalearning.experiments.base import Experiment
from simple_metalearning.metrics.corr import safe_pearsonr, safe_spearmanr
from simple_metalearning.metrics.metric_store import MetricStore
from simple_metalearning.metrics.regret import compute_regret
from simple_metalearning.metrics.tabrepo import compute_improvability, compute_normalized_regret

log = logging.getLogger(__name__)


class BOSelection(Experiment):
    """Bayesian Optimization selection experiment with k-fold cross-validation."""

    def __init__(
        self,
        max_visible: int,
        n_folds: int = 5,
        n_folds_used: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        task_limit: int | None = None,
        exclude_invalid: bool = False,
        cold_start_steps: int = 0,
        parallel_backend: str = "loky",
    ):
        super().__init__()
        self.max_visible = max_visible
        self.n_folds = n_folds
        self.n_folds_used = n_folds_used
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.task_limit = task_limit
        self.exclude_invalid = exclude_invalid
        # Cold-start ablation: force the first ``cold_start_steps`` picks to be
        # random for *every* estimator, so the initialisation is identical and
        # AUC differences reflect the surrogate rather than a warm first pick
        # (some estimators make an informed first pick from the matrix while
        # others fall back to a constant).  0 = each estimator's native start.
        self.cold_start_steps = cold_start_steps
        # joblib backend: "loky" (process, default) or "threading" (shared
        # memory).  Some torch models (e.g. BoTorch MultiTaskGP) segfault in
        # loky's spawned workers; "threading" avoids that while keeping some
        # parallelism (torch releases the GIL for linear algebra).
        self.parallel_backend = parallel_backend

    def _run_task(
        self,
        fold_idx: int,
        task_idx: int,
        estimator: Any,
        dataset: MetaDataset,
        status_dict: dict,
        bar_idx: int,
    ) -> list[dict]:
        rng = np.random.default_rng(self.random_state + fold_idx * 10000 + task_idx)
        target_estimator = copy.deepcopy(estimator)
        task_logs = []

        # Get available ground truth for this task
        ground_truth = dataset.performance.iloc[:, task_idx].values
        available_mask = ~np.isnan(ground_truth)

        # Optionally restrict to genuinely-evaluated flows (exclude imputed dummies).
        # When exclude_invalid=True, the BO loop only considers flows that actually
        # ran on this task; imputed dummy-baseline entries are excluded from selection
        # and from regret reference computations.
        if self.exclude_invalid and dataset.valid_mask is not None:
            valid = dataset.valid_mask.iloc[:, task_idx].values.astype(bool)
            available_mask = available_mask & valid

        available_indices = np.where(available_mask)[0]

        available_gt = ground_truth[available_mask]
        optimal_overall = np.max(available_gt)

        # Test performance column (None when val_test_split was not used).
        test_col: np.ndarray | None = None
        test_available_gt: np.ndarray | None = None
        test_optimal: float | None = None
        if dataset.test_performance is not None:
            test_col = dataset.test_performance.iloc[:, task_idx].values
            test_avail_mask = ~np.isnan(test_col)
            test_available_gt = test_col[test_avail_mask]
            if len(test_available_gt) > 0:
                test_optimal = float(np.max(test_available_gt))
            else:
                test_col = None  # no test data for this task

        # Per-flow evaluation cost for this task (train + infer seconds).
        # None when the dataset has no cost matrix.
        task_cost: np.ndarray | None = None
        if dataset.cost is not None:
            task_cost = dataset.cost.iloc[:, task_idx].values

        revealed_indices = []
        revealed_performances = []
        cumulative_cost_s: float = 0.0  # total wall-clock cost spent so far

        # Iteratively suggest and reveal
        for num_visible in range(self.max_visible + 1):
            # 1. Obtain predictions and compute metrics for current state
            predictions = target_estimator.predict(task_idx)

            remaining_available = np.setdiff1d(available_indices, revealed_indices)

            if revealed_performances:
                best_so_far = float(np.max(revealed_performances))
                regret = optimal_overall - best_so_far
            else:
                best_so_far = float("nan")
                regret = float("nan")  # undefined before first evaluation
            normalized_regret = compute_normalized_regret(
                available_gt, revealed_performances
            )
            improvability = compute_improvability(available_gt, revealed_performances)

            # Test metrics: select best val-revealed flow, evaluate on test performance.
            test_regret = float("nan")
            test_normalized_regret = float("nan")
            test_improvability = float("nan")
            if test_col is not None and revealed_performances:
                best_val_flow_idx = revealed_indices[int(np.argmax(revealed_performances))]
                test_perf_of_best = float(test_col[best_val_flow_idx])
                test_regret = test_optimal - test_perf_of_best
                test_normalized_regret = compute_normalized_regret(
                    test_available_gt, [test_perf_of_best]
                )
                test_improvability = compute_improvability(
                    test_available_gt, [test_perf_of_best]
                )

            # 2. Suggest next flow to reveal
            suggestions = target_estimator.suggest(task_idx, predictions)

            # Mask already revealed indices and indices with NaN ground truth
            mask = np.ones(len(suggestions), dtype=bool)
            mask[revealed_indices] = False
            mask[~available_mask] = False

            internal_regret = None
            next_flow_idx = None
            if np.any(mask):
                if num_visible < self.cold_start_steps:
                    # Cold-start ablation: ignore the surrogate and pick at random
                    # among the still-available flows (identical init for all).
                    next_flow_idx = rng.choice(np.where(mask)[0])
                else:
                    masked_suggestions = np.where(mask, suggestions, -np.inf)
                    max_suggestion = np.max(masked_suggestions)
                    best_suggestion_indices = np.where(
                        masked_suggestions == max_suggestion
                    )[0]

                    next_flow_idx = rng.choice(best_suggestion_indices)

                    # Internal Regret: difference between best predicted and suggested flow
                    masked_predictions = np.where(mask, predictions, -np.inf)
                    max_predicted_value = np.max(masked_predictions)
                    suggested_predicted_value = predictions[next_flow_idx]
                    internal_regret = max_predicted_value - suggested_predicted_value

            log_entry = {
                "fold_idx": fold_idx,
                "task_idx": task_idx,
                "visible_num": num_visible,
                # cumulative_cost_s is the total evaluation time spent *before*
                # this iteration (i.e. the cost of the first num_visible flows).
                # It is logged as both a metric and an alternative x-axis so
                # that curves can be plotted against wall-clock budget as well
                # as iteration count.
                "cumulative_cost_s": cumulative_cost_s,
                "regret": regret,
                "normalized_regret": normalized_regret,
                "improvability": improvability,
            }
            if test_col is not None:
                log_entry["test_regret"] = test_regret
                log_entry["test_normalized_regret"] = test_normalized_regret
                log_entry["test_improvability"] = test_improvability
            if internal_regret is not None:
                log_entry["internal_regret"] = internal_regret
            if len(remaining_available) > 0:
                test_predictions = predictions[remaining_available]
                test_ground_truth = ground_truth[remaining_available]
                pearson = safe_pearsonr(test_ground_truth, test_predictions)
                spearman = safe_spearmanr(test_ground_truth, test_predictions)
                if pearson is not None:
                    log_entry["pearson_corr"] = pearson
                if spearman is not None:
                    log_entry["spearman_corr"] = spearman
            task_logs.append(log_entry)

            if num_visible == self.max_visible:
                status_dict[bar_idx] = num_visible + 1
                break

            if not np.any(mask):
                status_dict[bar_idx] = self.max_visible + 1
                break

            # 3. Reveal and fit
            next_performance = ground_truth[next_flow_idx]
            target_estimator.partial_fit([next_flow_idx], task_idx, [next_performance])
            revealed_indices.append(next_flow_idx)
            revealed_performances.append(next_performance)

            # Accumulate cost of the newly revealed flow.
            if task_cost is not None:
                flow_cost = task_cost[next_flow_idx]
                if np.isfinite(flow_cost) and flow_cost >= 0:
                    cumulative_cost_s += float(flow_cost)

            status_dict[bar_idx] = num_visible + 1

        return task_logs

    def run(self, dataset: MetaDataset, estimator_factory: Callable):
        self.dataset = dataset
        self.estimator_factory = estimator_factory

        num_tasks = self.dataset.performance.shape[1]

        log.info(f"Number of flows: {self.dataset.performance.shape[0]}")
        log.info(f"Number of tasks: {num_tasks}")
        log.info(f"Cross-validation: {self.n_folds_used}/{self.n_folds} folds")

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(np.arange(num_tasks)))[: self.n_folds_used]

        # Build flat list of (fold_idx, task_idx, estimator) across all used folds
        tasks_to_run: list[tuple[int, int, Any]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            if self.task_limit is not None and len(train_idx) > self.task_limit:
                rng_limit = np.random.default_rng(self.random_state + fold_idx)
                train_idx = np.sort(
                    rng_limit.choice(train_idx, size=self.task_limit, replace=False)
                )
            log.info(
                f"Fold {fold_idx + 1}: train={len(train_idx)}, test={len(test_idx)}"
            )
            masked_dataset = copy.deepcopy(self.dataset)
            # Mask all tasks except the selected training subset
            all_task_idx = np.arange(self.dataset.performance.shape[1])
            non_train_idx = np.setdiff1d(all_task_idx, train_idx)
            masked_dataset.performance.iloc[:, non_train_idx] = np.nan
            estimator = estimator_factory(masked_dataset)
            estimator.fit()
            for task_idx in test_idx:
                tasks_to_run.append((fold_idx, task_idx, estimator))

        metric_store = MetricStore()

        results = self._run_parallel(tasks_to_run, len(folds))

        for task_logs in results:
            for log_entry in task_logs:
                metric_store.log(log_entry)

        # Primary x-axis: iteration count (discrete, equal-length for all tasks).
        # Exclude cumulative_cost_s here — it is a secondary x-axis, not a
        # metric, and averaging it by visible_num would produce a meaningless
        # (and non-monotonic) curve in the cost-time chart.
        metric_store.wandb_plot(
            x_axis="visible_num",
            exclude_cols=["cumulative_cost_s"],
        )

        # Secondary x-axis: cumulative evaluation cost in seconds.
        # Uses forward-fill interpolation: for each (fold, task) series, the
        # metric value at any cost point is the last observed value at or
        # before that cost — i.e. "best result achievable within budget X".
        # This avoids binning artefacts and produces a monotonically increasing
        # x-axis.  Skips automatically when cost is all-zero (no cost matrix).
        metric_store.wandb_plot_interpolated(
            x_axis="cumulative_cost_s",
            exclude_cols=["visible_num"],
        )

        metric_store.log_artifact()

    def _run_parallel(
        self,
        tasks_to_run: list[tuple[int, int, Any]],
        n_folds_used: int,
    ) -> list:
        """Run all (fold, task) combinations in parallel with progress bars."""
        with Manager() as manager:
            status_dict = manager.dict()
            for i in range(len(tasks_to_run)):
                status_dict[i] = 0

            # Disable the live progress bar when stdout is not a real terminal
            # (e.g. SLURM .out / wandb console capture). Otherwise rich rewrites
            # the bar every 0.1 s, exploding redirected log files.
            with Progress(disable=not sys.stdout.isatty()) as progress:
                bars = [
                    progress.add_task(
                        f"Fold {fold_idx + 1} Task {i + 1}/{len(tasks_to_run)}",
                        total=self.max_visible + 1,
                    )
                    for i, (fold_idx, task_idx, _) in enumerate(tasks_to_run)
                ]

                results = []
                worker_error: list[BaseException] = []

                def run_parallel_inner():
                    nonlocal results
                    try:
                        results = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(
                            delayed(self._run_task)(
                                fold_idx, task_idx, estimator, self.dataset, status_dict, i
                            )
                            for i, (fold_idx, task_idx, estimator) in enumerate(
                                tasks_to_run
                            )
                        )
                    except BaseException as exc:  # surface, don't swallow
                        worker_error.append(exc)

                thread = Thread(target=run_parallel_inner)
                thread.start()

                def _refresh_bars():
                    # The Manager proxy can raise (EOFError/BrokenPipe) if a
                    # worker died; don't let the progress bar mask the real
                    # error — just stop updating and let thread.join surface it.
                    try:
                        for i, bar_id in enumerate(bars):
                            progress.update(bar_id, completed=status_dict[i])
                    except (EOFError, BrokenPipeError, ConnectionError):
                        pass

                while thread.is_alive():
                    _refresh_bars()
                    time.sleep(0.1)

                _refresh_bars()

                thread.join()
                # Re-raise any worker exception in the main thread; otherwise a
                # single failed (fold, task) would silently yield empty results
                # (the loop above only reassigns ``results`` on success).
                if worker_error:
                    raise worker_error[0]
                return results
