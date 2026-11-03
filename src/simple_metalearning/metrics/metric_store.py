from typing import Any

import numpy as np
import pandas as pd

import wandb


class MetricStore:
    """Collects raw per-step metrics and uploads aggregated curves to W&B.

    W&B does not natively average multiple runs of the same x-axis step, so
    this class accumulates all raw log entries, aggregates them in pandas, and
    then uploads the mean curves and area-under-curve summaries.

    Two plotting modes are available:

    ``wandb_plot(x_axis, exclude_cols)``
        Intended for **discrete** x-axes where every (fold, task) pair logs
        exactly the same set of x values (e.g. ``visible_num = 0, 1, …, K``).
        Groups by the exact x value and takes the mean.  Pass ``exclude_cols``
        to suppress columns that are being used as a secondary x-axis
        (e.g. ``cumulative_cost_s``) from appearing as spurious metrics in
        this chart.

    ``wandb_plot_interpolated(x_axis, group_cols)``
        Intended for **continuous** x-axes (e.g. ``cumulative_cost_s``) where
        each (fold, task) pair takes a different set of x values.

        Algorithm (same as the "union-then-forward-fill" approach):

        1. Collect the **union** of every x value logged across all
           (fold × task) series and sort it.
        2. For each individual series, **reindex** to that union of x values
           and **forward-fill** — the metric value at any x where this series
           has not yet logged a new observation stays at its last known value.
           This models "what is the best result we had achieved by time X?"
        3. **Average** the forward-filled values across all series at each x.
        4. Upload to W&B and compute AUC summaries.

        This produces a smooth, monotonically advancing x-axis without binning
        artefacts, and is the correct way to compare BO curves over a wall-
        clock budget.  AUC keys use ``area/{x_axis}/{metric}`` so they do not
        collide with keys from ``wandb_plot``.
    """

    def __init__(self):
        self.logs: list[dict[str, Any]] = []
        self.df: pd.DataFrame | None = None

    def log(self, metrics: dict[str, Any]):
        self.logs.append(metrics)

    def combine_logs(self) -> pd.DataFrame:
        self.df = pd.DataFrame(self.logs)

    def save(self, path: str):
        if self.df is None:
            self.combine_logs()
        self.df.to_csv(path)

    def wandb_plot(self, x_axis: str, exclude_cols: list[str] | None = None):
        """Upload mean metric curves aggregated over a discrete x-axis.

        Args:
            x_axis: Column name to use as the x-axis (e.g. ``"visible_num"``).
                Every fold/task must have logged the same set of values for
                this column.
            exclude_cols: Additional columns to drop before aggregating.
                Use this to suppress columns that serve as a secondary x-axis
                (e.g. ``["cumulative_cost_s"]``) so they do not appear as
                spurious flat/non-monotonic metric curves in this chart.
        """
        if self.df is None:
            self.combine_logs()

        drop = list(exclude_cols or [])
        df = self.df.drop(columns=[c for c in drop if c in self.df.columns])

        if df.empty or x_axis not in df.columns:
            import logging
            logging.getLogger(__name__).warning(
                "wandb_plot: no data logged for x_axis=%r — skipping upload. "
                "This usually means all worker tasks failed.", x_axis
            )
            return

        # aggregate by x_axis
        df_aggregated = df.groupby(x_axis).mean(numeric_only=True)
        df_aggregated = df_aggregated.sort_index()

        # log curves step by step
        for index, row in df_aggregated.iterrows():
            wandb.log({
                x_axis: index,
                **row.to_dict()
            })

        self._log_auc(df_aggregated, x_axis)

    def wandb_plot_interpolated(
        self,
        x_axis: str,
        group_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
    ):
        """Upload mean metric curves for a continuous x-axis via forward-fill.

        This is the correct aggregation strategy when each (fold, task) series
        has a different set of x values (e.g. cumulative evaluation cost),
        which makes a plain ``groupby`` useless.

        The algorithm is:

        1. Group the raw logs into individual series identified by
           ``group_cols`` (default: ``["fold_idx", "task_idx"]``).
        2. Build the **union** of every x value that appears in any series
           and sort it.
        3. For each series, reindex to the union of x values and
           **forward-fill**: a series that has not yet logged a new value at x
           keeps its last known value.  Rows before the series' first
           observation are left as NaN (not back-filled) so they do not
           contribute to the mean — this avoids artificially low averages at
           the very start of the budget axis.
        4. Average across all series at each x value (``nanmean`` semantics
           via ``groupby().mean()``).
        5. Upload to W&B and store AUC under ``area/{x_axis}/{metric}``.

        Args:
            x_axis: Column name of the continuous x-axis
                (e.g. ``"cumulative_cost_s"``).
            group_cols: Columns that identify an individual series.
                Defaults to ``["fold_idx", "task_idx"]`` when both are
                present in the logged data.
            exclude_cols: Additional columns to drop before building metric
                curves.  Use this to suppress columns that serve as a
                discrete x-axis in a companion ``wandb_plot`` call (e.g.
                ``["visible_num"]``) so they do not appear as noisy /
                non-monotonic metric curves in this chart.
        """
        if self.df is None:
            self.combine_logs()

        # Determine series grouping columns
        default_group = [c for c in ["fold_idx", "task_idx"] if c in self.df.columns]
        group_cols = group_cols if group_cols is not None else default_group

        # Columns that are neither the x-axis nor grouping keys nor excluded
        non_metric = set(group_cols) | {x_axis} | set(exclude_cols or [])
        metric_cols = [
            c for c in self.df.columns
            if c not in non_metric and pd.api.types.is_numeric_dtype(self.df[c])
        ]

        if not metric_cols or self.df.empty or x_axis not in self.df.columns:
            return

        # 1. Union of all x values across every series
        all_x = np.sort(self.df[x_axis].dropna().unique())

        if len(all_x) < 2:
            return

        # 2 & 3. Forward-fill each series onto the union grid
        filled_series: list[pd.DataFrame] = []
        groups = self.df.groupby(group_cols) if group_cols else [(None, self.df)]
        for _, group_df in groups:
            # Sort within the series and set x as index
            series = (
                group_df[[x_axis] + metric_cols]
                .sort_values(x_axis)
                .set_index(x_axis)
            )
            # Deduplicate: multiple iterations can share the same x value
            # (e.g. cumulative_cost_s=0 before any flow is revealed).
            # Keep the last row at each x — the metric state after all
            # observations at that cost level have been processed.
            series = series[~series.index.duplicated(keep="last")]
            # Reindex to the full union; forward-fill (no back-fill so that
            # the series contributes NaN before its first observation)
            series = series.reindex(all_x, method="ffill")
            filled_series.append(series)

        # 4. Average across series (nanmean at each x value)
        df_stacked = pd.concat(filled_series)
        df_mean = df_stacked.groupby(level=0).mean()
        df_mean.index.name = x_axis

        # 5. Upload and compute AUC
        for x_val, row in df_mean.iterrows():
            wandb.log({x_axis: float(x_val), **row.to_dict()})

        self._log_auc(df_mean, x_axis)

    def log_artifact(self, name: str = "raw-metrics") -> None:
        """Persist all raw, unaggregated log entries as ``metrics.json``.

        Always writes a **plain JSON file** (a normal file write, which works
        even on permission-restricted network filesystems where wandb's
        artifact staging ``chmod`` fails).  The wandb artifact upload is then
        attempted best-effort and never fails the run.

        Destination of the plain file:
          - ``$METRICS_OUT_DIR/metrics_<run>.json`` if that env var is set
            (used on the cluster to collect one file per job into one dir), else
          - ``<wandb run dir>/metrics.json`` (kept inside the offline run), else
          - ``./metrics.json``.

        Args:
            name: Artifact name shown in W&B (default ``"raw-metrics"``).
        """
        import json
        import logging
        import os

        import numpy as np

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        # Resolve a unique-ish run tag for filenames / central collection.
        run_tag = None
        try:
            if wandb.run is not None:
                run_tag = wandb.run.name or wandb.run.id
        except Exception:
            run_tag = None

        out_dir = os.environ.get("METRICS_OUT_DIR")
        if out_dir:
            fname = f"metrics_{run_tag or 'run'}.json"
        else:
            try:
                out_dir = wandb.run.dir if wandb.run is not None else "."
            except Exception:
                out_dir = "."
            fname = "metrics.json"
        os.makedirs(out_dir, exist_ok=True)
        plain_path = os.path.join(out_dir, fname)

        # 1. Plain write — the result we actually rely on for aggregation.
        with open(plain_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, cls=_NumpyEncoder)

        # 2. Best-effort wandb artifact; skip silently if the FS rejects the
        #    chmod wandb does on its staging copy (common on CIFS/NFS mounts).
        try:
            artifact = wandb.Artifact(name=name, type="raw_metrics")
            artifact.add_file(plain_path, name="metrics.json")
            wandb.log_artifact(artifact)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "wandb artifact upload skipped (metrics saved to %s): %s",
                plain_path, exc,
            )

    def _log_auc(self, df_aggregated: pd.DataFrame, x_axis: str) -> None:
        """Compute and upload area-under-curve for every metric column.

        Keys are stored as ``wandb.summary["area/{x_axis}/{metric}"]`` so
        that summaries from different x-axes (``visible_num``,
        ``cumulative_cost_s``) do not collide.
        """
        x_vals = df_aggregated.index.to_numpy(dtype=float)
        metric_cols = [c for c in df_aggregated.columns if c != x_axis]
        for metric in metric_cols:
            y_vals = df_aggregated[metric].to_numpy(dtype=float)
            # only compute AUC over steps where the metric was actually logged
            valid = np.isfinite(y_vals)
            if valid.sum() >= 2:
                auc = float(np.trapezoid(y_vals[valid], x_vals[valid]))
            elif valid.sum() == 1:
                auc = float(y_vals[valid][0])
            else:
                auc = float("nan")
            wandb.summary[f"area/{x_axis}/{metric}"] = auc
