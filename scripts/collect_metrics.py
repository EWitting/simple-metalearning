"""Aggregate the continuous-BO cluster array's per-task metric files into one
W&B run per estimator — identical in form to a discrete ``BOSelection`` run —
so ``scripts/plot_comparison.py`` can overlay continuous and discrete curves
with no special-casing.

Background
----------
Each SLURM array job ran exactly one ``(estimator, fold, task)`` and wrote
``metrics_<safe_est>-f<fold>-<task>.json`` (the raw per-step log list) into the
shared ``cluster_io/metrics`` dir, where ``safe_est`` is the estimator config
key with ``/`` replaced by ``-`` (e.g. ``continuous-rf_ei``).

A *discrete* run, by contrast, accumulates the log entries for ALL its folds and
tasks in a single in-process ``MetricStore`` and uploads them as one
``raw_metrics`` artifact.  ``plot_comparison.py`` reads exactly that artifact.

This script reconstructs the discrete shape: it groups the per-task files by
estimator, concatenates their log lists (each entry already carries ``fold_idx``
and ``task_idx``), and replays the merged list through ``MetricStore`` —
``wandb_plot`` (visible_num), ``wandb_plot_interpolated`` (cumulative_cost_s) and
``log_artifact`` (the raw_metrics artifact).  The resulting W&B run is
indistinguishable from a discrete run to downstream tooling.

Usage
-----
After copying ``cluster_io/metrics`` to your machine::

    # offline (no login needed); sync later with `wandb sync wandb/offline-run-*`
    uv run python scripts/collect_metrics.py --metrics-dir cluster_io/metrics

    # straight to the cloud for manual visualisation
    export WANDB_ENTITY=<you>
    uv run python scripts/collect_metrics.py --metrics-dir cluster_io/metrics \
        --mode online --tag continuous

    # no W&B at all — just print the grouping and write the merged CSV
    uv run python scripts/collect_metrics.py --metrics-dir cluster_io/metrics --dry-run

The runs are named ``<prefix><safe_est>`` (default prefix empty, so e.g.
``continuous-rf_ei``) and tagged so you can fetch them together::

    python scripts/plot_comparison.py --tags continuous <discrete run names...>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Filename → (estimator, fold, task).  ``.+`` for the estimator is greedy and
# backtracks to the single ``-f<digit>-`` separator; estimator keys never contain
# that pattern, so the split is unambiguous even for hyphenated task names
# (credit-g) and estimator keys (continuous-oof_ridge_mmr).
_FNAME_RE = re.compile(r"^metrics_(?P<est>.+)-f(?P<fold>\d+)-(?P<task>.+)\.json$")


def _load_grouped(metrics_dir: Path) -> dict[str, list[dict]]:
    """Group every ``metrics_*.json`` file's log entries by estimator.

    Returns ``{safe_est: merged_log_entries}``.  Failure-marker entries (written
    by a crashed task, though with ``raise_on_task_failure=true`` these normally
    abort the job before any file is written) are dropped defensively.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    files = sorted(metrics_dir.glob("metrics_*.json"))
    if not files:
        raise SystemExit(f"No metrics_*.json files found in {metrics_dir}")

    n_tasks_per_est: dict[str, set] = defaultdict(set)
    empty_per_est: dict[str, int] = defaultdict(int)  # parsed but zero usable rows
    for path in files:
        m = _FNAME_RE.match(path.name)
        if not m:
            print(f"  ! skipping unparseable filename: {path.name}")
            continue
        est = m.group("est")
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        entries = [e for e in entries if e.get("status") != "failed"]
        if not entries:
            print(f"  ! {path.name}: no usable entries (all failed/empty)")
            empty_per_est[est] += 1
            continue
        grouped[est].extend(entries)
        n_tasks_per_est[est].add((m.group("fold"), m.group("task")))

    # Estimators whose files ALL came back empty never reach `grouped`, so they
    # would silently vanish from the summary below — surface them loudly.
    wiped = sorted(set(empty_per_est) - set(grouped))
    for est in wiped:
        print(f"  !! {est:<30} ALL {empty_per_est[est]} files empty/failed — "
              f"estimator produced no usable results")

    counts = {est: len(n_tasks_per_est[est]) for est in grouped}
    max_runs = max(counts.values()) if counts else 0
    print(f"Found {len(files)} files across {len(grouped)} estimators:")
    for est in sorted(grouped):
        flag = ""
        # Flag estimators whose (fold,task) coverage trails the leader: their
        # mean curves are averaged over fewer / different tasks, so cross-
        # estimator comparison is biased until coverage evens out.
        if max_runs and counts[est] < 0.8 * max_runs:
            flag = f"  <-- only {counts[est]/max_runs:.0%} of max ({max_runs}); coverage imbalance"
        print(f"  {est:<32} {counts[est]:>3} (fold,task) runs, "
              f"{len(grouped[est]):>5} log rows{flag}")
    return grouped


def _write_csv(grouped: dict[str, list[dict]], out_csv: Path) -> None:
    """Write a tidy mean-curve CSV: one row per (estimator, visible_num, metric).

    A quick offline path for plotting without W&B.  Mirrors ``wandb_plot``: the
    mean of each numeric metric across all (fold, task) at each ``visible_num``.
    The cumulative-cost axis needs the forward-fill interpolation in
    ``MetricStore`` and is only produced on the W&B path.
    """
    rows = []
    for est, logs in grouped.items():
        df = pd.DataFrame(logs)
        if "visible_num" not in df.columns:
            continue
        drop = [c for c in ("cumulative_cost_s",) if c in df.columns]
        agg = df.drop(columns=drop).groupby("visible_num").mean(numeric_only=True)
        for visible_num, row in agg.iterrows():
            for metric, value in row.items():
                if metric in ("fold_idx", "task_idx"):
                    continue
                rows.append({"estimator": est, "visible_num": int(visible_num),
                             "metric": metric, "value": float(value)})
    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote mean-curve CSV: {out_csv} ({len(out_df)} rows)")


def _upload(grouped: dict[str, list[dict]], project: str, mode: str,
            tags: list[str], prefix: str) -> None:
    """Replay each estimator's merged logs through MetricStore → one W&B run.

    Reuses the exact aggregation a discrete run uses, so the produced
    ``raw_metrics`` artifact and curves are consumed identically by
    ``plot_comparison.py``.
    """
    import wandb
    from simple_metalearning.metrics.metric_store import MetricStore

    for est in sorted(grouped):
        run_name = f"{prefix}{est}"
        print(f"Uploading {run_name} ({len(grouped[est])} rows)...")
        wandb.init(project=project, name=run_name, tags=tags,
                   mode=mode, reinit=True)
        ms = MetricStore()
        ms.logs = list(grouped[est])
        # Same two-axis aggregation as BOSelection.run().
        ms.wandb_plot(x_axis="visible_num", exclude_cols=["cumulative_cost_s"])
        ms.wandb_plot_interpolated(x_axis="cumulative_cost_s",
                                   exclude_cols=["visible_num"])
        ms.log_artifact()  # raw_metrics artifact (metrics.json) for plot_comparison
        wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-dir", default="cluster_io/metrics", type=Path,
                    help="Dir containing metrics_*.json from the cluster array.")
    ap.add_argument("--project", default="simple-metalearning")
    ap.add_argument("--mode", default="offline",
                    choices=["online", "offline", "disabled"],
                    help="W&B mode (default offline; sync later or use online).")
    ap.add_argument("--tag", action="append", default=None,
                    help="W&B tag to attach (repeatable). Default: ['continuous'].")
    ap.add_argument("--prefix", default="",
                    help="Prefix for run names (default empty → e.g. continuous-rf_ei).")
    ap.add_argument("--out-csv", default="cluster_io/continuous_mean_curves.csv",
                    type=Path, help="Where to write the tidy mean-curve CSV.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only group + write CSV; no W&B upload.")
    args = ap.parse_args()

    grouped = _load_grouped(args.metrics_dir)
    _write_csv(grouped, args.out_csv)

    if args.dry_run:
        print("Dry run — skipping W&B upload.")
        return

    tags = args.tag if args.tag is not None else ["continuous"]
    _upload(grouped, args.project, args.mode, tags, args.prefix)
    print("Done. Overlay with discrete via, e.g.:")
    print(f"  python scripts/plot_comparison.py --tags {tags[0]} <discrete run names>")


if __name__ == "__main__":
    main()
