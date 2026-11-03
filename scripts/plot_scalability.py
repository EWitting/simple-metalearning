"""Scalability analysis: how metric shifts over training-task limit for each method.

Fetches all W&B runs tagged with the scalability experiment tags, groups them
by method (by stripping the "-TL{N}" suffix from run names), and for each
method produces a figure with one curve per task_limit coloured with a hue
gradient (light = few tasks, dark = many tasks).

Usage
-----
    python scripts/plot_scalability.py [options]

Options
-------
--project   W&B project  (default: simple-metalearning)
--entity    W&B entity   (default: $WANDB_ENTITY env var, or your default W&B entity)
--tags      Tags that all scalability runs must have (default: scalability)
--x-axis    X-axis column (default: visible_num)
--metric    Metric column (default: test_normalized_regret)
--tl-sep    Separator before task-limit number in run name (default: -TL)
--out-dir   Output directory (default: plots/scalability)
--ci        Confidence interval level (default: 0.95)
--band      'ci' or 'std' (default: ci)

Output
------
One PNG per method saved to --out-dir, e.g.:
  RF-Perf_scalability.png
  TwoTower-Perf_scalability.png
  ...
Plus a combined PNG with all methods in a grid:
  all_methods_scalability.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.stats as st
import wandb
import yaml


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _load_style(path: str | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"Warning: style file '{path}' not found, using defaults.", file=sys.stderr)
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _hue_colors_from_base(n: int, base_color: str) -> list:
    """Return n colors as a light-to-dark gradient anchored at base_color."""
    r, g, b = mcolors.to_rgb(base_color)
    colors = []
    for i in range(n):
        t = 0.35 + 0.65 * i / max(n - 1, 1)
        colors.append((1.0 - t * (1.0 - r), 1.0 - t * (1.0 - g), 1.0 - t * (1.0 - b)))
    return colors


# ---------------------------------------------------------------------------
# W&B helpers (shared with plot_comparison.py)
# ---------------------------------------------------------------------------

def _find_runs_by_tags(
    tags: list[str],
    entity: str,
    project: str,
) -> list[wandb.apis.public.Run]:
    """Return all runs that have ALL of the specified tags."""
    api = wandb.Api()
    filters: dict = {"tags": {"$in": tags}}
    runs = api.runs((f"{entity}/{project}" if entity else project), filters=filters, order="-created_at")
    tags_set = set(tags)
    matched = []
    seen: set[str] = set()
    for run in runs:
        run_tags = set(run.tags or [])
        if tags_set.issubset(run_tags) and run.displayName not in seen:
            matched.append(run)
            seen.add(run.displayName)
    print(f"Found {len(matched)} runs with tags {tags}", file=sys.stderr)
    return matched


def _load_artifact(run: wandb.apis.public.Run) -> list[dict]:
    artifacts = run.logged_artifacts()
    for art in artifacts:
        if art.type == "raw_metrics":
            with tempfile.TemporaryDirectory() as tmpdir:
                art.download(root=tmpdir)
                with open(Path(tmpdir) / "metrics.json", encoding="utf-8") as f:
                    return json.load(f)
    raise ValueError(f"No raw_metrics artifact found for run '{run.name}'")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_series(
    logs: list[dict],
    x_col: str,
    metric_col: str,
) -> dict[tuple, pd.Series]:
    df = pd.DataFrame(logs)
    required = {x_col, metric_col, "fold_idx", "task_idx"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Columns missing from artifact: {missing}")
    series: dict[tuple, pd.Series] = {}
    for (fold, task), group in df.groupby(["fold_idx", "task_idx"]):
        s = group.set_index(x_col)[metric_col].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        series[(fold, task)] = s
    return series


def _interpolate_to_grid(
    series_dict: dict[tuple, pd.Series],
    x_grid: np.ndarray,
) -> np.ndarray:
    rows = []
    for s in series_dict.values():
        reindexed = (
            s.reindex(pd.Index(x_grid).union(s.index))
            .sort_index()
            .ffill()
            .reindex(x_grid)
        )
        rows.append(reindexed.values)
    return np.array(rows, dtype=float)


def _ci(data: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.sum(~np.isnan(data), axis=0)
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    alpha = 1.0 - level
    t_crit = st.t.ppf(1.0 - alpha / 2, df=np.maximum(n - 1, 1))
    return mean - t_crit * sem, mean + t_crit * sem


def _std_band(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1)
    return mean - std, mean + std


def _band(data, band_type, ci_level):
    return _std_band(data) if band_type == "std" else _ci(data, ci_level)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_run_name(
    run_name: str,
    tl_sep: str,
) -> tuple[str, int] | None:
    """Return (method_name, task_limit) or None if the pattern is not matched."""
    pattern = re.escape(tl_sep) + r"(\d+)$"
    m = re.search(pattern, run_name)
    if not m:
        return None
    task_limit = int(m.group(1))
    method = run_name[: m.start()]
    return method, task_limit


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def _hue_colors(n: int, cmap_name: str = "Blues") -> list:
    """Return n colors from a colormap, avoiding the very pale end."""
    cmap = cm.get_cmap(cmap_name)
    return [cmap(0.35 + 0.65 * i / max(n - 1, 1)) for i in range(n)]


def _method_colors(method: str, n: int, style: dict) -> list:
    """Return n hue-gradient colors for a method, using base color from style if available."""
    base = style.get(method, {}).get("color")
    if base:
        return _hue_colors_from_base(n, base)
    return _hue_colors(n)


def _method_plot(
    method: str,
    tl_series: dict[int, dict[tuple, pd.Series]],
    x_grid: np.ndarray,
    metric_col: str,
    ci_level: float,
    band_type: str,
    out_path: Path,
    style: dict | None = None,
) -> None:
    """Plot all task_limit curves for a single method with a hue gradient."""
    style = style or {}
    task_limits = sorted(tl_series.keys())
    colors = _method_colors(method, len(task_limits), style)
    title = style.get(method, {}).get("label", method)

    fig, ax = plt.subplots(figsize=(6, 4))
    all_means = []

    for tl, color in zip(task_limits, colors):
        mat = _interpolate_to_grid(tl_series[tl], x_grid)
        mean = np.nanmean(mat, axis=0)
        lo, hi = _band(mat, band_type, ci_level)
        ax.plot(x_grid, mean, label=f"TL={tl}", color=color)
        ax.fill_between(x_grid, lo, hi, alpha=0.15, color=color)
        all_means.append(mean)

    means_stack = np.array(all_means)
    y_min = np.nanmin(means_stack)
    y_max = np.nanmax(means_stack)
    ax.set_ylim(y_min * 0.8, y_max * 1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(title="Training tasks", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _combined_plot(
    method_data: dict[str, dict[int, dict[tuple, pd.Series]]],
    x_grid: np.ndarray,
    metric_col: str,
    ci_level: float,
    band_type: str,
    out_path: Path,
    style: dict | None = None,
) -> None:
    """Grid of per-method plots in a single figure with shared y-axis."""
    style = style or {}
    methods = sorted(method_data.keys())
    ncols = min(3, len(methods))
    nrows = (len(methods) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        sharey=True,
    )

    for idx, method in enumerate(methods):
        ax = axes[idx // ncols][idx % ncols]
        tl_series = method_data[method]
        task_limits = sorted(tl_series.keys())
        colors = _method_colors(method, len(task_limits), style)
        title = style.get(method, {}).get("label", method)

        for tl, color in zip(task_limits, colors):
            mat = _interpolate_to_grid(tl_series[tl], x_grid)
            mean = np.nanmean(mat, axis=0)
            lo, hi = _band(mat, band_type, ci_level)
            ax.plot(x_grid, mean, label=f"TL={tl}", color=color)
            ax.fill_between(x_grid, lo, hi, alpha=0.15, color=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("# evaluations")
        if idx % ncols == 0:
            ax.set_ylabel(metric_col.replace("_", " ").title())
        ax.legend(title="Training tasks", ncol=2, fontsize=7)

    # Hide unused axes
    for idx in range(len(methods), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Scalability: metric vs training-task limit", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot metric vs visible_num for each method across task limits."
    )
    parser.add_argument("--project", default="simple-metalearning")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["scalability"],
        help="Tags that all scalability runs must have (default: scalability)",
    )
    parser.add_argument("--x-axis", default="visible_num", dest="x_axis")
    parser.add_argument("--metric", default="test_normalized_regret")
    parser.add_argument(
        "--tl-sep",
        default="-TL",
        dest="tl_sep",
        help="Separator before task-limit number in run name (default: -TL)",
    )
    parser.add_argument("--out-dir", default="plots/scalability", dest="out_dir")
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--band", choices=["ci", "std"], default="ci")
    parser.add_argument("--style", default="conf/plot_style.yaml",
                        help="YAML file mapping method names to label/color "
                             "(default: conf/plot_style.yaml)")
    args = parser.parse_args()

    style = _load_style(args.style)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _find_runs_by_tags(args.tags, args.entity, args.project)
    if not runs:
        print("No runs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Parse run names and group by method
    # method_data: {method: {task_limit: {(fold,task): pd.Series}}}
    method_data: dict[str, dict[int, dict[tuple, pd.Series]]] = {}
    skipped = []

    for run in runs:
        parsed = _parse_run_name(run.displayName, args.tl_sep)
        if parsed is None:
            skipped.append(run.displayName)
            continue
        method, task_limit = parsed
        print(f"Loading artifact for run '{run.displayName}' (method={method}, TL={task_limit})...")
        try:
            logs = _load_artifact(run)
        except ValueError as e:
            print(f"  Skipping: {e}", file=sys.stderr)
            continue
        series = _build_series(logs, args.x_axis, args.metric)
        method_data.setdefault(method, {})[task_limit] = series

    if skipped:
        print(f"Skipped (no TL pattern): {skipped}", file=sys.stderr)

    if not method_data:
        print("No usable runs after parsing. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Drop excluded methods (matched against base method name, before -TL{N})
    excluded = set(style.get("exclude", []) or [])
    if excluded:
        removed = set(method_data) & excluded
        if removed:
            print(f"Excluding methods (from style config): {sorted(removed)}", file=sys.stderr)
        method_data = {k: v for k, v in method_data.items() if k not in excluded}

    if not method_data:
        print("No methods remaining after exclusions. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Build shared x_grid across all runs
    x_vals: set[float] = set()
    for tl_series in method_data.values():
        for s_dict in tl_series.values():
            for s in s_dict.values():
                x_vals.update(s.index.tolist())
    if args.x_axis == "visible_num":
        x_vals = {x for x in x_vals if x > 0}
    x_grid = np.sort(np.array(list(x_vals)))

    # Per-method plots
    for method, tl_series in method_data.items():
        safe_name = re.sub(r"[^\w\-]", "_", method)
        _method_plot(
            method,
            tl_series,
            x_grid,
            args.metric,
            args.ci,
            args.band,
            out_dir / f"{safe_name}_scalability.png",
            style=style,
        )

    # Combined grid plot
    _combined_plot(
        method_data,
        x_grid,
        args.metric,
        args.ci,
        args.band,
        out_dir / "all_methods_scalability.png",
        style=style,
    )


if __name__ == "__main__":
    main()
