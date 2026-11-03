"""Adaptive overfitting plots: val vs test normalised regret per run.

For each W&B run found via tags (or run names), produces one plot showing
``normalised_regret`` (val) and ``test_normalized_regret`` (test) vs
``visible_num`` on the same log-log axes with 95 % CI bands.  One PNG +
PDF is saved per run.  The val/test gap reveals how much a model exploits
the validation signal at the expense of held-out test performance.

Usage
-----
    # By tags (fetches all runs that have ALL listed tags):
    python scripts/plot_adaptive_overfitting.py --tags cash  --out-dir plots/adaptive_overfitting/cash
    python scripts/plot_adaptive_overfitting.py --tags hpo   --out-dir plots/adaptive_overfitting/hpo

    # By explicit run names:
    python scripts/plot_adaptive_overfitting.py GP-Hyper GP-Perf --out-dir plots/adaptive_overfitting/cash

Options
-------
--project   W&B project  (default: simple-metalearning)
--entity    W&B entity   (default: $WANDB_ENTITY env var, or your default W&B entity)
--x-axis    X-axis column (default: visible_num)
--out-dir   Directory to write PNG/PDF files (default: plots/adaptive_overfitting)
--tags      Filter runs by W&B tags.
--ci        Confidence interval level, e.g. 0.95 (default: 0.95)
--band      Band type: 'ci' (t-distribution CI) or 'std' (±1 std, default: ci)
--style     YAML style config (default: conf/plot_style.yaml)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
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


def _run_style(style: dict, run_name: str, index: int) -> tuple[str, str]:
    """Return (label, color) for a run, falling back to auto defaults."""
    entry = style.get(run_name, {})
    label = entry.get("label", run_name)
    color = entry.get("color", COLORS[index % len(COLORS)])
    return label, color


# ---------------------------------------------------------------------------
# W&B artifact loading
# ---------------------------------------------------------------------------

def _find_runs(
    run_names: list[str],
    entity: str,
    project: str,
    tags: list[str] | None = None,
) -> dict[str, wandb.apis.public.Run]:
    api = wandb.Api()

    if not run_names and tags:
        filters: dict = {"tags": {"$in": tags}}
        runs = api.runs((f"{entity}/{project}" if entity else project), filters=filters, order="-created_at")
        name_to_run: dict[str, wandb.apis.public.Run] = {}
        tags_set = set(tags)
        for run in runs:
            run_tags = set(run.tags or [])
            if tags_set.issubset(run_tags):
                if run.displayName not in name_to_run:
                    name_to_run[run.displayName] = run
        print(f"Found {len(name_to_run)} runs with tags {tags}", file=sys.stderr)
        return name_to_run

    filters = {"displayName": {"$in": run_names}}
    if tags:
        filters["tags"] = {"$in": tags}
    runs = api.runs((f"{entity}/{project}" if entity else project), filters=filters, order="-created_at")
    name_to_run = {}
    for run in runs:
        if run.displayName not in name_to_run:
            name_to_run[run.displayName] = run
    missing = set(run_names) - set(name_to_run)
    if missing:
        print(f"Warning: could not find runs: {missing}", file=sys.stderr)
    return name_to_run


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
# Series construction
# ---------------------------------------------------------------------------

def _build_series(
    logs: list[dict],
    x_col: str,
    metric_col: str,
) -> dict[tuple, pd.Series]:
    """Build per-(fold, task) series of metric vs x."""
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
    """Forward-fill each series onto x_grid; return (n_series, n_x) array."""
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


# ---------------------------------------------------------------------------
# Confidence bands
# ---------------------------------------------------------------------------

def _ci_band(data: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """t-distribution CI for the mean across axis 0."""
    n = np.sum(~np.isnan(data), axis=0)
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    alpha = 1.0 - level
    t_crit = st.t.ppf(1.0 - alpha / 2, df=np.maximum(n - 1, 1))
    margin = t_crit * sem
    return mean - margin, mean + margin


def _std_band(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1)
    return mean - std, mean + std


def _band(
    data: np.ndarray, band_type: str, ci_level: float
) -> tuple[np.ndarray, np.ndarray]:
    if band_type == "std":
        return _std_band(data)
    return _ci_band(data, ci_level)


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# Per-run plot
# ---------------------------------------------------------------------------

def _overfitting_plot(
    run_name: str,
    val_series: dict[tuple, pd.Series],
    test_series: dict[tuple, pd.Series],
    x_grid: np.ndarray,
    ci_level: float,
    out_path: Path,
    band_type: str = "ci",
    style: dict | None = None,
    style_index: int = 0,
    best_test_regret: float | None = None,
) -> None:
    """Plot val and test normalised regret vs visible_num for a single run."""
    style = style or {}
    label, color = _run_style(style, run_name, style_index)

    fig, ax = plt.subplots()

    # Val curve — solid
    val_mat = _interpolate_to_grid(val_series, x_grid)
    val_mean = np.nanmean(val_mat, axis=0)
    val_lo, val_hi = _band(val_mat, band_type, ci_level)
    ax.plot(x_grid, val_mean, color=color, linestyle="solid", label="Val")
    ax.fill_between(x_grid, val_lo, val_hi, alpha=0.2, color=color)

    # Test curve — dashed
    test_mat = _interpolate_to_grid(test_series, x_grid)
    test_mean = np.nanmean(test_mat, axis=0)
    test_lo, test_hi = _band(test_mat, band_type, ci_level)
    ax.plot(x_grid, test_mean, color=color, linestyle="dashed", label="Test")
    ax.fill_between(x_grid, test_lo, test_hi, alpha=0.12, color=color)

    # Horizontal reference line: best mean test regret across all models
    if best_test_regret is not None and best_test_regret > 0:
        ax.axhline(best_test_regret, color="black", linestyle="dotted",
                   linewidth=1.0, label="Best (any model)")

    # Axis limits: fit the foreground curves, guard against log(0) when
    # normalised regret reaches zero (common in HPO / small search spaces).
    all_means = np.concatenate([val_mean[np.isfinite(val_mean)],
                                 test_mean[np.isfinite(test_mean)]])
    if all_means.size > 0:
        y_max = np.nanmax(all_means)
        positive = all_means[all_means > 0]
        if best_test_regret is not None and best_test_regret > 0:
            positive = np.append(positive, best_test_regret)
        y_min_pos = float(np.nanmin(positive)) if positive.size > 0 else 1e-4
        ax.set_ylim(y_min_pos * 0.5, y_max * 1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Normalised regret")
    ax.set_title(label)
    ax.legend()
    fig.tight_layout()

    for suffix in (".png", ".pdf"):
        p = out_path.with_suffix(suffix)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined overview figure
# ---------------------------------------------------------------------------

def _slot_order(
    style: dict,
    run_names: set[str],
    ncols: int = 4,
) -> list[str | None]:
    """Build subplot slot order from style: families first (row-aligned), then baselines.

    Family methods (Perf/Hyper variants) are placed consecutively so they end
    up side-by-side in the grid.  Background/naive baselines go in the last
    row, padded with None to fill the grid.
    """
    families: dict[str, list[str]] = style.get("families", {})
    background: list[str] = list(style.get("background_methods", []))

    slots: list[str | None] = []

    # Family methods in YAML order
    for methods in families.values():
        for m in methods:
            if m in run_names:
                slots.append(m)

    # Pad family section to a complete row so baselines start fresh
    rem = len(slots) % ncols
    if rem:
        slots.extend([None] * (ncols - rem))

    # Background/baseline methods
    for m in background:
        if m in run_names:
            slots.append(m)

    # Any runs not captured above
    accounted = {s for s in slots if s is not None}
    for name in sorted(run_names - accounted):
        slots.append(name)

    # Pad to complete rows
    rem = len(slots) % ncols
    if rem:
        slots.extend([None] * (ncols - rem))

    return slots


def _combined_overfitting_plot(
    run_data: list[tuple[int, str, dict, dict, np.ndarray]],
    ci_level: float,
    out_path: Path,
    band_type: str = "ci",
    style: dict | None = None,
    best_test_regret: float | None = None,
    ncols: int = 4,
) -> None:
    """One-page overview figure: all runs in a shared-ylim grid."""
    from matplotlib.lines import Line2D

    style = style or {}
    run_lookup = {name: (idx, val_s, test_s, xg)
                  for idx, name, val_s, test_s, xg in run_data}

    slots = _slot_order(style, set(run_lookup.keys()), ncols=ncols)
    nrows = len(slots) // ncols

    rc = {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 14,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3.8, nrows * 3.2),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        # Pre-compute curves for every active slot
        all_positive: list[float] = []
        y_max_global = 0.0
        cache: dict[str, tuple] = {}  # name -> (label, color, x_grid, v_mean, v_lo, v_hi, t_mean, t_lo, t_hi)
        for name, (idx, val_s, test_s, x_grid) in run_lookup.items():
            label, color = _run_style(style, name, idx)

            val_mat = _interpolate_to_grid(val_s, x_grid)
            v_mean = np.nanmean(val_mat, axis=0)
            v_lo, v_hi = _band(val_mat, band_type, ci_level)

            test_mat = _interpolate_to_grid(test_s, x_grid)
            t_mean = np.nanmean(test_mat, axis=0)
            t_lo, t_hi = _band(test_mat, band_type, ci_level)

            for arr in [v_mean, t_mean]:
                finite = arr[np.isfinite(arr)]
                pos = finite[finite > 0]
                all_positive.extend(pos.tolist())
                if finite.size > 0:
                    y_max_global = max(y_max_global, float(np.nanmax(finite)))

            cache[name] = (label, color, x_grid, v_mean, v_lo, v_hi, t_mean, t_lo, t_hi)

        # Global y limits
        if all_positive:
            all_pos_arr = np.array(all_positive)
            if best_test_regret is not None and best_test_regret > 0:
                all_pos_arr = np.append(all_pos_arr, best_test_regret)
            y_lo = float(np.nanmin(all_pos_arr)) * 0.5
        else:
            y_lo = 1e-4
        y_hi = y_max_global * 1.2

        # Draw subplots
        for slot_i, slot_name in enumerate(slots):
            ax = axes_flat[slot_i]
            row, col = divmod(slot_i, ncols)

            if slot_name is None:
                ax.set_visible(False)
                continue

            label, color, x_grid, v_mean, v_lo, v_hi, t_mean, t_lo, t_hi = cache[slot_name]

            ax.plot(x_grid, v_mean, color=color, linestyle="solid")
            ax.fill_between(x_grid, v_lo, v_hi, alpha=0.2, color=color)
            ax.plot(x_grid, t_mean, color=color, linestyle="dashed")
            ax.fill_between(x_grid, t_lo, t_hi, alpha=0.12, color=color)

            if best_test_regret is not None and best_test_regret > 0:
                ax.axhline(best_test_regret, color="black", linestyle="dotted",
                           linewidth=1.0)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(label)

            if col == 0:
                ax.set_ylabel("Normalised regret")
            else:
                ax.tick_params(labelleft=False)

            if row == nrows - 1:
                ax.set_xlabel("Number of evaluations")

        # Shared legend below the figure
        legend_handles = [
            Line2D([0], [0], color="black", linestyle="solid",  label="Val"),
            Line2D([0], [0], color="black", linestyle="dashed", label="Test"),
        ]
        if best_test_regret is not None and best_test_regret > 0:
            legend_handles.append(
                Line2D([0], [0], color="black", linestyle="dotted",
                       linewidth=1.0, label="Best (any model)")
            )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.10)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_handles),
            frameon=False,
        )

        for suffix in (".png", ".pdf"):
            p = out_path.with_suffix(suffix)
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved: {p}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-run val vs test regret plots (adaptive overfitting)."
    )
    parser.add_argument("run_names", nargs="*",
                        help="W&B run names to plot (omit to use --tags alone)")
    parser.add_argument("--project", default="simple-metalearning")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--x-axis", default="visible_num", dest="x_axis")
    parser.add_argument("--out-dir", default="plots/adaptive_overfitting", dest="out_dir")
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Fetch all runs that have ALL specified tags.")
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--band", choices=["ci", "std"], default="ci",
                        help="Band type: 'ci' (t-distribution) or 'std' (±1 std)")
    parser.add_argument("--style", default="conf/plot_style.yaml")
    args = parser.parse_args()

    if not args.run_names and not args.tags:
        parser.error("Provide run names and/or --tags to select runs.")

    style = _load_style(args.style)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_names:
        print(f"Fetching {len(args.run_names)} runs ...", file=sys.stderr)
    else:
        print(f"Fetching runs with tags {args.tags} ...", file=sys.stderr)
    name_to_run = _find_runs(args.run_names, args.entity, args.project, tags=args.tags)
    if not name_to_run:
        print("No runs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    excluded = set(style.get("exclude", []) or [])

    # Pass 1: load all artifacts and build series
    run_data: list[tuple[int, str, dict, dict, np.ndarray]] = []
    for idx, (run_name, run) in enumerate(name_to_run.items()):
        if run_name in excluded:
            print(f"Skipping excluded run '{run_name}'", file=sys.stderr)
            continue

        print(f"Loading artifact for '{run_name}' ...", file=sys.stderr)
        try:
            logs = _load_artifact(run)
        except ValueError as exc:
            print(f"  Skipping: {exc}", file=sys.stderr)
            continue

        df = pd.DataFrame(logs)
        if "test_normalized_regret" not in df.columns:
            print(f"  Skipping '{run_name}': no test_normalized_regret in artifact "
                  f"(val_test_split was likely not used).", file=sys.stderr)
            continue
        if "normalized_regret" not in df.columns:
            print(f"  Skipping '{run_name}': no normalized_regret in artifact.",
                  file=sys.stderr)
            continue

        val_series = _build_series(logs, args.x_axis, "normalized_regret")
        test_series = _build_series(logs, args.x_axis, "test_normalized_regret")

        x_vals: set[float] = set()
        for s in val_series.values():
            x_vals.update(s.index.tolist())
        if args.x_axis == "visible_num":
            x_vals = {x for x in x_vals if x > 0}
        x_grid = np.sort(np.array(list(x_vals)))

        run_data.append((idx, run_name, val_series, test_series, x_grid))

    # Compute global minimum mean test regret across all runs and x-grid points
    best_test_regret: float | None = None
    for _, _, _, test_series, x_grid in run_data:
        test_mat = _interpolate_to_grid(test_series, x_grid)
        test_mean = np.nanmean(test_mat, axis=0)
        positive = test_mean[test_mean > 0]
        if positive.size > 0:
            run_min = float(np.nanmin(positive))
            if best_test_regret is None or run_min < best_test_regret:
                best_test_regret = run_min

    print(f"Best mean test regret across all runs: {best_test_regret}", file=sys.stderr)

    # Pass 2: individual plots
    for idx, run_name, val_series, test_series, x_grid in run_data:
        safe_name = run_name.replace("/", "_").replace(" ", "_")
        out_path = out_dir / safe_name

        _overfitting_plot(
            run_name=run_name,
            val_series=val_series,
            test_series=test_series,
            x_grid=x_grid,
            ci_level=args.ci,
            out_path=out_path,
            band_type=args.band,
            style=style,
            style_index=idx,
            best_test_regret=best_test_regret,
        )

    # Combined overview figure
    _combined_overfitting_plot(
        run_data=run_data,
        ci_level=args.ci,
        out_path=out_dir / "overview",
        band_type=args.band,
        style=style,
        best_test_regret=best_test_regret,
    )


if __name__ == "__main__":
    main()
