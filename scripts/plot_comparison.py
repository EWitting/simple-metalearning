"""Comparison plots across W&B runs using raw-metrics artifacts.

Usage
-----
    # By run names (optionally further filtered by tags):
    python scripts/plot_comparison.py run_name_1 run_name_2 ... [options]

    # By tags only (no run names — fetches all runs with ALL given tags):
    python scripts/plot_comparison.py --tags cash [options]

Options
-------
--project   W&B project (default: simple-metalearning)
--entity    W&B entity  (default: $WANDB_ENTITY env var, or your default W&B entity)
--x-axis    X-axis column (default: visible_num)
--metric    Metric column for the regret plot (default: test_normalized_regret)
--out-dir   Directory to write PNG files (default: plots/)
--tags      Filter runs by W&B tags.  When no run names are given, fetches all
            runs that have ALL specified tags.  When run names are given, keeps
            only those whose tag set includes at least one of the given tags.
--ci        Confidence interval level, e.g. 0.95 (default: 0.95)
--band      Band type: 'ci' (t-distribution CI) or 'std' (±1 std, default: ci)

Output
------
Two PNG files saved to --out-dir:
  regret_vs_evals.png   — mean metric + CI bands per run, log-log axes
  rank_vs_evals.png     — mean rank + CI bands per run, log x-axis
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
    """Load run style config from YAML. Returns {} if path is None or missing."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"Warning: style file '{path}' not found, using defaults.", file=sys.stderr)
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_style(style: dict, run_name: str, index: int) -> tuple[str, str, str, str]:
    """Return (label, color, linestyle, marker) for a run, falling back to auto defaults."""
    entry = style.get(run_name, {})
    label = entry.get("label", run_name)
    color = entry.get("color", COLORS[index % len(COLORS)])
    linestyle = entry.get("linestyle", "solid")
    marker = entry.get("marker", "")
    return label, color, linestyle, marker


def _families_from_style(style: dict) -> tuple[dict[str, list[str]], list[str]]:
    """Return (families_dict, background_methods) from the style config.

    families_dict maps family name -> list of foreground method run names.
    background_methods is the list of run names to draw faded behind all families.
    """
    families = {k: list(v) for k, v in style.get("families", {}).items()}
    background = list(style.get("background_methods", []))
    return families, background


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _find_runs(
    run_names: list[str],
    entity: str,
    project: str,
    tags: list[str] | None = None,
) -> dict[str, wandb.apis.public.Run]:
    """Return a dict mapping run_name -> wandb Run object.

    When ``run_names`` is empty and ``tags`` is provided, fetches all runs that
    have ALL of the specified tags (tag-only mode).  When ``run_names`` is non-
    empty, fetches those specific runs and optionally post-filters by tags.
    """
    api = wandb.Api()

    if not run_names and tags:
        # Tag-only mode: fetch all runs that have every specified tag.
        # W&B's $in filter matches any tag, so we fetch runs containing the
        # first tag and filter client-side for the rest.
        filters: dict = {"tags": {"$in": tags}}
        runs = api.runs(
            (f"{entity}/{project}" if entity else project),
            filters=filters,
            order="-created_at",
        )
        name_to_run: dict[str, wandb.apis.public.Run] = {}
        tags_set = set(tags)
        for run in runs:
            run_tags = set(run.tags or [])
            if tags_set.issubset(run_tags):  # must have ALL tags
                if run.displayName not in name_to_run:
                    name_to_run[run.displayName] = run
        print(f"Found {len(name_to_run)} runs with tags {tags}", file=sys.stderr)
        return name_to_run

    # Name-based mode (original behaviour).
    filters = {"displayName": {"$in": run_names}}
    if tags:
        filters["tags"] = {"$in": tags}
    runs = api.runs(
        (f"{entity}/{project}" if entity else project),
        filters=filters,
        order="-created_at",
    )
    name_to_run = {}
    for run in runs:
        if run.displayName not in name_to_run:
            name_to_run[run.displayName] = run
    missing = set(run_names) - set(name_to_run)
    if missing:
        print(f"Warning: could not find runs: {missing}", file=sys.stderr)
    return name_to_run


def _load_artifact(run: wandb.apis.public.Run) -> list[dict]:
    """Download and parse the raw-metrics artifact for a run."""
    artifacts = run.logged_artifacts()
    raw_artifact = None
    for art in artifacts:
        if art.type == "raw_metrics":
            raw_artifact = art
            break
    if raw_artifact is None:
        raise ValueError(f"No raw_metrics artifact found for run '{run.name}'")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_artifact.download(root=tmpdir)
        metrics_path = Path(tmpdir) / "metrics.json"
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_series(
    logs: list[dict],
    x_col: str,
    metric_col: str,
) -> dict[tuple, pd.Series]:
    """Build per-(fold, task) series of metric vs x.

    Returns a dict mapping (fold_idx, task_idx) -> pd.Series indexed by x values.
    """
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
        reindexed = s.reindex(
            pd.Index(x_grid).union(s.index)
        ).sort_index().ffill()
        reindexed = reindexed.reindex(x_grid)
        rows.append(reindexed.values)
    return np.array(rows, dtype=float)


def _ci(data: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) confidence interval bounds across axis 0.

    Uses the t-distribution; gracefully handles all-NaN columns.
    """
    n = np.sum(~np.isnan(data), axis=0)
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    alpha = 1.0 - level
    t_crit = st.t.ppf(1.0 - alpha / 2, df=np.maximum(n - 1, 1))
    margin = t_crit * sem
    return mean - margin, mean + margin


def _std_band(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) as mean ± 1 std across axis 0."""
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1)
    return mean - std, mean + std


def _band(data: np.ndarray, band_type: str, ci_level: float) -> tuple[np.ndarray, np.ndarray]:
    if band_type == "std":
        return _std_band(data)
    return _bootstrap_ci(data, ci_level)


def _bootstrap_ci(
    data: np.ndarray, level: float, n_bootstrap: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap percentile CI for the mean across axis 0.

    Preferred over the t-distribution for ordinal (rank) data, which is
    discrete and bounded and therefore not normally distributed.
    """
    rng = np.random.default_rng(42)
    n = data.shape[0]
    boots = np.array([
        np.nanmean(data[rng.integers(0, n, size=n)], axis=0)
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - level) / 2
    return np.nanquantile(boots, alpha, axis=0), np.nanquantile(boots, 1.0 - alpha, axis=0)


def _rank_band(
    data: np.ndarray, band_type: str, ci_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Band for rank data: bootstrap CI (not t-distribution) or ±1 std."""
    if band_type == "std":
        return _std_band(data)
    return _bootstrap_ci(data, ci_level)


# ---------------------------------------------------------------------------
# Rank computation
# ---------------------------------------------------------------------------

def _compute_ranks(
    all_series: dict[str, dict[tuple, pd.Series]],
    x_grid: np.ndarray,
    metric_col: str,
) -> dict[str, np.ndarray]:
    """Compute per-task rank of each run at every x in x_grid.

    Lower regret → better rank.  Rank 1 = best (lowest regret / highest metric).
    Returns dict mapping run_name -> (n_tasks, n_x) rank array.
    """
    run_names = list(all_series.keys())
    # Collect all (fold, task) keys
    all_keys: set[tuple] = set()
    for s_dict in all_series.values():
        all_keys.update(s_dict.keys())

    # For each run, build (n_keys, n_x) matrix on the shared x_grid
    matrices: dict[str, np.ndarray] = {}
    key_list = sorted(all_keys)
    for run_name, s_dict in all_series.items():
        rows = []
        for key in key_list:
            if key in s_dict:
                s = s_dict[key]
                reindexed = s.reindex(
                    pd.Index(x_grid).union(s.index)
                ).sort_index().ffill().reindex(x_grid)
                rows.append(reindexed.values)
            else:
                rows.append(np.full(len(x_grid), np.nan))
        matrices[run_name] = np.array(rows, dtype=float)

    # Stack into (n_runs, n_keys, n_x), rank along run dimension per (key, x)
    stacked = np.stack([matrices[r] for r in run_names], axis=0)  # (R, K, X)
    R, K, X = stacked.shape

    # Direction: lower is better for regret/loss/error metrics.
    ascending = "regret" in metric_col or "loss" in metric_col or "error" in metric_col

    # Rank with average-ties and NaN-preservation.
    # Reshape so each column of the DataFrame is one (task, x) slot; rank across rows (runs).
    flat = stacked.reshape(R, K * X)          # (R, K*X)
    ranked_df = pd.DataFrame(flat.T).rank(    # (K*X, R) → rank per row
        axis=1,
        method="average",
        ascending=ascending,   # True → rank 1 = lowest (best for regret)
        na_option="keep",
    )
    ranked = ranked_df.values.T.reshape(R, K, X)

    rank_dict: dict[str, np.ndarray] = {}
    for i, run_name in enumerate(run_names):
        rank_dict[run_name] = ranked[i]  # (n_keys, n_x)
    return rank_dict


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def _save_summary_json(
    all_series: dict[str, dict[tuple, pd.Series]],
    rank_dict: dict[str, np.ndarray],
    x_grid: np.ndarray,
    metric_col: str,
    out_path: Path,
) -> None:
    """Save mean regret and mean rank (averaged across tasks and x-axis) as JSON.

    Each entry is the nanmean over the full (n_tasks × n_x) matrix, giving a
    single comparable scalar per run.  Results are sorted by avg_rank ascending.
    """
    summary: dict[str, dict[str, float]] = {}

    for run_name, s_dict in all_series.items():
        mat = _interpolate_to_grid(s_dict, x_grid)         # (n_tasks, n_x)
        avg_metric = float(np.nanmean(mat))

        ranks = rank_dict.get(run_name)
        avg_rank = float(np.nanmean(ranks)) if ranks is not None else float("nan")

        summary[run_name] = {
            f"avg_{metric_col}": round(avg_metric, 6),
            "avg_rank": round(avg_rank, 4),
        }

    # Sort by avg_rank ascending (best first)
    summary = dict(sorted(summary.items(), key=lambda kv: kv[1]["avg_rank"]))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _compute_auc_per_series(
    series_dict: dict[tuple, pd.Series],
    x_grid: np.ndarray,
) -> dict[tuple, float]:
    """Trapezoidal AUC for each (fold, task) series on x_grid."""
    aucs: dict[tuple, float] = {}
    for key, s in series_dict.items():
        reindexed = (
            s.reindex(pd.Index(x_grid).union(s.index))
            .sort_index()
            .ffill()
            .reindex(x_grid)
        )
        vals = reindexed.values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() >= 2:
            aucs[key] = float(np.trapezoid(vals[valid], x_grid[valid]))
    return aucs


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; NaNs pass through unchanged."""
    corrected = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    valid = pvals[mask]
    n = len(valid)
    if n == 0:
        return corrected
    order = np.argsort(valid)
    adjusted = np.minimum(1.0, valid[order] * n / np.arange(1, n + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]  # enforce monotonicity
    unsorted = np.empty(n)
    unsorted[order] = adjusted
    corrected[mask] = unsorted
    return corrected


def _wilcoxon_paired_auc(
    aucs_a: dict[tuple, float],
    aucs_b: dict[tuple, float],
) -> tuple[float, float, int]:
    """Wilcoxon signed-rank test on paired per-task AUC differences.

    Returns (p_value_two_sided, mean_diff, n_pairs).
    mean_diff = mean(AUC_A - AUC_B); negative = A lower (better for regret).
    """
    common = sorted(set(aucs_a) & set(aucs_b))
    n = len(common)
    if n < 5:
        return float("nan"), float("nan"), n
    diffs = np.array([aucs_a[k] - aucs_b[k] for k in common])
    if np.all(diffs == 0):
        return 1.0, 0.0, n
    result = st.wilcoxon(diffs, alternative="two-sided")
    return float(result.pvalue), float(np.mean(diffs)), n


def _bootstrap_auc_diff(
    aucs_a: dict[tuple, float],
    aucs_b: dict[tuple, float],
    n_bootstrap: int = 10_000,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for mean(AUC_A - AUC_B).

    Returns (mean_diff, ci_lo, ci_hi).
    """
    common = sorted(set(aucs_a) & set(aucs_b))
    if len(common) < 2:
        return float("nan"), float("nan"), float("nan")
    diffs = np.array([aucs_a[k] - aucs_b[k] for k in common])
    rng = np.random.default_rng(42)
    n = len(diffs)
    boot_means = np.array([
        np.mean(diffs[rng.integers(0, n, size=n)])
        for _ in range(n_bootstrap)
    ])
    return (
        float(np.mean(diffs)),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def _per_x_wilcoxon(
    series_a: dict[tuple, pd.Series],
    series_b: dict[tuple, pd.Series],
    x_grid: np.ndarray,
    lower_is_better: bool,
) -> tuple[float, float, np.ndarray]:
    """Per-x-point Wilcoxon test with BH FDR correction.

    Returns (pct_a_better, pct_b_better, corrected_pvals).
    Percentages are fraction of valid x-points where A (or B) is
    significantly better after FDR correction.
    """
    common_keys = sorted(set(series_a) & set(series_b))
    if not common_keys:
        return float("nan"), float("nan"), np.array([])

    mat_a = _interpolate_to_grid({k: series_a[k] for k in common_keys}, x_grid)
    mat_b = _interpolate_to_grid({k: series_b[k] for k in common_keys}, x_grid)

    raw_pvals = np.full(len(x_grid), np.nan)
    mean_diffs = np.zeros(len(x_grid))

    for xi in range(len(x_grid)):
        a_vals, b_vals = mat_a[:, xi], mat_b[:, xi]
        mask = ~(np.isnan(a_vals) | np.isnan(b_vals))
        if mask.sum() < 5:
            continue
        diffs = a_vals[mask] - b_vals[mask]
        mean_diffs[xi] = float(np.mean(diffs))
        if np.all(diffs == 0):
            raw_pvals[xi] = 1.0
        else:
            raw_pvals[xi] = float(st.wilcoxon(diffs, alternative="two-sided").pvalue)

    corrected = _bh_fdr(raw_pvals)
    n_valid = float(np.sum(~np.isnan(corrected)))
    if n_valid == 0:
        return float("nan"), float("nan"), corrected

    # lower_is_better: negative mean_diff means A has lower values (better)
    if lower_is_better:
        sig_a = (corrected < 0.05) & (mean_diffs < 0)
        sig_b = (corrected < 0.05) & (mean_diffs > 0)
    else:
        sig_a = (corrected < 0.05) & (mean_diffs > 0)
        sig_b = (corrected < 0.05) & (mean_diffs < 0)

    return (
        float(np.sum(sig_a) / n_valid * 100),
        float(np.sum(sig_b) / n_valid * 100),
        corrected,
    )


def _run_statistical_tests(
    all_series: dict[str, dict[tuple, pd.Series]],
    x_grid: np.ndarray,
    metric_col: str,
    out_path: Path,
    style: dict | None = None,
) -> None:
    """Pairwise statistical tests for all run pairs; saves results to JSON.

    For each pair (A, B):
      - Wilcoxon signed-rank on paired per-task AUC values (two-sided)
      - Bootstrap 95% CI for mean(AUC_A - AUC_B)
      - Per-x Wilcoxon with BH FDR → % of x-axis where each is significantly better
    """
    style = style or {}
    lower_is_better = (
        "regret" in metric_col or "loss" in metric_col or "error" in metric_col
    )

    aucs: dict[str, dict[tuple, float]] = {
        name: _compute_auc_per_series(s_dict, x_grid)
        for name, s_dict in all_series.items()
    }

    run_names = list(all_series.keys())
    results: dict[str, dict] = {}

    for i, name_a in enumerate(run_names):
        for name_b in run_names[i + 1:]:
            pval, mean_diff, n_pairs = _wilcoxon_paired_auc(aucs[name_a], aucs[name_b])
            _, ci_lo, ci_hi = _bootstrap_auc_diff(aucs[name_a], aucs[name_b])
            pct_a, pct_b, _ = _per_x_wilcoxon(
                all_series[name_a], all_series[name_b], x_grid, lower_is_better
            )

            label_a = style.get(name_a, {}).get("label", name_a)
            label_b = style.get(name_b, {}).get("label", name_b)

            if not np.isnan(mean_diff):
                better_name = name_a if (lower_is_better == (mean_diff < 0)) else name_b
                better_label = label_a if better_name == name_a else label_b
            else:
                better_name, better_label = "unknown", "unknown"

            sig = not np.isnan(pval) and pval < 0.05
            print(
                f"  {label_a} vs {label_b}: p={pval:.4f}{'*' if sig else ' '}, "
                f"mean_diff={mean_diff:+.4f}, "
                f"better={better_label}, "
                f"pct_x: {label_a}={pct_a:.1f}% {label_b}={pct_b:.1f}%",
                file=sys.stderr,
            )

            results[f"{name_a}_vs_{name_b}"] = {
                "label_a": label_a,
                "label_b": label_b,
                "n_pairs": n_pairs,
                "mean_auc_diff (A-B)": round(mean_diff, 6) if not np.isnan(mean_diff) else None,
                "bootstrap_ci_95": [
                    round(ci_lo, 6) if not np.isnan(ci_lo) else None,
                    round(ci_hi, 6) if not np.isnan(ci_hi) else None,
                ],
                "wilcoxon_pval": round(pval, 6) if not np.isnan(pval) else None,
                "significant_p05": sig,
                "better_run": better_name,
                "better_label": better_label,
                f"pct_x_{name_a}_better": round(pct_a, 1) if not np.isnan(pct_a) else None,
                f"pct_x_{name_b}_better": round(pct_b, 1) if not np.isnan(pct_b) else None,
            }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",  # math symbols that match Times
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

_MARKER_EVERY = 0.15   # fraction of line length between markers (display-space, good for log axes)
_MARKER_SIZE = 7


def _plot_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    color: str,
    linestyle: str,
    marker: str,
    **extra,
) -> None:
    """Plot a line with optional hollow markers, using consistent style."""
    kw: dict = dict(label=label, color=color, linestyle=linestyle)
    kw.update(extra)
    if marker:
        kw.update(
            marker=marker,
            markevery=_MARKER_EVERY,
            markersize=_MARKER_SIZE,
        )
    ax.plot(x, y, **kw)


def _regret_plot(
    all_series: dict[str, dict[tuple, pd.Series]],
    x_grid: np.ndarray,
    metric_col: str,
    ci_level: float,
    out_path: Path,
    band_type: str = "ci",
    style: dict | None = None,
) -> None:
    style = style or {}
    fig, ax = plt.subplots()

    all_means = []
    for i, (run_name, s_dict) in enumerate(all_series.items()):
        mat = _interpolate_to_grid(s_dict, x_grid)
        mean = np.nanmean(mat, axis=0)
        lo, hi = _band(mat, band_type, ci_level)
        label, color, linestyle, marker = _run_style(style, run_name, i)
        _plot_line(ax, x_grid, mean, label, color, linestyle, marker)
        ax.fill_between(x_grid, lo, hi, alpha=0.2, color=color)
        all_means.append(mean)

    means_stack = np.array(all_means)
    y_min = np.nanmin(means_stack)
    y_max = np.nanmax(means_stack)
    ax.set_ylim(y_min * 0.8, y_max * 1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlabel(_x_label(x_grid))
    ax.set_ylabel(_metric_label(metric_col))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _rank_plot(
    rank_dict: dict[str, np.ndarray],
    x_grid: np.ndarray,
    ci_level: float,
    out_path: Path,
    band_type: str = "ci",
    style: dict | None = None,
) -> None:
    style = style or {}
    fig, ax = plt.subplots()

    for i, (run_name, ranks) in enumerate(rank_dict.items()):
        data = ranks.astype(float)
        mean = np.nanmean(data, axis=0)
        lo, hi = _rank_band(data, band_type, ci_level)
        label, color, linestyle, marker = _run_style(style, run_name, i)
        _plot_line(ax, x_grid, mean, label, color, linestyle, marker)
        ax.fill_between(x_grid, lo, hi, alpha=0.2, color=color)

    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_xlabel(_x_label(x_grid))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _family_regret_plot(
    family_name: str,
    foreground: list[str],
    background: list[str],
    all_series: dict[str, dict[tuple, pd.Series]],
    x_grid: np.ndarray,
    metric_col: str,
    ci_level: float,
    out_path: Path,
    band_type: str = "ci",
    style: dict | None = None,
) -> None:
    style = style or {}
    fig, ax = plt.subplots()

    # Background: naive baselines, faded lines only (no fill)
    for i, run_name in enumerate(background):
        if run_name not in all_series:
            continue
        mat = _interpolate_to_grid(all_series[run_name], x_grid)
        mean = np.nanmean(mat, axis=0)
        label, color, linestyle, marker = _run_style(style, run_name, i)
        _plot_line(ax, x_grid, mean, label, color, linestyle, marker,
                   alpha=0.6, linewidth=1.5)

    # Foreground: family methods, full style with CI bands
    all_means = []
    for i, run_name in enumerate(foreground):
        if run_name not in all_series:
            continue
        mat = _interpolate_to_grid(all_series[run_name], x_grid)
        mean = np.nanmean(mat, axis=0)
        lo, hi = _band(mat, band_type, ci_level)
        label, color, linestyle, marker = _run_style(style, run_name, len(background) + i)
        _plot_line(ax, x_grid, mean, label, color, linestyle, marker)
        ax.fill_between(x_grid, lo, hi, alpha=0.2, color=color)
        all_means.append(mean)

    if all_means:
        means_stack = np.array(all_means)
        y_min, y_max = np.nanmin(means_stack), np.nanmax(means_stack)
        ax.set_ylim(y_min * 0.8, y_max * 1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlabel(_x_label(x_grid))
    ax.set_ylabel(_metric_label(metric_col))
    ax.set_title(family_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _family_rank_plot(
    family_name: str,
    foreground: list[str],
    background: list[str],
    rank_dict: dict[str, np.ndarray],
    x_grid: np.ndarray,
    ci_level: float,
    out_path: Path,
    band_type: str = "ci",
    style: dict | None = None,
) -> None:
    style = style or {}
    fig, ax = plt.subplots()

    for i, run_name in enumerate(background):
        if run_name not in rank_dict:
            continue
        data = rank_dict[run_name].astype(float)
        mean = np.nanmean(data, axis=0)
        label, color, linestyle, marker = _run_style(style, run_name, i)
        _plot_line(ax, x_grid, mean, label, color, linestyle, marker,
                   alpha=0.35, linewidth=1.0)

    for i, run_name in enumerate(foreground):
        if run_name not in rank_dict:
            continue
        data = rank_dict[run_name].astype(float)
        mean = np.nanmean(data, axis=0)
        lo, hi = _rank_band(data, band_type, ci_level)
        label, color, linestyle, marker = _run_style(style, run_name, len(background) + i)
        _plot_line(ax, x_grid, mean, label, color, linestyle, marker)
        ax.fill_between(x_grid, lo, hi, alpha=0.2, color=color)

    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_xlabel(_x_label(x_grid))
    ax.set_title(family_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _combined_family_rank_plot(
    all_series: dict[str, dict[tuple, pd.Series]],
    rank_dict: dict[str, np.ndarray],
    x_grid: np.ndarray,
    metric_col: str,
    families: dict[str, list[str]],
    background: list[str],
    ci_level: float,
    out_path: Path,
    band_type: str = "ci",
    style: dict | None = None,
) -> None:
    """Single wide figure: one regret panel per family + overall rank, shared legend below."""
    style = style or {}
    family_items = list(families.items())   # [(name, [methods]), ...]
    n_panels = len(family_items) + 1        # +1 for rank

    rc = {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.5))

        # ── Regret panels ──────────────────────────────────────────
        for ax, (family_name, foreground) in zip(axes[:-1], family_items):
            # Background baselines (faded, no fill)
            for i, run_name in enumerate(background):
                if run_name not in all_series:
                    continue
                mat = _interpolate_to_grid(all_series[run_name], x_grid)
                mean = np.nanmean(mat, axis=0)
                label, color, linestyle, marker = _run_style(style, run_name, i)
                _plot_line(ax, x_grid, mean, label, color, linestyle, marker,
                           alpha=0.6, linewidth=1.5)

            # Foreground family methods
            all_means = []
            for i, run_name in enumerate(foreground):
                if run_name not in all_series:
                    continue
                mat = _interpolate_to_grid(all_series[run_name], x_grid)
                mean = np.nanmean(mat, axis=0)
                lo, hi = _band(mat, band_type, ci_level)
                label, color, linestyle, marker = _run_style(style, run_name, len(background) + i)
                _plot_line(ax, x_grid, mean, label, color, linestyle, marker)
                ax.fill_between(x_grid, lo, hi, alpha=0.2, color=color)
                all_means.append(mean)

            if all_means:
                ms = np.array(all_means)
                ax.set_ylim(np.nanmin(ms) * 0.8, np.nanmax(ms) * 1.2)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_xlabel(_x_label(x_grid))
            ax.set_title(family_name)

        axes[0].set_ylabel(_metric_label(metric_col))

        # ── Synchronise y-limits across regret panels ───────────────
        ylims = [ax.get_ylim() for ax in axes[:-1]]
        global_ymin = min(lo for lo, hi in ylims)
        global_ymax = max(hi for lo, hi in ylims)
        for ax in axes[:-1]:
            ax.set_ylim(global_ymin, global_ymax)
        for ax in axes[1:-1]:
            ax.tick_params(labelleft=False)

        # ── Rank panel ─────────────────────────────────────────────
        ax_rank = axes[-1]
        for i, (run_name, ranks) in enumerate(rank_dict.items()):
            data = ranks.astype(float)
            mean = np.nanmean(data, axis=0)
            lo, hi = _rank_band(data, band_type, ci_level)
            label, color, linestyle, marker = _run_style(style, run_name, i)
            _plot_line(ax_rank, x_grid, mean, label, color, linestyle, marker)
            ax_rank.fill_between(x_grid, lo, hi, alpha=0.2, color=color)

        ax_rank.set_xscale("log")
        ax_rank.grid(True, alpha=0.3, linewidth=0.5)
        ax_rank.set_ylabel("Rank (1 = best)")
        ax_rank.set_xlabel(_x_label(x_grid))
        ax_rank.set_title("Overall Rank")

        # ── Shared legend ──────────────────────────────────────────
        # Collect all handles; overwrite so the rank panel (full opacity) wins for
        # baselines. Then reorder: background methods first, family methods after.
        seen_labels: list[str] = []
        seen_handles: dict[str, object] = {}
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen_handles:
                    seen_labels.append(l)
                seen_handles[l] = h  # overwrite → rank panel wins for baselines

        background_labels = [style.get(rn, {}).get("label", rn) for rn in background]
        ordered_labels = [l for l in background_labels if l in seen_handles]
        ordered_labels += [l for l in seen_labels if l not in set(ordered_labels)]

        fig.tight_layout()
        legend = fig.legend(
            [seen_handles[l] for l in ordered_labels],
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=len(ordered_labels),
            frameon=False,
            fontsize=17,
            handlelength=2.0,
            handletextpad=0.4,
            columnspacing=0.8,
        )

        # bbox_extra_artists tells savefig to expand the canvas to include the legend
        for suffix in (".png", ".pdf"):
            p = out_path.with_suffix(suffix)
            fig.savefig(p, dpi=150, bbox_inches="tight", bbox_extra_artists=(legend,))
            print(f"Saved: {p}")
        plt.close(fig)


def _x_label(x_grid: np.ndarray) -> str:
    return "Number of evaluations"


_METRIC_LABELS: dict[str, str] = {
    "test_normalized_regret": "Normalised regret",
    "normalized_regret": "Normalised regret",
}


def _metric_label(metric_col: str) -> str:
    return _METRIC_LABELS.get(metric_col, metric_col.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate regret and rank comparison plots from W&B run artifacts."
    )
    parser.add_argument("run_names", nargs="*",
                        help="W&B run names to compare (omit to use --tags alone)")
    parser.add_argument("--project", default="simple-metalearning")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--x-axis", default="visible_num", dest="x_axis")
    parser.add_argument("--metric", default="test_normalized_regret")
    parser.add_argument("--out-dir", default="plots", dest="out_dir")
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Filter by tags.  Without run names, fetches all runs "
                             "that have ALL specified tags.")
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--band", choices=["ci", "std"], default="ci",
                        help="Band type: 'ci' (t-distribution CI) or 'std' (±1 std)")
    parser.add_argument("--style", default="conf/plot_style.yaml",
                        help="YAML file mapping run names to label/color/linestyle "
                             "(default: conf/plot_style.yaml)")
    args = parser.parse_args()

    if not args.run_names and not args.tags:
        parser.error("Provide run names and/or --tags to select runs.")

    style = _load_style(args.style)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_names:
        print(f"Fetching {len(args.run_names)} runs from {args.entity}/{args.project}...")
    else:
        print(f"Fetching all runs with tags {args.tags} from {args.entity}/{args.project}...")
    name_to_run = _find_runs(args.run_names, args.entity, args.project, tags=args.tags)
    if not name_to_run:
        print("No runs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Load artifacts and build per-task series for each run
    all_series: dict[str, dict[tuple, pd.Series]] = {}
    for run_name, run in name_to_run.items():
        print(f"Loading artifact for run '{run_name}'...")
        try:
            logs = _load_artifact(run)
            all_series[run_name] = _build_series(logs, args.x_axis, args.metric)
        except Exception as e:
            print(f"  SKIP '{run_name}' (id={run.id}): {e}", file=sys.stderr)

    # Drop excluded methods before any computation (ranking, plots, stats)
    excluded = set(style.get("exclude", []) or [])
    if excluded:
        removed = set(all_series) & excluded
        if removed:
            print(f"Excluding (from style config): {sorted(removed)}", file=sys.stderr)
        all_series = {k: v for k, v in all_series.items() if k not in excluded}

    # Filter excluded methods from families/background so no empty panels are drawn
    families, background = _families_from_style(style)
    if excluded:
        families = {
            name: [m for m in methods if m not in excluded]
            for name, methods in families.items()
        }
        families = {name: methods for name, methods in families.items() if methods}
        background = [m for m in background if m not in excluded]

    # Build shared x_grid: union of all x values across all runs/tasks
    x_vals: set[float] = set()
    for s_dict in all_series.values():
        for s in s_dict.values():
            x_vals.update(s.index.tolist())
    # if x is visible_num, exclude zero
    if args.x_axis == "visible_num":
        x_vals = {x for x in x_vals if x > 0}
    x_grid = np.sort(np.array(list(x_vals)))

    # 1. Regret plot
    _regret_plot(
        all_series,
        x_grid,
        args.metric,
        args.ci,
        out_dir / "regret_vs_evals.png",
        band_type=args.band,
        style=style,
    )

    # 2. Rank plot
    rank_dict = _compute_ranks(all_series, x_grid, args.metric)
    _rank_plot(
        rank_dict,
        x_grid,
        args.ci,
        out_dir / "rank_vs_evals.png",
        band_type=args.band,
        style=style,
    )

    # 3. Summary JSON
    _save_summary_json(all_series, rank_dict, x_grid, args.metric, out_dir / "summary.json")

    # 4. Statistical tests
    print("Running statistical tests...", file=sys.stderr)
    _run_statistical_tests(all_series, x_grid, args.metric, out_dir / "stats_tests.json", style=style)

    # 5. Per-family plots (if families defined in style)
    for family_name, foreground in families.items():
        safe = family_name.replace(" ", "_").replace("-", "_")
        _family_regret_plot(
            family_name, foreground, background,
            all_series, x_grid, args.metric, args.ci,
            out_dir / f"{safe}_regret_vs_evals.png",
            band_type=args.band, style=style,
        )
        _family_rank_plot(
            family_name, foreground, background,
            rank_dict, x_grid, args.ci,
            out_dir / f"{safe}_rank_vs_evals.png",
            band_type=args.band, style=style,
        )

    # 6. Combined wide figure (all families + rank, shared legend)
    if families:
        _combined_family_rank_plot(
            all_series, rank_dict, x_grid, args.metric,
            families, background,
            args.ci,
            out_dir / "combined_family_rank.png",
            band_type=args.band, style=style,
        )


if __name__ == "__main__":
    main()
