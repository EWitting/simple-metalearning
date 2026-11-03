"""Scalability AUC plot: training-task limit vs average test-normalised regret.

For each method with TL variants (scalability runs), plots one curve over TL
values [1,5,10,20,30,40] where each point is the mean AUC of test_normalized_regret
over the 200-step BO horizon, divided by the horizon (= time-averaged regret).
CI bands are computed across (fold, task) pairs.

Methods without TL variants (RF-Hyper, Linear-Hyper) are fetched from the CASH
experiment and shown as a horizontal reference line + CI band.

TwoTower-Hyper has TL variants (scalability runs) so it gets a full curve.

Outputs (to --out-dir, default plots/scalability_auc/):
  {Family}_auc.png  — per-family subplot (Linear, RF, TwoTower, Naive)
  all_methods_auc.png — combined grid

Usage
-----
    python scripts/plot_scalability_auc.py [options]

Options
-------
--project   W&B project  (default: simple-metalearning)
--entity    W&B entity   (default: $WANDB_ENTITY env var, or your default W&B entity)
--out-dir   Output directory (default: plots/scalability_auc)
--metric    Metric column (default: test_normalized_regret)
--x-axis    X-axis column (default: visible_num)
--ci        Confidence interval level (default: 0.95)
--band      'ci' or 'std' (default: ci)
--style     Style YAML file (default: conf/plot_style.yaml)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as st
import wandb
import yaml


# ---------------------------------------------------------------------------
# Hardcoded experiment config
# ---------------------------------------------------------------------------

TL_VALUES = [1, 2, 4, 8, 16, 32]

# Methods that have per-TL scalability runs
SCALABILITY_METHODS = [
    "Naive-Random",
    "Naive-Mean",
    "Naive-Mean-MMR",
    "Linear-Perf",
    "RF-Perf",
    "GP-Perf",
    "TwoTower-Perf",
    "TwoTower-Hyper",
]

# Methods drawn from the CASH experiment as horizontal reference lines
CASH_REFERENCE_METHODS = ["Linear-Hyper", "RF-Hyper", "GP-Hyper"]

# Per-family groupings for family plots: {family_title: [method_names]}
FAMILIES: dict[str, list[str]] = {
    "Linear": ["Linear-Perf", "Linear-Hyper"],
    "Gaussian Process": ["GP-Perf", "GP-Hyper"],
    "Random Forest": ["RF-Perf", "RF-Hyper"],
    "Two-Tower": ["TwoTower-Perf", "TwoTower-Hyper"],
    "Naive": ["Naive-Random", "Naive-Mean", "Naive-Mean-MMR"],
}


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


COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _method_style(style: dict, method: str, index: int) -> tuple[str, str, str, str]:
    entry = style.get(method, {})
    label = entry.get("label", method)
    color = entry.get("color", COLORS[index % len(COLORS)])
    linestyle = entry.get("linestyle", "solid")
    marker = entry.get("marker", "")
    return label, color, linestyle, marker


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _find_runs_by_tags(tags: list[str], entity: str, project: str) -> list:
    api = wandb.Api()
    runs = api.runs((f"{entity}/{project}" if entity else project), filters={"tags": {"$in": tags}}, order="-created_at")
    tags_set = set(tags)
    matched, seen = [], set()
    for run in runs:
        if tags_set.issubset(set(run.tags or [])) and run.displayName not in seen:
            matched.append(run)
            seen.add(run.displayName)
    print(f"Found {len(matched)} runs with tags {tags}", file=sys.stderr)
    return matched


def _load_artifact(run) -> list[dict]:
    for art in run.logged_artifacts():
        if art.type == "raw_metrics":
            with tempfile.TemporaryDirectory() as tmpdir:
                art.download(root=tmpdir)
                with open(Path(tmpdir) / "metrics.json", encoding="utf-8") as f:
                    return json.load(f)
    raise ValueError(f"No raw_metrics artifact for run '{run.displayName}'")


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------

def _auc_per_task(
    logs: list[dict],
    x_col: str,
    metric_col: str,
) -> np.ndarray:
    """Return array of time-averaged metric values, one per (fold, task) pair.

    AUC is computed via the trapezoidal rule and divided by the x-range to give
    the average metric value over the BO horizon (unit-comparable across methods).
    """
    df = pd.DataFrame(logs)
    required = {x_col, metric_col, "fold_idx", "task_idx"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Columns missing from artifact: {missing}")

    aucs = []
    for _, group in df.groupby(["fold_idx", "task_idx"]):
        s = group.set_index(x_col)[metric_col].sort_index()
        s = s[~s.index.duplicated(keep="last")].dropna()
        if len(s) >= 2:
            x = s.index.values.astype(float)
            y = s.values.astype(float)
            auc = np.trapezoid(y, x) / (x[-1] - x[0])
            aucs.append(auc)
    return np.array(aucs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_scalability_data(
    entity: str,
    project: str,
    x_col: str,
    metric: str,
    scalability_tags: list[str] | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Fetch scalability runs and return {method: {tl: auc_array}}."""
    tags = scalability_tags if scalability_tags is not None else ["scalability"]
    runs = _find_runs_by_tags(tags, entity, project)

    data: dict[str, dict[int, np.ndarray]] = {}
    for run in runs:
        name = run.displayName
        # Parse method and TL from name like "RF-Perf-TL10"
        if "-TL" not in name:
            continue
        sep_idx = name.rfind("-TL")
        method = name[:sep_idx]
        try:
            tl = int(name[sep_idx + 3:])
        except ValueError:
            continue
        if method not in SCALABILITY_METHODS:
            continue
        if tl not in TL_VALUES:
            continue
        print(f"  Loading {name}...")
        try:
            logs = _load_artifact(run)
            aucs = _auc_per_task(logs, x_col, metric)
            data.setdefault(method, {})[tl] = aucs
        except (ValueError, KeyError) as e:
            print(f"  Skipping {name}: {e}", file=sys.stderr)
    return data


def _load_cash_reference_data(
    entity: str,
    project: str,
    x_col: str,
    metric: str,
    reference_tags: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Fetch reference runs (CASH or HPO) for non-scalability methods."""
    tags = reference_tags if reference_tags is not None else ["cash"]
    runs = _find_runs_by_tags(tags, entity, project)

    data: dict[str, np.ndarray] = {}
    for run in runs:
        name = run.displayName
        if name not in CASH_REFERENCE_METHODS:
            continue
        if name in data:
            continue  # take the most recent (runs are ordered by -created_at)
        print(f"  Loading CASH reference {name}...")
        try:
            logs = _load_artifact(run)
            data[name] = _auc_per_task(logs, x_col, metric)
        except (ValueError, KeyError) as e:
            print(f"  Skipping {name}: {e}", file=sys.stderr)
    return data


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean_ci(aucs: np.ndarray, ci_level: float) -> tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci)."""
    n = len(aucs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(aucs))
    if n == 1:
        return mean, mean, mean
    sem = float(np.std(aucs, ddof=1) / np.sqrt(n))
    t = float(st.t.ppf(1.0 - (1.0 - ci_level) / 2, df=n - 1))
    return mean, mean - t * sem, mean + t * sem


def _mean_std(aucs: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(aucs))
    std = float(np.std(aucs, ddof=1))
    return mean, mean - std, mean + std


def _summarise(aucs: np.ndarray, band: str, ci: float) -> tuple[float, float, float]:
    return _mean_std(aucs) if band == "std" else _mean_ci(aucs, ci)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def _plot_tl_curve(
    ax,
    method: str,
    tl_data: dict[int, np.ndarray],
    band: str,
    ci: float,
    label: str,
    color: str,
    linestyle: str,
    marker: str = "o",
) -> None:
    tls = sorted(tl_data.keys())
    means, los, his = [], [], []
    for tl in tls:
        m, lo, hi = _summarise(tl_data[tl], band, ci)
        means.append(m)
        los.append(lo)
        his.append(hi)
    ax.plot(tls, means, label=label, color=color, linestyle=linestyle, marker=marker, markersize=7)
    ax.fill_between(tls, los, his, alpha=0.2, color=color)


def _plot_reference_line(
    ax,
    aucs: np.ndarray,
    tl_range: list[int],
    band: str,
    ci: float,
    label: str,
    color: str,
    linestyle: str,
) -> None:
    """Draw a horizontal mean line + CI band spanning the TL x-axis range."""
    mean, lo, hi = _summarise(aucs, band, ci)
    x = [tl_range[0], tl_range[-1]]
    ax.plot(x, [mean, mean], label=label, color=color,
            linestyle="dashed", linewidth=1.5)
    ax.fill_between(x, [lo, lo], [hi, hi], alpha=0.15, color=color)


def _finish_ax(ax, ylabel: str, tl_values: list[int]) -> None:
    ax.set_xlabel("# Training Tasks")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(tl_values)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend()


# ---------------------------------------------------------------------------
# Per-family plot
# ---------------------------------------------------------------------------

def _family_plot(
    family: str,
    methods: list[str],
    scalability_data: dict[str, dict[int, np.ndarray]],
    cash_data: dict[str, np.ndarray],
    tl_values: list[int],
    metric_col: str,
    band: str,
    ci: float,
    style: dict,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ylabel = "Avg. " + metric_col.replace("_", " ").title()

    for i, method in enumerate(methods):
        label, color, linestyle, marker = _method_style(style, method, i)
        if method in scalability_data:
            _plot_tl_curve(ax, method, scalability_data[method], band, ci, label, color, linestyle, marker)
        elif method in cash_data:
            _plot_reference_line(ax, cash_data[method], tl_values, band, ci, label, color, linestyle)
        else:
            print(f"  No data for {method} in family '{family}', skipping.", file=sys.stderr)

    _finish_ax(ax, ylabel, tl_values)
    ax.set_title(family)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Combined overview plot (one row, shared legend, log y-axis)
# ---------------------------------------------------------------------------

def _combined_overview_plot(
    scalability_data: dict[str, dict[int, np.ndarray]],
    cash_data: dict[str, np.ndarray],
    tl_values: list[int],
    metric_col: str,
    band: str,
    ci: float,
    style: dict,
    out_path: Path,
    families: dict[str, list[str]] | None = None,
) -> None:
    """Four family panels in one row with shared legend below and log y-axis."""
    family_items = list((families if families is not None else FAMILIES).items())
    n_panels = len(family_items)
    ylabel = "Avg. " + metric_col.replace("_", " ").title()

    rc = {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.5), sharey=True)

        for ax, (family, methods) in zip(axes, family_items):
            for i, method in enumerate(methods):
                label, color, linestyle, marker = _method_style(style, method, i)
                if method in scalability_data:
                    _plot_tl_curve(ax, method, scalability_data[method],
                                   band, ci, label, color, linestyle, marker)
                elif method in cash_data:
                    _plot_reference_line(ax, cash_data[method], tl_values,
                                        band, ci, label, color, linestyle)
                else:
                    print(f"  No data for {method} in '{family}', skipping.",
                          file=sys.stderr)

            _finish_ax(ax, ylabel, tl_values)
            ax.get_legend().remove()
            ax.set_yscale("log")
            ax.set_title(family)

        # Only leftmost panel gets the y-axis label
        for ax in axes[1:]:
            ax.set_ylabel("")

        # Shared legend: all methods, baselines first, then family methods
        seen_labels: list[str] = []
        seen_handles: dict[str, object] = {}
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen_handles:
                    seen_labels.append(l)
                seen_handles[l] = h  # overwrite → last panel wins

        background = list(style.get("background_methods", []))
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

        for suffix in (".png", ".pdf"):
            p = out_path.with_suffix(suffix)
            fig.savefig(p, dpi=150, bbox_inches="tight", bbox_extra_artists=(legend,))
            print(f"Saved: {p}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Combined grid plot
# ---------------------------------------------------------------------------

def _combined_plot(
    scalability_data: dict[str, dict[int, np.ndarray]],
    cash_data: dict[str, np.ndarray],
    tl_values: list[int],
    metric_col: str,
    band: str,
    ci: float,
    style: dict,
    out_path: Path,
    families: dict[str, list[str]] | None = None,
) -> None:
    families = list((families if families is not None else FAMILIES).items())
    ncols = min(2, len(families))
    nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False, sharey=False)
    ylabel = "Avg. " + metric_col.replace("_", " ").title()

    for idx, (family, methods) in enumerate(families):
        ax = axes[idx // ncols][idx % ncols]
        for i, method in enumerate(methods):
            label, color, linestyle, marker = _method_style(style, method, i)
            if method in scalability_data:
                _plot_tl_curve(ax, method, scalability_data[method], band, ci, label, color, linestyle, marker)
            elif method in cash_data:
                _plot_reference_line(ax, cash_data[method], tl_values, band, ci, label, color, linestyle)
        _finish_ax(ax, ylabel, tl_values)
        ax.set_title(family)

    for idx in range(len(families), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Scalability: avg. regret over BO horizon vs training-task limit", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AUC of test_normalized_regret vs training-task limit."
    )
    parser.add_argument("--project", default="simple-metalearning")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--metric", default="test_normalized_regret")
    parser.add_argument("--x-axis", default="visible_num", dest="x_axis")
    parser.add_argument("--out-dir", default="plots/scalability_auc", dest="out_dir")
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--band", choices=["ci", "std"], default="ci")
    parser.add_argument("--style", default="conf/plot_style.yaml")
    parser.add_argument(
        "--scalability-tags",
        nargs="+",
        default=["scalability"],
        dest="scalability_tags",
        help="Tags that scalability runs must have (default: scalability)",
    )
    parser.add_argument(
        "--reference-tags",
        nargs="+",
        default=["cash"],
        dest="reference_tags",
        help="Tags that reference runs must have (default: cash)",
    )
    args = parser.parse_args()

    style = _load_style(args.style)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scalability runs...")
    scalability_data = _load_scalability_data(
        args.entity, args.project, args.x_axis, args.metric, args.scalability_tags
    )

    print(f"Loading reference runs (tags: {args.reference_tags})...")
    cash_data = _load_cash_reference_data(
        args.entity, args.project, args.x_axis, args.metric, args.reference_tags
    )

    # Drop excluded methods from all data and family groupings
    excluded = set(style.get("exclude", []) or [])
    if excluded:
        removed_s = set(scalability_data) & excluded
        removed_c = set(cash_data) & excluded
        if removed_s or removed_c:
            print(f"Excluding (from style config): {sorted(removed_s | removed_c)}", file=sys.stderr)
        scalability_data = {k: v for k, v in scalability_data.items() if k not in excluded}
        cash_data = {k: v for k, v in cash_data.items() if k not in excluded}
    active_families = {
        fam: [m for m in methods if m not in excluded]
        for fam, methods in FAMILIES.items()
    }
    active_families = {fam: ms for fam, ms in active_families.items() if ms}

    # Collect the actual TL values present in the data
    present_tls = sorted({tl for m in scalability_data.values() for tl in m})
    tl_values = present_tls if present_tls else TL_VALUES

    # Per-family plots
    for family, methods in active_families.items():
        safe = family.replace(" ", "_").replace("-", "_")
        _family_plot(
            family, methods,
            scalability_data, cash_data,
            tl_values, args.metric, args.band, args.ci, style,
            out_dir / f"{safe}_auc.png",
        )

    # Combined grid
    _combined_plot(
        scalability_data, cash_data,
        tl_values, args.metric, args.band, args.ci, style,
        out_dir / "all_methods_auc.png",
        families=active_families,
    )

    # Combined overview (one row, shared legend, log y)
    _combined_overview_plot(
        scalability_data, cash_data,
        tl_values, args.metric, args.band, args.ci, style,
        out_dir / "combined_overview.png",
        families=active_families,
    )


if __name__ == "__main__":
    main()
