"""TEMPORARY first-insight plot: fair continuous-vs-discrete comparison.

The continuous-BO cluster array is still running, so estimators have uneven task
coverage and the discrete runs (on W&B) cover all tasks.  To compare apples to
apples we restrict every curve to the **intersection of (fold, task) pairs that
all selected runs have in common**, then average over only those tasks.

Continuous data is read from the local per-task JSONs in ``cluster_io/metrics``
(scp'ed off the cluster); discrete data is pulled from the W&B ``raw_metrics``
artifact of each mapped run.  ``(fold_idx, task_idx)`` is a global key shared by
both (same dataset + ``KFold(random_state=42)``), so the join is exact.

Why ABSOLUTE best-score, not normalized_regret
----------------------------------------------
``normalized_regret`` divides regret by a per-run reference span, and the
continuous runs evaluate at low fidelity (1+1 fold, 8 bag) so their raw scores
routinely *exceed* the full-fidelity discrete-matrix optimum -> negative regret
that blows up under normalization.  The continuous and discrete curves then
never share a baseline, even though both just random-sample the same space for
the first ``n_init`` iterations and should start together.  So we reconstruct
the reference-free absolute best-score

    best_score = per_task_optimum - regret        (higher = better; -err/dummy)

using the matrix optimum each run logged its regret against (the full XGBoost
matrix for both continuous and the discrete hpo_v2 runs).  This is exactly what
scripts/compare_cont_vs_subset.py did.  Pass ``--normalized`` to see the old,
non-comparable view for contrast.

This is a throwaway diagnostic — the real comparison waits until the array
finishes and ``collect_metrics.py`` uploads the continuous runs proper.

Usage
-----
    uv run python scripts/compare_continuous_discrete.py \
        --metrics-dir cluster_io/metrics --entity <your-entity> \
        --max-visible 100 --out plots/cont_vs_disc.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import wandb

_FNAME_RE = re.compile(r"^metrics_(?P<est>.+)-f(?P<fold>\d+)-(?P<task>.+)\.json$")

# continuous estimator key (as in the cluster filenames, minus the "continuous-"
# prefix) -> (display label, discrete W&B run name, colour)
DEFAULT_MAP = {
    "rf_ei":        ("RF-EI",       "RF-Hyper",    "tab:blue"),
    "gp_ei":        ("GP-EI",       "GP-Hyper",    "tab:orange"),
    "oof_ridge_ei": ("OOFRidge-EI", "OOFRidge-EI", "tab:green"),
}

# Discrete-only methods (no continuous counterpart) overlaid as extra dashed
# curves.  Linear-Hyper = ridge_ei on the raw matrix (not OOF predictions).
DISCRETE_ONLY = {
    "Linear-Hyper": ("Linear-Hyper", "tab:purple"),
}


def _series_from_logs(logs: list[dict], x_col: str, metric_col: str
                      ) -> dict[tuple, pd.Series]:
    """{(fold_idx, task_idx): Series(metric indexed by x)} from a flat log list."""
    df = pd.DataFrame(logs)
    if not {x_col, metric_col, "fold_idx", "task_idx"}.issubset(df.columns):
        return {}
    out: dict[tuple, pd.Series] = {}
    for (fold, task), g in df.groupby(["fold_idx", "task_idx"]):
        s = g.set_index(x_col)[metric_col].sort_index()
        out[(int(fold), int(task))] = s[~s.index.duplicated(keep="last")]
    return out


def load_continuous(metrics_dir: Path, x_col: str, metric_col: str
                    ) -> dict[str, dict[tuple, pd.Series]]:
    """Group local metrics_*.json by estimator -> per-(fold,task) series."""
    by_est: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(metrics_dir.glob("metrics_*.json")):
        m = _FNAME_RE.match(path.name)
        if not m:
            continue
        est = m.group("est")
        if est.startswith("continuous-"):
            est = est[len("continuous-"):]
        with open(path, encoding="utf-8") as f:
            entries = [e for e in json.load(f) if e.get("status") != "failed"]
        by_est[est].extend(entries)
    return {est: _series_from_logs(logs, x_col, metric_col)
            for est, logs in by_est.items()}


def load_discrete(run_name: str, entity: str, project: str, x_col: str,
                  metric_col: str, tag: str | None = None) -> dict[tuple, pd.Series]:
    """Pull a discrete run's raw_metrics artifact -> per-(fold,task) series.

    ``tag`` pins the fidelity: the same display name (e.g. RF-Hyper) exists at
    both full fidelity (hpo_v2) and single-fold fidelity (hpo_v2_fsub11), so we
    must filter by tag or we silently grab the wrong one.
    """
    api = wandb.Api()
    filters: dict = {"displayName": run_name}
    if tag:
        filters["tags"] = {"$in": [tag]}
    runs = api.runs((f"{entity}/{project}" if entity else project), filters=filters, order="-created_at")
    run = next(iter(runs), None)
    if run is None:
        print(f"  ! discrete run not found: {run_name}")
        return {}
    art = next((a for a in run.logged_artifacts() if a.type == "raw_metrics"), None)
    if art is None:
        print(f"  ! no raw_metrics artifact on {run_name}")
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        art.download(root=tmp)
        logs = json.load(open(Path(tmp) / "metrics.json", encoding="utf-8"))
    return _series_from_logs(logs, x_col, metric_col)


def load_matrix_stats(fold_subset: bool = False):
    """Per-task (val_opt, test_opt, val_median, test_median) from the matrix.

    ``optimum - regret`` recovers a run's raw best-score, but the optimum must
    match the matrix that run logged its regret against:

      * continuous cluster runs (and full hpo_v2) -> FULL matrix optima,
      * fidelity-matched hpo_v2_fsub11 runs        -> SUBSET (first-fold) optima.

    The medians anchor the normalization scale (see ``recompute_normalized``).
    ``fold_subset`` selects the first val + first test fold (matches
    ``n_val_folds=n_test_folds=1``).  All arrays indexed by global task_idx.
    """
    from simple_metalearning.datasets.tabarena import TabArenaDatasetFactory
    kw = dict(fold_subset_n_val=1, fold_subset_n_test=1) if fold_subset else {}
    ds = TabArenaDatasetFactory(methods="XGBoost", normalization="dummy",
                                flip_sign=True, val_test_split=True, **kw).load()
    v, t = ds.performance.to_numpy(), ds.test_performance.to_numpy()
    return (np.nanmax(v, 0), np.nanmax(t, 0),
            np.nanmedian(v, 0), np.nanmedian(t, 0))


def reconstruct_best(series: dict[tuple, pd.Series], optimum: np.ndarray
                     ) -> dict[tuple, pd.Series]:
    """best_score = per_task_optimum - regret, per (fold, task) series."""
    return {(f, t): optimum[t] - s for (f, t), s in series.items()}


def _to_grid(series: dict[tuple, pd.Series], keys: list[tuple],
             grid: np.ndarray) -> np.ndarray:
    """Forward-fill each key's series onto grid -> (n_keys, n_x) matrix."""
    rows = []
    for k in keys:
        s = series[k]
        r = s.reindex(pd.Index(grid).union(s.index)).sort_index().ffill().reindex(grid)
        rows.append(r.values)
    return np.array(rows, dtype=float)


def _bootstrap_ci(data: np.ndarray, level: float, n_boot: int = 2000):
    """Bootstrap CI of the nanmean over axis 0 (tasks).

    Narrows ~1/sqrt(n_tasks) as coverage grows, unlike a fixed ±std band.
    ``data`` is (n_tasks, n_x); returns (mean, lo, hi) each (n_x,).
    """
    rng = np.random.default_rng(42)
    n = data.shape[0]
    boots = np.empty((n_boot, data.shape[1]))
    for b in range(n_boot):
        boots[b] = np.nanmean(data[rng.integers(0, n, n)], axis=0)
    a = (1.0 - level) / 2
    return (np.nanmean(data, axis=0),
            np.nanquantile(boots, a, axis=0),
            np.nanquantile(boots, 1.0 - a, axis=0))


def recompute_normalized(raw_mats: dict[str, np.ndarray], task_of_key: np.ndarray,
                         median: np.ndarray, matrix_opt: np.ndarray
                         ) -> dict[str, np.ndarray]:
    """Fair normalized regret with a SHARED optimum across all methods.

    Per task: median comes from the matrix (scale anchor); the optimum is the
    best achieved by ANY method (continuous or discrete) on that task, or the
    matrix optimum if higher — so the winner sits at ~0 and nothing goes
    negative (the failure mode of normalizing against the matrix optimum alone,
    which continuous routinely beats).

        normalized_best = clip((best - median) / (opt - median), 0, None)
        normalized_regret = 1 - normalized_best

    ``raw_mats`` maps run -> (n_keys, n_x) best-so-far scores on the grid.
    """
    med_k = median[task_of_key]                       # (K,)
    # Shared optimum per key = best any method reached, vs the matrix optimum.
    opt_k = matrix_opt[task_of_key].astype(float).copy()
    for mat in raw_mats.values():
        opt_k = np.fmax(opt_k, np.nanmax(mat, axis=1))
    denom = opt_k - med_k
    denom[denom <= 0] = np.nan                         # degenerate task -> drop
    out = {}
    for run, mat in raw_mats.items():
        nb = np.clip((mat - med_k[:, None]) / denom[:, None], 0.0, None)
        out[run] = 1.0 - nb
    return out


def _ranks(nr_mats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per (task, x) rank of each run by normalized regret (1 = best/lowest)."""
    runs = list(nr_mats)
    stacked = np.stack([nr_mats[r] for r in runs], 0)          # (R, K, X)
    R, K, X = stacked.shape
    flat = stacked.reshape(R, K * X)
    ranked = pd.DataFrame(flat.T).rank(axis=1, method="average",
                                       ascending=True, na_option="keep").values.T
    ranked = ranked.reshape(R, K, X)
    return {r: ranked[i] for i, r in enumerate(runs)}


# Drawing order / styling shared by both panels.
def _style(run: str) -> tuple[str, str, float, object]:
    """run key -> (label, color, lw, linestyle)."""
    if run.startswith("rand:"):
        return "Naive-Random", "dimgray", 1.4, (0, (1, 1))
    if run.startswith("donly:"):
        name = run.split(":", 1)[1]
        label, color = DISCRETE_ONLY[name]
        return f"{label} (discrete)", color, 1.6, "--"
    kind, est = run.split(":", 1)
    label, disc_run, color = DEFAULT_MAP[est]
    if kind == "cont":
        return f"{label} (continuous)", color, 2.0, "solid"
    return f"{label} (discrete: {disc_run})", color, 1.6, "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-dir", default="cluster_io/metrics", type=Path)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--project", default="simple-metalearning")
    ap.add_argument("--view", default="test", choices=["test", "val"],
                    help="Which side to normalize/plot (default test).")
    ap.add_argument("--absolute", action="store_true",
                    help="Plot reconstructed absolute best-score instead of the "
                         "fair normalized regret (single panel).")
    ap.add_argument("--discrete-tag", default="hpo_v2_fsub11",
                    help="W&B tag pinning discrete fidelity. Default hpo_v2_fsub11 "
                         "(single-fold, matches continuous). Use hpo_v2 for the "
                         "full-fidelity (non-comparable) discrete runs.")
    ap.add_argument("--add-random", action="store_true", default=True,
                    help="Overlay Naive-Random (same tag) as a baseline anchor.")
    ap.add_argument("--x-axis", default="visible_num")
    ap.add_argument("--max-visible", default=100, type=int)
    ap.add_argument("--ci", default=0.95, type=float)
    ap.add_argument("--out", default="plots/cont_vs_disc.png", type=Path)
    ap.add_argument("--no-discrete", action="store_true",
                    help="Only the continuous estimators (skip W&B pulls).")
    args = ap.parse_args()

    grid = np.arange(1, args.max_visible + 1)  # metric is NaN at visible_num=0
    disc_fold_subset = args.discrete_tag.endswith("fsub11")

    # We always reconstruct absolute best-score from regret first (it is the
    # comparable quantity); normalized regret is then recomputed from it with a
    # shared optimum.  So both modes read the raw regret column.
    col = "test_regret" if args.view == "test" else "regret"
    cv, ct, cvm, ctm = load_matrix_stats(fold_subset=False)        # continuous ref
    cont_opt = ct if args.view == "test" else cv
    if disc_fold_subset:
        dv, dt, dvm, dtm = load_matrix_stats(fold_subset=True)     # fsub11 ref
        disc_opt = dt if args.view == "test" else dv
        median = dtm if args.view == "test" else dvm               # matrix median
        matrix_opt = disc_opt
    else:
        disc_opt, median, matrix_opt = cont_opt, (ctm if args.view == "test" else cvm), cont_opt

    cont = load_continuous(args.metrics_dir, args.x_axis, col)
    ests = [e for e in DEFAULT_MAP if e in cont and cont[e]]

    collections: dict[str, dict[tuple, pd.Series]] = {}
    for est in ests:
        collections[f"cont:{est}"] = cont[est]
    if not args.no_discrete:
        for est in ests:
            _, disc_run, _ = DEFAULT_MAP[est]
            print(f"Fetching discrete '{disc_run}' [{args.discrete_tag}] for {est}...")
            d = load_discrete(disc_run, args.entity, args.project, args.x_axis,
                              col, tag=args.discrete_tag)
            if d:
                collections[f"disc:{est}"] = d
        for run_name in DISCRETE_ONLY:
            print(f"Fetching discrete-only '{run_name}' [{args.discrete_tag}]...")
            d = load_discrete(run_name, args.entity, args.project, args.x_axis,
                              col, tag=args.discrete_tag)
            if d:
                collections[f"donly:{run_name}"] = d
        if args.add_random:
            r = load_discrete("Naive-Random", args.entity, args.project,
                              args.x_axis, col, tag=args.discrete_tag)
            if r:
                collections["rand:Naive-Random"] = r

    # Reconstruct raw best-score against each run's own logged optimum.
    collections = {
        k: reconstruct_best(v, cont_opt if k.startswith("cont:") else disc_opt)
        for k, v in collections.items()
    }

    common = None
    for s in collections.values():
        common = set(s) if common is None else (common & set(s))
    common = sorted(common or [])
    print(f"\nCommon (fold,task) pairs across all {len(collections)} curves: "
          f"{len(common)}")
    if not common:
        raise SystemExit("No tasks common to all runs — cannot compare fairly.")
    print("  by fold:", pd.Series([f for f, _ in common]).value_counts().sort_index().to_dict())

    task_of_key = np.array([t for _, t in common])
    raw_mats = {run: _to_grid(s, common, grid) for run, s in collections.items()}
    order = [f"cont:{e}" for e in ests] + \
            [f"disc:{e}" for e in ests if f"disc:{e}" in raw_mats] + \
            [f"donly:{r}" for r in DISCRETE_ONLY if f"donly:{r}" in raw_mats] + \
            (["rand:Naive-Random"] if "rand:Naive-Random" in raw_mats else [])

    if args.absolute:
        fig, ax_r = plt.subplots(figsize=(8, 5.5))
        axes = [ax_r]
        plot_mats = raw_mats
        ax_r.axhline(float(np.nanmean(matrix_opt[task_of_key])), color="black",
                     lw=0.8, ls=":", label=f"matrix optimum ({args.discrete_tag})")
        ax_r.set_ylabel(f"{args.view} best score  (-err/dummy, higher = better)")
    else:
        plot_mats = recompute_normalized(raw_mats, task_of_key, median, matrix_opt)
        fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(13, 5.5))
        axes = [ax_r, ax_k]
        # Rank panel (plot_comparison style): mean rank + bootstrap CI, log-x.
        rank_mats = _ranks(plot_mats)
        for run in order:
            label, color, lw, ls = _style(run)
            m, lo, hi = _bootstrap_ci(rank_mats[run], args.ci)
            ax_k.plot(grid, m, color=color, lw=lw, ls=ls, label=label)
            ax_k.fill_between(grid, lo, hi, color=color, alpha=0.10)
        ax_k.set_xscale("log")
        ax_k.set_xlabel(args.x_axis)
        ax_k.set_ylabel("mean rank (1 = best)")
        ax_k.set_title("Rank across methods")
        ax_k.grid(alpha=0.3, which="both")
        ax_r.set_ylabel(f"{args.view} normalized regret  (shared optimum)")
        ax_r.set_yscale("log")

    # Regret / best-score panel: mean + bootstrap CI, log-x (plot_comparison style).
    for run in order:
        label, color, lw, ls = _style(run)
        m, lo, hi = _bootstrap_ci(plot_mats[run], args.ci)
        ax_r.plot(grid, m, color=color, lw=lw, ls=ls, label=label)
        ax_r.fill_between(grid, lo, hi, color=color, alpha=0.12)
    ax_r.set_xscale("log")
    ax_r.set_xlabel(args.x_axis)
    ax_r.set_title("Continuous vs discrete  "
                   f"({len(common)} common tasks, {args.discrete_tag})")
    ax_r.legend(fontsize=7, ncol=1)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
