"""Scalability AUC (v3): training-task limit vs time-averaged test-normalized regret.

Two data sources, same plotting style and output format either way:

* ``run``       -- local (no W&B), simulated fresh against the CORRECT
                   flip_sign_v2 matrices with the linear estimator's cold-start
                   default. Use for methods without a W&B scalability runner.
* ``run-wandb`` -- pulls pre-run scalability sweeps from W&B (tags
                   scalability_v2 / scalability-hpo_v2, produced by
                   experiments/scalability_{cash,hpo}.{sh,bat}) plus flat
                   reference lines from the main comparison runs (tags
                   cash_v2 / hpo_v2). Reproducing via W&B is fine as long as
                   the runner and this script agree on tags -- see
                   experiments/scalability_{cash,hpo}.{sh,bat} for the
                   producing side.

Either way, ``plot`` reads the same on-disk .npz format, so it needs no
changes regardless of which source was used to populate plots/scalability_auc_v3/.

Conventions
-----------
* Methods that use prior tasks (meta-learning) get a CURVE over #training tasks.
* Methods that do not (the *-Hyper baselines: they fit per-task on revealed
  points only) get a flat REFERENCE line at their full-data AUC.
* ABLR always uses prior tasks -> its own single-curve family panel
  (Meta-learning, alongside Active-Testing).
* RGPE-Oracle is intentionally not covered here: excluded from the main
  paper's Figure 3, referenced only as a small ablation side note.
* Skipped in local mode (cost): full GP-based RGPE everywhere; TwoTower
  everywhere; GP-Perf / GP-Hyper on CASH.

Metric matches scripts/plot_scalability_auc.py: per (fold, task) trapezoidal AUC
of test_normalized_regret over visible_num, divided by the x-range (time-average).

Usage
-----
    python scripts/scalability_auc_v3.py run        --bench hpo  [--methods all|name,name]
    python scripts/scalability_auc_v3.py run-wandb  --bench hpo  [--methods all|name,name] [--entity ... --project ...]
    python scripts/scalability_auc_v3.py plot       --bench hpo
"""
import argparse
import copy
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as st
import wandb
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from simple_metalearning.experiments.bo_selection import BOSelection
from simple_metalearning.estimators.bayesian_ridge_ei import BayesianRidgeEIEstimator
from simple_metalearning.estimators.task_only_rf_ei import TaskOnlyRandomForestEIEstimator
from simple_metalearning.estimators.task_only_botorch_gp_ei import TaskOnlyBoTorchGPEIEstimator
from simple_metalearning.estimators.ablr import ABLREstimator
from simple_metalearning.estimators.mean import Mean
from simple_metalearning.estimators.mean_mmr import MeanMMR
from simple_metalearning.estimators.random import Random
from simple_metalearning.estimators.active_testing import ActiveTestingEstimator
from simple_metalearning.estimators.acquisition import EIGaussianAcquisition, EIForestAcquisition, GreedyAcquisition
from simple_metalearning.datasets.metafeature_transforms import (
    StandardScalerTransform, CompositeTransform, PerformanceVectorTransform,
)

MAX_VISIBLE = 200
N_FOLDS = 5
RANDOM_STATE = 42
N_JOBS = 8
TLS = [1, 2, 4, 8, 16, 32]
OUT = Path("plots/scalability_auc_v3")

SRC = {
    "cash": "data/processed/tabarena_full_dummy_hash_e18a963c18_flip_sign_v2_valsplit.pkl",
    "hpo":  "data/processed/tabarena_full_dummy_XGBoost_flip_sign_v2_valsplit.pkl",
}

# W&B tags: must match experiments/scalability_{cash,hpo}.{sh,bat} (scalability
# sweeps, -TL{N} run names) and experiments/{cash,hpo}.{sh,bat} (main
# comparison, used for the flat *-Hyper reference lines).
SCALABILITY_TAGS = {"cash": "scalability_v2", "hpo": "scalability-hpo_v2"}
REFERENCE_TAGS = {"cash": "cash_v2", "hpo": "hpo_v2"}
WANDB_PROJECT = "simple-metalearning"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")


def _eig():
    return EIGaussianAcquisition(xi=0.0)


def _eif():
    return EIForestAcquisition(xi=0.0)


def _pvt():
    return PerformanceVectorTransform(n_components=None, impute_strategy="mean")


# kind: "curve" (uses prior tasks) or "ref" (flat reference line).
# benches: None = both, else set of allowed benches.
METHODS = {
    "Naive-Random":   dict(kind="curve", family="Naive", make=lambda d: Random(d, random_state=RANDOM_STATE)),
    "Naive-Mean":     dict(kind="curve", family="Naive", make=lambda d: Mean(d)),
    "Naive-Mean-MMR": dict(kind="curve", family="Naive", make=lambda d: MeanMMR(d, use_mmr=True, mmr_lambda=0.05)),
    "Linear-Perf":    dict(kind="curve", family="Linear", make=lambda d: BayesianRidgeEIEstimator(
        d, fit_intercept=False, use_metadata=False, prior_equal_weight=False,
        prior_shape=2.0, f_star_from_predictions=False, acquisition_fn=_eig())),
    "Linear-Hyper":   dict(kind="ref", family="Linear", make=lambda d: BayesianRidgeEIEstimator(
        d, fit_intercept=False, use_metadata=True, prior_equal_weight=False,
        prior_shape=2.0, f_star_from_predictions=False, acquisition_fn=_eig())),
    "RF-Perf":        dict(kind="curve", family="Random Forest", make=lambda d: TaskOnlyRandomForestEIEstimator(
        d, rf_model=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        performance_transform=_pvt(), use_metadata=False, f_star_from_predictions=False,
        acquisition_fn=_eif())),
    "RF-Hyper":       dict(kind="ref", family="Random Forest", make=lambda d: TaskOnlyRandomForestEIEstimator(
        d, rf_model=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, min_samples_leaf=3),
        use_metadata=True, f_star_from_predictions=False, acquisition_fn=_eif())),
    "GP-Perf":        dict(kind="curve", family="Gaussian Process", benches={"hpo"}, make=lambda d: TaskOnlyBoTorchGPEIEstimator(
        d, n_restarts=0, f_star_from_predictions=False, use_metadata=False,
        normalize_inputs=True, standardize_outputs=True, performance_transform=_pvt(),
        acquisition_fn=_eig())),
    "GP-Hyper":       dict(kind="ref", family="Gaussian Process", benches={"hpo"}, make=lambda d: TaskOnlyBoTorchGPEIEstimator(
        d, n_restarts=0, f_star_from_predictions=False, use_metadata=True,
        normalize_inputs=True, standardize_outputs=True, acquisition_fn=_eig())),
    "ABLR":           dict(kind="curve", family="Meta-learning", make=lambda d: ABLREstimator(
        d, feature_dim=50, hidden=[50, 50], n_epochs=300, lr=0.01, use_metadata=True,
        random_state=RANDOM_STATE, acquisition_fn=_eig())),
    "Active-Testing": dict(kind="curve", family="Meta-learning", make=lambda d: ActiveTestingEstimator(
        d, sim_sharpness=1.0, acquisition_fn=GreedyAcquisition())),
}

# Family panel composition for plotting (order matters within a panel).
# RGPE-Oracle is intentionally not covered here: it's excluded from the main
# paper's Figure 3, referenced only as a small ablation side note in the
# Discussion/supplement.
FAMILIES = {
    "Linear": ["Linear-Perf", "Linear-Hyper"],
    "Random Forest": ["RF-Perf", "RF-Hyper"],
    "Gaussian Process": ["GP-Perf", "GP-Hyper"],
    "Model-free": ["Naive-Random", "Naive-Mean", "Naive-Mean-MMR"],
    "Meta-learning": ["ABLR", "Active-Testing"],
}
BACKGROUND = ["Naive-Random", "Naive-Mean", "Naive-Mean-MMR"]


def _load_ds(bench):
    ds = pickle.load(open(SRC[bench], "rb"))
    ds.flow_metadata = CompositeTransform([StandardScalerTransform(fill_na=True)]).fit_transform(ds.flow_metadata)
    ds.task_metadata = CompositeTransform([StandardScalerTransform(fill_na=True)]).fit_transform(ds.task_metadata)
    return ds


def _auc_one(task_logs):
    df = pd.DataFrame(task_logs)
    if "test_normalized_regret" not in df:
        return np.nan
    s = df.set_index("visible_num")["test_normalized_regret"].sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna()
    if len(s) >= 2:
        x = s.index.values.astype(float)
        y = s.values.astype(float)
        return float(np.trapezoid(y, x) / (x[-1] - x[0]))
    return np.nan


def _run_one_config(ds, make, tl, n_jobs):
    n_tasks = ds.performance.shape[1]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    jobs = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_tasks))):
        if tl is not None and len(train_idx) > tl:
            rngl = np.random.default_rng(RANDOM_STATE + fold_idx)
            train_idx = np.sort(rngl.choice(train_idx, size=tl, replace=False))
        masked = copy.deepcopy(ds)
        non = np.setdiff1d(np.arange(n_tasks), train_idx)
        masked.performance.iloc[:, non] = np.nan
        est = make(masked)
        est.fit()
        for t in test_idx:
            jobs.append((fold_idx, int(t), est))
    exp = BOSelection(max_visible=MAX_VISIBLE, n_folds=N_FOLDS, n_folds_used=N_FOLDS,
                      random_state=RANDOM_STATE, n_jobs=1)
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(exp._run_task)(f, t, e, ds, {}, 0) for f, t, e in jobs
    )
    aucs = np.array([a for a in (_auc_one(tl_logs) for tl_logs in res) if not np.isnan(a)])
    return aucs


def run(bench, method_names, n_jobs):
    ds = _load_ds(bench)
    print(f"{bench}: matrix {ds.performance.shape}")
    outd = OUT / bench
    outd.mkdir(parents=True, exist_ok=True)
    import time
    for name in method_names:
        spec = METHODS[name]
        if spec.get("benches") and bench not in spec["benches"]:
            print(f"  skip {name} (not for {bench})")
            continue
        tls = TLS if spec["kind"] == "curve" else [None]
        result = {}
        for tl in tls:
            t0 = time.time()
            aucs = _run_one_config(ds, spec["make"], tl, n_jobs)
            key = str(tl) if tl is not None else "ref"
            result[key] = aucs
            print(f"  {name:14s} tl={str(tl):4s}  mean_auc*100={np.mean(aucs)*100:6.2f}  ({time.time()-t0:5.0f}s, n={len(aucs)})", flush=True)
            np.savez(outd / f"{name}.npz", **result)
    print(f"DONE {bench}")


# --------------------------------------------------------------------------
# W&B data source
# --------------------------------------------------------------------------

def _find_runs_by_tags(tags, entity, project):
    """Fetch runs carrying ALL of `tags`, deduplicated by name (most recent kept)."""
    api = wandb.Api()
    runs = api.runs((f"{entity}/{project}" if entity else project), filters={"tags": {"$in": tags}}, order="-created_at")
    tags_set = set(tags)
    by_name = {}
    for run in runs:
        if tags_set.issubset(set(run.tags or [])) and run.displayName not in by_name:
            by_name[run.displayName] = run
    print(f"Found {len(by_name)} runs with tags {tags}", file=sys.stderr)
    return by_name


def _load_artifact(run):
    for art in run.logged_artifacts():
        if art.type == "raw_metrics":
            with tempfile.TemporaryDirectory() as tmpdir:
                art.download(root=tmpdir)
                with open(Path(tmpdir) / "metrics.json", encoding="utf-8") as f:
                    return json.load(f)
    raise ValueError(f"No raw_metrics artifact for run '{run.displayName}'")


def _auc_array_from_logs(logs):
    """Same trapezoidal-AUC metric as _auc_one, but grouped over all (fold, task)
    pairs present in a single run's combined logs (matches plot_scalability_auc.py)."""
    df = pd.DataFrame(logs)
    required = {"visible_num", "test_normalized_regret", "fold_idx", "task_idx"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Columns missing from artifact: {missing}")
    aucs = []
    for _, group in df.groupby(["fold_idx", "task_idx"]):
        s = group.set_index("visible_num")["test_normalized_regret"].sort_index()
        s = s[~s.index.duplicated(keep="last")].dropna()
        if len(s) >= 2:
            x = s.index.values.astype(float)
            y = s.values.astype(float)
            aucs.append(float(np.trapezoid(y, x) / (x[-1] - x[0])))
    return np.array(aucs)


def run_wandb(bench, method_names, entity, project):
    scal_tag = SCALABILITY_TAGS[bench]
    ref_tag = REFERENCE_TAGS[bench]
    scal_runs = _find_runs_by_tags([scal_tag], entity, project)
    ref_runs = _find_runs_by_tags([ref_tag], entity, project)

    outd = OUT / bench
    outd.mkdir(parents=True, exist_ok=True)
    for name in method_names:
        spec = METHODS[name]
        if spec.get("benches") and bench not in spec["benches"]:
            print(f"  skip {name} (not for {bench})")
            continue
        result = {}
        if spec["kind"] == "curve":
            for tl in TLS:
                run_name = f"{name}-TL{tl}"
                run = scal_runs.get(run_name)
                if run is None:
                    print(f"  missing W&B run '{run_name}' (tag {scal_tag}) -- skipped")
                    continue
                aucs = _auc_array_from_logs(_load_artifact(run))
                result[str(tl)] = aucs
                print(f"  {name:14s} tl={tl:<4d}  mean_auc*100={np.mean(aucs)*100:6.2f}  (n={len(aucs)})")
        else:
            run = ref_runs.get(name)
            if run is None:
                print(f"  missing W&B run '{name}' (tag {ref_tag}) -- skipped")
                continue
            aucs = _auc_array_from_logs(_load_artifact(run))
            result["ref"] = aucs
            print(f"  {name:14s} ref       mean_auc*100={np.mean(aucs)*100:6.2f}  (n={len(aucs)})")
        if result:
            np.savez(outd / f"{name}.npz", **result)
    print(f"DONE {bench} (wandb)")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _style():
    with open("conf/plot_style.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _mean_ci(a, level=0.95):
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(a))
    if n == 1:
        return m, m, m
    sem = float(np.std(a, ddof=1) / np.sqrt(n))
    t = float(st.t.ppf(1 - (1 - level) / 2, df=n - 1))
    return m, m - t * sem, m + t * sem


def _load_method(bench, name):
    p = OUT / bench / f"{name}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _ent(style, name, i):
    e = style.get(name, {})
    return (e.get("label", name), e.get("color", f"C{i}"),
            e.get("linestyle", "solid"), e.get("marker", ""))


def _plot_curve(ax, data, label, color, ls, marker):
    tls = sorted(int(k) for k in data if k != "ref")
    ms, los, his = [], [], []
    for tl in tls:
        m, lo, hi = _mean_ci(data[str(tl)] * 100)
        ms.append(m); los.append(lo); his.append(hi)
    ax.plot(tls, ms, label=label, color=color, linestyle=ls, marker=marker or "o", markersize=6)
    ax.fill_between(tls, los, his, alpha=0.18, color=color)


def _plot_ref(ax, data, label, color):
    m, lo, hi = _mean_ci(data["ref"] * 100)
    x = [TLS[0], TLS[-1]]
    ax.plot(x, [m, m], label=label, color=color, linestyle="dashed", lw=1.5)
    ax.fill_between(x, [lo, lo], [hi, hi], alpha=0.12, color=color)


def plot(bench, exclude_families=()):
    style = _style()
    outd = OUT / bench
    families = {f: ms for f, ms in FAMILIES.items()
                if f not in exclude_families
                and any((outd / f"{m}.npz").exists() for m in ms)}
    bg = {m: _load_method(bench, m) for m in BACKGROUND}

    panels = list(families.items())
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.3), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (fam, methods) in zip(axes, panels):
        # faded model-free background (skip in the Model-free panel itself)
        if fam != "Model-free":
            for i, (bn, bd) in enumerate(bg.items()):
                if bd is None:
                    continue
                _, c, ls, mk = _ent(style, bn, i)
                tls = sorted(int(k) for k in bd if k != "ref")
                if tls:
                    ys = [np.mean(bd[str(t)]) * 100 for t in tls]
                    ax.plot(tls, ys, color=c, linestyle=ls, lw=1.0, alpha=0.3, zorder=1)
        for i, m in enumerate(methods):
            d = _load_method(bench, m)
            if d is None:
                continue
            label, color, ls, marker = _ent(style, m, i)
            if "ref" in d:
                _plot_ref(ax, d, label, color)
            else:
                _plot_curve(ax, d, label, color, ls, marker)
        ax.set_title(fam)
        ax.set_xlabel("# Training Tasks")
        ax.set_xscale("log", base=2)
        ax.set_xticks(TLS)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Avg. test normalized regret ×100")
    # No suptitle: the caption in the paper carries the description.
    fig.tight_layout()
    p = outd / "combined_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "run-wandb", "plot"])
    ap.add_argument("--bench", choices=["cash", "hpo"], required=True)
    ap.add_argument("--methods", default="all")
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    ap.add_argument("--entity", default=WANDB_ENTITY, help="W&B entity (run-wandb only)")
    ap.add_argument("--project", default=WANDB_PROJECT, help="W&B project (run-wandb only)")
    ap.add_argument("--exclude-families", default="",
                    help="comma-separated family names to drop from the plot (e.g. ABLR,RGPE)")
    args = ap.parse_args()
    if args.mode == "run":
        names = list(METHODS) if args.methods == "all" else args.methods.split(",")
        run(args.bench, names, args.n_jobs)
    elif args.mode == "run-wandb":
        names = list(METHODS) if args.methods == "all" else args.methods.split(",")
        run_wandb(args.bench, names, args.entity, args.project)
    else:
        excl = [s for s in args.exclude_families.split(",") if s]
        plot(args.bench, excl)


if __name__ == "__main__":
    main()
