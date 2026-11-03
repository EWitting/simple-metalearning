"""Ablation: Linear-Perf with BRR (sklearn BayesianRidge, per-fit evidence max)
vs BLR (fixed-precision Bayesian linear regression, ABLR-style head).

Runs both through the exact local harness used for the scalability/main numbers
(BOSelection._run_task on the cached flip_sign_v2 matrices, cold-start linear),
at the FULL training-task set, and reports avg test-normalized regret (summary.json
metric) plus a paired per-(fold,task) comparison.

    python scripts/ablation_blr.py --bench both --n-jobs 8
"""
import argparse
import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from simple_metalearning.experiments.bo_selection import BOSelection
from simple_metalearning.estimators.bayesian_ridge_ei import BayesianRidgeEIEstimator
from simple_metalearning.estimators.blr import BLRFixedEIEstimator
from simple_metalearning.estimators.acquisition import EIGaussianAcquisition
from simple_metalearning.datasets.metafeature_transforms import (
    StandardScalerTransform, CompositeTransform,
)

N_FOLDS = 5
MAX_VISIBLE = 200
RANDOM_STATE = 42
SRC = {
    "cash": "data/processed/tabarena_full_dummy_hash_e18a963c18_flip_sign_v2_valsplit.pkl",
    "hpo":  "data/processed/tabarena_full_dummy_XGBoost_flip_sign_v2_valsplit.pkl",
}


def _eig():
    return EIGaussianAcquisition(xi=0.0)


def make_brr(d):
    return BayesianRidgeEIEstimator(
        d, fit_intercept=False, use_metadata=False, prior_equal_weight=False,
        prior_shape=2.0, f_star_from_predictions=False, acquisition_fn=_eig())


def make_blr(d):
    # Fixed-precision BLR head; precisions auto-estimated once per fold (one
    # sklearn BayesianRidge reference fit), then frozen across the BO loop.
    return BLRFixedEIEstimator(
        d, fit_intercept=False, use_metadata=False, prior_equal_weight=False,
        prior_shape=2.0, f_star_from_predictions=False, acquisition_fn=_eig())


def _load_ds(bench):
    ds = pickle.load(open(SRC[bench], "rb"))
    ds.flow_metadata = CompositeTransform([StandardScalerTransform(fill_na=True)]).fit_transform(ds.flow_metadata)
    ds.task_metadata = CompositeTransform([StandardScalerTransform(fill_na=True)]).fit_transform(ds.task_metadata)
    return ds


def _series(task_logs):
    """(per-series mean regret, final regret) for one (fold,task)."""
    df = pd.DataFrame(task_logs)
    if "test_normalized_regret" not in df:
        return None
    s = df.set_index("visible_num")["test_normalized_regret"].sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna()
    if len(s) < 2:
        return None
    return float(s.mean()), float(s.iloc[-1])


def _run(ds, make, n_jobs):
    n_tasks = ds.performance.shape[1]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    jobs = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_tasks))):
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
    keys = [(f, t) for f, t, _ in jobs]
    out = {}
    for (f, t), logs in zip(keys, res):
        v = _series(logs)
        if v is not None:
            out[(f, t)] = v
    return out


def run_bench(bench, n_jobs):
    ds = _load_ds(bench)
    print(f"\n=== {bench.upper()}  matrix {ds.performance.shape} ===", flush=True)
    brr = _run(ds, make_brr, n_jobs)
    blr = _run(ds, make_blr, n_jobs)
    common = sorted(set(brr) & set(blr))
    brr_mean = np.array([brr[k][0] for k in common])
    blr_mean = np.array([blr[k][0] for k in common])
    brr_fin = np.array([brr[k][1] for k in common])
    blr_fin = np.array([blr[k][1] for k in common])

    def line(tag, a, b):
        # a=BRR, b=BLR (lower regret is better)
        diff = b - a  # >0 => BLR worse
        try:
            p = st.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        except Exception:
            p = float("nan")
        better = "BRR" if a.mean() < b.mean() else "BLR"
        print(f"  {tag:16s} BRR={a.mean()*100:6.2f}  BLR={b.mean()*100:6.2f}  "
              f"diff(BLR-BRR)={diff.mean()*100:+6.2f}  better={better}  "
              f"wilcoxon_p={p:.4f}  n={len(a)}", flush=True)

    line("avg-regret x100", brr_mean, blr_mean)
    line("final-regret x100", brr_fin, blr_fin)
    return dict(bench=bench, n=len(common),
                brr_avg=float(brr_mean.mean()), blr_avg=float(blr_mean.mean()),
                brr_final=float(brr_fin.mean()), blr_final=float(blr_fin.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="both", choices=["cash", "hpo", "both"])
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    benches = ["cash", "hpo"] if args.bench == "both" else [args.bench]
    rows = [run_bench(b, args.n_jobs) for b in benches]
    print("\n=== SUMMARY (avg test normalized regret x100, lower=better) ===")
    for r in rows:
        print(f"  {r['bench']:5s}  BRR avg={r['brr_avg']*100:6.2f}  BLR avg={r['blr_avg']*100:6.2f}"
              f"   |  BRR final={r['brr_final']*100:6.2f}  BLR final={r['blr_final']*100:6.2f}  (n={r['n']})")
    Path("plots").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("plots/ablation_blr_summary.csv", index=False)
    print("saved plots/ablation_blr_summary.csv")


if __name__ == "__main__":
    main()
