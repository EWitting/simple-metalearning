"""Setup (one-time pretraining) vs per-task (BO loop) timing split.

Re-runs each estimator on the cached flip_sign_v2 matrices and times, serially:
  - setup    = constructing + fit()  (the one-time, fold-shared pretraining:
               ABLR's NN, RGPE's base GPs, MTGP's aux block; ~0 for Linear/RF/GP)
  - per-task = one target task's full 200-iteration BO loop via BOSelection._run_task
               (deep-copies the fitted estimator, then suggest+partial_fit x200)

These are serial (n_jobs=1) wall-times, so setup and per-task are directly comparable
and the amortized "total/#tasks" = setup/#test_tasks + per_task can be reconstructed.

    python scripts/timing_split.py --bench hpo --methods all --n-task 1
"""
import argparse, copy, pickle, time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from simple_metalearning.experiments.bo_selection import BOSelection
from simple_metalearning.estimators.bayesian_ridge_ei import BayesianRidgeEIEstimator
from simple_metalearning.estimators.task_only_rf_ei import TaskOnlyRandomForestEIEstimator
from simple_metalearning.estimators.task_only_botorch_gp_ei import TaskOnlyBoTorchGPEIEstimator
from simple_metalearning.estimators.ablr import ABLREstimator
from simple_metalearning.estimators.rgpe import RGPEEstimator
from simple_metalearning.estimators.multitask_gp_ei import MultiTaskGPEIEstimator
from simple_metalearning.estimators.active_testing import ActiveTestingEstimator
from simple_metalearning.estimators.mean import Mean
from simple_metalearning.estimators.mean_mmr import MeanMMR
from simple_metalearning.estimators.random import Random
from simple_metalearning.estimators.acquisition import (
    EIGaussianAcquisition, EIForestAcquisition, GreedyAcquisition)
from simple_metalearning.datasets.metafeature_transforms import (
    StandardScalerTransform, CompositeTransform, PerformanceVectorTransform)

RS = 42
SRC = {
    "cash": "data/processed/tabarena_full_dummy_hash_e18a963c18_flip_sign_v2_valsplit.pkl",
    "hpo":  "data/processed/tabarena_full_dummy_XGBoost_flip_sign_v2_valsplit.pkl",
}
def eig(): return EIGaussianAcquisition(xi=0.0)
def eif(): return EIForestAcquisition(xi=0.0)
def pvt(): return PerformanceVectorTransform(n_components=None, impute_strategy="mean")

# (constructor, set of benches it runs on)
METHODS = {
    "Linear-Perf": (lambda d: BayesianRidgeEIEstimator(d, fit_intercept=False, use_metadata=False,
        prior_equal_weight=False, prior_shape=2.0, f_star_from_predictions=False, acquisition_fn=eig()), {"cash","hpo"}),
    "RF-Perf": (lambda d: TaskOnlyRandomForestEIEstimator(d,
        rf_model=RandomForestRegressor(n_estimators=100, random_state=RS),
        performance_transform=pvt(), use_metadata=False, f_star_from_predictions=False, acquisition_fn=eif()), {"cash","hpo"}),
    "Linear-Hyper": (lambda d: BayesianRidgeEIEstimator(d, fit_intercept=False, use_metadata=True,
        prior_equal_weight=False, prior_shape=2.0, f_star_from_predictions=False, acquisition_fn=eig()), {"cash","hpo"}),
    "RF-Hyper": (lambda d: TaskOnlyRandomForestEIEstimator(d,
        rf_model=RandomForestRegressor(n_estimators=100, random_state=RS, min_samples_leaf=3),
        use_metadata=True, f_star_from_predictions=False, acquisition_fn=eif()), {"cash","hpo"}),
    "GP-Perf": (lambda d: TaskOnlyBoTorchGPEIEstimator(d, n_restarts=0, f_star_from_predictions=False,
        use_metadata=False, normalize_inputs=True, standardize_outputs=True, performance_transform=pvt(),
        acquisition_fn=eig()), {"cash","hpo"}),
    "GP-Hyper": (lambda d: TaskOnlyBoTorchGPEIEstimator(d, n_restarts=0, f_star_from_predictions=False,
        use_metadata=True, normalize_inputs=True, standardize_outputs=True, acquisition_fn=eig()), {"cash","hpo"}),
    "Naive-Random": (lambda d: Random(d, random_state=RS), {"cash","hpo"}),
    "Naive-Mean": (lambda d: Mean(d), {"cash","hpo"}),
    "Naive-Mean-MMR": (lambda d: MeanMMR(d, use_mmr=True, mmr_lambda=0.05), {"cash","hpo"}),
    "ABLR": (lambda d: ABLREstimator(d, feature_dim=50, hidden=[50,50], n_epochs=300, lr=0.01,
        use_metadata=True, random_state=RS, acquisition_fn=eig()), {"cash","hpo"}),
    "Active-Testing": (lambda d: ActiveTestingEstimator(d, sim_sharpness=1.0, acquisition_fn=GreedyAcquisition()), {"cash","hpo"}),
    "RGPE": (lambda d: RGPEEstimator(d, n_bootstrap_samples=256, target_cv_folds=5, use_base_variance=True,
        prevent_weight_dilution=True, horizon=200, max_base_tasks=None, normalize_inputs=True,
        standardize_outputs=True, random_state=RS, acquisition_fn=eig()), {"hpo"}),
    "MTGP": (lambda d: MultiTaskGPEIEstimator(d, n_aux_tasks=6, n_context_flows=40, rank=None,
        refit_factor=3.0, use_metadata=True, standardize_outputs=True, random_state=RS, acquisition_fn=eig()), {"hpo"}),
}


def load_ds(bench):
    ds = pickle.load(open(SRC[bench], "rb"))
    ds.flow_metadata = CompositeTransform([StandardScalerTransform(fill_na=True)]).fit_transform(ds.flow_metadata)
    ds.task_metadata = CompositeTransform([StandardScalerTransform(fill_na=True)]).fit_transform(ds.task_metadata)
    return ds


def time_method(name, make, ds, n_task):
    n_tasks = ds.performance.shape[1]
    kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    train_idx, test_idx = next(iter(kf.split(np.arange(n_tasks))))
    masked = copy.deepcopy(ds)
    masked.performance.iloc[:, np.setdiff1d(np.arange(n_tasks), train_idx)] = np.nan

    t0 = time.perf_counter()
    est = make(masked); est.fit()
    setup = time.perf_counter() - t0

    exp = BOSelection(max_visible=200, n_folds=5, n_folds_used=5, random_state=RS, n_jobs=1)
    per = []
    for t in test_idx[:n_task]:
        t0 = time.perf_counter()
        exp._run_task(0, int(t), est, ds, {}, 0)
        per.append(time.perf_counter() - t0)
    return setup, float(np.mean(per)), len(train_idx), len(test_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["cash","hpo","both"], default="both")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--n-task", type=int, default=1)
    args = ap.parse_args()
    benches = ["cash","hpo"] if args.bench == "both" else [args.bench]
    names = list(METHODS) if args.methods == "all" else args.methods.split(",")
    for b in benches:
        ds = load_ds(b)
        print(f"\n=== {b.upper()}  matrix {ds.performance.shape} ===", flush=True)
        print(f"  {'method':16s} {'setup(s)':>10s} {'per-task(s)':>12s} {'amort/task(s)':>14s}  (n_train, n_test)", flush=True)
        for name in names:
            make, benchset = METHODS[name]
            if b not in benchset:
                print(f"  {name:16s}  (skip on {b})", flush=True); continue
            try:
                setup, per, ntr, nte = time_method(name, make, ds, args.n_task)
                amort = setup / max(nte, 1) + per   # one-time setup amortized over the fold's test tasks
                print(f"  {name:16s} {setup:10.2f} {per:12.2f} {amort:14.2f}  ({ntr}, {nte})", flush=True)
            except Exception as e:
                print(f"  {name:16s}  ERROR: {repr(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
