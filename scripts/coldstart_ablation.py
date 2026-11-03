"""Ablation: do Linear-Perf / Linear-Hyper lead because of the representation+BO,
or because of their mean-best warm start?

The linear estimators fall back to row-means (each config's mean performance over
prior tasks) while <2 observations exist on the target task, so their first TWO
picks are the global mean-best configs (identical to the Mean baseline). RF-EI and
GP-EI instead return zeros at cold start -> random first picks. This script reruns
the linear variants with a NEUTRALIZED cold start (constant score while <2 obs ->
the BO loop's random tie-break picks the first two flows at random, like RF/GP),
leaving the fitted-model BO from pick 3 onward completely unchanged.

Protocol is the real one: BOSelection._run_task, 5-fold CV (all folds),
max_visible=200, val/test split, on the cached v2 (correct, flipped) matrices.
"""
import argparse
import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from simple_metalearning.experiments.bo_selection import BOSelection
from simple_metalearning.estimators.bayesian_ridge_ei import BayesianRidgeEIEstimator
from simple_metalearning.estimators.mean import Mean
from simple_metalearning.estimators.random import Random
from simple_metalearning.estimators.acquisition import EIGaussianAcquisition
from simple_metalearning.datasets.metafeature_transforms import (
    StandardScalerTransform,
    CompositeTransform,
)

MAX_VISIBLE = 200
N_FOLDS = 5
RANDOM_STATE = 42
N_JOBS = 8


class ColdStartNeutralBayesianRidgeEI(BayesianRidgeEIEstimator):
    """Linear estimator without the mean-best warm start.

    While the per-task model is unfit (<2 target observations) we return a
    constant score for every flow, so the experiment loop's random tie-break
    chooses the flow uniformly at random -- exactly the cold-start behaviour of
    RF-EI / GP-EI. Once >=2 observations exist the model fits and behaviour is
    identical to the unmodified estimator.
    """

    def predict(self, task_index, return_std=False):
        if task_index not in self.models:
            self._train_model(task_index)
        model, _ = self.models[task_index]
        if model is None:
            n = self.dataset.performance.shape[0]
            preds = np.zeros(n)
            return (preds, np.zeros(n)) if return_std else preds
        return super().predict(task_index, return_std=return_std)

    def suggest(self, task_index, predictions=None):
        if task_index not in self.models:
            self._train_model(task_index)
        model, _ = self.models[task_index]
        if model is None:
            # constant -> loop picks uniformly at random among unrevealed flows
            return np.zeros(self.dataset.performance.shape[0])
        return super().suggest(task_index, predictions)


def linear_factory(cls, use_metadata):
    def make(ds):
        return cls(
            ds,
            fit_intercept=False,
            use_metadata=use_metadata,
            prior_equal_weight=False,
            prior_shape=2.0,
            f_star_from_predictions=False,
            acquisition_fn=EIGaussianAcquisition(xi=0.0),
        )
    return make


METHODS = {
    "Linear-Perf":        linear_factory(BayesianRidgeEIEstimator, False),
    "Linear-Perf (cold)": linear_factory(ColdStartNeutralBayesianRidgeEI, False),
    "Linear-Hyper":        linear_factory(BayesianRidgeEIEstimator, True),
    "Linear-Hyper (cold)": linear_factory(ColdStartNeutralBayesianRidgeEI, True),
    "Naive-Mean":   lambda ds: Mean(ds),
    "Naive-Random": lambda ds: Random(ds, random_state=RANDOM_STATE),
}


def run_method(full_ds, factory):
    exp = BOSelection(max_visible=MAX_VISIBLE, n_folds=N_FOLDS, n_folds_used=N_FOLDS,
                      random_state=RANDOM_STATE, n_jobs=1)
    n_tasks = full_ds.performance.shape[1]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    jobs = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_tasks))):
        masked = copy.deepcopy(full_ds)
        non_train = np.setdiff1d(np.arange(n_tasks), train_idx)
        masked.performance.iloc[:, non_train] = np.nan
        est = factory(masked)
        est.fit()
        for task_idx in test_idx:
            jobs.append((fold_idx, int(task_idx), est))
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(exp._run_task)(f, t, e, full_ds, {}, 0) for f, t, e in jobs
    )
    rows = [r for task_logs in results for r in task_logs]
    df = pd.DataFrame(rows)
    return df[["visible_num", "test_normalized_regret"]].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["cash", "hpo"], required=True)
    args = ap.parse_args()

    src = {
        "cash": "data/processed/tabarena_full_dummy_hash_e18a963c18_flip_sign_v2_valsplit.pkl",
        "hpo":  "data/processed/tabarena_full_dummy_XGBoost_flip_sign_v2_valsplit.pkl",
    }[args.bench]
    ds = pickle.load(open(src, "rb"))
    print(f"{args.bench}: matrix {ds.performance.shape} (flows x tasks)")

    # Replicate main.py: +metafeature_transforms=standardscaler (fill_na=True).
    # This converts nullable flow_metadata (pd.NA) to plain floats that the
    # Linear-Hyper estimator can consume, exactly as in the real runs.
    ds.flow_metadata = CompositeTransform(
        [StandardScalerTransform(fill_na=True)]
    ).fit_transform(ds.flow_metadata)
    ds.task_metadata = CompositeTransform(
        [StandardScalerTransform(fill_na=True)]
    ).fit_transform(ds.task_metadata)

    out = Path("plots/coldstart_ablation")
    out.mkdir(parents=True, exist_ok=True)

    curves, aucs = {}, {}
    for name, fac in METHODS.items():
        df = run_method(ds, fac)
        auc = df["test_normalized_regret"].mean() * 100  # nanmean over tasks & steps
        curve = df.groupby("visible_num")["test_normalized_regret"].mean()
        curves[name] = curve
        aucs[name] = auc
        print(f"  {name:22s} AUC test-norm-regret x100 = {auc:.2f}")

    pd.Series(aucs).to_csv(out / f"{args.bench}_auc.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    styles = {
        "Linear-Perf": ("C0", "-"), "Linear-Perf (cold)": ("C0", "--"),
        "Linear-Hyper": ("C1", "-"), "Linear-Hyper (cold)": ("C1", "--"),
        "Naive-Mean": ("0.4", ":"), "Naive-Random": ("0.7", ":"),
    }
    for name, curve in curves.items():
        c, ls = styles[name]
        x = curve.index.values
        m = x >= 1
        ax.plot(x[m], curve.values[m] * 100, label=f"{name}  (AUC {aucs[name]:.1f})",
                color=c, linestyle=ls, lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("visible_num (evaluations)")
    ax.set_ylabel("test normalized regret x100")
    ax.set_title(f"Cold-start ablation ({args.bench.upper()}): solid=mean-best warm start, dashed=random init")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / f"{args.bench}_coldstart.png", dpi=150)
    print("saved", out / f"{args.bench}_coldstart.png")


if __name__ == "__main__":
    main()
