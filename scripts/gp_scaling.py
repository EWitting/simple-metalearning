"""Empirically confirm the GP-fit scaling exponent for the BoTorch SingleTaskGP
path used by RGPE base models, on the real CASH hyperparameter features.

Times fit_gpytorch_mll (same Normalize+Standardize transforms as RGPE._make_gp)
at increasing observation counts n, fits the log-log exponent, and extrapolates
to the full CASH n=3623 and to RGPE's ~per-fold base-GP setup.

Sub-sampling configs from the full CASH matrix is equivalent to varying the
number of selected models (each model contributes ~200 configs), at the full
CASH feature dimension d.

    python scripts/gp_scaling.py
"""
import time, pickle, warnings
import numpy as np, torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from simple_metalearning.datasets.metafeature_transforms import StandardScalerTransform, CompositeTransform

CASH = "data/processed/tabarena_full_dummy_hash_e18a963c18_flip_sign_v2_valsplit.pkl"
ds = pickle.load(open(CASH, "rb"))
feats = CompositeTransform([StandardScalerTransform(fill_na=True)]).fit_transform(ds.flow_metadata)
X = feats.values.astype("float64")
y = np.nan_to_num(ds.performance.values[:, 0].astype("float64"))
d = X.shape[1]
print(f"CASH features: n_max={X.shape[0]}, d={d}", flush=True)


def fit_time(n, reps=1):
    ts = []
    for r in range(reps):
        idx = np.random.default_rng(100 + r).choice(X.shape[0], n, replace=False)
        Xt = torch.tensor(X[idx], dtype=torch.float64)
        yt = torch.tensor(y[idx], dtype=torch.float64).unsqueeze(-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            m = SingleTaskGP(Xt, yt, input_transform=Normalize(d=d), outcome_transform=Standardize(m=1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


ns = [600, 900, 1200, 1500]   # all > d=475 -> cubic (Cholesky) regime, clean exponent
times = []
for n in ns:
    t = fit_time(n)
    times.append(t)
    print(f"  n={n:5d}  fit_time={t:8.3f}s", flush=True)

slope, intercept = np.polyfit(np.log(ns), np.log(times), 1)
print(f"\nempirical fit-time exponent (log-log slope) = {slope:.2f}   (exact Cholesky = 3.0)")
t_full = float(np.exp(intercept) * 3623 ** slope)
print(f"extrapolated single base-GP fit at full CASH n=3623: {t_full:.1f}s ({t_full/60:.1f} min)")
print(f"RGPE setup (~40 base GPs, one per training task): {t_full*40/60:.0f} min ({t_full*40/3600:.1f} h) per fold")
