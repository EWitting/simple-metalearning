"""Generate the appendix figure on signed Bayesian-ridge weights (v2 data).

Two panels:
  (left)  fraction of learned weights that are positive vs negative, vs #observations
  (right) posterior weight density at a representative observation count,
          relative to 0 and the equal-weight (1/N) averaging solution.

Saves a single PDF into the article figures directory.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

CACHE = "data/processed/tabarena_full_dummy_hash_e18a963c18_flip_sign_v2_valsplit.pkl"
OUT = os.environ.get("BAYESRIDGE_FIG_OUT", "plots/linear/bayesridge_weights.pdf")
TEST_TASK, SEED = 5, 42
os.makedirs(os.path.dirname(OUT), exist_ok=True)

md = pickle.load(open(CACHE, "rb"))
perf = md.performance.values.astype(float)
n_flows, n_tasks = perf.shape
cols = [j for j in range(n_tasks) if j != TEST_TASK]
X, y = perf[:, cols].copy(), perf[:, TEST_TASK].copy()
ok = ~np.isnan(y)
X, y = X[ok], y[ok]
cm = np.nanmean(X, axis=0)
for j in range(X.shape[1]):
    X[np.isnan(X[:, j]), j] = cm[j]
N = X.shape[1]
eqw = 1.0 / N
rng = np.random.default_rng(SEED)
o = rng.permutation(X.shape[0])
X, y = X[o], y[o]

steps = [2**k for k in range(1, int(np.floor(np.log2(X.shape[0]))) + 1)]
res = {}
for n in steps:
    m = BayesianRidge(fit_intercept=False)
    m.fit(X[:n], y[:n])
    res[n] = (m.coef_.copy(), np.sqrt(np.diag(m.sigma_)).copy())
ns = sorted(res)

fig, axl = plt.subplots(figsize=(7, 4))

fpos = [np.mean(res[n][0] > 0) for n in ns]
fneg = [np.mean(res[n][0] < 0) for n in ns]
axl.plot(ns, fpos, marker="o", color="#4477AA", label="weight $> 0$")
axl.plot(ns, fneg, marker="s", color="#CC6677", label="weight $< 0$")
axl.axhline(0.5, color="gray", lw=0.8, ls=":")
axl.set_xscale("log", base=2); axl.set_xticks(ns); axl.set_xticklabels(ns)
axl.set_ylim(-0.02, 1.02)
axl.set_xlabel("number of observed configurations")
axl.set_ylabel("fraction of learned weights")
axl.legend(fontsize=9, loc="center right")

fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
print(f"N (prior tasks/features)={N}; at n={ns[-1]}: pos={fpos[-1]:.2f} neg={fneg[-1]:.2f}")
print("neg fraction across steps:", [f"{n}:{v:.2f}" for n, v in zip(ns, fneg)])
