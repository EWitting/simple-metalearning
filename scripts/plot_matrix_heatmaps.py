"""Diagnostic heatmaps of the CASH performance matrix to inspect the flip_sign bug.

Four variants share ONE flow ordering (grouped by estimator type, lowest mean
dummy-normalized error on top) so panels are directly comparable:

  V1  dummy-normalized score, NO flip      = error/dummy            (lower = better)
  V3  dummy-normalized score, FLIPPED      = -(error/dummy)         (higher = better)
  V2  normalized test regret, NO flip      (buggy: max-error treated as optimal)
  V4  normalized test regret, FLIPPED      (correct: min-error is optimal)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simple_metalearning.datasets.tabarena import TabArenaDatasetFactory

OUT = Path("plots/heatmaps")
OUT.mkdir(parents=True, exist_ok=True)

# Unflipped dummy-normalized CASH matrix. The pre-bugfix "flip_sign" cache holds
# the UNflipped matrix, because the bug meant the flip was never applied with
# dummy normalization.
SRC = "data/processed/tabarena_full_dummy_hash_e18a963c18_flip_sign_valsplit.pkl"
ds = pickle.load(open(SRC, "rb"))
M = ds.test_performance.copy()          # (flows x tasks); error/dummy, lower=better
print("matrix:", M.shape)

# Map each flow to its estimator type via longest known-method prefix.
methods = sorted(TabArenaDatasetFactory.BASE_METHODS, key=len, reverse=True)


def method_of(name: str) -> str:
    for m in methods:
        if name == m or name.startswith(m + "_"):
            return m
    return name.split("_")[0]


flow_method = pd.Series({f: method_of(f) for f in M.index})
print("methods:", flow_method.nunique(), sorted(flow_method.unique()))

# Shared ordering: groups (methods) by mean error asc; within group flows by mean error asc.
flow_mean_err = M.mean(axis=1)
grp_mean_err = flow_mean_err.groupby(flow_method).mean().sort_values()
ordered_flows, group_bounds = [], []
for g in grp_mean_err.index:
    fs = flow_mean_err[flow_method == g].sort_values().index.tolist()
    start = len(ordered_flows)
    ordered_flows += fs
    group_bounds.append((g, start, len(ordered_flows)))
task_order = M.mean(axis=0).sort_values().index.tolist()

base = M.loc[ordered_flows, task_order]


def per_cell_regret(mat: pd.DataFrame) -> pd.DataFrame:
    """Per-cell normalized regret using the metric-reporting formula, per task column."""
    out = np.empty_like(mat.values, dtype=float)
    for j, t in enumerate(mat.columns):
        col = mat[t].values
        optimal, median = np.nanmax(col), np.nanmedian(col)
        if optimal == median:
            out[:, j] = 0.0
            continue
        nb = np.clip((col - median) / (optimal - median), 0, None)
        out[:, j] = 1.0 - nb
    return pd.DataFrame(out, index=mat.index, columns=mat.columns)


V1 = base
V3 = -base
V2 = per_cell_regret(base)
V4 = per_cell_regret(-base)

err_vmax = float(np.nanpercentile(base.values, 97))


def draw(ax, data, title, cmap, vmin, vmax, cbar_label):
    im = ax.imshow(data.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("tasks (sorted by mean error →)", fontsize=8)
    yt, yl = [], []
    for g, s, e in group_bounds:
        if s > 0:
            ax.axhline(s - 0.5, color="white", lw=0.4)
        yt.append((s + e) / 2)
        yl.append(g)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=5)
    ax.set_xticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


specs = [
    ("variant1_dummy_noflip", V1, "V1  dummy-normalized score (NO flip)\nerror/dummy  (LOWER = better)",
     "viridis_r", 0, err_vmax, "error / dummy"),
    ("variant2_regret_noflip", V2, "V2  normalized test regret (NO flip)\nBUG: max-error treated as optimal",
     "magma", 0, 1, "regret (0=best)"),
    ("variant3_dummy_flip", V3, "V3  dummy-normalized score (FLIPPED)\n-(error/dummy)  (HIGHER = better)",
     "viridis", -err_vmax, 0, "-(error/dummy)"),
    ("variant4_regret_flip", V4, "V4  normalized test regret (FLIPPED)\nCORRECT: min-error is optimal",
     "magma", 0, 1, "regret (0=best)"),
]

for fname, data, title, cmap, vmin, vmax, clab in specs:
    fig, ax = plt.subplots(figsize=(7, 9))
    draw(ax, data, title, cmap, vmin, vmax, clab)
    fig.suptitle("CASH test matrix — flows grouped by estimator, lowest mean error on top", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / f"{fname}.png", dpi=150)
    plt.close(fig)
    print("saved", OUT / f"{fname}.png")

fig, axes = plt.subplots(2, 2, figsize=(15, 18))
panels = [(V1, specs[0]), (V3, specs[2]), (V2, specs[1]), (V4, specs[3])]
for ax, (data, sp) in zip(axes.ravel(), panels):
    _, _, title, cmap, vmin, vmax, clab = sp
    draw(ax, data, title, cmap, vmin, vmax, clab)
fig.suptitle(
    "CASH test matrix — same flow ordering across panels (grouped by estimator, lowest mean error on top)\n"
    "Top row: dummy-normalized score (no flip vs flipped).  Bottom row: per-cell normalized regret (buggy no-flip vs correct flip).",
    fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "combined_2x2.png", dpi=130)
plt.close(fig)
print("saved", OUT / "combined_2x2.png")
print("DONE")
