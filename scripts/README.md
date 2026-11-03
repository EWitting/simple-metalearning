# scripts/ — results & plotting pipeline

## Core pipeline (intended use)

| Script | Source | Produces | Notes |
|---|---|---|---|
| `plot_comparison.py` | W&B (`--tags`) | `summary.json` (avg regret + avg rank), `stats_tests.json` (pairwise Wilcoxon), per-family + **`combined_family_rank`** plots | **Main results.** Uses `conf/plot_style.yaml` (families, colours, `exclude`). |
| `scalability_auc_v3.py` | local cached matrices (`run`) or W&B (`run-wandb`) | scalability vs #training-tasks (`combined_overview`) | Cold-start scalability figure. `run`/`run-wandb` then `plot`; `--exclude-families` to drop a panel. |
| `plot_adaptive_overfitting.py` | W&B (`--tags`) | validation-vs-test regret (`overview`) | Adaptive-overfitting appendix figure. |
| `plot_matrix_heatmaps.py` | local | performance-matrix heatmaps | Sign/orientation diagnostic for the performance matrix. |
| `coldstart_ablation.py` | local | warm- vs cold-start linear ablation | |
| `ablation_blr.py` | local | fixed-precision Bayesian linear head vs Bayesian ridge ablation | See `estimators/blr.py`. |
| `compare_continuous_discrete.py` | local | continuous vs discrete search space comparison | Appendix experiment. |
| `gp_scaling.py` | local | empirical GP-fit scaling exponent (BoTorch `SingleTaskGP`) | Produces the fit-time-vs-n figures cited in the supplement's GP cost analysis. |
| `fetch_compute.py` | W&B | total wall-clock compute per run family | Produces the supplement's compute table. |
| `fig_bayesridge_weights.py` | local | sign of learned Bayesian-ridge weights over prior tasks | Produces the supplement's weight-sign figure. |
| `timing_split.py` | local | per-step timing breakdown | |
| `pull_results.py` | W&B | raw run artifacts → CSV | |
| `collect_metrics.py` | W&B/local | metric aggregation | |
| `pick_best_gp_perf.py` | W&B | selects PCA `n_components` for GP-Perf from a sweep, writes it into the estimator configs | The sweep concluded no dimensionality reduction (`n_components: null`) performs best; current configs reflect that. |

## W&B tags
- `cash_v2`, `hpo_v2` — main experiments. `plot_comparison.py --tags cash_v2`.
- `scalability_v2` (CASH), `scalability-hpo_v2` (HPO) — scalability sweeps, `-TL{N}` run names.
  Producer: `experiments/scalability_{cash,hpo}.{sh,bat}`. Consumer:
  `scalability_auc_v3.py run-wandb`.
- ABLR and Active-Testing have `-TL{N}` scalability sweeps alongside the original 8 methods.
  RGPE-Oracle does not: it's excluded from the main paper's Figure 3 and referenced only as a small
  ablation side note, so it's intentionally left out of all scalability code.

## Main plot excludes the SOTA baselines
`conf/plot_style.yaml` has an `exclude:` list holding the dedicated meta-learning / AS baselines
(ABLR, RGPE, RGPE-Oracle, TST-R, Active-Testing, MTGP). With them excluded, `plot_comparison.py`
gives the main figure/summary/ranks matching the paper. For the separate **baselines** plot/table,
copy the style with `exclude: []` (or empty the list). Style: Perf = solid + ▲; Hyper = dashed (per
family colour).

## Superseded
`plot_scalability.py` and `plot_scalability_auc.py` are earlier W&B-based scalability plotting
scripts, no longer called by any `experiments/*.{sh,bat}` (superseded by `scalability_auc_v3.py
run-wandb` + `plot`). Kept for reference.

## Removed scripts
Earlier debug/exploration scripts (flip-sign comparisons, mask/oof checks, runtime estimators, GP
variant exploration) were removed rather than archived. Recoverable from git history if needed.
