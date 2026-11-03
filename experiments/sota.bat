@echo off
REM ============================================================
REM  sota.bat — SOTA transfer / meta-BO baselines
REM
REM  Adds state-of-the-art transfer-learning surrogates on top of
REM  the canonical CASH / HPO comparison.  Kept in a separate runner
REM  while under development; runs are tagged with the SAME experiment
REM  tag as cash.bat / hpo.bat (cash_v2 / hpo_v2) so they fold straight
REM  into the existing comparison + bootstrap-rank plots, plus a
REM  dedicated "sota_v1" tag for focused views.
REM
REM  Methods:
REM    - RGPE  (Ranking-Weighted GP Ensemble; Feurer et al. 2018)
REM            content/task_only/rgpe — flow-hyperparameter features,
REM            one base GP per training task + target GP, ranking weights.
REM
REM  NOTE: RGPE fits one GP per training task once per fold (~40 GPs).
REM  On HPO (201 configs) this is feasible; on CASH (3623 configs, 475-dim
REM  features) it is not, so RGPE and MTGP are only run for HPO here
REM  (see paper supplement, cost analysis).
REM ============================================================

REM One BLAS/torch thread per worker — avoids severe oversubscription when many
REM joblib workers each spin up a full torch thread pool (GP / MTGP runs).
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1

set N_JOBS_FAST=10
set N_JOBS_RGPE=6
set FAST=experiment.n_jobs=%N_JOBS_FAST%
set RGPE=experiment.n_jobs=%N_JOBS_RGPE%

REM ── HPO (XGBoost, 201 configs) ───────────────────────────────
set COMMON_HPO=^
experiment=bo_selection ^
experiment.n_folds=5 ^
experiment.n_folds_used=5 ^
experiment.max_visible=200 ^
dataset=tabarena ^
dataset.methods=XGBoost ^
dataset.normalization=dummy ^
dataset.val_test_split=true ^
+dataset.flip_sign=true ^
+metafeature_transforms=standardscaler ^
+wandb.tags=["hpo_v2","sota_v1"]

uv run python -m simple_metalearning.main %COMMON_HPO% %RGPE% estimator=content/task_only/rgpe +wandb.name=RGPE
REM Surrogate-free discrete baselines (no GP — fast):
uv run python -m simple_metalearning.main %COMMON_HPO% %FAST% estimator=content/task_only/rgpe_oracle +wandb.name=RGPE-Oracle
uv run python -m simple_metalearning.main %COMMON_HPO% %FAST% estimator=content/task_only/tstr_discrete +wandb.name=TST-R
uv run python -m simple_metalearning.main %COMMON_HPO% %FAST% estimator=content/task_only/active_testing +wandb.name=Active-Testing
REM Multi-Task GP (ICM) — set OMP_NUM_THREADS=1 to avoid torch oversubscription:
REM MTGP: n_jobs=1 (BoTorch MultiTaskGP segfaults in loky's spawned workers).
uv run python -m simple_metalearning.main %COMMON_HPO% experiment.n_jobs=1 estimator=content/task_only/mtgp_hpo +wandb.name=MTGP
uv run python -m simple_metalearning.main %COMMON_HPO% %FAST% estimator=content/task_only/ablr +wandb.name=ABLR

REM ── CASH (all base methods, 3623 configs) ────────────────────
set COMMON_CASH=^
experiment=bo_selection ^
experiment.n_folds=5 ^
experiment.n_folds_used=5 ^
experiment.max_visible=200 ^
dataset=tabarena ^
dataset.methods=base ^
dataset.normalization=dummy ^
dataset.val_test_split=true ^
+dataset.flip_sign=true ^
+metafeature_transforms=standardscaler ^
+wandb.tags=["cash_v2","sota_v1"]

REM RGPE and MTGP on CASH are not run: GP-based base fitting does not stay
REM computationally feasible at 3623 configs (see paper supplement, cost analysis).
REM Surrogate-free discrete baselines (no GP — fast even on 3623 configs):
uv run python -m simple_metalearning.main %COMMON_CASH% %FAST% estimator=content/task_only/rgpe_oracle +wandb.name=RGPE-Oracle
uv run python -m simple_metalearning.main %COMMON_CASH% %FAST% estimator=content/task_only/tstr_discrete +wandb.name=TST-R
uv run python -m simple_metalearning.main %COMMON_CASH% %FAST% estimator=content/task_only/active_testing +wandb.name=Active-Testing
uv run python -m simple_metalearning.main %COMMON_CASH% %FAST% estimator=content/task_only/ablr +wandb.name=ABLR

REM ── Plots: regenerate canonical comparison (now incl. RGPE) ──
uv run python scripts/plot_comparison.py --tags hpo_v2  --out-dir plots/hpo
uv run python scripts/plot_comparison.py --tags cash_v2 --out-dir plots/cash
