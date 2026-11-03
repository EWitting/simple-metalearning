@echo off
REM ============================================================
REM  hpo.bat — HPO experiment
REM
REM  Identical setup to cash.bat but restricted to
REM  XGBoost only (201 configs).  Tests whether methods benefit
REM  from the reduced, homogeneous search space.
REM
REM  All model families are included so CASH vs HPO comparison
REM  is a direct apples-to-apples swap of dataset.methods.
REM ============================================================

set N_JOBS_FAST=10
set N_JOBS_SLOW=4
set FAST=experiment.n_jobs=%N_JOBS_FAST%
set SLOW=experiment.n_jobs=%N_JOBS_SLOW%

set COMMON=^
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
+wandb.tags=["hpo_v2"]

REM ── Naive baselines ─────────────────────────────────────────
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=naive/random +wandb.name=Naive-Random
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=naive/mean +wandb.name=Naive-Mean
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=naive/mean_mmr estimator.mmr_lambda=0.05 +wandb.name=Naive-Mean-MMR

REM ── Collaborative (performance-based, EI acquisition) ────────
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=content/task_only/perf_vector_ridge_ei +wandb.name=Linear-Perf

REM ── Content-based (hyperparameter-based, EI acquisition) ─────
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=content/task_only/ridge_ei +wandb.name=Linear-Hyper
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=content/task_only/rf_ei +wandb.name=RF-Hyper
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=content/task_only/perf_vector_rf_ei +estimator.use_metadata=false estimator.performance_transform.n_components=null +wandb.name=RF-Perf

REM ── GP family (thread-limited to avoid BLAS/torch oversubscription when
REM    many joblib workers each spin up a full internal thread pool) ────────
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=content/task_only/botorch_gp_ei +wandb.name=GP-Hyper
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=content/task_only/botorch_gp_perf_hpo +wandb.name=GP-Perf
set OMP_NUM_THREADS=
set MKL_NUM_THREADS=
set OPENBLAS_NUM_THREADS=

REM ── Deep models (PyTorch — use fewer parallel jobs) ──────────
uv run python -m simple_metalearning.main %COMMON% %SLOW% estimator=deep/two_tower_perf_hpo estimator.update_task_only=true +wandb.name=TwoTower-Perf
uv run python -m simple_metalearning.main %COMMON% %SLOW% estimator=deep/two_tower_hyper_hpo estimator.update_task_only=true +wandb.name=TwoTower-Hyper

REM ── Plots ────────────────────────────────────────────────────
uv run python scripts/plot_comparison.py --tags hpo_v2 --out-dir plots/hpo
uv run python scripts/plot_adaptive_overfitting.py --tags hpo_v2 --out-dir plots/adaptive_overfitting/hpo
