@echo off
REM ============================================================
REM  cash.bat — CASH experiment
REM
REM  Setup: TabArena base methods, 5-fold CV, val/test split,
REM         200 BO iterations, dummy normalisation, flip_sign.
REM
REM  Compares all model families on the full CASH search space:
REM    - Naive baselines (random, mean, mean+MMR)
REM    - Linear-Perf     (BayesianRidge EI on perf-matrix columns, CF)
REM    - Linear-Hyper    (BayesianRidge EI on flow hyperparams, content)
REM    - RF-Hyper        (RF EI on flow hyperparams, SMAC-style)
REM    - RF-Perf         (RF EI on raw perf vectors, no PCA)
REM    - GP-Hyper        (BoTorch GP EI on flow hyperparams, content)
REM    - GP-Perf         (BoTorch GP EI on raw perf vectors, CF, no PCA)
REM    - TwoTower-Perf   (embed-only CF, EmbeddingMMR, dim=32)
REM    - TwoTower-Hyper  (asym: task embed + flow MLP, EmbeddingMMR)
REM
REM  Order: baselines first, deep models last.
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
dataset.methods=base ^
dataset.normalization=dummy ^
dataset.val_test_split=true ^
+dataset.flip_sign=true ^
+metafeature_transforms=standardscaler ^
+wandb.tags=["cash_v2"]

REM ── Naive baselines ──────────────────────────────────────────
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
uv run python -m simple_metalearning.main %COMMON% %FAST% estimator=content/task_only/botorch_gp_perf_cash +wandb.name=GP-Perf
set OMP_NUM_THREADS=
set MKL_NUM_THREADS=
set OPENBLAS_NUM_THREADS=

REM ── Deep models (PyTorch — use fewer parallel jobs) ──────────
uv run python -m simple_metalearning.main %COMMON% %SLOW% estimator=deep/two_tower_perf_cash estimator.update_task_only=true +wandb.name=TwoTower-Perf
uv run python -m simple_metalearning.main %COMMON% %SLOW% estimator=deep/two_tower_hyper_cash estimator.update_task_only=true +wandb.name=TwoTower-Hyper

REM ── Plots ────────────────────────────────────────────────────
uv run python scripts/plot_comparison.py --tags cash_v2 --out-dir plots/cash
uv run python scripts/plot_adaptive_overfitting.py --tags cash_v2 --out-dir plots/adaptive_overfitting/cash
