@echo off
REM ============================================================
REM  demo.bat — Quick sanity check / minimal replication demo
REM
REM  Compares two methods on the HPO (XGBoost, 201 configs) setting:
REM    1. Naive-Random   — random selection baseline
REM    2. Linear-Perf    — BayesianRidge EI on performance vectors
REM
REM  Settings: 2 CV folds (1 evaluated), 10 BO steps, wandb offline.
REM  Expected runtime: ~1-2 minutes on CPU.
REM  Results are saved by Hydra to outputs\<date>\<time>\.
REM ============================================================

set COMMON=^
experiment=bo_selection ^
experiment.n_folds=2 ^
experiment.n_folds_used=1 ^
experiment.max_visible=10 ^
experiment.n_jobs=4 ^
dataset=tabarena ^
dataset.methods=XGBoost ^
dataset.normalization=dummy ^
dataset.val_test_split=true ^
+dataset.flip_sign=true ^
+metafeature_transforms=standardscaler ^
wandb.mode=disabled

echo === Demo run 1/2: Naive-Random ===
uv run python -m simple_metalearning.main %COMMON% estimator=naive/random +wandb.name=demo-random

echo.
echo === Demo run 2/2: Linear-Perf (BayesianRidge EI on perf vectors) ===
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ridge_ei +wandb.name=demo-linear-perf

echo.
echo Demo complete. Results saved to outputs\
