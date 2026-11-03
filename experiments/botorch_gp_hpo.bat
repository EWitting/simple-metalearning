@echo off
REM ============================================================
REM  botorch_gp_hpo.bat
REM
REM  Compare BoTorch GP vs sklearn GP on HPO (XGBoost configs).
REM  Four conditions:
REM    - sklearn GP-Hyper   (no quantile transform)
REM    - BoTorch GP-Hyper   (no quantile transform)
REM    - sklearn GP-Hyper-Q (with quantile transform)
REM    - BoTorch GP-Hyper-Q (with quantile transform)
REM
REM  Tags: botorch-gp  (filter in W&B to compare these runs)
REM ============================================================

set N_JOBS=10

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
dataset.drop_default_configs=true ^
experiment.n_jobs=%N_JOBS%

REM --- Without quantile transform ---
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/gp_ei         +wandb.name=Sklearn-GP-Hyper   "+wandb.tags=[botorch-gp_v2]"
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/botorch_gp_ei +wandb.name=BoTorch-GP-Hyper   "+wandb.tags=[botorch-gp_v2]"

REM --- With quantile transform ---
uv run python -m simple_metalearning.main %COMMON% +metafeature_transforms=quantile estimator=content/task_only/gp_ei         +wandb.name=Sklearn-GP-Hyper-Q   "+wandb.tags=[botorch-gp_v2]"
uv run python -m simple_metalearning.main %COMMON% +metafeature_transforms=quantile estimator=content/task_only/botorch_gp_ei +wandb.name=BoTorch-GP-Hyper-Q   "+wandb.tags=[botorch-gp_v2]"
