@echo off
REM ============================================================
REM  linear_models_cash.bat -- Linear model comparison (CASH / base methods)
REM
REM  All models are performance-vector-based (collaborative filtering).
REM  No hyperparameter/content features used.
REM
REM  Model x Acquisition matrix:
REM    OLS           : Greedy, MMR
REM    Ridge         : Greedy, MMR
REM    Ensemble OLS  : EI, LogEI
REM    Ensemble Ridge: EI
REM    BayesRidge    : Greedy, MMR, EI, LogEI
REM    OOFRidge      : Greedy, MMR, EI, LogEI, OOF-MMR
REM
REM  Tags: linear-models-cash
REM ============================================================

set N_JOBS=10

set COMMON=^
experiment=bo_selection ^
experiment.n_folds=5 ^
experiment.n_folds_used=5 ^
experiment.max_visible=200 ^
experiment.n_jobs=%N_JOBS% ^
dataset=tabarena ^
dataset.methods=base ^
dataset.normalization=dummy ^
dataset.val_test_split=true ^
+dataset.flip_sign=true ^
+metafeature_transforms=standardscaler ^
+wandb.tags=[linear-models-cash_v2]

REM -- OLS --
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ols_greedy +wandb.name=Linear-Greedy
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ols_mmr +wandb.name=Linear-MMR

REM -- Plain Ridge --
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_sklridge_greedy +wandb.name=Ridge-Greedy
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_sklridge_mmr +wandb.name=Ridge-MMR

REM -- Ensemble OLS --
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ensemble_ols_ei +wandb.name=Ensemble-Linear-EI
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ensemble_ols_log_ei +wandb.name=Ensemble-Linear-LogEI

REM -- Ensemble Ridge --
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ensemble_ridge_ei +wandb.name=Ensemble-Ridge-EI

REM -- BayesianRidge --
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ridge_greedy +wandb.name=BayesRidge-Greedy
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ridge_mmr +wandb.name=BayesRidge-MMR
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ridge_ei +wandb.name=BayesRidge-EI
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_ridge_log_ei +wandb.name=BayesRidge-LogEI

REM -- OOF Ridge (BayesRidge on OOF RF predictions) --
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_oof_ridge_greedy +wandb.name=OOFRidge-Greedy
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_oof_ridge_mmr +wandb.name=OOFRidge-MMR
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_oof_ridge_ei +wandb.name=OOFRidge-EI
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_oof_ridge_log_ei +wandb.name=OOFRidge-LogEI
uv run python -m simple_metalearning.main %COMMON% estimator=content/task_only/perf_vector_oof_ridge_oof_mmr +wandb.name=OOFRidge-OOF-MMR

REM -- Plots --
uv run python scripts/plot_comparison.py --tags linear-models-cash_v2 --out-dir plots/linear_models_cash
