@echo off
REM ============================================================
REM  scalability_cash.bat — Task-limit scalability (CASH)
REM
REM  CASH setup (full base method set) repeated at 6 training-
REM  task limits: 1, 2, 4, 8, 16, 32 (powers of 2).
REM
REM  Only performance-based models and naive baselines are run
REM  (hyperparameter-based models are excluded — they don't rely
REM  on the performance matrix and are unaffected by task count).
REM
REM  Run names follow the pattern {Method}-TL{N} so that the
REM  plot_scalability.py script can parse method and task limit.
REM ============================================================

timeout /t 1800 /nobreak

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
+wandb.tags=["scalability_v2"]

REM ── task_limit=1 ─────────────────────────────────────────────
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=1 estimator=naive/random +wandb.name=Naive-Random-TL1
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=1 estimator=naive/mean +wandb.name=Naive-Mean-TL1
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=1 estimator=naive/mean_mmr estimator.mmr_lambda=0.05 +wandb.name=Naive-Mean-MMR-TL1
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=1 estimator=content/task_only/perf_vector_ridge_ei +wandb.name=Linear-Perf-TL1
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=1 estimator=deep/two_tower_perf_cash estimator.update_task_only=true +wandb.name=TwoTower-Perf-TL1
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=1 estimator=deep/two_tower_hyper_cash estimator.update_task_only=true +wandb.name=TwoTower-Hyper-TL1
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=1 estimator=content/task_only/perf_vector_rf_ei +estimator.use_metadata=false estimator.performance_transform.n_components=null +wandb.name=RF-Perf-TL1
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=1 estimator=content/task_only/botorch_gp_perf_cash +wandb.name=GP-Perf-TL1
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=1 estimator=content/task_only/ablr +wandb.name=ABLR-TL1
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=1 estimator=content/task_only/active_testing +wandb.name=Active-Testing-TL1

REM ── task_limit=2 ─────────────────────────────────────────────
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=2 estimator=naive/random +wandb.name=Naive-Random-TL2
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=2 estimator=naive/mean +wandb.name=Naive-Mean-TL2
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=2 estimator=naive/mean_mmr estimator.mmr_lambda=0.05 +wandb.name=Naive-Mean-MMR-TL2
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=2 estimator=content/task_only/perf_vector_ridge_ei +wandb.name=Linear-Perf-TL2
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=2 estimator=content/task_only/perf_vector_rf_ei +estimator.use_metadata=false estimator.performance_transform.n_components=null +wandb.name=RF-Perf-TL2
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=2 estimator=deep/two_tower_perf_cash estimator.update_task_only=true +wandb.name=TwoTower-Perf-TL2
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=2 estimator=deep/two_tower_hyper_cash estimator.update_task_only=true +wandb.name=TwoTower-Hyper-TL2
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=2 estimator=content/task_only/botorch_gp_perf_cash +wandb.name=GP-Perf-TL2
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=2 estimator=content/task_only/ablr +wandb.name=ABLR-TL2
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=2 estimator=content/task_only/active_testing +wandb.name=Active-Testing-TL2

REM ── task_limit=4 ─────────────────────────────────────────────
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=4 estimator=naive/random +wandb.name=Naive-Random-TL4
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=4 estimator=naive/mean +wandb.name=Naive-Mean-TL4
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=4 estimator=naive/mean_mmr estimator.mmr_lambda=0.05 +wandb.name=Naive-Mean-MMR-TL4
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=4 estimator=content/task_only/perf_vector_ridge_ei +wandb.name=Linear-Perf-TL4
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=4 estimator=content/task_only/perf_vector_rf_ei +estimator.use_metadata=false estimator.performance_transform.n_components=null +wandb.name=RF-Perf-TL4
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=4 estimator=deep/two_tower_perf_cash estimator.update_task_only=true +wandb.name=TwoTower-Perf-TL4
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=4 estimator=deep/two_tower_hyper_cash estimator.update_task_only=true +wandb.name=TwoTower-Hyper-TL4
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=4 estimator=content/task_only/botorch_gp_perf_cash +wandb.name=GP-Perf-TL4
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=4 estimator=content/task_only/ablr +wandb.name=ABLR-TL4
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=4 estimator=content/task_only/active_testing +wandb.name=Active-Testing-TL4

REM ── task_limit=8 ─────────────────────────────────────────────
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=8 estimator=naive/random +wandb.name=Naive-Random-TL8
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=8 estimator=naive/mean +wandb.name=Naive-Mean-TL8
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=8 estimator=naive/mean_mmr estimator.mmr_lambda=0.05 +wandb.name=Naive-Mean-MMR-TL8
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=8 estimator=content/task_only/perf_vector_ridge_ei +wandb.name=Linear-Perf-TL8
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=8 estimator=content/task_only/perf_vector_rf_ei +estimator.use_metadata=false estimator.performance_transform.n_components=null +wandb.name=RF-Perf-TL8
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=8 estimator=deep/two_tower_perf_cash estimator.update_task_only=true +wandb.name=TwoTower-Perf-TL8
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=8 estimator=deep/two_tower_hyper_cash estimator.update_task_only=true +wandb.name=TwoTower-Hyper-TL8
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=8 estimator=content/task_only/botorch_gp_perf_cash +wandb.name=GP-Perf-TL8
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=8 estimator=content/task_only/ablr +wandb.name=ABLR-TL8
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=8 estimator=content/task_only/active_testing +wandb.name=Active-Testing-TL8

REM ── task_limit=16 ────────────────────────────────────────────
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=16 estimator=naive/random +wandb.name=Naive-Random-TL16
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=16 estimator=naive/mean +wandb.name=Naive-Mean-TL16
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=16 estimator=naive/mean_mmr estimator.mmr_lambda=0.05 +wandb.name=Naive-Mean-MMR-TL16
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=16 estimator=content/task_only/perf_vector_ridge_ei +wandb.name=Linear-Perf-TL16
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=16 estimator=content/task_only/perf_vector_rf_ei +estimator.use_metadata=false estimator.performance_transform.n_components=null +wandb.name=RF-Perf-TL16
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=16 estimator=deep/two_tower_perf_cash estimator.update_task_only=true +wandb.name=TwoTower-Perf-TL16
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=16 estimator=deep/two_tower_hyper_cash estimator.update_task_only=true +wandb.name=TwoTower-Hyper-TL16
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=16 estimator=content/task_only/botorch_gp_perf_cash +wandb.name=GP-Perf-TL16
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=16 estimator=content/task_only/ablr +wandb.name=ABLR-TL16
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=16 estimator=content/task_only/active_testing +wandb.name=Active-Testing-TL16

REM ── task_limit=32 ────────────────────────────────────────────
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=32 estimator=naive/random +wandb.name=Naive-Random-TL32
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=32 estimator=naive/mean +wandb.name=Naive-Mean-TL32
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=32 estimator=naive/mean_mmr estimator.mmr_lambda=0.05 +wandb.name=Naive-Mean-MMR-TL32
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=32 estimator=content/task_only/perf_vector_ridge_ei +wandb.name=Linear-Perf-TL32
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=32 estimator=content/task_only/perf_vector_rf_ei +estimator.use_metadata=false estimator.performance_transform.n_components=null +wandb.name=RF-Perf-TL32
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=32 estimator=deep/two_tower_perf_cash estimator.update_task_only=true +wandb.name=TwoTower-Perf-TL32
uv run python -m simple_metalearning.main %COMMON% %SLOW% experiment.task_limit=32 estimator=deep/two_tower_hyper_cash estimator.update_task_only=true +wandb.name=TwoTower-Hyper-TL32
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=32 estimator=content/task_only/botorch_gp_perf_cash +wandb.name=GP-Perf-TL32
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=32 estimator=content/task_only/ablr +wandb.name=ABLR-TL32
uv run python -m simple_metalearning.main %COMMON% %FAST% experiment.task_limit=32 estimator=content/task_only/active_testing +wandb.name=Active-Testing-TL32

REM ── Plots ────────────────────────────────────────────────────
uv run python scripts/scalability_auc_v3.py run-wandb --bench cash
uv run python scripts/scalability_auc_v3.py plot       --bench cash
