#!/usr/bin/env bash
# ============================================================
#  cash.sh — CASH experiment
#
#  Setup: TabArena base methods, 5-fold CV, val/test split,
#         200 BO iterations, dummy normalisation, flip_sign.
#
#  Compares all model families on the full CASH search space:
#    - Naive baselines (random, mean, mean+MMR)
#    - Linear-Perf     (BayesianRidge EI on perf-matrix columns, CF)
#    - Linear-Hyper    (BayesianRidge EI on flow hyperparams, content)
#    - RF-Hyper        (RF EI on flow hyperparams, SMAC-style)
#    - RF-Perf         (RF EI on raw perf vectors, no PCA)
#    - GP-Hyper        (BoTorch GP EI on flow hyperparams, content)
#    - GP-Perf         (BoTorch GP EI on raw perf vectors, CF, no PCA)
#    - TwoTower-Perf   (embed-only CF, EmbeddingMMR, dim=32)
#    - TwoTower-Hyper  (asym: task embed + flow MLP, EmbeddingMMR)
#
#  Order: baselines first, deep models last.
# ============================================================
set -e

N_JOBS_FAST=10
N_JOBS_SLOW=4

COMMON="experiment=bo_selection \
experiment.n_folds=5 \
experiment.n_folds_used=5 \
experiment.max_visible=200 \
dataset=tabarena \
dataset.methods=base \
dataset.normalization=dummy \
dataset.val_test_split=true \
+dataset.flip_sign=true \
+metafeature_transforms=standardscaler \
'+wandb.tags=[cash_v2]'"

# ── Naive baselines ─────────────────────────────────────────
# uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_FAST estimator=naive/random '+wandb.name=Naive-Random'
# uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_FAST estimator=naive/mean '+wandb.name=Naive-Mean'
# uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_FAST estimator=naive/mean_mmr estimator.mmr_lambda=0.05 '+wandb.name=Naive-Mean-MMR'

# ── Collaborative (performance-based, EI acquisition) ────────
# uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_FAST estimator=content/task_only/perf_vector_ridge_ei '+wandb.name=Linear-Perf'

# ── Content-based (hyperparameter-based, EI acquisition) ─────
# uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_FAST estimator=content/task_only/ridge_ei '+wandb.name=Linear-Hyper'
# uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_FAST estimator=content/task_only/rf_ei '+wandb.name=RF-Hyper'
# uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_FAST estimator=content/task_only/perf_vector_rf_ei +estimator.use_metadata=false 'estimator.performance_transform.n_components=null' '+wandb.name=RF-Perf'

# ── GP family (BoTorch; thread-limited to avoid BLAS/torch oversubscription
#    when many joblib workers each spin up a full internal thread pool) ──────
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_FAST estimator=content/task_only/botorch_gp_ei '+wandb.name=GP-Hyper'
uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_FAST estimator=content/task_only/botorch_gp_perf_cash '+wandb.name=GP-Perf'
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

# ── Deep models (PyTorch — use fewer parallel jobs) ──────────
uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_SLOW estimator=deep/two_tower_perf_cash estimator.update_task_only=true '+wandb.name=TwoTower-Perf'
uv run python -m simple_metalearning.main $COMMON experiment.n_jobs=$N_JOBS_SLOW estimator=deep/two_tower_hyper_cash estimator.update_task_only=true '+wandb.name=TwoTower-Hyper'

# ── Plots ────────────────────────────────────────────────────
uv run python scripts/plot_comparison.py --tags cash_v2 --out-dir plots/cash
uv run python scripts/plot_adaptive_overfitting.py --tags cash_v2 --out-dir plots/adaptive_overfitting/cash
