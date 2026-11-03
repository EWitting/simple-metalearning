#!/usr/bin/env bash
# ============================================================
#  sota.sh — SOTA transfer / meta-BO baselines (Linux / cluster)
#
#  Linux/macOS counterpart of sota.bat.  See that file for notes.
#  Runs are tagged with the canonical experiment tag (cash_v2 / hpo_v2)
#  plus "sota_v1" so they fold into the existing comparison plots.
#
#  RGPE fits one GP per training task once per fold. On HPO (201 configs)
#  this is feasible; on CASH (3623 configs) it is not, so RGPE and MTGP
#  are only run for HPO here (see paper supplement, cost analysis).
# ============================================================
set -euo pipefail

# One BLAS/torch thread per worker — avoids severe oversubscription when many
# joblib workers each spin up a full torch thread pool (GP/MTGP runs).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

N_JOBS_RGPE=6
RGPE="experiment.n_jobs=${N_JOBS_RGPE}"
WANDB_ENT="${WANDB_ENTITY:-}"

COMMON_HPO=(
  experiment=bo_selection
  experiment.n_folds=5
  experiment.n_folds_used=5
  experiment.max_visible=200
  dataset=tabarena
  dataset.methods=XGBoost
  dataset.normalization=dummy
  dataset.val_test_split=true
  +dataset.flip_sign=true
  +metafeature_transforms=standardscaler
  '+wandb.tags=[hpo_v2,sota_v1]'
)

COMMON_CASH=(
  experiment=bo_selection
  experiment.n_folds=5
  experiment.n_folds_used=5
  experiment.max_visible=200
  dataset=tabarena
  dataset.methods=base
  dataset.normalization=dummy
  dataset.val_test_split=true
  +dataset.flip_sign=true
  +metafeature_transforms=standardscaler
  '+wandb.tags=[cash_v2,sota_v1]'
)

SF="experiment.n_jobs=10"  # surrogate-free models: no GP, run wide
uv run python -m simple_metalearning.main "${COMMON_HPO[@]}"  $RGPE estimator=content/task_only/rgpe           +wandb.name=RGPE
uv run python -m simple_metalearning.main "${COMMON_HPO[@]}"  $SF   estimator=content/task_only/rgpe_oracle    +wandb.name=RGPE-Oracle
uv run python -m simple_metalearning.main "${COMMON_HPO[@]}"  $SF   estimator=content/task_only/tstr_discrete  +wandb.name=TST-R
uv run python -m simple_metalearning.main "${COMMON_HPO[@]}"  $SF   estimator=content/task_only/active_testing +wandb.name=Active-Testing
# MTGP: n_jobs=1 (BoTorch MultiTaskGP segfaults in loky's spawned workers).
uv run python -m simple_metalearning.main "${COMMON_HPO[@]}"  experiment.n_jobs=1 estimator=content/task_only/mtgp_hpo +wandb.name=MTGP
uv run python -m simple_metalearning.main "${COMMON_HPO[@]}"  $SF   estimator=content/task_only/ablr           +wandb.name=ABLR
# RGPE and MTGP on CASH are not run: GP-based base fitting does not stay
# computationally feasible at 3623 configs (see paper supplement, cost analysis).
uv run python -m simple_metalearning.main "${COMMON_CASH[@]}" $SF   estimator=content/task_only/rgpe_oracle    +wandb.name=RGPE-Oracle
uv run python -m simple_metalearning.main "${COMMON_CASH[@]}" $SF   estimator=content/task_only/tstr_discrete  +wandb.name=TST-R
uv run python -m simple_metalearning.main "${COMMON_CASH[@]}" $SF   estimator=content/task_only/active_testing +wandb.name=Active-Testing
uv run python -m simple_metalearning.main "${COMMON_CASH[@]}" $SF   estimator=content/task_only/ablr           +wandb.name=ABLR

uv run python scripts/plot_comparison.py --entity "$WANDB_ENT" --tags hpo_v2  --out-dir plots/hpo
uv run python scripts/plot_comparison.py --entity "$WANDB_ENT" --tags cash_v2 --out-dir plots/cash
