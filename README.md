# Simple Meta-Learning for HPO and CASH Using Performance Representations

Code accompanying the paper *Simple Meta-Learning for HPO and CASH Using Performance Representations* by Emiel Witting, Cheng Yan and Tom Julian Viering (Delft University of Technology).

## Overview

This repository implements a meta-learning framework for **algorithm selection and hyperparameter optimisation (HPO)**. Given a new dataset (task), the system recommends which ML pipeline/hyperparameter configuration (flow) to evaluate first, by learning from a performance matrix of past evaluations across many tasks.

The core contribution is a Bayesian optimisation loop that uses various surrogate models — from simple baselines to GP-based and deep neural collaborative filtering — to sequentially recommend configurations for a new task.

## Quick Start

A two-run demo takes ~1-2 minutes on CPU and requires no wandb account:

```bash
# Linux / macOS
bash experiments/demo.sh

# Windows
experiments\demo.bat
```

This compares **Naive-Random** against **Linear-Perf** (BayesianRidge EI) on the XGBoost HPO setting (10 BO steps, 1 CV fold, wandb offline). Results are saved by Hydra to `outputs/<date>/<time>/`.

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA 11.8 (optional, for GPU-accelerated TwoTower models)
- A [Weights & Biases](https://wandb.ai) account (optional — experiments run offline by default)

## Setup

### 1. Clone and install

```bash
git clone https://github.com/EWitting/simple-metalearning.git
cd simple-metalearning

# Clone the TabArena data package (required)
git clone https://github.com/autogluon/tabarena external_repos/tabarena

# Install all dependencies
uv sync
```

### 2. Data

The TabArena benchmark dataset is downloaded automatically from OpenML on the first run and cached to `data/processed/`. This requires an internet connection on first use. Subsequent runs use the local cache.

### 3. wandb (optional)

By default all runs log offline to `outputs/`. To enable online logging:

```bash
wandb login
export WANDB_ENTITY=your-entity-name
# Then run with: +wandb.mode=online
```

## Running Experiments

All experiments use the Hydra CLI via `uv run python -m simple_metalearning.main`.

### Linux / macOS

```bash
# Full HPO experiment (all methods on XGBoost search space)
bash experiments/hpo.sh

# Full CASH experiment (all methods on full method set)
bash experiments/cash.sh

# Scalability study
bash experiments/scalability_cash.sh
bash experiments/scalability_hpo.sh

# Dedicated meta-learning / algorithm-selection baselines (ABLR, RGPE,
# RGPE-Oracle, TST-R, Active-Testing, MTGP)
bash experiments/sota.sh

# Regenerate all plots from existing W&B runs
bash experiments/plot_all.sh
```

### Windows

```batch
experiments\hpo.bat
experiments\cash.bat
experiments\scalability_cash.bat
experiments\scalability_hpo.bat
experiments\sota.bat
experiments\plot_all.bat

REM Linear model family ablations (OLS/Ridge/Ensemble/BayesRidge x
REM Greedy/MMR/EI/LogEI) and a sklearn-vs-BoTorch GP side comparison
experiments\linear_models_cash.bat
experiments\linear_models_hpo.bat
experiments\botorch_gp_hpo.bat
```

### Single run example

```bash
uv run python -m simple_metalearning.main \
  experiment=bo_selection \
  experiment.n_folds=5 \
  experiment.max_visible=200 \
  dataset=tabarena \
  dataset.methods=XGBoost \
  dataset.normalization=dummy \
  dataset.val_test_split=true \
  +dataset.flip_sign=true \
  +metafeature_transforms=standardscaler \
  estimator=naive/random \
  +wandb.name=my-run
```

## Docker

```bash
# Build
docker build -t simple-metalearning .

# Run with GPU (recommended for TwoTower models)
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  simple-metalearning \
  bash experiments/hpo.sh

# Run CPU-only
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  simple-metalearning \
  bash experiments/hpo.sh

# With wandb online logging
docker run --gpus all \
  -e WANDB_API_KEY=<your-key> \
  -e WANDB_ENTITY=<your-entity> \
  -v $(pwd)/data:/app/data \
  simple-metalearning \
  bash experiments/hpo.sh
```

## Project Structure

```
conf/                        # Hydra YAML configs
  config.yaml                # Entry point (experiment, estimator, dataset defaults)
  dataset/tabarena.yaml      # TabArena dataset config
  estimator/                 # One YAML per estimator variant
    naive/                   # random, mean, mean+MMR
    content/task_only/       # GP-EI, RF-EI, BayesianRidge-EI variants
    deep/                    # TwoTower variants (CASH/HPO x Perf/Hyper)
  experiment/bo_selection.yaml
  metafeature_transforms/    # Feature preprocessing pipelines
  acquisition/               # Acquisition function configs

src/simple_metalearning/
  main.py                    # Hydra entry point
  datasets/
    dataset.py               # MetaDataset dataclass + factory base
    tabarena.py              # TabArena benchmark loader
    metafeature_transforms.py
  estimators/
    estimator.py             # Abstract base: fit / partial_fit / predict / suggest
    bayesian_ridge_ei.py     # BayesianRidge + EI (Linear-Perf / Linear-Hyper)
    task_only_gp_ei.py       # Per-task sklearn GP + EI (superseded for GP-Hyper/GP-Perf, see below)
    task_only_botorch_gp_ei.py  # Per-task BoTorch GP + EI (GP-Hyper / GP-Perf, used by default)
    task_only_rf_ei.py       # Per-task RF + EI (RF-Hyper / RF-Perf)
    two_tower.py             # Two-tower neural CF (TwoTower-Perf / TwoTower-Hyper)
    torch_trainer.py         # PyTorch training loop
    random.py / mean.py / mean_mmr.py   # Naive baselines
    acquisition/             # EI (Gaussian, Forest), MMR, UCB, EmbeddingMMR
  experiments/
    bo_selection.py          # Iterative BO loop with k-fold CV
  metrics/
    metric_store.py          # Logging + wandb upload

experiments/                 # Experiment runner scripts
  cash.{bat,sh}
  hpo.{bat,sh}
  scalability_cash.{bat,sh}
  scalability_hpo.{bat,sh}
  sota.{bat,sh}              # Dedicated meta-learning / AS baselines
  plot_all.{bat,sh}          # Regenerate all plots from existing W&B runs
  demo.{bat,sh}
  linear_models_cash.bat     # Linear model family ablation (OLS/Ridge/...)
  linear_models_hpo.bat
  botorch_gp_hpo.{bat,sh}    # sklearn-vs-BoTorch GP side comparison (HPO)
  sweep_*_two_tower.yaml     # W&B sweep configs for TwoTower tuning

scripts/
  plot_comparison.py         # Regret/rank plots from wandb runs
  plot_scalability.py        # Scalability curves per method
  plot_scalability_auc.py    # AUC vs training-task-limit plots
```

## Estimators

| Name | Config key | Description |
|------|-----------|-------------|
| Naive-Random | `naive/random` | Random selection baseline |
| Naive-Mean | `naive/mean` | Per-flow global mean |
| Naive-Mean-MMR | `naive/mean_mmr` | Mean + MMR diversity |
| Linear-Perf | `content/task_only/perf_vector_ridge_ei` | BayesianRidge EI on performance vectors |
| Linear-Hyper | `content/task_only/ridge_ei` | BayesianRidge EI on flow hyperparams |
| RF-Hyper | `content/task_only/rf_ei` | Random Forest EI on flow hyperparams |
| RF-Perf | `content/task_only/perf_vector_rf_ei` | Random Forest EI on performance vectors |
| GP-Hyper | `content/task_only/botorch_gp_ei` | BoTorch GP (Matern-5/2, ARD) + EI on flow hyperparams |
| GP-Perf | `content/task_only/botorch_gp_perf_{cash,hpo}` | BoTorch GP + EI on raw performance vectors, no dimensionality reduction |
| TwoTower-Perf | `deep/two_tower_perf_{cash,hpo}` | Embedding-only collaborative filtering |
| TwoTower-Hyper | `deep/two_tower_hyper_{cash,hpo}` | Asymmetric two-tower (task embed + flow MLP) |

Dedicated meta-learning / algorithm-selection baselines (run via `experiments/sota.{sh,bat}`; RGPE and MTGP are HPO-only, not computationally feasible on the 3623-config CASH search space):

| Name | Config key | Description |
|------|-----------|-------------|
| ABLR | `content/task_only/ablr` | NN hyperparameter embedding + per-task Bayesian linear head (Perrone et al. 2018) |
| RGPE | `content/task_only/rgpe` | Ranking-weighted GP ensemble over per-task base GPs (Feurer et al. 2018). HPO only. |
| RGPE-Oracle | `content/task_only/rgpe_oracle` | RGPE ablation: base-GP predictions replaced with true performance columns |
| TST-R | `content/task_only/tstr_discrete` | Two-stage transfer surrogate (Wistuba, Schilling, and Schmidt-Thieme 2018) |
| Active-Testing | `content/task_only/active_testing` | Adaptive algorithm selection from cross-task performance similarity (Leite, Brazdil, and Vanschoren 2012) |
| MTGP | `content/task_only/mtgp_{hpo,cash}` | Multi-task GP with a joint task kernel (Swersky, Snoek, and Adams 2013). HPO only. |

## Key CLI overrides

| Override | Default | Description |
|----------|---------|-------------|
| `experiment.max_visible` | 200 | Max BO iterations |
| `experiment.n_folds` | 5 | Number of CV folds |
| `experiment.n_jobs` | 1 | Parallel workers (use 10 for non-PyTorch, 4 for PyTorch) |
| `experiment.task_limit` | — | Limit training tasks (scalability study) |
| `dataset.methods` | — | Filter methods: `base` (CASH) or `XGBoost` (HPO) |
| `wandb.mode` | offline | `offline` or `online` |
| `+wandb.name` | — | Run name in wandb |
| `+wandb.tags` | — | Run tags, e.g. `'[cash]'` |

## Results

Raw per-step metrics for all paper runs are in `results/`. Files are organized by experiment (`cash/`, `hpo/`, `scalability/`, etc.) with one CSV per method. Each CSV contains one row per (fold, task, BO step) with columns `fold_idx`, `task_idx`, `visible_num`, `test_normalized_regret`, and related metrics.

To re-download from wandb (fetches the most recent run per method per experiment):

```bash
export WANDB_ENTITY=your-entity
python scripts/pull_results.py --entity $WANDB_ENTITY
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
