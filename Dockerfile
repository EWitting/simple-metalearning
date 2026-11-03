# ============================================================
#  Dockerfile — simple-metalearning
#
#  Builds a CUDA 11.8 image with uv-managed Python environment.
#  GPU is optional; CPU-only runs are supported but will be
#  slower for the PyTorch-based estimators (TwoTower-*).
#
#  Build:
#    docker build -t metalearning-submission .
#
#  Run (GPU):
#    docker run --gpus all -v $(pwd)/data:/app/data \
#               -v $(pwd)/outputs:/app/outputs \
#               metalearning-submission \
#               bash experiments/greenlight_hpo.sh
#
#  Run (CPU):
#    docker run -v $(pwd)/data:/app/data \
#               -v $(pwd)/outputs:/app/outputs \
#               metalearning-submission \
#               bash experiments/greenlight_hpo.sh
#
#  wandb (optional):
#    Pass -e WANDB_API_KEY=<key> -e WANDB_ENTITY=<entity>
#    to enable online logging. Without these, runs are logged
#    offline to outputs/ (default mode: offline).
# ============================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv via official binary (no curl needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON=/usr/bin/python3.11

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Clone tabarena dependency
# The local path dep (external_repos/tabarena) is gitignored, so we clone it here.
# Package root with pyproject.toml is at external_repos/tabarena/tabarena/
RUN git clone https://github.com/autogluon/tabarena external_repos/tabarena

# Install Python environment
RUN uv sync --frozen

# Copy source code
COPY src/ src/
COPY conf/ conf/
COPY experiments/ experiments/
COPY scripts/ scripts/

# Make shell scripts executable
RUN chmod +x experiments/*.sh

# Data and outputs are mounted at runtime (see docker run -v above)
# Cache dir for processed datasets defaults to data/processed/
RUN mkdir -p data/processed outputs plots

# Default: run a quick smoke-test (1 fold, 5 BO steps, no wandb)
CMD ["uv", "run", "python", "-m", "simple_metalearning.main", \
     "experiment=bo_selection", \
     "experiment.n_folds=2", \
     "experiment.n_folds_used=1", \
     "experiment.max_visible=5", \
     "dataset=tabarena", \
     "dataset.methods=XGBoost", \
     "dataset.normalization=dummy", \
     "dataset.val_test_split=true", \
     "+dataset.flip_sign=true", \
     "+metafeature_transforms=standardscaler", \
     "estimator=naive/random", \
     "+wandb.name=smoke-test", \
     "wandb.mode=disabled"]
