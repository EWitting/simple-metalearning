#!/usr/bin/env bash
# ============================================================
#  plot_all.sh — regenerate all plots from W&B artifacts
# ============================================================
set -e

uv run python scripts/plot_comparison.py --tags cash_v2 --out-dir plots/cash
uv run python scripts/plot_comparison.py --tags hpo_v2 --out-dir plots/hpo

uv run python scripts/plot_adaptive_overfitting.py --tags cash_v2 --out-dir plots/adaptive_overfitting/cash
uv run python scripts/plot_adaptive_overfitting.py --tags hpo_v2 --out-dir plots/adaptive_overfitting/hpo

uv run python scripts/scalability_auc_v3.py run-wandb --bench cash
uv run python scripts/scalability_auc_v3.py plot       --bench cash

uv run python scripts/scalability_auc_v3.py run-wandb --bench hpo
uv run python scripts/scalability_auc_v3.py plot       --bench hpo
