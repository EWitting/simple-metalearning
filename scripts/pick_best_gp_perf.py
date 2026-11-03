"""Select optimal PCA n_components for BoTorch GP-Perf and update YAML configs.

Fetches all runs tagged 'gp-perf-sweep-botorch' from W&B, picks the
n_components with lowest mean area/visible_num/test_normalized_regret for
each context (CASH / HPO), then writes the result into the two YAML configs.

Usage:
    uv run python scripts/pick_best_gp_perf.py
"""

import os
import re
import sys
from pathlib import Path

import wandb

ENTITY = os.environ.get("WANDB_ENTITY")
PROJECT = "simple-metalearning"
TAG = "gp-perf-sweep-botorch"
METRIC = "area/visible_num/test_normalized_regret"

YAML_CASH = Path("conf/estimator/content/task_only/botorch_gp_perf_cash.yaml")
YAML_HPO  = Path("conf/estimator/content/task_only/botorch_gp_perf_hpo.yaml")

# Maps run-name suffix → n_components value written to YAML (null = raw)
NAME_TO_COMPONENTS = {
    "PCA8":  "8",
    "PCA16": "16",
    "PCA32": "32",
    "Raw":   "null",
}


def parse_run_name(name: str):
    """Return (variant, context) e.g. ('PCA16', 'CASH') or None."""
    m = re.match(r"GP-Perf-(PCA8|PCA16|PCA32|Raw)-(CASH|HPO)$", name)
    if not m:
        return None
    return m.group(1), m.group(2)


api = wandb.Api()
print(f"Fetching runs tagged '{TAG}' from {ENTITY}/{PROJECT} ...")
runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": {"$in": [TAG]}})

scores: dict[str, dict[str, list[float]]] = {"CASH": {}, "HPO": {}}
missing = []

for run in runs:
    parsed = parse_run_name(run.name)
    if parsed is None:
        continue
    variant, context = parsed
    value = run.summary.get(METRIC)
    if value is None:
        missing.append(f"  {run.name} ({run.id}) — metric missing")
        continue
    scores[context].setdefault(variant, []).append(float(value))

if missing:
    print("\nWarning — runs missing the metric:")
    for m in missing:
        print(m)

results = {}
for context in ("CASH", "HPO"):
    if not scores[context]:
        print(f"\nNo completed runs found for {context}. Skipping.")
        continue

    avgs = {v: sum(vs) / len(vs) for v, vs in scores[context].items()}
    ranked = sorted(avgs.items(), key=lambda x: x[1])

    print(f"\n{context} ranking ({METRIC}, lower = better):")
    for rank, (variant, avg) in enumerate(ranked, 1):
        n = len(scores[context][variant])
        marker = "  <-- BEST" if rank == 1 else ""
        print(f"  {rank}. {variant:<6}  {avg:.4f}  (n={n}){marker}")

    results[context] = ranked[0][0]

# Update YAML files
def update_yaml(path: Path, n_components_str: str, context: str):
    text = path.read_text()
    updated = re.sub(
        r"(n_components:\s*)(\d+|null)",
        rf"\g<1>{n_components_str}",
        text,
    )
    if updated == text:
        print(f"\n{context}: n_components already {n_components_str} in {path}")
    else:
        path.write_text(updated)
        print(f"\n{context}: updated {path} → n_components: {n_components_str}")

if "CASH" in results:
    comp_str = NAME_TO_COMPONENTS[results["CASH"]]
    update_yaml(YAML_CASH, comp_str, "CASH")

if "HPO" in results:
    comp_str = NAME_TO_COMPONENTS[results["HPO"]]
    update_yaml(YAML_HPO, comp_str, "HPO")

print("\nDone. Re-run gp_only.bat to use the updated configs.")
