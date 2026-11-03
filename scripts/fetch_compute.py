"""Sum W&B wall-clock (_runtime) across the paper's run tags for the compute table."""
import os
import wandb
from collections import defaultdict

ENTITY, PROJECT = os.environ.get("WANDB_ENTITY"), "simple-metalearning"
TAGS = ["cash_v2", "hpo_v2", "scalability", "scalability-hpo"]
api = wandb.Api()

seen = {}  # run.id -> (name, runtime, tag)
per_tag = defaultdict(float)
for tag in TAGS:
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": {"$in": [tag]}})
    for r in runs:
        rt = r.summary.get("_runtime")
        if rt is None:
            continue
        per_tag[tag] += rt
        if r.id not in seen:
            seen[r.id] = (r.name, rt, tag)


def fam(n):
    n = n or ""
    if n.startswith("Linear"): return "Linear"
    if n.startswith("RF") or "Random Forest" in n: return "Random Forest"
    if n.startswith("GP"): return "Gaussian Process"
    if "Tower" in n: return "Two-Tower"
    if n in {"Random", "Mean", "MeanMMR"}: return "Model-free"
    if n in {"ABLR", "RGPE", "RGPE-Oracle", "MTGP", "Active-Testing", "TST-R"}: return "Meta-learning baselines"
    return f"OTHER:{n}"


def hm(sec): return f"{int(sec//3600)}h {int((sec%3600)//60):02d}m"


per_fam = defaultdict(float)
per_name = defaultdict(float)
for _id, (name, rt, tag) in seen.items():
    per_fam[fam(name)] += rt
    per_name[name] += rt

print("=== per tag (sum, may double-count multi-tag runs) ===")
for t in TAGS:
    print(f"  {t:20s} {hm(per_tag[t])}")
total = sum(rt for _, rt, _ in seen.values())
print(f"\n=== per family (deduped by run id) ===")
for f, s in sorted(per_fam.items(), key=lambda x: -x[1]):
    print(f"  {f:26s} {hm(s)}")
print(f"\nTOTAL (deduped): {hm(total)}  ({total/3600:.1f} h, {len(seen)} runs)")
print("\n=== per run name ===")
for n, s in sorted(per_name.items(), key=lambda x: -x[1]):
    print(f"  {n:20s} {hm(s)}")
