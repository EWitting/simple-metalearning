"""Download raw per-step metrics for all paper experiments from W&B.

Queries each experiment's tag set, keeps only the most recent run per
(name, tags) combination, and saves artifacts as CSV files organized by
experiment:

    results/
      cash/                   <- tags: [cash]
        Naive-Random.csv
        GP-Hyper.csv
        ...
      hpo/                    <- tags: [hpo]
        ...
      scalability/            <- tags: [scalability]
        ...
      scalability-hpo/        <- tags: [scalability-hpo]
        ...
      gp-perf-sweep-cash/     <- tags: [gp-perf-sweep, cash]
        ...
      gp-perf-sweep-hpo/
        ...

Usage
-----
    export WANDB_ENTITY=your-entity
    python scripts/pull_results.py

Options
-------
--project   W&B project (default: simple-metalearning)
--entity    W&B entity  (default: $WANDB_ENTITY env var, or your default W&B entity)
--out-dir   Root output directory (default: results/)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import wandb

# All tag sets used across the experiment bat files.
# Each entry is one experiment group; runs in wandb must have ALL listed tags.
EXPERIMENT_TAG_SETS: list[list[str]] = [
    ["cash"],
    ["hpo"],
    ["scalability"],
    ["scalability-hpo"],
]


def _subdir_name(tags: list[str]) -> str:
    return "-".join(tags)


def _model_family(run_name: str) -> str:
    """Coarse family from run name prefix (before first '-')."""
    return run_name.split("-")[0]


def _runtime_seconds(run: wandb.apis.public.Run) -> float | None:
    rt = run.summary.get("_runtime")
    return float(rt) if rt is not None else None


def _fetch_latest_per_name(
    tags: list[str],
    entity: str,
    project: str,
) -> dict[str, wandb.apis.public.Run]:
    """Return the most recent run for each distinct name within this tag set."""
    api = wandb.Api()
    runs = api.runs(
        (f"{entity}/{project}" if entity else project),
        filters={"tags": {"$all": tags}},
        order="-created_at",
    )
    latest: dict[str, wandb.apis.public.Run] = {}
    for run in runs:
        if run.name not in latest:
            latest[run.name] = run
    return latest


def _download_artifact(run: wandb.apis.public.Run) -> list[dict]:
    artifacts = run.logged_artifacts()
    raw_artifact = None
    for art in artifacts:
        if art.type == "raw_metrics":
            raw_artifact = art
            break
    if raw_artifact is None:
        raise ValueError(f"No raw_metrics artifact found for run '{run.name}'")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_artifact.download(root=tmpdir)
        with open(Path(tmpdir) / "metrics.json", encoding="utf-8") as f:
            return json.load(f)


def _format_hours(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m"


def _build_compute_report(
    runtime_rows: list[tuple[str, str, str, float | None]],
) -> str:
    """Build a human-readable compute report from (experiment, run_name, run_id, seconds) rows."""
    lines: list[str] = []

    # Per-experiment breakdown
    lines.append("=== Compute by experiment ===\n")
    from itertools import groupby as _groupby
    for exp, group in _groupby(runtime_rows, key=lambda r: r[0]):
        group = list(group)
        known = [r[3] for r in group if r[3] is not None]
        exp_total = sum(known)
        missing = len(group) - len(known)
        lines.append(f"  [{exp}]  {_format_hours(exp_total)}"
                     + (f"  (+ {missing} run(s) with no runtime)" if missing else ""))
        for _, run_name, run_id, rt in group:
            rt_str = _format_hours(rt) if rt is not None else "N/A"
            lines.append(f"    {run_name:<35s} {rt_str}  (id={run_id})")
        lines.append("")

    # Per model family
    lines.append("=== Compute by model family ===\n")
    family_totals: dict[str, float] = {}
    for _, run_name, _, rt in runtime_rows:
        fam = _model_family(run_name)
        family_totals[fam] = family_totals.get(fam, 0.0) + (rt or 0.0)
    for fam, total in sorted(family_totals.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {fam:<15s} {_format_hours(total)}")
    lines.append("")

    # Grand total
    grand = sum(r[3] for r in runtime_rows if r[3] is not None)
    lines.append(f"=== Total compute: {_format_hours(grand)} ===")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw W&B run artifacts for all paper experiments "
                    "and report total compute time."
    )
    parser.add_argument("--project", default="simple-metalearning")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--out-dir", default="results", dest="out_dir")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    total_saved = 0
    # (experiment_label, run_name, run_id, runtime_seconds_or_None)
    runtime_rows: list[tuple[str, str, str, float | None]] = []

    for tags in EXPERIMENT_TAG_SETS:
        subdir = out_root / _subdir_name(tags)
        exp_label = _subdir_name(tags)
        print(f"\n[{' + '.join(tags)}] → {subdir}/")
        latest = _fetch_latest_per_name(tags, args.entity, args.project)
        if not latest:
            print("  No runs found, skipping.")
            continue

        subdir.mkdir(parents=True, exist_ok=True)
        for run_name, run in sorted(latest.items()):
            out_path = subdir / f"{run_name}.csv"
            rt = _runtime_seconds(run)
            runtime_rows.append((exp_label, run_name, run.id, rt))
            print(f"  {run_name} (id={run.id}, created={run.created_at[:10]})...",
                  end=" ", flush=True)
            try:
                logs = _download_artifact(run)
                pd.DataFrame(logs).to_csv(out_path, index=False)
                print(f"saved ({len(logs)} rows)")
                total_saved += 1
            except ValueError as e:
                print(f"SKIP — {e}")

    print(f"\nDone. {total_saved} CSV(s) written to {out_root}/")

    if runtime_rows:
        report = _build_compute_report(runtime_rows)
        print("\n" + report)
        report_path = out_root / "compute_report.txt"
        out_root.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"\nCompute report saved to {report_path}")


if __name__ == "__main__":
    main()
