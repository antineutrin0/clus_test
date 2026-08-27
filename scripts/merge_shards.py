"""Combine multiple sharded pipeline runs (each produced by a separate
``run_pipeline.py --index-start/--index-end`` invocation) into one unified
report.

Why this exists: running the full 500-task EvoEval set in one process is
slow and, on a platform like Kaggle with runtime limits, not always
possible in one sitting. Each shard is a self-contained, resumable unit --
if shard 3 crashes at task 60/100, only that shard needs rerunning, not the
other 400 tasks. This script then reuses the pipeline's own aggregation
functions (``_aggregate_summaries``, ``_aggregate_comparison_rows``) and the
statistics/figure generator (``generate_statistical_report``) over the
concatenated per-shard results, so the combined report is produced by
exactly the same code path a single full run would have used -- it does not
duplicate or reimplement any aggregation logic.

Usage
-----
    python scripts/merge_shards.py \
        --shards results_shard0 results_shard1 results_shard2 results_shard3 results_shard4 \
        --output results_full

Each ``--shards`` entry must be a results directory produced by
``run_pipeline.py`` (i.e. containing a ``report/`` and ``raw/`` subdirectory,
per the report/raw output tiering). The combined output directory gets its
own ``report/`` and ``raw/`` -- ``raw/`` entries are copied (or symlinked)
in from every shard so the full per-problem diagnostic trace remains
available under one root, and ``report/`` is regenerated from the
concatenated per-problem data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

# Allow running as `python scripts/merge_shards.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import _aggregate_comparison_rows, _aggregate_summaries, _write_rows_csv, log_run_summary  # noqa: E402
from src.utils import config  # noqa: E402


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _link_or_copy_tree(src: Path, dst: Path, *, use_symlinks: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            # Name collision across shards (shouldn't happen for per-problem
            # dirs, which are uniquely named by task_id, but could happen for
            # shared subdirectories like "metrics" or "layer1"). Merge
            # directories recursively; for files, the first shard wins and a
            # warning is printed so a genuine collision isn't silently lost.
            if item.is_dir() and target.is_dir():
                _link_or_copy_tree(item, target, use_symlinks=use_symlinks)
            elif item.is_file():
                print(f"  [warn] '{item}' already exists at destination; keeping the first shard's copy, skipping this one")
            continue
        if use_symlinks:
            target.symlink_to(item.resolve())
        elif item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def merge_shards(shard_dirs: List[Path], output_dir: Path, *, symlink_raw: bool = False, seed: int = 42) -> Dict:
    output_dir = Path(output_dir)
    out_report = output_dir / config.REPORT_DIRNAME
    out_raw = output_dir / config.RAW_DIRNAME
    out_report.mkdir(parents=True, exist_ok=True)
    out_raw.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict] = []
    all_comparison_rows: List[Dict] = []
    all_selected_rows: List[Dict] = []
    per_shard_manifests: List[Dict] = []
    dataset_names: set[str] = set()
    dataset_subsets: set[str] = set()
    subset_counts: Dict[str, int] = {}
    llm_configuration: Dict = {}

    for shard_dir in shard_dirs:
        shard_dir = Path(shard_dir)
        shard_report = shard_dir / config.REPORT_DIRNAME
        shard_raw = shard_dir / config.RAW_DIRNAME
        if not shard_report.exists():
            # Back-compat: allow a shard produced before the report/raw
            # split existed, where everything lived directly under shard_dir.
            shard_report = shard_dir
            shard_raw = shard_dir

        summaries = _read_json(shard_report / "aggregate_summary.json", [])
        all_summaries.extend(summaries)

        comparison_rows = _read_json(shard_report / "comparison" / "proposed_vs_baseline_summary.json", [])
        all_comparison_rows.extend(comparison_rows)

        selected_rows = _read_json(shard_report / "selected_tasks.json", [])
        all_selected_rows.extend(selected_rows)

        manifest = _read_json(shard_report / "run_manifest.json", {})
        per_shard_manifests.append({"shard_dir": str(shard_dir), **manifest})
        dataset_names.update(manifest.get("dataset_names", []))
        dataset_subsets.update(manifest.get("dataset_subsets", []))
        for subset, count in (manifest.get("subset_counts") or {}).items():
            subset_counts[subset] = subset_counts.get(subset, 0) + count
        llm_configuration.update(manifest.get("llm_configuration", {}))

        if shard_raw.exists():
            print(f"Merging raw artifacts from {shard_raw} -> {out_raw}")
            _link_or_copy_tree(shard_raw, out_raw, use_symlinks=symlink_raw)

    # Re-run the pipeline's own aggregation logic over the concatenated
    # per-problem data -- this is the same function a single full run would
    # have called, so the combined report is produced by one code path
    # regardless of how many processes generated the underlying data.
    with open(out_report / "selected_tasks.json", "w", encoding="utf-8") as f:
        json.dump(all_selected_rows, f, indent=2)
    _write_rows_csv(all_selected_rows, out_report / "selected_tasks.csv")

    with open(out_report / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    with open(out_report / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_aggregate_summaries(all_summaries), f, indent=2)
    _write_rows_csv(all_summaries, out_report / "aggregate_summary.csv")

    if all_comparison_rows:
        comparison_dir = out_report / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        _write_rows_csv(all_comparison_rows, comparison_dir / "proposed_vs_baseline_summary.csv")
        with open(comparison_dir / "proposed_vs_baseline_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_comparison_rows, f, indent=2)
        with open(comparison_dir / "proposed_vs_baseline_aggregate.json", "w", encoding="utf-8") as f:
            json.dump(_aggregate_comparison_rows(all_comparison_rows), f, indent=2)

    combined_manifest = {
        "run_name": "merged",
        "shards": [str(s) for s in shard_dirs],
        "shard_manifests": per_shard_manifests,
        "selected_problems": len(all_selected_rows),
        "executed_problems": sum(1 for row in all_summaries if not row.get("skipped")),
        "skipped_problems": sum(1 for row in all_summaries if row.get("skipped")),
        "dataset_names": sorted(dataset_names),
        "dataset_subsets": sorted(dataset_subsets),
        "subset_counts": subset_counts,
        "llm_configuration": llm_configuration,
    }
    with open(out_report / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(combined_manifest, f, indent=2)

    try:
        from src.evaluation.statistical_analysis import generate_statistical_report

        generate_statistical_report(out_report, seed=seed, raw_dir=out_raw)
    except Exception as exc:  # pragma: no cover - best-effort, same as pipeline.py
        print(f"[warn] statistical report generation failed: {exc}; combined raw/report data is still available")

    log_run_summary("merged", all_summaries, all_comparison_rows)
    return combined_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=Path, nargs="+", required=True, help="Shard result directories, in any order")
    parser.add_argument("--output", type=Path, required=True, help="Combined output directory")
    parser.add_argument("--symlink-raw", action="store_true", help="Symlink raw/ artifacts instead of copying (saves disk, but breaks if a shard directory is later moved/deleted)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = merge_shards(args.shards, args.output, symlink_raw=args.symlink_raw, seed=args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
