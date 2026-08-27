"""Compare CLUSE-Test generated tests with benchmark reference tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set

from src.layers.common import mutant_copies
from src.mutation.mutation_engine import Mutant, run_suite_against_mutants, verify_no_false_positives
from src.utils.config import RESULTS_DIR
from src.utils.logger import get_logger
from src.utils.metrics import BenchmarkComparison

log = get_logger(__name__)


def _killed_ids(evaluated: List[Mutant]) -> Set[str]:
    return {m.mutant_id for m in evaluated if m.is_killed}


def _classification_metrics(cluse_killed_ids: Set[str], official_killed_ids: Set[str], all_ids: Set[str]) -> dict:
    tp = len(cluse_killed_ids & official_killed_ids)
    tn = len((all_ids - cluse_killed_ids) & (all_ids - official_killed_ids))
    fp = len(cluse_killed_ids - official_killed_ids)
    fn = len(official_killed_ids - cluse_killed_ids)
    total = len(all_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compare_with_official_test(problem_id: str, cluse_tests: List[str], official_test: str,
                               canonical_source: str, entry_point: str,
                               all_mutants: List[Mutant],
                               official_killed_ids: Set[str] | None = None,
                               cluse_killed_ids: Set[str] | None = None,
                               sanity_already_checked: bool = False) -> BenchmarkComparison:
    log.info("[Benchmark] %s — comparing final CLUSE suite with benchmark reference test", problem_id)

    all_ids = {m.mutant_id for m in all_mutants}
    if official_killed_ids is None:
        official_eval = run_suite_against_mutants([official_test], mutant_copies(all_mutants), entry_point)
        official_ids = _killed_ids(official_eval)
    else:
        official_ids = set(official_killed_ids)
    if cluse_killed_ids is None:
        cluse_eval = run_suite_against_mutants(cluse_tests, mutant_copies(all_mutants), entry_point)
        cluse_ids = _killed_ids(cluse_eval)
    else:
        cluse_ids = set(cluse_killed_ids)
    metrics = _classification_metrics(cluse_ids, official_ids, all_ids)

    total = len(all_mutants)
    equivalent_ids = {
        m.mutant_id for m in all_mutants
        if getattr(m, "equivalence_status", "UNKNOWN") == "STATIC_EQUIVALENT"
    }
    adjusted_ids = all_ids - equivalent_ids
    cluse_score = round(len(cluse_ids) / total, 4) if total else 0.0
    official_score = round(len(official_ids) / total, 4) if total else 0.0
    adjusted_total = len(adjusted_ids)
    cluse_adjusted = round(len(cluse_ids & adjusted_ids) / adjusted_total, 4) if adjusted_total else 0.0
    official_adjusted = round(len(official_ids & adjusted_ids) / adjusted_total, 4) if adjusted_total else 0.0
    if sanity_already_checked:
        sanity = {"all_passed": True, "failures": []}
        sanity_notes = "Generated tests were validated on the canonical solution before acceptance"
    else:
        sanity = verify_no_false_positives(cluse_tests, canonical_source, entry_point)
        sanity_notes = "All generated tests passed on canonical solution" if sanity["all_passed"] else "; ".join(sanity["failures"][:5])

    log.info("[Benchmark] %s — CLUSE=%.3f official=%.3f agreement=%.3f",
             problem_id, cluse_score, official_score, metrics["accuracy"])

    return BenchmarkComparison(
        problem_id=problem_id,
        cluse_mutation_score=cluse_score,
        official_mutation_score=official_score,
        cluse_killed=len(cluse_ids),
        official_killed=len(official_ids),
        total_mutants=total,
        kill_agreement_accuracy=metrics["accuracy"],
        kill_precision=metrics["precision"],
        kill_recall=metrics["recall"],
        kill_f1=metrics["f1"],
        cluse_wins=cluse_score >= official_score,
        sanity_check_passed=bool(sanity["all_passed"]),
        sanity_check_notes=sanity_notes,
        equivalent_mutants=len(equivalent_ids),
        adjusted_total_mutants=adjusted_total,
        cluse_equivalent_adjusted_score=cluse_adjusted,
        official_equivalent_adjusted_score=official_adjusted,
    )


def save_benchmark_result(comparison: BenchmarkComparison, output_dir: Path = RESULTS_DIR / "benchmark") -> Path:
    out_dir = Path(output_dir) / comparison.problem_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison.__dict__, f, indent=2)
    return out_path


def aggregate_benchmark_results(benchmark_dir: Path = RESULTS_DIR / "benchmark") -> dict:
    rows = []
    for p in sorted(Path(benchmark_dir).rglob("benchmark_comparison.json")):
        try:
            with open(p, encoding="utf-8") as f:
                rows.append(json.load(f))
        except Exception:
            continue
    if not rows:
        return {"n_problems": 0}
    n = len(rows)
    return {
        "n_problems": n,
        "avg_cluse_score": round(sum(r["cluse_mutation_score"] for r in rows) / n, 4),
        "avg_official_score": round(sum(r["official_mutation_score"] for r in rows) / n, 4),
        "avg_kill_agreement_accuracy": round(sum(r["kill_agreement_accuracy"] for r in rows) / n, 4),
        "cluse_win_rate": round(sum(1 for r in rows if r["cluse_wins"]) / n, 4),
        "sanity_pass_rate": round(sum(1 for r in rows if r["sanity_check_passed"]) / n, 4),
    }
