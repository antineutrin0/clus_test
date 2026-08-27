"""End-to-end CLUSE-Test pipeline for normalized coding benchmark tasks."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.baselines.iterative_mutant_baseline import IterativeMutantBaseline
from src.clustering.cluster_mutants import MutantClusterer
from src.clustering.representative_selector import RepresentativeSelector, RepresentativeStack
from src.evaluation.benchmark_comparator import compare_with_official_test
from src.layers.common import mutant_copies
from src.layers.layer1_generator import Layer1Generator
from src.layers.layer2_refiner import Layer2Refiner
from src.layers.layer3_frontier import Layer3Frontier
from src.llm.llm_client import BaseLLMClient, get_baseline_client, get_client
from src.mutation.mutation_engine import (
    Mutant,
    attach_behavior_signatures,
    generate_mutants,
    mark_obvious_equivalents,
    run_suite_against_mutants,
    save_mutants,
)
from src.utils import config
from src.utils.dataset_loader import Problem, select_problems
from src.utils.logger import get_logger
from src.utils.metrics import ExperimentTracker
from src.utils.probes import build_probe_exprs

log = get_logger(__name__)


def _cluster_and_select(mutants: List[Mutant]) -> Tuple[Dict[int, List[Mutant]], List[RepresentativeStack]]:
    """Create the single cluster partition used by every generation layer."""
    clusters = MutantClusterer().fit_predict(mutants)
    stacks = RepresentativeSelector().select_all(clusters)
    return clusters, stacks


def _surviving_clusters_from_initial(
    initial_clusters: Dict[int, List[Mutant]],
    surviving_mutants: List[Mutant],
) -> Dict[int, List[Mutant]]:
    """Filter the initial partition without recomputing cluster assignments.

    Layers 2 and 3 retain the semantic grouping established before Layer 1.
    When an original representative is killed, ``choose_dynamic_targets``
    selects the most informative live member from that same cluster.

    Not used to write a cluster-dump file anymore (see `_surviving_cluster_metadata`
    for the lightweight, id-only version that actually gets persisted), but
    kept as the documented, unit-tested reference implementation of "filter,
    don't recompute."
    """
    live_ids = {mutant.mutant_id for mutant in surviving_mutants}
    return {
        int(cluster_id): [member for member in members if member.mutant_id in live_ids]
        for cluster_id, members in initial_clusters.items()
        if any(member.mutant_id in live_ids for member in members)
    }


def _surviving_cluster_metadata(
    stacks: List[RepresentativeStack],
    surviving_mutants: List[Mutant],
) -> List[Dict]:
    live_ids = {mutant.mutant_id for mutant in surviving_mutants}
    rows: List[Dict] = []
    for stack in stacks:
        members = list(stack.representatives) + list(stack.non_representatives)
        live_members = [member for member in members if member.mutant_id in live_ids]
        if not live_members:
            continue
        live_representatives = [
            member.mutant_id for member in stack.representatives if member.mutant_id in live_ids
        ]
        rows.append({
            "cluster_id": stack.cluster_id,
            "original_cluster_size": len(members),
            "surviving_cluster_size": len(live_members),
            "original_representative_ids": [member.mutant_id for member in stack.representatives],
            "live_original_representative_ids": live_representatives,
            "live_member_ids": [member.mutant_id for member in live_members],
            "strategy": stack.strategy,
            "cluster_assignment_reused": True,
        })
    return rows


def _metadata_for_stacks(stacks: List[RepresentativeStack]) -> List[Dict]:
    rows: List[Dict] = []
    for stack in stacks:
        reps = [m.mutant_id for m in stack.representatives]
        rows.append({
            "cluster_id": stack.cluster_id,
            "representative_ids": reps,
            "representative_count": len(reps),
            "non_representative_count": len(stack.non_representatives),
            "cluster_size": len(reps) + len(stack.non_representatives),
            "strategy": stack.strategy,
        })
    return rows


def _aggregate_summaries(summaries: List[Dict]) -> Dict:
    if not summaries:
        return {"n_problems": 0, "n_executed": 0, "n_skipped": 0}
    all_rows = list(summaries)
    summaries = [row for row in all_rows if not bool(row.get("skipped"))]
    if not summaries:
        return {"n_problems": len(all_rows), "n_executed": 0, "n_skipped": len(all_rows)}
    n = len(summaries)
    total_mutants = sum(int(r.get("total_mutants") or 0) for r in summaries)
    total_killed = sum(int(r.get("killed_mutants") or 0) for r in summaries)
    total_official_killed = sum(int(r.get("official_killed") or 0) for r in summaries)
    total_equivalent = sum(int(r.get("equivalent_mutants") or 0) for r in summaries)
    adjusted_total = max(0, total_mutants - total_equivalent)
    adjusted_killed = sum(
        min(int(r.get("killed_mutants") or 0), max(0, int(r.get("total_mutants") or 0) - int(r.get("equivalent_mutants") or 0)))
        for r in summaries
    )
    adjusted_official_killed = sum(
        min(int(r.get("official_killed") or 0), max(0, int(r.get("total_mutants") or 0) - int(r.get("equivalent_mutants") or 0)))
        for r in summaries
    )
    total_tokens = sum(int(r.get("total_tokens") or 0) for r in summaries)
    paid_tokens = sum(int(r.get("paid_api_tokens") or 0) for r in summaries)
    total_cost = sum(float(r.get("estimated_cost_usd") or 0.0) for r in summaries)
    paid_cost = sum(float(r.get("paid_api_cost_usd") or 0.0) for r in summaries)
    total_runtime = sum(float(r.get("runtime_sec") or 0.0) for r in summaries)

    def avg(key: str) -> float:
        return round(sum(float(r.get(key) or 0.0) for r in summaries) / n, 6)

    return {
        "n_problems": len(all_rows),
        "n_executed": n,
        "n_skipped": len(all_rows) - n,
        "macro_avg_final_score": avg("final_score"),
        "micro_final_score": round(total_killed / total_mutants, 6) if total_mutants else 0.0,
        "macro_avg_official_score": avg("official_score"),
        "micro_official_score": round(total_official_killed / total_mutants, 6) if total_mutants else 0.0,
        "macro_avg_equivalent_adjusted_score": avg("equivalent_adjusted_score"),
        "micro_equivalent_adjusted_score": round(adjusted_killed / adjusted_total, 6) if adjusted_total else 0.0,
        "macro_avg_official_equivalent_adjusted_score": avg("official_equivalent_adjusted_score"),
        "micro_official_equivalent_adjusted_score": round(adjusted_official_killed / adjusted_total, 6) if adjusted_total else 0.0,
        "total_equivalent_mutants": total_equivalent,
        "adjusted_total_mutants": adjusted_total,
        "avg_kill_agreement_accuracy": avg("kill_agreement_accuracy"),
        "total_mutants": total_mutants,
        "total_killed_mutants": total_killed,
        "total_tokens": total_tokens,
        "paid_api_tokens": paid_tokens,
        "total_estimated_cost_usd": round(total_cost, 8),
        "paid_api_cost_usd": round(paid_cost, 8),
        "total_runtime_sec": round(total_runtime, 4),
        "avg_runtime_sec": round(total_runtime / n, 4),
        "avg_generated_tests": avg("generated_tests"),
        "avg_llm_calls": avg("llm_calls"),
        "total_llm_calls": sum(int(r.get("llm_calls") or 0) for r in summaries),
        "productive_call_rate": round(
            sum(int(r.get("productive_calls") or 0) for r in summaries) /
            max(1, sum(int(r.get("llm_calls") or 0) for r in summaries)), 6
        ),
        "sanity_pass_rate": round(sum(1 for r in summaries if r.get("sanity_passed")) / n, 6),
    }


def _write_rows_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_comparison_rows(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n_problems": 0}
    n = len(rows)

    def avg(key: str) -> float:
        return round(sum(float(r.get(key) or 0.0) for r in rows) / n, 6)

    total_mutants = sum(int(r.get("total_mutants") or 0) for r in rows)
    proposed_killed = sum(int(r.get("proposed_killed") or 0) for r in rows)
    baseline_killed = sum(int(r.get("baseline_killed") or 0) for r in rows)
    total_equivalent = sum(int(r.get("equivalent_mutants") or 0) for r in rows)
    adjusted_total = max(0, total_mutants - total_equivalent)
    wins = sum(1 for r in rows if float(r.get("proposed_score") or 0) > float(r.get("baseline_score") or 0))
    ties = sum(1 for r in rows if float(r.get("proposed_score") or 0) == float(r.get("baseline_score") or 0))
    return {
        "n_problems": n,
        "macro_avg_proposed_score": avg("proposed_score"),
        "macro_avg_baseline_score": avg("baseline_score"),
        "macro_avg_official_score": avg("official_score"),
        "micro_proposed_score": round(proposed_killed / total_mutants, 6) if total_mutants else 0.0,
        "micro_baseline_score": round(baseline_killed / total_mutants, 6) if total_mutants else 0.0,
        "macro_avg_proposed_equivalent_adjusted_score": avg("proposed_equivalent_adjusted_score"),
        "macro_avg_baseline_equivalent_adjusted_score": avg("baseline_equivalent_adjusted_score"),
        "macro_avg_official_equivalent_adjusted_score": avg("official_equivalent_adjusted_score"),
        "adjusted_total_mutants": adjusted_total,
        "total_equivalent_mutants": total_equivalent,
        "avg_proposed_minus_baseline_score": avg("score_delta_proposed_minus_baseline"),
        "avg_proposed_tokens": avg("proposed_total_tokens"),
        "avg_baseline_tokens": avg("baseline_total_tokens"),
        "total_proposed_tokens": sum(int(r.get("proposed_total_tokens") or 0) for r in rows),
        "total_baseline_tokens": sum(int(r.get("baseline_total_tokens") or 0) for r in rows),
        "total_proposed_paid_tokens": sum(int(r.get("proposed_paid_tokens") or 0) for r in rows),
        "total_baseline_paid_tokens": sum(int(r.get("baseline_paid_tokens") or 0) for r in rows),
        "avg_baseline_tokens_at_first_final_score": avg("baseline_tokens_at_first_final_score"),
        "avg_baseline_tokens_at_plateau": avg("baseline_tokens_at_plateau"),
        "avg_baseline_score_at_matched_proposed_tokens": avg("baseline_score_at_matched_proposed_tokens"),
        "avg_baseline_score_at_matched_proposed_paid_tokens": avg("baseline_score_at_matched_proposed_paid_tokens"),
        "avg_proposed_runtime_sec": avg("proposed_runtime_sec"),
        "avg_baseline_runtime_sec": avg("baseline_runtime_sec"),
        "proposed_win_rate_vs_baseline": round(wins / n, 6),
        "tie_rate": round(ties / n, 6),
    }


def log_run_summary(run_name: str, summaries: List[Dict], comparison_rows: List[Dict]) -> None:
    """Print exactly one overall summary for a run (or a merged set of shards).

    Per-problem detail is intentionally not logged while a run is in
    progress (see `CLUSEPipeline.run`); this is the only place run-level
    numbers are printed, and it includes a token/cost breakdown per layer
    (proposed pipeline) and, when a baseline was run, a proposed-vs-baseline
    comparison. Standalone (module-level) so `scripts/merge_shards.py` can
    print the same summary for a combined multi-shard run without
    duplicating this logic.
    """
    executed = [row for row in summaries if not row.get("skipped")]
    skipped = len(summaries) - len(executed)
    if not executed:
        log.info("=== Run %s: no executed problems (skipped=%d) ===", run_name, skipped)
        return

    def _sum(key: str) -> float:
        return sum(float(row.get(key) or 0.0) for row in executed)

    total_tokens = _sum("total_tokens")
    paid_tokens = _sum("paid_api_tokens")
    total_cost = _sum("estimated_cost_usd")
    paid_cost = _sum("paid_api_cost_usd")
    avg_score = _sum("final_score") / len(executed)

    lines = [
        "",
        "=" * 64,
        f"RUN SUMMARY: {run_name}",
        "=" * 64,
        f"Problems executed: {len(executed)}   Skipped: {skipped}",
        f"Mean mutation score (proposed): {avg_score:.4f}",
        "",
        "Proposed pipeline -- token cost by layer:",
    ]
    for label, key in (("Layer 1 (local, free)", "layer1"), ("Layer 2 (API)", "layer2"), ("Layer 3 (API)", "layer3")):
        layer_tokens = _sum(f"{key}_tokens")
        layer_cost = _sum(f"{key}_cost_usd")
        layer_calls = _sum(f"{key}_calls")
        lines.append(
            f"  {label:<24} tokens={int(layer_tokens):>10,}   "
            f"calls={int(layer_calls):>6,}   cost=${layer_cost:,.4f}"
        )
    lines.append(
        f"  {'Proposed total':<24} tokens={int(total_tokens):>10,}   "
        f"paid_tokens={int(paid_tokens):>10,}   cost=${total_cost:,.4f}   paid_cost=${paid_cost:,.4f}"
    )

    if comparison_rows:
        n = len(comparison_rows)
        base_tokens = sum(int(r.get("baseline_total_tokens") or 0) for r in comparison_rows)
        base_paid_tokens = sum(int(r.get("baseline_paid_tokens") or 0) for r in comparison_rows)
        base_cost = sum(float(r.get("baseline_estimated_cost_usd") or 0.0) for r in comparison_rows)
        base_paid_cost = sum(float(r.get("baseline_paid_cost_usd") or 0.0) for r in comparison_rows)
        prop_tokens_matched = sum(int(r.get("proposed_total_tokens") or 0) for r in comparison_rows)
        prop_cost_matched = sum(float(r.get("proposed_estimated_cost_usd") or 0.0) for r in comparison_rows)
        token_savings = (1 - prop_tokens_matched / base_tokens) * 100 if base_tokens else 0.0
        cost_savings = (1 - prop_cost_matched / base_cost) * 100 if base_cost else 0.0
        lines += [
            "",
            f"Baseline pipeline (non-clustering, {n} problem(s) compared):",
            f"  {'Baseline total':<24} tokens={base_tokens:>10,}   "
            f"paid_tokens={base_paid_tokens:>10,}   cost=${base_cost:,.4f}   paid_cost=${base_paid_cost:,.4f}",
            "",
            f"Proposed vs baseline: {token_savings:+.1f}% tokens, {cost_savings:+.1f}% cost "
            f"({'proposed cheaper' if token_savings > 0 else 'baseline cheaper'})",
        ]
    lines.append("=" * 64)
    log.info("\n".join(lines))


class CLUSEPipeline:
    def __init__(
        self,
        results_dir: Path = config.RESULTS_DIR,
        max_layers: int = 3,
        max_mutants: int = config.MAX_MUTANTS_PER_PROBLEM,
        max_probes: int = config.MAX_PROBES_PER_PROBLEM,
        problem_limit: Optional[int] = config.PROBLEM_LIMIT,
        problem_percent: float = config.PROBLEM_PERCENT,
        sample_mode: str = config.SAMPLE_MODE,
        stratify_by: str = "dataset_subset",
        seed: int = config.RANDOM_SEED,
        run_name: str = "default",
        run_baseline: bool = config.RUN_BASELINE,
        baseline_only: bool = config.BASELINE_ONLY,
        layer_providers: Optional[Dict[int, str]] = None,
        layer_models: Optional[Dict[int, str]] = None,
        layer_fallback_models: Optional[Dict[int, List[str]]] = None,
        layer_max_tokens: Optional[Dict[int, int]] = None,
        baseline_provider: str = config.BASELINE_PROVIDER,
        baseline_model: str = config.BASELINE_MODEL,
        baseline_fallback_models: Optional[List[str]] = None,
        baseline_max_tokens: int = config.BASELINE_MAX_TOKENS,
        baseline_max_iterations: int = config.BASELINE_MAX_ITERATIONS,
        generate_statistics: bool = config.GENERATE_STATISTICS,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.max_layers = max_layers
        self.max_mutants = max_mutants
        self.max_probes = max_probes
        self.problem_limit = problem_limit
        self.problem_percent = problem_percent
        self.sample_mode = sample_mode
        self.stratify_by = stratify_by
        self.seed = seed
        self.run_name = run_name
        self.run_baseline = run_baseline
        self.baseline_only = baseline_only
        self.baseline_provider = baseline_provider
        self.baseline_model = baseline_model
        self.baseline_fallback_models = baseline_fallback_models
        self.baseline_max_tokens = baseline_max_tokens
        self.baseline_max_iterations = baseline_max_iterations
        self.generate_statistics = generate_statistics
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Output tiering, deliberately kept minimal: `report/` holds curated,
        # human-facing artifacts (aggregate metrics, comparison tables,
        # statistics, figures, final test suites, run manifest); `raw/`
        # holds only what a completed run actually needs to keep per task --
        # the final mutant set, the surviving mutants, and the full metrics
        # record (tokens, cost, layer breakdown) for the proposed pipeline
        # and, when run, the baseline. Intermediate per-layer/per-attempt
        # dumps (layer1/2/3 result+handoff files, the raw LLM prompt/response
        # trace, per-layer cluster snapshots, and standalone benchmark files)
        # are not written at all: every fact in them is either redundant with
        # final_mutants.json/survived_mutants.json or already captured,
        # compactly, inside the metrics JSON via `tracker.metadata` /
        # `tracker.record_layer` / `tracker.record_benchmark`.
        self.report_dir = self.results_dir / config.REPORT_DIRNAME
        self.raw_dir = self.results_dir / config.RAW_DIRNAME
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        providers = layer_providers or {}
        models = layer_models or {}
        fallbacks = layer_fallback_models or {}
        max_tokens = layer_max_tokens or {}
        self.layer_clients: Dict[int, Optional[BaseLLMClient]] = {1: None, 2: None, 3: None}
        if not self.baseline_only:
            for layer in range(1, self.max_layers + 1):
                self.layer_clients[layer] = get_client(
                    layer=layer,
                    provider=providers.get(layer),
                    model_name=models.get(layer),
                    fallback_models=fallbacks.get(layer),
                    max_output_tokens=max_tokens.get(layer),
                )
        self.baseline_llm = None
        if self.run_baseline or self.baseline_only:
            self.baseline_llm = get_baseline_client(
                model_name=self.baseline_model,
                provider=self.baseline_provider,
                fallback_models=self.baseline_fallback_models,
                max_output_tokens=self.baseline_max_tokens,
            )

        self.provider_config = {
            f"layer{layer}": {
                "provider": getattr(client, "provider_name", None),
                "model": getattr(client, "model_name", None),
            }
            for layer, client in self.layer_clients.items() if client is not None
        }
        if self.baseline_llm is not None:
            self.provider_config["baseline"] = {
                "provider": getattr(self.baseline_llm, "provider_name", self.baseline_provider),
                "model": getattr(self.baseline_llm, "model_name", self.baseline_model),
            }

    def run_baseline_problem(
        self,
        problem: Problem,
        mutants: List[Mutant],
        official_killed_ids: Optional[set[str]] = None,
        probe_exprs: Optional[List[str]] = None,
        probe_outcomes: Optional[List[List[str]]] = None,
    ) -> ExperimentTracker:
        # Nested under raw/metrics/ (not a sibling top-level folder) so
        # proposed- and baseline-run metrics live in one place; filenames
        # still disambiguate ("<task>_metrics.json" vs "baseline/<task>_metrics.json").
        tracker = ExperimentTracker(problem_id=problem.task_id, save_dir=self.raw_dir / "metrics" / "baseline")
        tracker.metadata.update({
            "entry_point": problem.entry_point,
            "mutant_count": len(mutants),
            "run_name": self.run_name,
            "dataset_name": problem.dataset_name,
            "dataset_metadata": problem.metadata(),
            "baseline": "iterative_full_mutant_prompting",
            "baseline_provider": getattr(self.baseline_llm, "provider_name", self.baseline_provider),
            "baseline_model": getattr(self.baseline_llm, "model_name", self.baseline_model),
            "baseline_max_iterations": self.baseline_max_iterations,
            "official_killed_count": len(official_killed_ids or set()),
        })
        baseline = IterativeMutantBaseline(
            problem_id=problem.task_id,
            source_code=problem.complete_source,
            entry_point=problem.entry_point,
            prompt_text=problem.prompt_text,
            official_test=problem.official_test,
            probe_exprs=probe_exprs,
            probe_outcomes=probe_outcomes,
            final_suite_dir=self.report_dir / "generated_tests_baseline",
            model_name=self.baseline_model,
            max_iterations=self.baseline_max_iterations,
            llm=self.baseline_llm,
        )
        final_tests, surviving, metrics, benchmark = baseline.run(
            mutants,
            official_killed_ids=official_killed_ids,
            tracker=tracker,
        )
        tracker.record_layer(metrics)
        tracker.record_benchmark(benchmark)

        problem_dir = self.raw_dir / problem.safe_id
        problem_dir.mkdir(parents=True, exist_ok=True)
        save_mutants(mutants, problem_dir / "baseline_final_mutants.json")
        save_mutants([m for m in mutants if not m.is_killed], problem_dir / "baseline_survived_mutants.json")

        tracker.save()
        return tracker

    def _save_proposed_vs_baseline(
        self,
        problem: Problem,
        proposed_tracker: ExperimentTracker,
        baseline_tracker: ExperimentTracker,
    ) -> Dict:
        proposed = proposed_tracker.summary()
        baseline = baseline_tracker.summary()
        total_mutants = int(proposed.get("total_mutants") or proposed_tracker.metadata.get("mutant_count", 0))
        reps = sum(
            len(item.get("representative_ids", []))
            for item in proposed_tracker.metadata.get("layer1_clusters", [])
        )
        proposed_runtime = round(sum(float(layer.total_time_sec or 0.0) for layer in proposed_tracker.layers), 4)
        row = {
            "problem_id": problem.task_id,
            "dataset_name": problem.dataset_name,
            "dataset_subset": problem.dataset_subset,
            "source_task_id": problem.source_task_id or problem.task_id,
            "parent_task_id": problem.parent_task_id,
            "total_mutants": total_mutants,
            "proposed_killed": int(proposed.get("killed_mutants") or 0),
            "baseline_killed": int(baseline.get("killed_mutants") or 0),
            "official_killed": int(proposed.get("official_killed") or baseline.get("official_killed") or 0),
            "proposed_score": float(proposed.get("final_score") or 0.0),
            "baseline_score": float(baseline.get("final_score") or 0.0),
            "official_score": float(proposed.get("official_score") or baseline.get("official_score") or 0.0),
            "equivalent_mutants": int(proposed.get("equivalent_mutants") or 0),
            "proposed_equivalent_adjusted_score": float(proposed.get("equivalent_adjusted_score") or 0.0),
            "baseline_equivalent_adjusted_score": float(baseline.get("equivalent_adjusted_score") or 0.0),
            "official_equivalent_adjusted_score": float(proposed.get("official_equivalent_adjusted_score") or 0.0),
            "score_delta_proposed_minus_baseline": round(float(proposed.get("final_score") or 0.0) - float(baseline.get("final_score") or 0.0), 6),
            "proposed_total_tokens": int(proposed.get("total_tokens") or 0),
            "baseline_total_tokens": int(baseline.get("total_tokens") or 0),
            "proposed_llm_calls": len(proposed_tracker.llm_calls),
            "baseline_llm_calls": len(baseline_tracker.llm_calls),
            "proposed_paid_tokens": proposed_tracker.paid_api_tokens(),
            "baseline_paid_tokens": baseline_tracker.paid_api_tokens(),
            "token_delta_proposed_minus_baseline": int(proposed.get("total_tokens") or 0) - int(baseline.get("total_tokens") or 0),
            "proposed_estimated_cost_usd": float(proposed.get("estimated_cost_usd") or 0.0),
            "baseline_estimated_cost_usd": float(baseline.get("estimated_cost_usd") or 0.0),
            "proposed_paid_cost_usd": proposed_tracker.paid_api_cost(),
            "baseline_paid_cost_usd": baseline_tracker.paid_api_cost(),
            "proposed_runtime_sec": proposed_runtime,
            "baseline_runtime_sec": float(baseline.get("runtime_sec") or 0.0),
            "proposed_generated_tests": int(proposed.get("generated_tests") or 0),
            "baseline_generated_tests": int(baseline.get("generated_tests") or 0),
            "proposed_kill_agreement_accuracy": proposed.get("kill_agreement_accuracy"),
            "baseline_kill_agreement_accuracy": baseline.get("kill_agreement_accuracy"),
            "layer1_num_clusters": len(proposed_tracker.metadata.get("layer1_clusters", [])),
            "layer1_representatives": reps,
            "cluster_compression_ratio": round(total_mutants / reps, 4) if reps else 0.0,
            "final_layer_used": max(
                [int(str(layer.layer).replace("Layer", "")) for layer in proposed_tracker.layers if str(layer.layer).startswith("Layer")] or [0]
            ),
            "layer1_killed_mutants": next((l.killed_mutants for l in proposed_tracker.layers if l.layer == "Layer1"), 0),
            "layer1_new_kills": next((l.new_kills for l in proposed_tracker.layers if l.layer == "Layer1"), 0),
            "layer2_new_kills": next((l.new_kills for l in proposed_tracker.layers if l.layer == "Layer2"), 0),
            "layer3_new_kills": next((l.new_kills for l in proposed_tracker.layers if l.layer == "Layer3"), 0),
            "layer2_used": any(l.layer == "Layer2" for l in proposed_tracker.layers),
            "layer3_used": any(l.layer == "Layer3" for l in proposed_tracker.layers),
            "layer1_iterations": proposed_tracker.metadata.get("layer1_iterations", 0),
            "baseline_iterations": baseline_tracker.metadata.get("baseline_iterations", 0),
            "baseline_first_final_score_iteration": baseline_tracker.metadata.get("first_final_score_iteration", 0),
            "baseline_tokens_at_first_final_score": baseline_tracker.metadata.get("tokens_at_first_final_score", 0),
            "baseline_first_plateau_iteration": baseline_tracker.metadata.get("first_plateau_iteration", 0),
            "baseline_tokens_at_plateau": baseline_tracker.metadata.get("tokens_at_plateau", 0),
            "baseline_score_at_matched_proposed_tokens": baseline_tracker.score_at_token_budget(
                int(proposed.get("total_tokens") or 0)
            ),
            "baseline_score_at_matched_proposed_paid_tokens": baseline_tracker.score_at_token_budget(
                proposed_tracker.paid_api_tokens(), paid=True
            ),
            "proposed_providers": proposed.get("providers", ""),
            "proposed_models": proposed.get("models", ""),
            "baseline_providers": baseline.get("providers", ""),
            "baseline_models": baseline.get("models", ""),
        }
        out_dir = self.report_dir / "comparison" / problem.safe_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "proposed_vs_baseline.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        proposed_tracker.metadata["baseline_comparison"] = row
        return row

    def run_problem(self, problem: Problem) -> ExperimentTracker:
        tracker = ExperimentTracker(problem_id=problem.task_id, save_dir=self.raw_dir / "metrics")
        problem_start = time.time()
        problem_dir = self.raw_dir / problem.safe_id
        problem_dir.mkdir(parents=True, exist_ok=True)

        missing_libraries = problem.missing_libraries()
        if missing_libraries and config.SKIP_MISSING_LIBRARIES:
            tracker.metadata.update({
                "dataset_name": problem.dataset_name,
                "dataset_metadata": problem.metadata(),
                "entry_point": problem.entry_point,
                "mutant_count": 0,
                "skipped": True,
                "skip_reason": "missing_declared_libraries",
                "missing_libraries": missing_libraries,
                "run_name": self.run_name,
            })
            (problem_dir / "environment_preflight.json").write_text(
                json.dumps({
                    "problem_id": problem.task_id,
                    "supported": False,
                    "missing_libraries": missing_libraries,
                    "action": "excluded_from_effectiveness_aggregates",
                }, indent=2),
                encoding="utf-8",
            )
            tracker.timing.total_problem_sec = round(time.time() - problem_start, 4)
            tracker.save()
            log.warning("Skipping %s because declared libraries are unavailable: %s", problem.task_id, missing_libraries)
            return tracker

        t0 = time.time()
        mutants = generate_mutants(problem.task_id, problem.complete_source, max_mutants=self.max_mutants)
        mark_obvious_equivalents(problem.complete_source, mutants)
        tracker.timing.mutant_generation_sec = round(time.time() - t0, 4)
        # No initial mutants.json snapshot: `mutants` is mutated in place as
        # layers run, and the same objects are saved as final_mutants.json /
        # survived_mutants.json below once generation finishes -- a separate
        # pre-generation dump would only be a stale, redundant copy.

        official_eval = run_suite_against_mutants([problem.official_test], mutant_copies(mutants), problem.entry_point)
        official_killed_ids = {m.mutant_id for m in official_eval if m.is_killed}

        t0 = time.time()
        probe_exprs = build_probe_exprs(problem.source_prompt, problem.entry_point, max_probes=self.max_probes)
        tracker.timing.probe_generation_sec = round(time.time() - t0, 4)
        t0 = time.time()
        probe_outcomes = attach_behavior_signatures(problem.complete_source, problem.entry_point, mutants, probe_exprs)
        tracker.timing.behavior_signature_sec = round(time.time() - t0, 4)
        baseline_seed_mutants = mutant_copies(mutants)

        tracker.metadata.update({
            "entry_point": problem.entry_point,
            "mutant_count": len(mutants),
            "probe_count": len(probe_exprs),
            "probe_exprs": probe_exprs,
            "canonical_probe_outcomes": [list(item) for item in probe_outcomes],
            "official_killed_count": len(official_killed_ids),
            "max_layers": self.max_layers,
            "max_mutants": self.max_mutants,
            "max_probes": self.max_probes,
            "run_name": self.run_name,
            "dataset_name": problem.dataset_name,
            "dataset_metadata": problem.metadata(),
            "llm_configuration": self.provider_config,
            "layer0_removed": True,
            "gem_removed": True,
            "dmsg_removed": True,
            "representative_strategy": "UBIG-RS",
            "generation_mode": "batched_representatives",
            "cluster_lifecycle": "single_initial_partition_reused_across_layers",
            "reclustering_disabled": True,
            "layer1_max_calls_per_problem": config.LAYER1_MAX_REFINEMENT,
            "layer2_max_calls_per_problem": config.LAYER2_MAX_REFINEMENT,
            "strict_productive_validation": config.REQUIRE_PRODUCTIVE_TEST,
            "static_equivalent_count": sum(
                1 for m in mutants if getattr(m, "equivalence_status", "UNKNOWN") == "STATIC_EQUIVALENT"
            ),
        })

        if not mutants:
            tracker.save()
            return tracker
        if self.baseline_only:
            return self.run_baseline_problem(
                problem, baseline_seed_mutants, official_killed_ids,
                probe_exprs=probe_exprs, probe_outcomes=probe_outcomes,
            )

        t0 = time.time()
        clusters, stacks = _cluster_and_select(mutants)
        elapsed = round(time.time() - t0, 4)
        tracker.timing.clustering_sec = elapsed
        tracker.timing.representative_selection_sec = elapsed
        # The cluster partition itself is not written to disk: it would mean
        # re-serializing every mutant's full record (including source) up to
        # three times per problem. `_metadata_for_stacks` below captures the
        # same cluster_id/representative_id structure at a fraction of the
        # size and is what actually lands in the metrics JSON.
        tracker.metadata["layer1_clusters"] = _metadata_for_stacks(stacks)

        all_tests: List[str] = []
        task_metadata = problem.metadata()
        task_metadata["dataset_name"] = problem.dataset_name
        layer1 = Layer1Generator(
            problem.task_id, problem.complete_source, problem.entry_point, problem.prompt_text,
            output_dir=self.raw_dir / "layer1", llm=self.layer_clients[1],
            probe_exprs=probe_exprs, probe_outcomes=probe_outcomes, task_metadata=task_metadata,
        )
        l1_tests, surviving, l1_metrics = layer1.run(stacks, mutants, all_tests, tracker=tracker)
        all_tests.extend(l1_tests)
        tracker.record_layer(l1_metrics)
        tracker.metadata["layer1_iterations"] = l1_metrics.llm_calls
        tracker.metadata["layer1_handoff"] = layer1.handoff
        previous_evaluated = mutants

        if not surviving:
            tracker.metadata["layer2_skipped_reason"] = "all_mutants_killed_by_layer1"
            tracker.metadata["layer3_skipped_reason"] = "all_mutants_killed_by_layer1"
            log.debug("[Pipeline] %s: Layer 1 killed all mutants; skipping Layers 2 and 3", problem.task_id)
        elif self.max_layers >= 2:
            tracker.metadata["layer2_clusters"] = _surviving_cluster_metadata(stacks, surviving)
            layer2 = Layer2Refiner(
                problem.task_id, problem.complete_source, problem.entry_point, problem.prompt_text,
                output_dir=self.raw_dir / "layer2", llm=self.layer_clients[2],
                probe_exprs=probe_exprs, probe_outcomes=probe_outcomes, task_metadata=task_metadata,
                previous_handoff=layer1.handoff,
            )
            # Reuse the original partition; only live members remain active.
            l2_tests, surviving, l2_metrics = layer2.run(stacks, mutants, all_tests, previous_evaluated, tracker=tracker)
            all_tests.extend(l2_tests)
            tracker.record_layer(l2_metrics)
            tracker.metadata["layer2_iterations"] = l2_metrics.llm_calls
            tracker.metadata["layer2_handoff"] = layer2.handoff
            previous_evaluated = mutants
        else:
            tracker.metadata["layer2_skipped_reason"] = "max_layers_below_2"

        if not surviving:
            if "layer3_skipped_reason" not in tracker.metadata:
                tracker.metadata["layer3_skipped_reason"] = "all_mutants_killed_before_layer3"
                log.debug("[Pipeline] %s: no survivors remain; skipping Layer 3", problem.task_id)
        elif self.max_layers >= 3:
            tracker.metadata["layer3_clusters"] = _surviving_cluster_metadata(stacks, surviving)
            layer3 = Layer3Frontier(
                problem.task_id, problem.complete_source, problem.entry_point, problem.prompt_text,
                output_dir=self.raw_dir / "layer3", llm=self.layer_clients[3],
                probe_exprs=probe_exprs, probe_outcomes=probe_outcomes, task_metadata=task_metadata,
                previous_handoff=layer2.handoff if "layer2" in locals() else layer1.handoff,
            )
            # Reuse the same original cluster IDs and choose a live member per cluster.
            l3_tests, surviving, l3_metrics = layer3.run(stacks, mutants, all_tests, previous_evaluated, tracker=tracker)
            all_tests.extend(l3_tests)
            tracker.record_layer(l3_metrics)
            tracker.metadata["layer3_iterations"] = l3_metrics.llm_calls
            tracker.metadata["layer3_handoff"] = layer3.handoff
        elif self.max_layers < 3:
            tracker.metadata["layer3_skipped_reason"] = "max_layers_below_3"

        save_mutants(mutants, problem_dir / "final_mutants.json")
        save_mutants([m for m in mutants if not m.is_killed], problem_dir / "survived_mutants.json")

        benchmark = compare_with_official_test(
            problem_id=problem.task_id,
            cluse_tests=all_tests,
            official_test=problem.official_test,
            canonical_source=problem.complete_source,
            entry_point=problem.entry_point,
            all_mutants=mutants,
            official_killed_ids=official_killed_ids,
            cluse_killed_ids={m.mutant_id for m in mutants if m.is_killed},
            sanity_already_checked=True,
        )
        # Not written to its own benchmark/ file: `tracker.record_benchmark`
        # already embeds this comparison in the per-problem metrics JSON
        # (raw/metrics/<task>_metrics.json, "benchmark" key), so a standalone
        # copy would just be the same data twice.
        tracker.record_benchmark(benchmark)

        if self.run_baseline:
            baseline_tracker = self.run_baseline_problem(
                problem, baseline_seed_mutants, official_killed_ids,
                probe_exprs=probe_exprs, probe_outcomes=probe_outcomes,
            )
            self._save_proposed_vs_baseline(problem, tracker, baseline_tracker)

        # Curated copy: just the final accepted test suite as runnable
        # Python, one file per task, under report/generated_tests/ -- this
        # is what a human reading results actually wants. (No separate raw
        # generated_tests.json dump: it would just be the same test strings
        # re-serialized as a JSON list next to this .py file.)
        final_suite_dir = self.report_dir / "generated_tests"
        final_suite_dir.mkdir(parents=True, exist_ok=True)
        suite_path = final_suite_dir / f"{problem.safe_id}.py"
        suite_path.write_text(
            f"# Final accepted test suite for {problem.task_id}\n"
            f"# {len(all_tests)} test function(s), mutation score computed over "
            f"{len(mutants)} mutant(s)\n\n" + "\n\n".join(all_tests) + "\n",
            encoding="utf-8",
        )

        # No normalized_task.json dump: task identity/subset fields already
        # live in tracker.metadata["dataset_metadata"] (saved in the metrics
        # JSON), and the canonical source / official test text are fully
        # recoverable from the dataset file itself -- re-copying them per
        # task added size without adding information not already kept.

        tracker.timing.total_problem_sec = round(time.time() - problem_start, 4)
        tracker.save()
        return tracker

    def run(self, problems: List[Problem]) -> List[Dict]:
        selected = select_problems(
            problems,
            indices=config.PROBLEM_INDICES,
            limit=self.problem_limit,
            percent=self.problem_percent,
            sample_mode=self.sample_mode,
            seed=self.seed,
            stratify_by=self.stratify_by,
        )
        summaries: List[Dict] = []
        comparison_rows: List[Dict] = []
        log.info(
            "Run %s: selected %d/%d problems, max_layers=%d, max_mutants=%d, max_probes=%d",
            self.run_name, len(selected), len(problems), self.max_layers, self.max_mutants, self.max_probes,
        )
        for index, problem in enumerate(selected, 1):
            # Progress heartbeat only -- no per-task score/token dump here.
            # Per-task detail (score, tokens, cost) is aggregated into the
            # one overall summary logged after the whole run finishes below,
            # and is always available in full per-problem in raw/metrics/.
            log.info("Running %s (%d/%d)...", problem.task_id, index, len(selected))
            tracker = self.run_problem(problem)
            summaries.append(tracker.summary())
            comparison = tracker.metadata.get("baseline_comparison")
            if comparison:
                comparison_rows.append(comparison)

        selected_rows = [
            {
                "selection_index": index,
                "task_id": problem.task_id,
                "dataset_name": problem.dataset_name,
                "dataset_subset": problem.dataset_subset,
                "source_task_id": problem.source_task_id or problem.task_id,
                "parent_task_id": problem.parent_task_id,
                "entry_point": problem.entry_point,
            }
            for index, problem in enumerate(selected, 1)
        ]
        _write_rows_csv(selected_rows, self.report_dir / "selected_tasks.csv")
        with open(self.report_dir / "selected_tasks.json", "w", encoding="utf-8") as f:
            json.dump(selected_rows, f, indent=2)

        with open(self.report_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        with open(self.report_dir / "aggregate_metrics.json", "w", encoding="utf-8") as f:
            json.dump(_aggregate_summaries(summaries), f, indent=2)
        _write_rows_csv(summaries, self.report_dir / "aggregate_summary.csv")

        if comparison_rows:
            comparison_dir = self.report_dir / "comparison"
            _write_rows_csv(comparison_rows, comparison_dir / "proposed_vs_baseline_summary.csv")
            with open(comparison_dir / "proposed_vs_baseline_summary.json", "w", encoding="utf-8") as f:
                json.dump(comparison_rows, f, indent=2)
            with open(comparison_dir / "proposed_vs_baseline_aggregate.json", "w", encoding="utf-8") as f:
                json.dump(_aggregate_comparison_rows(comparison_rows), f, indent=2)

        run_manifest = {
            "run_name": self.run_name,
            "results_dir": str(self.results_dir),
            "report_dir": str(self.report_dir),
            "raw_dir": str(self.raw_dir),
            "selected_problems": len(selected),
            "executed_problems": sum(1 for row in summaries if not row.get("skipped")),
            "skipped_problems": sum(1 for row in summaries if row.get("skipped")),
            "available_problems": len(problems),
            "dataset_names": sorted({p.dataset_name for p in selected}),
            "dataset_subsets": sorted({p.dataset_subset for p in selected if p.dataset_subset}),
            "subset_counts": {
                subset: sum(1 for p in selected if p.dataset_subset == subset)
                for subset in sorted({p.dataset_subset for p in selected if p.dataset_subset})
            },
            "sample_mode": self.sample_mode,
            "stratify_by": self.stratify_by,
            "llm_configuration": self.provider_config,
            "run_baseline": self.run_baseline,
            "baseline_only": self.baseline_only,
            "verbose_artifacts": config.VERBOSE_ARTIFACTS,
        }
        with open(self.report_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)

        if self.generate_statistics:
            try:
                from src.evaluation.statistical_analysis import generate_statistical_report

                generate_statistical_report(self.report_dir, seed=self.seed, raw_dir=self.raw_dir)
            except Exception:
                log.exception("Statistical report generation failed; raw metrics remain available")

        self._log_run_summary(summaries, comparison_rows)
        return summaries

    def _log_run_summary(self, summaries: List[Dict], comparison_rows: List[Dict]) -> None:
        log_run_summary(self.run_name, summaries, comparison_rows)
