"""Iterative full-survivor baseline with fairness/plateau instrumentation."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.evaluation.benchmark_comparator import compare_with_official_test
from src.layers.common import (
    apply_cumulative_kills,
    evaluate_single_generated_test,
    extract_check_function,
    summarize_generated_test,
    validation_failure_category,
)
from src.llm.llm_client import BaseLLMClient, get_baseline_client
from src.llm.prompt_builder import build_iterative_baseline_prompt
from src.mutation.mutation_engine import Mutant
from src.utils import config
from src.utils.logger import get_logger
from src.utils.metrics import BenchmarkComparison, ExperimentTracker, LLMCallRecord, LayerMetrics

log = get_logger(__name__)


class IterativeMutantBaseline:
    """Same-model, non-clustered prompting over every surviving mutant."""

    def __init__(
        self,
        problem_id: str,
        source_code: str,
        entry_point: str,
        prompt_text: str,
        official_test: str,
        probe_exprs: Optional[List[str]] = None,
        probe_outcomes: Optional[List[List[str]]] = None,
        final_suite_dir: Optional[Path] = None,
        model_name: str = config.BASELINE_MODEL,
        max_iterations: int = config.BASELINE_MAX_ITERATIONS,
        llm: Optional[BaseLLMClient] = None,
    ) -> None:
        self.problem_id = problem_id
        self.source_code = source_code
        self.entry_point = entry_point
        self.prompt_text = prompt_text
        self.official_test = official_test
        self.probe_exprs = list(probe_exprs or [])
        self.probe_outcomes = [list(item) for item in (probe_outcomes or [])]
        # Curated final-suite location only (report/generated_tests_baseline/
        # by default via the pipeline); no more raw per-problem trace dir.
        self.final_suite_dir = Path(final_suite_dir) if final_suite_dir is not None else None
        self.model_name = model_name
        self.max_iterations = max_iterations
        self._llm = llm if llm is not None else get_baseline_client(model_name)
        self.iteration_history: List[Dict] = []

    def run(
        self,
        all_mutants: List[Mutant],
        official_killed_ids: Optional[set[str]] = None,
        tracker: Optional[ExperimentTracker] = None,
    ) -> Tuple[List[str], List[Mutant], LayerMetrics, BenchmarkComparison]:
        layer_start = time.time()
        generation_time = evaluation_time = 0.0
        prompt_tokens = completion_tokens = thoughts_tokens = total_tokens = 0
        prompt_chars = response_chars = 0
        estimated_cost = 0.0
        accepted_tests: List[str] = []
        cumulative_killed_ids: set[str] = set()
        feedback: Dict = {"note": "Initial iteration; every surviving mutant is supplied."}
        no_gain_streak = 0
        first_plateau_iteration = 0
        tokens_at_plateau = 0
        accepted_calls = productive_calls = invalid_calls = zero_kill_calls = 0
        consecutive_api_errors = 0

        canonical_mutants = all_mutants
        for iteration in range(1, self.max_iterations + 1):
            surviving = [m for m in canonical_mutants if m.mutant_id not in cumulative_killed_ids]
            if not surviving:
                break

            prompt = build_iterative_baseline_prompt(
                source_code=self.source_code,
                entry_point=self.entry_point,
                prompt_text=self.prompt_text,
                surviving_mutants=surviving,
                previous_tests=accepted_tests,
                iteration=iteration,
                max_iterations=self.max_iterations,
                feedback=feedback,
                probe_exprs=self.probe_exprs,
                probe_outcomes=self.probe_outcomes,
            )
            call_record = LLMCallRecord(
                problem_id=self.problem_id,
                layer=f"BaselineIter{iteration}",
                model=getattr(self._llm, "model_name", self.model_name),
                provider=getattr(self._llm, "provider_name", ""),
                cluster_id=None,
                mutant_id=",".join(m.mutant_id for m in surviving[:25]),
                prompt=prompt,
                attempt=iteration,
                target_count=len(surviving),
                prompt_chars=len(prompt),
                batch_mode=False,
            )
            prompt_chars += len(prompt)
            if config.LOG_FULL_LLM_IO and config.should_trace_problem(self.problem_id):
                log.info("[Baseline Prompt] %s | iteration=%d | surviving=%d\n%s", self.problem_id, iteration, len(surviving), prompt)

            new_killed_ids: set[str] = set()
            err = ""
            test_code = ""
            response_summary: Dict[str, object] = {"candidate_calls": [], "assertions": []}
            try:
                t0 = time.time()
                response = self._llm.generate(prompt)
                generation_time += time.time() - t0
                consecutive_api_errors = 0
                call_record.response = response.text
                call_record.response_chars = len(response.text or "")
                response_chars += call_record.response_chars
                call_record.model = response.model
                call_record.provider = response.provider
                call_record.prompt_tokens = response.prompt_tokens
                call_record.completion_tokens = response.completion_tokens
                call_record.thoughts_tokens = response.thoughts_tokens
                call_record.total_tokens = response.total_tokens
                call_record.estimated_cost_usd = response.estimated_cost_usd
                call_record.latency_sec = response.latency_sec

                prompt_tokens += response.prompt_tokens
                completion_tokens += response.completion_tokens
                thoughts_tokens += response.thoughts_tokens
                total_tokens += response.total_tokens
                estimated_cost += response.estimated_cost_usd

                if config.LOG_FULL_LLM_IO and config.should_trace_problem(self.problem_id):
                    log.info("[Baseline Response] %s | iteration=%d | model=%s\n%s", self.problem_id, iteration, response.model, response.text)

                test_code = extract_check_function(response.text)
                call_record.extracted_test = test_code
                response_summary = summarize_generated_test(test_code)
                t0 = time.time()
                ok, killed_once, survived_once, err, killed_ids = evaluate_single_generated_test(
                    test_code,
                    self.source_code,
                    self.entry_point,
                    surviving,
                    require_kill=config.REQUIRE_PRODUCTIVE_TEST,
                )
                evaluation_time += time.time() - t0
                new_killed_ids = set(killed_ids) - cumulative_killed_ids
                call_record.passed_on_original = ok or err.startswith("valid but unproductive")
                call_record.killed_mutants = killed_once
                call_record.surviving_mutants = survived_once
                call_record.target_kills = len(new_killed_ids)
                call_record.new_kills = len(new_killed_ids)
                call_record.validation_reason = err

                if ok and new_killed_ids:
                    accepted_calls += 1
                    productive_calls += 1
                    accepted_tests.append(test_code)
                    cumulative_killed_ids.update(new_killed_ids)
                    call_record.status = "PRODUCTIVE"
                    no_gain_streak = 0
                else:
                    no_gain_streak += 1
                    if err.startswith("valid but unproductive"):
                        zero_kill_calls += 1
                        call_record.status = "VALID_ZERO_KILL"
                    else:
                        invalid_calls += 1
                        call_record.status = "REJECTED"
                    call_record.error = err
            except Exception as exc:
                invalid_calls += 1
                consecutive_api_errors += 1
                # Deliberately NOT incrementing no_gain_streak here: an
                # infrastructure error (API/timeout/OOM) means no test
                # strategy was actually evaluated, so it must not count as a
                # "tried and made no progress" attempt toward the plateau
                # stopping criterion -- the same distinction applied to the
                # proposed pipeline's layers (see PlateauTracker /
                # summarize_attempt_history in src/layers/common.py).
                call_record.status = "ERROR"
                call_record.error = f"{type(exc).__name__}: {exc!r}"
                call_record.validation_reason = call_record.error
                err = call_record.error
                log.exception("[Baseline] LLM call failed for %s iteration %d", self.problem_id, iteration)

            call_record.cumulative_kills = len(cumulative_killed_ids)
            call_record.cumulative_score = len(cumulative_killed_ids) / len(canonical_mutants) if canonical_mutants else 0.0
            if tracker:
                tracker.record_llm_call(call_record)

            history_row = {
                "iteration": iteration,
                "status": call_record.status,
                "new_kills": len(new_killed_ids),
                "cumulative_kills": len(cumulative_killed_ids),
                "cumulative_score": call_record.cumulative_score,
                "iteration_tokens": call_record.total_tokens,
                "cumulative_tokens": total_tokens,
                "cumulative_paid_tokens": total_tokens if call_record.provider not in {"hf", "mock", "local", ""} else 0,
                "surviving_count": len(canonical_mutants) - len(cumulative_killed_ids),
                "failure_category": (
                    "LLM_API_ERROR" if call_record.status == "ERROR"
                    else validation_failure_category(err, test_code)
                ),
                "candidate_calls": response_summary.get("candidate_calls", []),
                "assertions": response_summary.get("assertions", []),
                "error": err,
            }
            # The baseline normally uses one paid provider. Recompute cumulative
            # paid tokens from the tracker when available.
            if tracker:
                history_row["cumulative_paid_tokens"] = tracker.paid_api_tokens()
            self.iteration_history.append(history_row)

            if no_gain_streak >= max(1, config.BASELINE_PLATEAU_PATIENCE) and not first_plateau_iteration:
                first_plateau_iteration = iteration
                tokens_at_plateau = total_tokens
            feedback = {
                "previous_iteration": history_row,
                "no_gain_streak": no_gain_streak,
                "instruction": "Generate a genuinely different semantic or boundary test; do not repeat prior candidate calls.",
            }
            apply_cumulative_kills(canonical_mutants, cumulative_killed_ids, "Baseline:generated")

            if config.BASELINE_STOP_ON_PLATEAU and no_gain_streak >= max(1, config.BASELINE_PLATEAU_PATIENCE):
                break
            if consecutive_api_errors >= 2:
                log.error("[Baseline] stopping %s after two consecutive provider failures", self.problem_id)
                break

        evaluated = apply_cumulative_kills(canonical_mutants, cumulative_killed_ids, "Baseline:generated")
        killed = [m for m in evaluated if m.is_killed]
        surviving = [m for m in evaluated if not m.is_killed]
        final_kills = len(killed)
        first_final = next((row for row in self.iteration_history if row["cumulative_kills"] == final_kills), None)

        if not surviving:
            stop_reason = "zero_survivors"
        elif config.BASELINE_STOP_ON_PLATEAU and no_gain_streak >= max(1, config.BASELINE_PLATEAU_PATIENCE):
            stop_reason = f"plateau: {no_gain_streak} consecutive attempt(s) with zero new kills"
        elif consecutive_api_errors >= 2:
            stop_reason = "consecutive_infrastructure_errors"
        else:
            stop_reason = "budget_exhausted"

        if tracker:
            tracker.metadata.update({
                "baseline_iterations": len(self.iteration_history),
                "iteration_history": self.iteration_history,
                "first_final_score_iteration": int(first_final["iteration"]) if first_final else 0,
                "tokens_at_first_final_score": int(first_final["cumulative_tokens"]) if first_final else 0,
                "first_plateau_iteration": first_plateau_iteration,
                "tokens_at_plateau": tokens_at_plateau,
                "plateau_patience": config.BASELINE_PLATEAU_PATIENCE,
                "stopped_on_plateau": bool(config.BASELINE_STOP_ON_PLATEAU and first_plateau_iteration),
                "stop_reason": stop_reason,
            })

        metrics = LayerMetrics(
            layer="BaselineIterative",
            problem_id=self.problem_id,
            total_mutants=len(canonical_mutants),
            killed_mutants=len(killed),
            new_kills=len(killed),
            test_count=len(accepted_tests),
            cumulative_tests=len(accepted_tests),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thoughts_tokens=thoughts_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(estimated_cost, 8),
            generation_time_sec=round(generation_time, 4),
            evaluation_time_sec=round(evaluation_time, 4),
            total_time_sec=round(time.time() - layer_start, 4),
            cluster_kill_consistency=0.0,
            llm_calls=len(self.iteration_history),
            accepted_calls=accepted_calls,
            productive_calls=productive_calls,
            invalid_calls=invalid_calls,
            zero_kill_calls=zero_kill_calls,
            target_count=len(canonical_mutants),
            prompt_chars=prompt_chars,
            response_chars=response_chars,
            notes=(
                f"model={getattr(self._llm, 'model_name', self.model_name)} "
                f"max_iterations={self.max_iterations} actual_iterations={len(self.iteration_history)} "
                f"first_plateau={first_plateau_iteration} accepted_tests={len(accepted_tests)} "
                f"stop_reason={stop_reason}"
            ),
        )
        metrics.compute()

        benchmark = compare_with_official_test(
            problem_id=self.problem_id,
            cluse_tests=accepted_tests,
            official_test=self.official_test,
            canonical_source=self.source_code,
            entry_point=self.entry_point,
            all_mutants=canonical_mutants,
            official_killed_ids=official_killed_ids,
            cluse_killed_ids=set(cumulative_killed_ids),
            sanity_already_checked=True,
        )
        # Not written to its own file: `tracker.record_benchmark` (called by
        # the pipeline right after this returns) embeds it in the per-problem
        # baseline metrics JSON instead.
        if self.final_suite_dir is not None:
            self.assemble_final_suite(accepted_tests)
        log.debug(
            "[Baseline] %s score=%.3f tests=%d iterations=%d tokens=%d time=%.2fs",
            self.problem_id,
            metrics.mutation_score,
            len(accepted_tests),
            metrics.llm_calls,
            total_tokens,
            metrics.total_time_sec,
        )
        return accepted_tests, surviving, metrics, benchmark

    def assemble_final_suite(self, all_tests: List[str]) -> Path:
        """Curated final baseline suite, written straight into the report
        tier (report/generated_tests_baseline/<task>.py) -- no intermediate
        raw/baseline_final/ copy."""
        out_dir = Path(self.final_suite_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.problem_id.replace('/', '_')}.py"
        lines = [f"# Baseline iterative tests for {self.problem_id}", ""]
        for index, test in enumerate(all_tests):
            lines.append(re.sub(r"def\s+check\s*\(", f"def baseline_check_{index}(", test, count=1))
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
