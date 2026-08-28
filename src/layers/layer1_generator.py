"""Layer 1: batched local-model generation with at most three calls per task."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.clustering.representative_selector import RepresentativeStack
from src.layers.common import (
    PlateauTracker,
    all_probable_equivalent,
    apply_cumulative_kills,
    build_cluster_contexts,
    choose_dynamic_targets,
    compute_cluster_kill_consistency,
    evaluate_single_generated_test,
    extract_check_function,
    flatten_stacks,
    mutant_copies,
    reconcile_literal_oracle_assertions,
    summarize_generated_test,
    validation_failure_category,
)
from src.llm.llm_client import BaseLLMClient, get_client
from src.llm.prompt_builder import build_handoff_summary, build_layer1_batch_prompt
from src.mutation.mutation_engine import Mutant, run_suite_against_mutants
from src.utils import config
from src.utils.logger import get_logger
from src.utils.metrics import ExperimentTracker, LLMCallRecord, LayerMetrics

log = get_logger(__name__)


class Layer1Generator:
    def __init__(
        self,
        problem_id: str,
        source_code: str,
        entry_point: str,
        prompt_text: str,
        output_dir: Path = config.RESULTS_DIR / "layer1",
        llm: Optional[BaseLLMClient] = None,
        probe_exprs: Optional[Sequence[str]] = None,
        probe_outcomes: Optional[Sequence[Sequence[str]]] = None,
        task_metadata: Optional[Dict] = None,
    ) -> None:
        self.problem_id = problem_id
        self.source_code = source_code
        self.entry_point = entry_point
        self.prompt_text = prompt_text
        self.probe_exprs = list(probe_exprs or [])
        self.probe_outcomes = [list(item) for item in (probe_outcomes or [])]
        self.task_metadata = dict(task_metadata or {})
        # `output_dir` is accepted for backward-compatible call sites but no
        # longer written to: Layer 1's per-attempt trace was diagnostic-only
        # bloat (it duplicates data already captured, cheaply, in the
        # per-problem metrics JSON via `tracker.record_layer`/`record_llm_call`
        # and in `self.handoff`, which the pipeline reads in-memory to chain
        # into Layer 2). Kept as an attribute in case a caller inspects it.
        self.output_dir = Path(output_dir) / problem_id.replace("/", "_")
        self._llm = llm if llm is not None else get_client(layer=1)
        self.handoff: Dict = {}

    def run(
        self,
        stacks: List[RepresentativeStack],
        all_mutants: List[Mutant],
        prior_tests: List[str],
        tracker: Optional[ExperimentTracker] = None,
    ) -> Tuple[List[str], List[Mutant], LayerMetrics]:
        layer_start = time.time()
        generation_time = 0.0
        evaluation_time = 0.0
        representatives = flatten_stacks(stacks)

        prior_eval = run_suite_against_mutants(prior_tests, mutant_copies(
            all_mutants), self.entry_point) if prior_tests else []
        prior_killed_ids = {m.mutant_id for m in prior_eval if m.is_killed}
        cumulative_killed_ids = set(prior_killed_ids)
        new_tests: List[str] = []
        attempt_history: List[Dict] = []

        prompt_tokens = completion_tokens = thoughts_tokens = total_tokens = 0
        prompt_chars = response_chars = 0
        estimated_cost = 0.0
        accepted_calls = productive_calls = invalid_calls = zero_kill_calls = 0
        feedback: Dict = {
            "note": "Generate one joint test for all representative clusters."}

        max_attempts = max(1, config.LAYER1_MAX_REFINEMENT)
        plateau = PlateauTracker(
            patience=config.LAYER1_PLATEAU_PATIENCE, enabled=config.LAYER1_STOP_ON_PLATEAU)
        stop_reason = "budget_exhausted"
        last_target_ids: List[str] = []
        for attempt in range(1, max_attempts + 1):
            active = [
                m for m in all_mutants if m.mutant_id not in cumulative_killed_ids]
            if not active:
                stop_reason = "zero_survivors"
                break
            if config.SKIP_LAYER_WHEN_ALL_PROBABLE_EQUIVALENT and all_probable_equivalent(active):
                stop_reason = "all_remaining_probable_equivalent"
                log.debug("[Layer 1] %s: skipping remaining attempts, all %d survivors flagged PROBABLE_EQUIVALENT",
                          self.problem_id, len(active))
                break
            targets = choose_dynamic_targets(stacks, active)
            if not targets:
                targets = sorted(
                    active,
                    key=lambda m: (float(m.information_score or 0.0),
                                   float(m.centrality or 0.0)),
                    reverse=True,
                )[: min(config.MAX_CLUSTERS, len(active))]
            last_target_ids = [m.mutant_id for m in targets]
            contexts = build_cluster_contexts(stacks, active)
            prompt = build_layer1_batch_prompt(
                targets=targets,
                source_code=self.source_code,
                entry_point=self.entry_point,
                prompt_text=self.prompt_text,
                cluster_contexts=contexts,
                probe_exprs=self.probe_exprs,
                probe_outcomes=self.probe_outcomes,
                attempt=attempt,
                feedback=feedback,
                task_metadata=self.task_metadata,
            )
            call_record = LLMCallRecord(
                problem_id=self.problem_id,
                layer="Layer1",
                model=getattr(self._llm, "model_name", "layer1"),
                provider=getattr(self._llm, "provider_name", ""),
                cluster_id=None,
                mutant_id=",".join(m.mutant_id for m in targets),
                prompt=prompt,
                attempt=attempt,
                target_count=len(targets),
                prompt_chars=len(prompt),
                batch_mode=True,
            )
            prompt_chars += len(prompt)
            if config.LOG_FULL_LLM_IO and config.should_trace_problem(self.problem_id):
                log.info("[Layer 1 Batch Prompt] %s | attempt=%d | targets=%d\n%s",
                         self.problem_id, attempt, len(targets), prompt)

            try:
                t0 = time.time()
                response = self._llm.generate(prompt)
                generation_time += time.time() - t0
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
                    log.info("[Layer 1 Batch Response] %s | attempt=%d | model=%s\n%s",
                             self.problem_id, attempt, response.model, response.text)

                test_code = extract_check_function(response.text)
                # Layer-1-only fix: the local model frequently hallucinates the
                # exact literal it should be copying from <canonical_probe_oracles>
                # (wrong case/punctuation/spacing) even though the verified value
                # was handed to it verbatim in the prompt. Rather than relying on
                # prompt compliance alone, silently correct any assertion whose
                # `candidate(...)` input exactly matches an already-verified probe
                # -- this never invents a value, it only fixes a wrong constant for
                # an input we already know the true canonical answer to. Layer 2,
                # Layer 3, and the baseline are intentionally left untouched.
                test_code = reconcile_literal_oracle_assertions(
                    test_code, self.probe_exprs, self.probe_outcomes)
                call_record.extracted_test = test_code
                response_summary = summarize_generated_test(test_code)
                t0 = time.time()
                ok, killed_once, survived_once, err, killed_ids = evaluate_single_generated_test(
                    test_code,
                    self.source_code,
                    self.entry_point,
                    active,
                    require_kill=config.REQUIRE_PRODUCTIVE_TEST,
                )
                evaluation_time += time.time() - t0
                new_killed_ids = set(killed_ids) - cumulative_killed_ids
                target_ids = {m.mutant_id for m in targets}
                target_kills = len(new_killed_ids & target_ids)
                passed_original = ok or err.startswith(
                    "valid but unproductive")

                call_record.passed_on_original = passed_original
                call_record.killed_mutants = killed_once
                call_record.surviving_mutants = survived_once
                call_record.target_kills = target_kills
                call_record.new_kills = len(new_killed_ids)
                call_record.validation_reason = err

                if ok and new_killed_ids:
                    accepted_calls += 1
                    productive_calls += 1
                    new_tests.append(test_code)
                    cumulative_killed_ids.update(new_killed_ids)
                    call_record.status = "PRODUCTIVE"
                else:
                    if err.startswith("valid but unproductive"):
                        zero_kill_calls += 1
                        call_record.status = "VALID_ZERO_KILL"
                    else:
                        invalid_calls += 1
                        call_record.status = "REJECTED"
                    call_record.error = err

                call_record.cumulative_kills = len(cumulative_killed_ids)
                call_record.cumulative_score = len(
                    cumulative_killed_ids) / len(all_mutants) if all_mutants else 0.0
                attempt_row = {
                    "attempt": attempt,
                    "target_count": len(targets),
                    "target_kills": target_kills,
                    "new_kills": len(new_killed_ids),
                    "cumulative_kills": len(cumulative_killed_ids),
                    "status": call_record.status,
                    "failure_category": validation_failure_category(err, test_code),
                    "error": err,
                    "candidate_calls": response_summary.get("candidate_calls", []),
                    "assertions": response_summary.get("assertions", []),
                    "tokens": response.total_tokens,
                }
                attempt_history.append(attempt_row)
                feedback = {
                    "previous_attempt": attempt_row,
                    "remaining_mutants": len(all_mutants) - len(cumulative_killed_ids),
                    "instruction": "Generate novel assertions for the remaining exact deltas; do not repeat earlier candidate calls.",
                }
            except Exception as exc:
                invalid_calls += 1
                call_record.status = "ERROR"
                call_record.error = f"{type(exc).__name__}: {exc!r}"
                call_record.validation_reason = call_record.error
                call_record.cumulative_kills = len(cumulative_killed_ids)
                call_record.cumulative_score = len(
                    cumulative_killed_ids) / len(all_mutants) if all_mutants else 0.0
                attempt_history.append({
                    "attempt": attempt,
                    "target_count": len(targets),
                    "target_kills": 0,
                    "new_kills": 0,
                    "cumulative_kills": len(cumulative_killed_ids),
                    "status": "ERROR",
                    "failure_category": "LLM_API_ERROR",
                    "error": call_record.error,
                    "candidate_calls": [],
                    "assertions": [],
                    "tokens": 0,
                })
                feedback = {"previous_error": call_record.error,
                            "instruction": "Return only a valid check(candidate) function."}
                log.exception(
                    "[Layer 1] batched LLM call failed for %s attempt %d", self.problem_id, attempt)
            finally:
                if tracker:
                    tracker.record_llm_call(call_record)

            # Plateau check: an infrastructure-error attempt (no test was
            # actually evaluated) does not count as a zero-kill attempt for
            # plateau purposes -- only a genuine zero-new-kill *evaluation*
            # should count toward giving up early.
            if call_record.status != "ERROR":
                plateau.record(len(new_killed_ids))
                if plateau.should_stop():
                    stop_reason = plateau.stop_reason
                    break

        cumulative_tests = prior_tests + new_tests
        evaluated = apply_cumulative_kills(
            all_mutants, cumulative_killed_ids, "Layer1:generated")
        killed = [m for m in evaluated if m.is_killed]
        surviving = [m for m in evaluated if not m.is_killed]
        new_kills = len(cumulative_killed_ids - prior_killed_ids)

        self.handoff = build_handoff_summary(
            layer="Layer1",
            attempts=attempt_history,
            surviving_mutants=surviving,
            previous_tests=cumulative_tests,
            target_mutant_ids=last_target_ids,
        )
        self.handoff["stop_reason"] = stop_reason
        metrics = LayerMetrics(
            layer="Layer1",
            problem_id=self.problem_id,
            total_mutants=len(all_mutants),
            killed_mutants=len(killed),
            new_kills=new_kills,
            test_count=len(new_tests),
            cumulative_tests=len(cumulative_tests),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thoughts_tokens=thoughts_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(estimated_cost, 8),
            generation_time_sec=round(generation_time, 4),
            evaluation_time_sec=round(evaluation_time, 4),
            total_time_sec=round(time.time() - layer_start, 4),
            cluster_kill_consistency=compute_cluster_kill_consistency(
                evaluated, representatives),
            llm_calls=len(attempt_history),
            accepted_calls=accepted_calls,
            productive_calls=productive_calls,
            invalid_calls=invalid_calls,
            zero_kill_calls=zero_kill_calls,
            target_count=len(representatives),
            prompt_chars=prompt_chars,
            response_chars=response_chars,
            notes=(
                f"batched_representatives={len(representatives)} max_attempts={max_attempts} "
                f"actual_calls={len(attempt_history)} accepted_tests={len(new_tests)} "
                f"stop_reason={stop_reason}"
            ),
        )
        metrics.compute()
        log.debug(
            "[Layer 1] %s score=%.3f new_kills=%d batched_calls=%d tokens=%d time=%.2fs",
            self.problem_id,
            metrics.mutation_score,
            new_kills,
            metrics.llm_calls,
            total_tokens,
            metrics.total_time_sec,
        )
        return new_tests, surviving, metrics
