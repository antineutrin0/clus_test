"""Experiment metrics, immediate LLM-response artifacts, and summaries."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LLMCallRecord:
    problem_id: str
    layer: str
    model: str
    cluster_id: Optional[int]
    mutant_id: Optional[str]
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_sec: float = 0.0
    prompt: str = ""
    response: str = ""
    extracted_test: str = ""
    passed_on_original: bool = False
    killed_mutants: int = 0
    surviving_mutants: int = 0
    status: str = "OK"
    error: str = ""
    attempt: int = 1
    target_count: int = 0
    target_kills: int = 0
    new_kills: int = 0
    cumulative_kills: int = 0
    cumulative_score: float = 0.0
    prompt_chars: int = 0
    response_chars: int = 0
    validation_reason: str = ""
    batch_mode: bool = False


@dataclass
class LayerMetrics:
    layer: str
    problem_id: str
    total_mutants: int = 0
    killed_mutants: int = 0
    surviving_mutants: int = 0
    mutation_score: float = 0.0
    new_kills: int = 0
    test_count: int = 0
    cumulative_tests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    generation_time_sec: float = 0.0
    evaluation_time_sec: float = 0.0
    total_time_sec: float = 0.0
    cost_per_killed_mutant: float = 0.0
    cost_per_new_kill: float = 0.0
    time_per_killed_mutant: float = 0.0
    time_per_new_kill: float = 0.0
    tokens_per_new_kill: float = 0.0
    cluster_kill_consistency: float = 0.0
    llm_calls: int = 0
    accepted_calls: int = 0
    productive_calls: int = 0
    invalid_calls: int = 0
    zero_kill_calls: int = 0
    target_count: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    notes: str = ""

    def compute(self) -> None:
        self.mutation_score = round(self.killed_mutants / self.total_mutants, 6) if self.total_mutants else 0.0
        self.surviving_mutants = max(0, self.total_mutants - self.killed_mutants)
        self.cost_per_killed_mutant = round(self.estimated_cost_usd / self.killed_mutants, 8) if self.killed_mutants else 0.0
        self.cost_per_new_kill = round(self.estimated_cost_usd / self.new_kills, 8) if self.new_kills else 0.0
        self.time_per_killed_mutant = round(self.total_time_sec / self.killed_mutants, 4) if self.killed_mutants else 0.0
        self.time_per_new_kill = round(self.total_time_sec / self.new_kills, 4) if self.new_kills else 0.0
        self.tokens_per_new_kill = round(self.total_tokens / self.new_kills, 4) if self.new_kills else 0.0


@dataclass
class BenchmarkComparison:
    problem_id: str
    cluse_mutation_score: float
    official_mutation_score: float
    cluse_killed: int
    official_killed: int
    total_mutants: int
    kill_agreement_accuracy: float
    kill_precision: float
    kill_recall: float
    kill_f1: float
    cluse_wins: bool
    sanity_check_passed: bool
    sanity_check_notes: str = ""
    equivalent_mutants: int = 0
    adjusted_total_mutants: int = 0
    cluse_equivalent_adjusted_score: float = 0.0
    official_equivalent_adjusted_score: float = 0.0


@dataclass
class ProblemTiming:
    mutant_generation_sec: float = 0.0
    probe_generation_sec: float = 0.0
    behavior_signature_sec: float = 0.0
    clustering_sec: float = 0.0
    representative_selection_sec: float = 0.0
    total_problem_sec: float = 0.0


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "item"


@dataclass
class ExperimentTracker:
    problem_id: str
    save_dir: Path
    layers: List[LayerMetrics] = field(default_factory=list)
    llm_calls: List[LLMCallRecord] = field(default_factory=list)
    benchmark: Optional[BenchmarkComparison] = None
    timing: ProblemTiming = field(default_factory=ProblemTiming)
    metadata: Dict = field(default_factory=dict)
    _start: float = field(default_factory=time.time, repr=False)

    @property
    def results_root(self) -> Path:
        return Path(self.save_dir).parent

    def paid_api_cost(self) -> float:
        return round(sum(c.estimated_cost_usd for c in self.llm_calls if c.provider not in {"", "hf", "mock", "local"}), 8)

    def paid_api_tokens(self) -> int:
        return int(sum(c.total_tokens for c in self.llm_calls if c.provider not in {"", "hf", "mock", "local"}))

    def score_at_token_budget(self, token_budget: int, *, paid: bool = False) -> float:
        """Return the best cumulative baseline score attainable within a token budget."""
        history = self.metadata.get("iteration_history") or []
        key = "cumulative_paid_tokens" if paid else "cumulative_tokens"
        eligible = [row for row in history if int(row.get(key) or 0) <= max(0, int(token_budget))]
        if not eligible:
            return 0.0
        return round(max(float(row.get("cumulative_score") or 0.0) for row in eligible), 6)

    def record_layer(self, metrics: LayerMetrics) -> None:
        metrics.compute()
        self.layers.append(metrics)

    def record_llm_call(self, call: LLMCallRecord) -> None:
        """Store each response immediately as JSONL/text and optionally print it."""
        self.llm_calls.append(call)
        call_index = len(self.llm_calls)
        if not call.provider:
            model = (call.model or "").lower()
            call.provider = "gemini" if model.startswith("gemini") else "openai" if model.startswith(("gpt", "o1", "o3", "o4")) else "hf"

        if config.SAVE_LLM_RESPONSES:
            root = self.results_root / "llm_responses"
            problem_dir = root / _safe(self.problem_id)
            problem_dir.mkdir(parents=True, exist_ok=True)
            root.mkdir(parents=True, exist_ok=True)
            payload = asdict(call)
            payload["call_index"] = call_index
            with open(root / "llm_calls.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            artifact = problem_dir / f"{call_index:03d}_{_safe(call.layer)}_{_safe(call.provider)}_{_safe(call.model)}.txt"
            artifact.write_text(
                "\n".join([
                    f"problem_id: {call.problem_id}",
                    f"layer: {call.layer}",
                    f"provider: {call.provider}",
                    f"model: {call.model}",
                    f"status: {call.status}",
                    f"tokens: prompt={call.prompt_tokens}, completion={call.completion_tokens}, thoughts={call.thoughts_tokens}, total={call.total_tokens}",
                    f"latency_sec: {call.latency_sec}",
                    f"estimated_cost_usd: {call.estimated_cost_usd}",
                    f"attempt: {call.attempt}",
                    f"target_count: {call.target_count}",
                    f"new_kills: {call.new_kills}",
                    f"cumulative_kills: {call.cumulative_kills}",
                    f"cumulative_score: {call.cumulative_score}",
                    f"prompt_chars: {call.prompt_chars}",
                    f"response_chars: {call.response_chars}",
                    f"validation_reason: {call.validation_reason}",
                    "",
                    "===== PROMPT =====",
                    call.prompt,
                    "",
                    "===== RAW LLM RESPONSE =====",
                    call.response,
                    "",
                    "===== EXTRACTED TEST =====",
                    call.extracted_test,
                    "",
                    "===== EVALUATION =====",
                    f"passed_on_original: {call.passed_on_original}",
                    f"killed_mutants: {call.killed_mutants}",
                    f"surviving_mutants: {call.surviving_mutants}",
                    f"error: {call.error}",
                ]),
                encoding="utf-8",
            )

        if config.DISPLAY_LLM_RESPONSES:
            if config.should_trace_problem(call.problem_id):
                response = call.response or call.error or "<empty response>"
                limit = max(0, config.LLM_RESPONSE_PREVIEW_CHARS)
                preview = response if limit == 0 or len(response) <= limit else response[:limit] + "\n...[truncated in console; full response saved]"
                print(
                    f"\n{'=' * 20} LLM RESPONSE {call_index} {'=' * 20}\n"
                    f"problem={call.problem_id} layer={call.layer} attempt={call.attempt} "
                    f"provider={call.provider} model={call.model} targets={call.target_count}\n"
                    f"tokens={call.total_tokens} new_kills={call.new_kills} "
                    f"cumulative_score={call.cumulative_score:.3f} status={call.status}\n"
                    f"{preview}\n{'=' * 58}\n",
                    flush=True,
                )
            elif config.DISPLAY_COMPACT_CALL_SUMMARY:
                print(
                    f"[LLM] problem={call.problem_id} layer={call.layer} attempt={call.attempt} "
                    f"tokens={call.total_tokens} new_kills={call.new_kills} "
                    f"score={call.cumulative_score:.3f} status={call.status}",
                    flush=True,
                )

    def record_benchmark(self, comparison: BenchmarkComparison) -> None:
        self.benchmark = comparison

    def save(self, filename: Optional[str] = None) -> Path:
        self.timing.total_problem_sec = round(time.time() - self._start, 3)
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / (filename or f"{_safe(self.problem_id)}_metrics.json")
        payload = {
            "problem_id": self.problem_id,
            "metadata": self.metadata,
            "timing": asdict(self.timing),
            "layers": [asdict(m) for m in self.layers],
            "llm_calls": [asdict(c) for c in self.llm_calls],
            "benchmark": asdict(self.benchmark) if self.benchmark else None,
            "paid_api_cost_usd": self.paid_api_cost(),
            "paid_api_tokens": self.paid_api_tokens(),
            "summary": self.summary(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path

    def summary(self) -> Dict:
        if not self.layers:
            return {
                "problem_id": self.problem_id,
                "total_mutants": int(self.metadata.get("mutant_count", 0)),
                "dataset_name": self.metadata.get("dataset_name", "unknown"),
                "dataset_subset": (self.metadata.get("dataset_metadata") or {}).get("dataset_subset", ""),
                "source_task_id": (self.metadata.get("dataset_metadata") or {}).get("source_task_id", self.problem_id),
                "parent_task_id": (self.metadata.get("dataset_metadata") or {}).get("parent_task_id", ""),
                "skipped": bool(self.metadata.get("skipped", False)),
                "skip_reason": self.metadata.get("skip_reason", ""),
                "missing_libraries": ",".join(self.metadata.get("missing_libraries", [])),
                "runtime_sec": self.timing.total_problem_sec,
            }
        last = self.layers[-1]
        providers = sorted({c.provider for c in self.llm_calls if c.provider})
        models = sorted({c.model for c in self.llm_calls if c.model})

        def _layer(name: str, attr: str, default=0):
            m = next((l for l in self.layers if l.layer == name), None)
            return getattr(m, attr) if m is not None else default

        return {
            "problem_id": self.problem_id,
            "dataset_name": self.metadata.get("dataset_name", "unknown"),
            "dataset_subset": (self.metadata.get("dataset_metadata") or {}).get("dataset_subset", ""),
            "source_task_id": (self.metadata.get("dataset_metadata") or {}).get("source_task_id", self.problem_id),
            "parent_task_id": (self.metadata.get("dataset_metadata") or {}).get("parent_task_id", ""),
            "final_score": last.mutation_score,
            "total_mutants": last.total_mutants,
            "killed_mutants": last.killed_mutants,
            "surviving_mutants": last.surviving_mutants,
            "total_tokens": sum(l.total_tokens for l in self.layers),
            "paid_api_tokens": self.paid_api_tokens(),
            "prompt_tokens": sum(l.prompt_tokens for l in self.layers),
            "completion_tokens": sum(l.completion_tokens for l in self.layers),
            "thoughts_tokens": sum(l.thoughts_tokens for l in self.layers),
            "estimated_cost_usd": round(sum(l.estimated_cost_usd for l in self.layers), 8),
            "paid_api_cost_usd": self.paid_api_cost(),
            # Per-layer token/cost breakdown, used to build the end-of-run
            # cost-by-layer comparison without re-reading every raw metrics
            # file. Zero/absent when a layer was skipped for this problem.
            "layer1_tokens": _layer("Layer1", "total_tokens"),
            "layer1_cost_usd": _layer("Layer1", "estimated_cost_usd"),
            "layer1_calls": _layer("Layer1", "llm_calls"),
            "layer2_tokens": _layer("Layer2", "total_tokens"),
            "layer2_cost_usd": _layer("Layer2", "estimated_cost_usd"),
            "layer2_calls": _layer("Layer2", "llm_calls"),
            "layer3_tokens": _layer("Layer3", "total_tokens"),
            "layer3_cost_usd": _layer("Layer3", "estimated_cost_usd"),
            "layer3_calls": _layer("Layer3", "llm_calls"),
            "generated_tests": last.cumulative_tests,
            "llm_calls": len(self.llm_calls),
            "productive_calls": sum(1 for c in self.llm_calls if c.status == "PRODUCTIVE"),
            "invalid_calls": sum(1 for c in self.llm_calls if c.status in {"REJECTED", "ERROR"}),
            "zero_kill_calls": sum(1 for c in self.llm_calls if c.status == "VALID_ZERO_KILL"),
            "prompt_chars": sum(int(c.prompt_chars or len(c.prompt or "")) for c in self.llm_calls),
            "response_chars": sum(int(c.response_chars or len(c.response or "")) for c in self.llm_calls),
            "runtime_sec": self.timing.total_problem_sec,
            "providers": ",".join(providers),
            "models": ",".join(models),
            "official_score": self.benchmark.official_mutation_score if self.benchmark else None,
            "official_killed": self.benchmark.official_killed if self.benchmark else None,
            "equivalent_mutants": self.benchmark.equivalent_mutants if self.benchmark else 0,
            "equivalent_adjusted_score": self.benchmark.cluse_equivalent_adjusted_score if self.benchmark else None,
            "official_equivalent_adjusted_score": self.benchmark.official_equivalent_adjusted_score if self.benchmark else None,
            "kill_agreement_accuracy": self.benchmark.kill_agreement_accuracy if self.benchmark else None,
            "kill_precision": self.benchmark.kill_precision if self.benchmark else None,
            "kill_recall": self.benchmark.kill_recall if self.benchmark else None,
            "kill_f1": self.benchmark.kill_f1 if self.benchmark else None,
            "sanity_passed": self.benchmark.sanity_check_passed if self.benchmark else None,
        }


def load_metrics(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate_results(results_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(Path(results_dir).rglob("*_metrics.json")):
        try:
            data = load_metrics(path)
            summary = data.get("summary") or {}
            if summary:
                rows.append(summary)
        except Exception:
            continue
    return rows
