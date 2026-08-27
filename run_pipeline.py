"""Command-line entry point for the cost-aware Claus-Test pipeline.

Recommended EvoEval run::

    python run_pipeline.py \
      --evoeval-semantic --hf-split test --dataset-type evoeval \
      --percent 0.10 --sample-mode stratified --stratify-by dataset_subset \
      --layer1-provider hf --layer1-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --layer2-provider openai --layer2-model gpt-5-nano \
      --layer3-provider openai --layer3-model gpt-5-mini \
      --run-baseline --baseline-provider openai --baseline-model gpt-5-mini

Do not place angle brackets around model IDs. Use ``--mock`` for a no-cost
integration smoke test; mock responses validate the workflow, not effectiveness.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.pipeline import CLUSEPipeline
from src.utils import config
from src.utils.dataset_loader import (
    EVOEVAL_SEMANTIC_SUBSETS,
    load_dataset,
    load_evoeval_semantic_dataset,
    load_huggingface_dataset,
)


def _optional_int(value: str) -> Optional[int]:
    if value.lower() in {"none", "all", ""}:
        return None
    return int(value)


def _csv_models(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool01(value: int) -> bool:
    return bool(int(value))


def _layer_dict(args, suffix: str) -> Dict[int, object]:
    result: Dict[int, object] = {}
    for layer in (1, 2, 3):
        value = getattr(args, f"layer{layer}_{suffix}")
        if value not in {None, ""}:
            result[layer] = value
    return result


def _configure_runtime(args) -> None:
    if args.mock:
        os.environ["USE_MOCK_LLM"] = "1"
        config.USE_MOCK_LLM = True
    if args.display_llm_responses is not None:
        config.DISPLAY_LLM_RESPONSES = _bool01(args.display_llm_responses)
    if args.save_llm_responses is not None:
        config.SAVE_LLM_RESPONSES = _bool01(args.save_llm_responses)
    if args.log_full_llm_io is not None:
        config.LOG_FULL_LLM_IO = _bool01(args.log_full_llm_io)
    if args.display_first_problem_only is not None:
        config.DISPLAY_FIRST_PROBLEM_ONLY = _bool01(args.display_first_problem_only)
    if args.display_compact_call_summary is not None:
        config.DISPLAY_COMPACT_CALL_SUMMARY = _bool01(args.display_compact_call_summary)
    if args.layer1_max_attempts is not None:
        config.LAYER1_MAX_REFINEMENT = max(1, int(args.layer1_max_attempts))
    if args.layer2_max_attempts is not None:
        config.LAYER2_MAX_REFINEMENT = max(1, int(args.layer2_max_attempts))
    if args.layer3_max_attempts is not None:
        config.LAYER3_MAX_REFINEMENT = max(1, int(args.layer3_max_attempts))
    if args.require_productive_test is not None:
        config.REQUIRE_PRODUCTIVE_TEST = _bool01(args.require_productive_test)
    if args.skip_missing_libraries is not None:
        config.SKIP_MISSING_LIBRARIES = _bool01(args.skip_missing_libraries)
    if args.baseline_plateau_patience is not None:
        config.BASELINE_PLATEAU_PATIENCE = max(1, int(args.baseline_plateau_patience))
    if args.baseline_stop_on_plateau is not None:
        config.BASELINE_STOP_ON_PLATEAU = _bool01(args.baseline_stop_on_plateau)
    for layer in (1, 2, 3):
        patience = getattr(args, f"layer{layer}_plateau_patience")
        if patience is not None:
            setattr(config, f"LAYER{layer}_PLATEAU_PATIENCE", max(1, int(patience)))
        stop_on_plateau = getattr(args, f"layer{layer}_stop_on_plateau")
        if stop_on_plateau is not None:
            setattr(config, f"LAYER{layer}_STOP_ON_PLATEAU", _bool01(stop_on_plateau))
    if args.index_start is not None or args.index_end is not None:
        if args.index_start is None or args.index_end is None:
            raise SystemExit("--index-start and --index-end must be given together")
        if args.index_start < 0 or args.index_end <= args.index_start:
            raise SystemExit("--index-end must be greater than --index-start (both >= 0)")
        config.PROBLEM_INDICES = list(range(args.index_start, args.index_end))
    if args.llm_response_preview_chars is not None:
        config.LLM_RESPONSE_PREVIEW_CHARS = int(args.llm_response_preview_chars)
    if args.trace_problem:
        os.environ["TRACE_PROBLEM_ID"] = args.trace_problem
        config.TRACE_PROBLEM_ID = args.trace_problem
    if args.bootstrap_samples is not None:
        config.STATISTICS_BOOTSTRAP_SAMPLES = int(args.bootstrap_samples)
    if args.figure_pdf is not None:
        config.SAVE_FIGURE_PDF = _bool01(args.figure_pdf)
    if args.gemini_thinking_budget is not None:
        config.GEMINI_THINKING_BUDGET = int(args.gemini_thinking_budget)
    if args.openai_reasoning_effort:
        config.OPENAI_REASONING_EFFORT = str(args.openai_reasoning_effort).strip().lower()
    if args.openai_text_verbosity:
        config.OPENAI_TEXT_VERBOSITY = str(args.openai_text_verbosity).strip().lower()
    if args.verbose_artifacts is not None:
        config.VERBOSE_ARTIFACTS = _bool01(args.verbose_artifacts)
        config.SAVE_LLM_RESPONSES = config.VERBOSE_ARTIFACTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLUSE-Test on EvoEval, HumanEval, BigCodeBench, or compatible data")
    parser.add_argument("--dataset", type=Path, default=config.DATASET_PATH, help="Dataset path: parquet, csv, json, or jsonl")
    parser.add_argument("--hf-dataset", default="", help="Load one dataset directly from Hugging Face")
    parser.add_argument("--hf-split", default="test", help="Hugging Face split name")
    parser.add_argument("--evoeval-semantic", action="store_true", help="Load the finalized five-subset EvoEval semantic suite (500 tasks)")
    parser.add_argument("--evoeval-subsets", default=",".join(EVOEVAL_SEMANTIC_SUBSETS), help="Comma-separated EvoEval subset repositories")
    parser.add_argument("--dataset-type", default=config.DATASET_TYPE, choices=["auto", "evoeval", "humaneval", "bigcodebench", "generic"], help="Schema adapter; auto is recommended")
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR, help="Experiment output directory")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--max-layers", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--limit", type=_optional_int, default=None)
    parser.add_argument("--percent", type=float, default=0.0, help="Fraction in (0,1], e.g. 0.10")
    parser.add_argument(
        "--index-start", type=int, default=None,
        help="Run only problems[index_start:index_end] (0-based, end-exclusive). "
             "Lets a large dataset be run in resumable shards across sessions "
             "(e.g. Kaggle runtime limits) -- combine shard outputs afterward "
             "with scripts/merge_shards.py. Requires --index-end.",
    )
    parser.add_argument(
        "--index-end", type=int, default=None,
        help="End (exclusive) of the --index-start problem range.",
    )
    parser.add_argument("--sample-mode", default="first", choices=["first", "random", "stratified"])
    parser.add_argument("--stratify-by", default="dataset_subset", choices=["dataset_subset", "parent_task_id", "dataset_name"], help="Grouping field for stratified sampling")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--max-mutants", type=int, default=config.MAX_MUTANTS_PER_PROBLEM)
    parser.add_argument("--max-probes", type=int, default=config.MAX_PROBES_PER_PROBLEM)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses")

    for layer in (1, 2, 3):
        parser.add_argument(f"--layer{layer}-provider", dest=f"layer{layer}_provider", default="", help="auto, hf, gemini, openai, mock, or a registered custom provider")
        parser.add_argument(f"--layer{layer}-model", dest=f"layer{layer}_model", default="")
        parser.add_argument(f"--layer{layer}-fallback-models", dest=f"layer{layer}_fallback_models", default=None, help="Comma-separated same-provider fallback models")
        parser.add_argument(
            f"--layer{layer}-max-tokens", dest=f"layer{layer}_max_tokens", type=int, default=None,
            help="Optional output-token ceiling; 0 disables the project-level ceiling",
        )

    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--baseline-provider", default=config.BASELINE_PROVIDER)
    parser.add_argument("--baseline-model", default=config.BASELINE_MODEL)
    parser.add_argument("--baseline-fallback-models", default=None)
    parser.add_argument(
        "--baseline-max-tokens", type=int, default=config.BASELINE_MAX_TOKENS,
        help="Optional output-token ceiling; 0 disables the project-level ceiling",
    )
    parser.add_argument("--baseline-max-iterations", type=int, default=config.BASELINE_MAX_ITERATIONS)
    parser.add_argument("--baseline-plateau-patience", type=int, default=None, help="Consecutive no-gain iterations before stopping; default 1")
    parser.add_argument("--baseline-stop-on-plateau", type=int, choices=[0, 1], default=None, help="Default 1: stop once the baseline plateaus, same criterion applied to Layer 1/2/3, so cost/effectiveness comparisons are not confounded by asymmetric stopping rules")

    parser.add_argument("--layer1-max-attempts", type=int, default=None, help="Batched calls per task; default 3")
    parser.add_argument("--layer2-max-attempts", type=int, default=None, help="Batched refinement calls per task; default 2")
    parser.add_argument("--layer3-max-attempts", type=int, default=None, help="Final batched calls per task; default 1")
    for layer in (1, 2, 3):
        parser.add_argument(
            f"--layer{layer}-plateau-patience", dest=f"layer{layer}_plateau_patience", type=int, default=None,
            help=f"Consecutive zero-new-kill attempts before Layer {layer} stops early; default 1",
        )
        parser.add_argument(
            f"--layer{layer}-stop-on-plateau", dest=f"layer{layer}_stop_on_plateau", type=int, choices=[0, 1], default=None,
            help=f"Default 1: stop Layer {layer} once it plateaus rather than always spending its full attempt budget",
        )
    parser.add_argument("--require-productive-test", type=int, choices=[0, 1], default=None, help="Reject valid tests that kill zero active mutants")
    parser.add_argument("--skip-missing-libraries", type=int, choices=[0, 1], default=None, help="Report and exclude tasks whose declared imports are unavailable")

    parser.add_argument("--display-llm-responses", type=int, choices=[0, 1], default=None, help="Print raw responses to notebook/terminal")
    parser.add_argument("--save-llm-responses", type=int, choices=[0, 1], default=None, help="Save per-call TXT and JSONL artifacts")
    parser.add_argument("--llm-response-preview-chars", type=int, default=None, help="0 prints full response")
    parser.add_argument("--log-full-llm-io", type=int, choices=[0, 1], default=None)
    parser.add_argument("--display-first-problem-only", type=int, choices=[0, 1], default=None)
    parser.add_argument("--display-compact-call-summary", type=int, choices=[0, 1], default=None)
    parser.add_argument("--trace-problem", default="")
    parser.add_argument("--gemini-thinking-budget", type=int, default=None)
    parser.add_argument(
        "--openai-reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None,
        help="OpenAI reasoning effort; defaults to high",
    )
    parser.add_argument(
        "--openai-text-verbosity", choices=["low", "medium", "high"], default=None,
        help="OpenAI visible response verbosity; defaults to low",
    )

    parser.add_argument("--generate-statistics", type=int, choices=[0, 1], default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--figure-pdf", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--verbose-artifacts", type=int, choices=[0, 1], default=None,
        help="Default 0: skip writing the full per-attempt LLM prompt/response "
             "dump (raw/llm_responses/), the largest and least-needed-day-to-day "
             "artifact. Set to 1 when debugging a specific run.",
    )
    parser.add_argument("--zip-output", action="store_true", help="Create <results>.zip after the run")
    args = parser.parse_args()

    if not (0.0 <= args.percent <= 1.0):
        parser.error("--percent must be between 0.0 and 1.0")

    _configure_runtime(args)
    args.results.mkdir(parents=True, exist_ok=True)
    if args.evoeval_semantic:
        requested_subsets = [item.strip() for item in args.evoeval_subsets.split(",") if item.strip()]
        problems = load_evoeval_semantic_dataset(split=args.hf_split, subsets=requested_subsets)
    elif args.hf_dataset:
        problems = load_huggingface_dataset(args.hf_dataset, split=args.hf_split, dataset_type=args.dataset_type)
    else:
        problems = load_dataset(args.dataset, dataset_type=args.dataset_type)

    layer_providers = {int(k): str(v) for k, v in _layer_dict(args, "provider").items()}
    layer_models = {int(k): str(v) for k, v in _layer_dict(args, "model").items()}
    layer_fallbacks = {
        layer: _csv_models(getattr(args, f"layer{layer}_fallback_models"))
        for layer in (1, 2, 3)
        if getattr(args, f"layer{layer}_fallback_models") is not None
    }
    layer_max_tokens = {int(k): int(v) for k, v in _layer_dict(args, "max_tokens").items()}

    pipeline = CLUSEPipeline(
        results_dir=args.results,
        max_layers=args.max_layers,
        max_mutants=args.max_mutants,
        max_probes=args.max_probes,
        problem_limit=args.limit,
        problem_percent=args.percent,
        sample_mode=args.sample_mode,
        stratify_by=args.stratify_by,
        seed=args.seed,
        run_name=args.run_name,
        run_baseline=args.run_baseline,
        baseline_only=args.baseline_only,
        layer_providers=layer_providers,
        layer_models=layer_models,
        layer_fallback_models=layer_fallbacks,
        layer_max_tokens=layer_max_tokens,
        baseline_provider=args.baseline_provider,
        baseline_model=args.baseline_model,
        baseline_fallback_models=_csv_models(args.baseline_fallback_models),
        baseline_max_tokens=args.baseline_max_tokens,
        baseline_max_iterations=args.baseline_max_iterations,
        generate_statistics=_bool01(args.generate_statistics),
    )
    summaries = pipeline.run(problems)

    zip_path = None
    if args.zip_output:
        zip_path = shutil.make_archive(str(args.results), "zip", root_dir=args.results)

    # Compact machine-readable manifest only -- the full per-problem
    # breakdown is already written to report/aggregate_summary.json (and the
    # human-readable overall summary was logged by pipeline.run() above), so
    # printing the whole `summaries` list here again would just be console
    # noise duplicating both.
    executed = [row for row in summaries if not row.get("skipped")]
    print(json.dumps({
        "n_problems": len(summaries),
        "n_executed": len(executed),
        "n_skipped": len(summaries) - len(executed),
        "results_dir": str(args.results),
        "report_dir": str(pipeline.report_dir),
        "raw_dir": str(pipeline.raw_dir),
        "zip_path": zip_path,
        "aggregate_summary": str(pipeline.report_dir / "aggregate_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
