"""Central configuration for the dataset- and provider-agnostic CLUSE-Test pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def is_kaggle_runtime() -> bool:
    return Path("/kaggle/working").exists() or bool(os.environ.get("KAGGLE_URL_BASE"))


KAGGLE_RUNTIME = is_kaggle_runtime()
PROJECT_ROOT = Path(os.environ.get(
    "CLUSE_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("CLUSE_DATA_DIR", PROJECT_ROOT / "data"))

if KAGGLE_RUNTIME:
    RESULTS_DIR = Path(os.environ.get("CLUSE_RESULTS_DIR",
                       "/kaggle/working/cluse_results"))
    LOGS_DIR = Path(os.environ.get(
        "CLUSE_LOGS_DIR", "/kaggle/working/cluse_logs"))
else:
    RESULTS_DIR = Path(os.environ.get(
        "CLUSE_RESULTS_DIR", DATA_DIR / "results"))
    LOGS_DIR = Path(os.environ.get("CLUSE_LOGS_DIR",
                    PROJECT_ROOT / "experiments" / "logs"))


def _first_existing(candidates: list[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_dataset() -> Path:
    """Find an explicitly configured or likely local/Kaggle benchmark file."""
    explicit = os.environ.get(
        "CLUSE_DATASET") or os.environ.get("HUMANEVAL_PARQUET")
    if explicit:
        return Path(explicit)

    local_candidates = [
        PROJECT_ROOT / "data" / "evoeval" / "EvoEval_semantic_500.parquet",
        Path.cwd() / "EvoEval_semantic_500.parquet",
        PROJECT_ROOT / "data" / "test-00000-of-00001.parquet",
    ]
    found = _first_existing(local_candidates)
    if found:
        return found

    if Path("/kaggle/input").exists():
        supported: list[Path] = []
        for pattern in ("**/*.parquet", "**/*.jsonl", "**/*.json", "**/*.csv"):
            supported.extend(Path("/kaggle/input").glob(pattern))
        supported = sorted(set(supported))
        preferred = [
            p for p in supported
            if any(token in str(p).lower() for token in ("evoeval", "bigcodebench", "humaneval", "test-00000"))
        ]
        if preferred:
            return preferred[0]
        if supported:
            return supported[0]

    return local_candidates[0]


def discover_humaneval_parquet() -> Path:
    """Backward-compatible alias for older notebooks."""
    return discover_dataset()


def _csv_env(name: str, default: str = "") -> list[str]:
    value = os.environ.get(name, default).strip()
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def _json_env(name: str, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


DATASET_PATH = discover_dataset()
HUMANEVAL_PARQUET = DATASET_PATH  # compatibility
# auto | evoeval | humaneval | bigcodebench | generic
DATASET_TYPE = os.environ.get("CLUSE_DATASET_TYPE", "auto")

TRACE_PROBLEM_ID = os.environ.get("TRACE_PROBLEM_ID", "")
LOG_FULL_LLM_IO = os.environ.get("LOG_FULL_LLM_IO", "0") == "1"
# Off by default (see VERBOSE_ARTIFACTS below, further down this file, which
# these three follow unless explicitly set): printing every LLM call's
# response to the console is exactly the per-test-case noise a real run
# doesn't want -- only the one overall run summary prints by default now.
DISPLAY_LLM_RESPONSES = os.environ.get("DISPLAY_LLM_RESPONSES", "0") == "1"
DISPLAY_FIRST_PROBLEM_ONLY = os.environ.get(
    "DISPLAY_FIRST_PROBLEM_ONLY", "1") == "1"
DISPLAY_COMPACT_CALL_SUMMARY = os.environ.get(
    "DISPLAY_COMPACT_CALL_SUMMARY", "0") == "1"
LLM_RESPONSE_PREVIEW_CHARS = int(
    os.environ.get("LLM_RESPONSE_PREVIEW_CHARS", "6000"))
SAVE_LLM_RESPONSES = os.environ.get("SAVE_LLM_RESPONSES", "1") == "1"
LAYER1_MAX_REFINEMENT = int(os.environ.get("LAYER1_MAX_REFINEMENT", "3"))
LAYER2_MAX_REFINEMENT = int(os.environ.get("LAYER2_MAX_REFINEMENT", "2"))
LAYER3_MAX_REFINEMENT = int(os.environ.get("LAYER3_MAX_REFINEMENT", "1"))
REQUIRE_PRODUCTIVE_TEST = os.environ.get("REQUIRE_PRODUCTIVE_TEST", "1") == "1"
SKIP_MISSING_LIBRARIES = os.environ.get("SKIP_MISSING_LIBRARIES", "1") == "1"
PROMPT_SPEC_CHARS = int(os.environ.get("PROMPT_SPEC_CHARS", "2200"))
_FIRST_TRACE_PROBLEM_ID = ""


def should_trace_problem(problem_id: str) -> bool:
    """Return whether full prompt/response content should be shown for a task."""
    global _FIRST_TRACE_PROBLEM_ID
    if TRACE_PROBLEM_ID:
        return problem_id == TRACE_PROBLEM_ID
    if not DISPLAY_FIRST_PROBLEM_ONLY:
        return True
    if not _FIRST_TRACE_PROBLEM_ID:
        _FIRST_TRACE_PROBLEM_ID = problem_id
    return problem_id == _FIRST_TRACE_PROBLEM_ID


# Problem selection.
# Index-range selection lets a 500-task dataset be run in resumable shards
# across multiple sessions (e.g. Kaggle's runtime limits): run
# --index-start 0 --index-end 100, then 100-200, etc., each to its own
# --results dir, then combine with scripts/merge_shards.py.
_index_start_env = os.environ.get("PROBLEM_INDEX_START", "").strip()
_index_end_env = os.environ.get("PROBLEM_INDEX_END", "").strip()
if _index_start_env and _index_end_env:
    PROBLEM_INDICES = list(range(int(_index_start_env), int(_index_end_env)))
else:
    PROBLEM_INDICES = None
PROBLEM_LIMIT = None if os.environ.get("PROBLEM_LIMIT", "").lower() in {
    "", "none", "all"} else int(os.environ["PROBLEM_LIMIT"])
PROBLEM_PERCENT = float(os.environ.get("PROBLEM_PERCENT", "0.0"))
SAMPLE_MODE = os.environ.get("SAMPLE_MODE", "first")

# Mutation testing.
MUTATION_TIMEOUT_SEC = int(os.environ.get("MUTATION_TIMEOUT_SEC", "2"))
PROBE_TIMEOUT_SEC = int(os.environ.get("PROBE_TIMEOUT_SEC", "1"))
MAX_MUTANTS_PER_PROBLEM = int(os.environ.get("MAX_MUTANTS_PER_PROBLEM", "25"))
MAX_PROBES_PER_PROBLEM = int(os.environ.get("MAX_PROBES_PER_PROBLEM", "6"))
MUTATION_OPERATORS = [
    "COMPARISON_FLIP", "BOUNDARY_SHIFT", "BOOLEAN_FLIP", "CONSTANT_CHANGE",
    "ARITHMETIC_FLIP", "RETURN_VALUE_CHANGE", "NEGATE_CONDITION",
    "INDEX_BOUNDARY", "MEMBERSHIP_FLIP", "IDENTITY_FLIP", "ARGUMENT_SWAP",
    "RANGE_BOUNDARY",
]

# Clustering and representative selection.
MAX_CLUSTERS = int(os.environ.get("MAX_CLUSTERS", "5"))
MIN_CLUSTER_SIZE = int(os.environ.get("MIN_CLUSTER_SIZE", "2"))
CENTRALITY_QUANTILE = float(os.environ.get("CENTRALITY_QUANTILE", "0.40"))
REPRESENTATIVES_PER_CLUSTER = int(
    os.environ.get("REPRESENTATIVES_PER_CLUSTER", "1"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

# Provider/model selection. Any layer can independently use hf, gemini, openai, or mock.
LAYER1_PROVIDER = os.environ.get("LAYER1_PROVIDER", "hf")
LAYER2_PROVIDER = os.environ.get("LAYER2_PROVIDER", "openai")
LAYER3_PROVIDER = os.environ.get("LAYER3_PROVIDER", "openai")
BASELINE_PROVIDER = os.environ.get("BASELINE_PROVIDER", "openai")

LAYER1_MODEL = os.environ.get(
    "LAYER1_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
LAYER2_MODEL = os.environ.get("LAYER2_MODEL", "gpt-5-mini")
LAYER3_MODEL = os.environ.get("LAYER3_MODEL", "gpt-5-mini")
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "gpt-5-mini")

LAYER1_FALLBACK_MODEL = os.environ.get(
    "LAYER1_FALLBACK_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
LAYER2_FALLBACK_MODELS = _csv_env("LAYER2_FALLBACK_MODELS", "")
LAYER3_FALLBACK_MODELS = _csv_env("LAYER3_FALLBACK_MODELS", "")
BASELINE_FALLBACK_MODELS = _csv_env("BASELINE_FALLBACK_MODELS", "")

# 0 means that Claus-Test does not impose a fixed output-token ceiling.
# API providers use their model/context defaults; local HF generation can use
# the remaining natural context window.
LAYER1_MAX_TOKENS = int(os.environ.get("LAYER1_MAX_TOKENS", "0"))
LAYER1_MAX_INPUT_TOKENS = int(
    os.environ.get("LAYER1_MAX_INPUT_TOKENS", "4096"))
# Local HF generation safety net. When LAYER1_MAX_TOKENS == 0, the client
# still computes an "unbounded" budget from the model's own context window
# (see LocalHFClient._generate_once) -- for a model with a large context
# (e.g. 32k) that budget can be tens of thousands of tokens. Greedy decoding
# on a small local model occasionally fails to emit EOS and instead drifts
# into a long repetitive tail, which previously meant a single call could
# run for a very long time and make the whole pipeline look "stuck." These
# two settings bound that regardless of LAYER1_MAX_TOKENS:
#   - HF_MAX_NEW_TOKENS_HARD_CAP: absolute ceiling on generated tokens.
#   - HF_GENERATION_TIMEOUT_SEC: wall-clock ceiling per call; generation is
#     stopped (not errored) once exceeded, and whatever text was produced so
#     far is returned/decoded normally.
HF_MAX_NEW_TOKENS_HARD_CAP = int(
    os.environ.get("HF_MAX_NEW_TOKENS_HARD_CAP", "1536"))
HF_GENERATION_TIMEOUT_SEC = float(
    os.environ.get("HF_GENERATION_TIMEOUT_SEC", "180"))
LAYER2_MAX_TOKENS = int(os.environ.get("LAYER2_MAX_TOKENS", "0"))
LAYER3_MAX_TOKENS = int(os.environ.get("LAYER3_MAX_TOKENS", "0"))
BASELINE_MAX_TOKENS = int(os.environ.get("BASELINE_MAX_TOKENS", "0"))

BASELINE_MAX_ITERATIONS = int(os.environ.get("BASELINE_MAX_ITERATIONS", "10"))
BASELINE_PLATEAU_PATIENCE = int(
    os.environ.get("BASELINE_PLATEAU_PATIENCE", "1"))
# Default ON: both the proposed pipeline's layers and this baseline must share
# one stopping criterion (stop after `*_PLATEAU_PATIENCE` consecutive attempts
# that produce zero new kills) so cost/effectiveness comparisons are not
# confounded by one arm being allowed to plateau-spend and the other not.
BASELINE_STOP_ON_PLATEAU = os.environ.get(
    "BASELINE_STOP_ON_PLATEAU", "1") == "1"
RUN_BASELINE = os.environ.get("RUN_BASELINE", "0") == "1"
BASELINE_ONLY = os.environ.get("BASELINE_ONLY", "0") == "1"

# Shared plateau-based stopping criterion for Layer 1/2/3 and the baseline.
# An attempt that kills zero *new* mutants counts toward the patience budget;
# once `*_PLATEAU_PATIENCE` consecutive attempts fail to make progress, the
# layer stops even if its `*_MAX_REFINEMENT`/`BASELINE_MAX_ITERATIONS` budget
# has not been exhausted. This replaces "always spend the full fixed budget"
# with "stop spending once the model stops making progress," applied
# identically to every layer and to the baseline.
LAYER1_STOP_ON_PLATEAU = os.environ.get("LAYER1_STOP_ON_PLATEAU", "1") == "1"
LAYER2_STOP_ON_PLATEAU = os.environ.get("LAYER2_STOP_ON_PLATEAU", "1") == "1"
LAYER3_STOP_ON_PLATEAU = os.environ.get("LAYER3_STOP_ON_PLATEAU", "1") == "1"
LAYER1_PLATEAU_PATIENCE = int(os.environ.get("LAYER1_PLATEAU_PATIENCE", "1"))
LAYER2_PLATEAU_PATIENCE = int(os.environ.get("LAYER2_PLATEAU_PATIENCE", "1"))
LAYER3_PLATEAU_PATIENCE = int(os.environ.get("LAYER3_PLATEAU_PATIENCE", "1"))
# Skip spending an LLM call at all when every remaining survivor going into a
# layer is already flagged PROBABLE_EQUIVALENT by fuzz-probe analysis -- a
# call is very unlikely to find a real kill against a batch that is already
# believed equivalent, so this is a stopping criterion based on *what's
# left*, not *how many attempts have run*.
SKIP_LAYER_WHEN_ALL_PROBABLE_EQUIVALENT = os.environ.get(
    "SKIP_LAYER_WHEN_ALL_PROBABLE_EQUIVALENT", "1") == "1"

# Provider credentials/options. Kaggle Secrets with the same names are also read.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_ORGANIZATION = os.environ.get("OPENAI_ORGANIZATION", "")
OPENAI_PROJECT = os.environ.get("OPENAI_PROJECT", "")
OPENAI_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
OPENAI_TIMEOUT_SEC = float(os.environ.get("OPENAI_TIMEOUT_SEC", "120"))
OPENAI_REASONING_EFFORT = os.environ.get(
    "OPENAI_REASONING_EFFORT", "high").strip().lower() or "high"
OPENAI_TEXT_VERBOSITY = os.environ.get(
    "OPENAI_TEXT_VERBOSITY", "low").strip().lower() or "low"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

GEMINI_THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "4"))
GEMINI_MAX_RETRY_SLEEP_SEC = float(
    os.environ.get("GEMINI_MAX_RETRY_SLEEP_SEC", "60"))
USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "0") == "1"

# USD per 1K tokens. These are the reason `estimated_cost_usd` was reporting
# 0.0 for every real API call: the table previously defaulted every model to
# $0.0/$0.0 unless an env var overrode it, and nothing ever did. Ship real
# public list-price defaults for the models this project actually uses, and
# keep the env-var/JSON override path so prices can be corrected the moment
# OpenAI changes them -- VERIFY these against
# https://platform.openai.com/docs/pricing before a paid run; prices move.
_KNOWN_MODEL_PRICES_USD_PER_1K = {
    # OpenAI, published pricing as of Aug 2026.
    "gpt-5-mini": {"input": 0.00025, "output": 0.002},
    "gpt-5-nano": {"input": 0.00005, "output": 0.0004},
    "gpt-5": {"input": 0.00125, "output": 0.01},
    "gpt-5.2": {"input": 0.00175, "output": 0.014},
    "gpt-5.6-terra": {"input": 0.002, "output": 0.012},
    "gpt-5.6-luna": {"input": 0.0002, "output": 0.0012},
    # Local Hugging Face models run on your own GPU -- genuinely $0 API cost.
    # (Compute/electricity/GPU-rental cost is real but out of scope for this
    # per-token estimate; report it separately if you want a fully-loaded
    # cost comparison against the API-based layers.)
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {"input": 0.0, "output": 0.0},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"input": 0.0, "output": 0.0},
}


def _price_for(model: str, input_env: str, output_env: str) -> dict:
    default = _KNOWN_MODEL_PRICES_USD_PER_1K.get(
        model, {"input": 0.0, "output": 0.0})
    return {
        "input": float(os.environ.get(input_env, str(default["input"]))),
        "output": float(os.environ.get(output_env, str(default["output"]))),
    }


MODEL_PRICING_USD_PER_1K_TOKENS = {
    LAYER1_MODEL: _price_for(LAYER1_MODEL, "LAYER1_INPUT_USD_PER_1K", "LAYER1_OUTPUT_USD_PER_1K"),
    LAYER2_MODEL: _price_for(LAYER2_MODEL, "LAYER2_INPUT_USD_PER_1K", "LAYER2_OUTPUT_USD_PER_1K"),
    LAYER3_MODEL: _price_for(LAYER3_MODEL, "LAYER3_INPUT_USD_PER_1K", "LAYER3_OUTPUT_USD_PER_1K"),
    BASELINE_MODEL: _price_for(BASELINE_MODEL, "BASELINE_INPUT_USD_PER_1K", "BASELINE_OUTPUT_USD_PER_1K"),
}
# Also register any known model not currently selected as a layer/baseline
# model, so cost estimation still works correctly if a fallback model kicks
# in (e.g. LAYER3_MODEL configured as gpt-5-mini but a run actually falls
# back to gpt-5.6-luna -- see LAYER3_FALLBACK_MODELS).
for _model_name, _prices in _KNOWN_MODEL_PRICES_USD_PER_1K.items():
    MODEL_PRICING_USD_PER_1K_TOKENS.setdefault(_model_name, _prices)
MODEL_PRICING_USD_PER_1K_TOKENS.update(_json_env("MODEL_PRICING_JSON", {}))

# Evaluation/report generation.
GENERATE_STATISTICS = os.environ.get("GENERATE_STATISTICS", "1") == "1"
STATISTICS_BOOTSTRAP_SAMPLES = int(
    os.environ.get("STATISTICS_BOOTSTRAP_SAMPLES", "5000"))
SAVE_FIGURE_PDF = os.environ.get("SAVE_FIGURE_PDF", "1") == "1"

# Output tiering. `report/` holds the curated, human-facing artifacts
# (aggregate metrics, comparison tables, statistics, figures, final test
# suites, run manifest). `raw/` holds only the minimal per-problem
# diagnostic trace actually needed to audit a run: `metrics/` (proposed
# pipeline, one `<task>_metrics.json` per problem, holding tokens, cost,
# per-layer breakdown, and the embedded benchmark comparison) and
# `metrics/baseline/` (same, for the non-clustering baseline), plus, per
# problem, `final_mutants.json` and `survived_mutants.json` (and their
# `baseline_*` counterparts when `--run-baseline` is set). Per-layer
# prompt/response traces, standalone cluster dumps, and a separate
# benchmark/ file are not written -- their contents were either fully
# redundant with the above or of diagnostic-only value not worth the size.
REPORT_DIRNAME = os.environ.get("CLUSE_REPORT_DIRNAME", "report")
RAW_DIRNAME = os.environ.get("CLUSE_RAW_DIRNAME", "raw")
# Off by default: the full per-attempt LLM prompt/response dump
# (llm_responses/) is the largest and least-needed-day-to-day artifact.
# Turn on for debugging a specific run; leave off for pilot/full runs whose
# results you intend to cite, to keep the archive a manageable size.
VERBOSE_ARTIFACTS = os.environ.get("CLUSE_VERBOSE_ARTIFACTS", "0") == "1"
# SAVE_LLM_RESPONSES (defined earlier, defaulting to "1") controls whether
# the full per-attempt LLM prompt/response dump under llm_responses/ gets
# written -- the single largest, least-needed-day-to-day artifact. Make it
# follow VERBOSE_ARTIFACTS by default so there is one on/off switch, while
# still allowing an explicit SAVE_LLM_RESPONSES env var to override it.
if "SAVE_LLM_RESPONSES" not in os.environ:
    SAVE_LLM_RESPONSES = VERBOSE_ARTIFACTS

DEVICE = os.environ.get("DEVICE", "cuda")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "1") == "1"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
PYTHON_EXECUTABLE = os.environ.get("PYTHON_EXECUTABLE", "")
PYTHONHASHSEED = os.environ.get("PYTHONHASHSEED", str(RANDOM_SEED))
