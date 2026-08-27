# Validation Report

## Scope validated

- One initial mutant partition is computed before Layer 1.
- Layers 2 and 3 filter that partition to surviving members without reclustering.
- A live member from the same original cluster replaces a killed representative when necessary.
- Layer 2/3 are skipped when no mutants remain, and skipped (without spending a call) when every remaining survivor is already flagged `PROBABLE_EQUIVALENT` (`SKIP_LAYER_WHEN_ALL_PROBABLE_EQUIVALENT`).
- Layer 1/2/3 and the baseline all stop as soon as a shared plateau criterion is met (`N` consecutive zero-new-kill attempts, default `N=1`) rather than always spending their full fixed attempt/iteration budget -- this criterion is implemented once (`PlateauTracker` in `src/layers/common.py`) and applied identically to every layer and to the baseline, specifically so a cost/effectiveness comparison between them is not confounded by one arm being allowed to plateau-spend while the other is not.
- Infrastructure failures (API/timeout/OOM errors) are distinguished from genuine zero-kill evaluations: an infrastructure error does not count toward the plateau streak, and is not framed to the next layer/iteration as a rejected test-writing hypothesis (`summarize_attempt_history`).
- Layer 1/2/3 and baseline prompts use the standardized structured (XML-tag) format, with internal pipeline bookkeeping (`cluster_id`, `centrality`, `information_score`, `source_location`) excluded from what is sent to the LLM, and duplicate/uninformative probe evidence collapsed rather than repeated per target.
- The baseline prompt omits cluster metadata (`original_cluster_id`, `behavior_signature`), consistent with its own "no clustering" design, and deduplicates mutants with an identical exact diff.
- Probe generation is grounded in literal example calls found in a task's spec text (or doctest lines) before falling back to generic type-hint-derived values, avoiding the previously observed failure mode where an untyped, string-based function received integer-only probes that errored out before reaching any mutated code.
- Per-token cost estimation uses real default prices for the models this project uses (`gpt-5-mini`, `gpt-5-nano`, `gpt-5.6-*`, local Qwen models at $0), rather than defaulting every unlisted model to $0.
- Output is split into a curated `report/` tier (aggregate metrics, comparison tables, statistics, figures, final test suites, run manifest) and a full diagnostic `raw/` tier (per-problem mutant/cluster dumps, per-layer directories, and -- gated by `VERBOSE_ARTIFACTS`, off by default -- the full LLM prompt/response log).
- A dataset can be run in resumable index-range shards (`--index-start`/`--index-end`) and recombined with `scripts/merge_shards.py`, which reuses the pipeline's own aggregation functions rather than reimplementing them.
- Model responses are extracted with AST line boundaries, including fenced code and trailing prose cases.
- OpenAI requests use high reasoning effort and omit `max_output_tokens` when the configured limit is `0`.
- The updated Kaggle notebook has valid Python code cells.

## Automated tests

Command:

```bash
python -m unittest -v tests.test_refactor
```

Result: **13 tests passed.**

(Two of these were previously failing: `test_layer1_batches_all_representatives_and_limits_calls`
and `test_prompt_uses_single_task_spec_and_targets_near_tail` asserted on
all-caps section headers -- e.g. `"CANONICAL IMPLEMENTATION"`, `"OUTPUT
CONTRACT"` -- left over from before the prompt format was refactored to XML
tags. The prompts themselves were already correct; only the test assertions
were stale. They now check for `<canonical_implementation>`,
`<output_contract>`, and `<targets>`, matching what the prompt builders
actually emit.)

The suite covers:

- vacuous-test rejection;
- productive-test enforcement;
- batched Layer 1 calls;
- fixed initial-cluster filtering;
- Layer 2 reuse of the initial partition;
- Layer 3 early skip after Layer 2 kills all remaining mutants;
- AST-aware response extraction;
- high OpenAI reasoning effort;
- omission of the project output-token ceiling;
- model-ID/fallback validation;
- EvoEval normalization and balanced sampling;
- equivalent-mutant detection;
- missing-library preflight.

## Compilation

```bash
python -m compileall -q src run_pipeline.py scripts tests
```

Result: passed.

## Notebook validation

Validated code-cell syntax for:

- `notebooks/cluse_evoeval_kaggle.ipynb`
- `claus-test-fixed-clusters-high-effort.ipynb`

Result: no Python syntax errors in executable code cells.

## Dataset

This project no longer includes a dataset-build step. The 500-task EvoEval
Parquet file is expected to already exist (see `data/EVOEVAL_DATASET.md`);
`src/utils/dataset_loader.py` only reads it.

## End-to-end CLI smoke tests

**Single run, mock layers, with baseline, index-range chunking, and the
report/raw output split:**

```bash
python run_pipeline.py --dataset <smoke jsonl> --dataset-type evoeval --mock \
  --results <dir> --run-baseline --index-start 0 --index-end 3
```

Result: exit status 0. Confirmed:

- `<dir>/report/` contains `aggregate_metrics.json`, `aggregate_summary.csv`,
  `comparison/`, `statistics/`, `figures/`, `generated_tests/*.py`,
  `run_manifest.json`.
- `<dir>/raw/` contains per-problem folders, `layer1/2/3/`,
  `baseline_iterative/`, `metrics/`, `baseline_metrics/`, `benchmark/`.
- `raw/llm_responses/` is **not** written by default; passing
  `--verbose-artifacts 1` causes it to be written.
- Per-task metrics show `stop_reason=plateau: ...` for Layer 1/2/3 and the
  baseline, confirming the shared plateau criterion fires identically on
  both arms rather than the baseline always exhausting its full iteration
  budget.

**Sharded run + merge:**

```bash
python run_pipeline.py --dataset <smoke jsonl> --dataset-type evoeval --mock \
  --results shard0 --index-start 0 --index-end 3 --run-baseline
python run_pipeline.py --dataset <smoke jsonl> --dataset-type evoeval --mock \
  --results shard1 --index-start 3 --index-end 6 --run-baseline
python scripts/merge_shards.py --shards shard0 shard1 --output merged
```

Result: exit status 0. Confirmed the merged `report/aggregate_summary.json`
contains all 6 problems from both shards, `subset_counts` sum correctly
across shards, per-problem `raw/` folders from both shards are present
under the combined `raw/` without collision, and the statistics/figures
report regenerates correctly over the combined data.

## Important limitation

No live Qwen/OpenAI research experiment was run in this offline validation
environment. High-effort API behavior, real mutation scores, token use,
latency, and cost must be measured with the configured models and API
credentials. The cost-estimation defaults shipped in `config.py`
(`MODEL_PRICING_USD_PER_1K_TOKENS`) should be verified against
https://platform.openai.com/docs/pricing immediately before a paid run, as
provider prices change over time.
