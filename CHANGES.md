# Fixed-Cluster High-Effort Update

## Pipeline

- Removed Layer 2 and Layer 3 reclustering.
- The Ward/silhouette partition is computed once before Layer 1 and reused across all layers.
- Later layers receive only surviving members of the original clusters.
- When an original representative is killed, the most informative live member of that same cluster is selected without changing cluster membership.
- Added explicit metadata and survivor snapshots proving that cluster assignments were reused.
- Added explicit early-exit metadata and logs: Layer 2/3 are skipped when an earlier layer kills every mutant.

## Prompting and response handling

- Replaced all Layer 1, Layer 2, Layer 3, and baseline prompts with a shared structured format: role, objective, task, canonical code, prior evidence, mutation evidence, failure feedback, reasoning strategy, and output contract.
- Added original and mutated statements, exact diffs, canonical probe oracles, cluster survival counts, equivalence signals, accepted-test digests, and categorized prior failures.
- Added candidate-call and assertion summaries to every handoff and baseline iteration.
- Made `check(candidate)` extraction AST-aware so valid code is preserved despite fences, preambles, or trailing prose.

## Model execution

- OpenAI Responses API calls now use `reasoning.effort=high` by default and low visible verbosity.
- A max-token value of `0` means no project-imposed output-token ceiling. OpenAI/Gemini requests omit the max-output field; local HF generation uses the remaining model context window.
- Added CLI options `--openai-reasoning-effort` and `--openai-text-verbosity`.

## Kaggle

- Updated the notebook, shell runner, and Python runner for fixed clusters, high reasoning effort, and no project output cap.
- Fixed direct execution of `scripts/build_evoeval_dataset.py` by adding the project root to `sys.path`.

# Correctness, fairness, and cost-efficiency pass

## Correctness fixes

- Fixed `tests/test_refactor.py`: two assertions still checked for all-caps section headers (e.g. `"CANONICAL IMPLEMENTATION"`, `"OUTPUT CONTRACT"`) left over from before the prompt format was refactored to XML tags. Updated to check `<canonical_implementation>`, `<output_contract>`, `<targets>`. All 13 tests now pass (was 11/13).
- Fixed `estimated_cost_usd` reporting `0.0` for every real API call: `MODEL_PRICING_USD_PER_1K_TOKENS` previously defaulted every model to `$0.0/$0.0` unless an env var overrode it. Added real default list prices for `gpt-5-mini`, `gpt-5-nano`, `gpt-5`, `gpt-5.6-sol/terra/luna`, and $0 for local Hugging Face models.
- Fixed a probe-generation domain mismatch: `_split_params` previously split on every comma including ones inside quoted strings (e.g. `"1,234"` was mis-parsed as four arguments), and untyped signatures fell back to generic int-heavy probes that errored out before reaching any mutated code. Fixed the splitter to be quote-aware and added `extract_example_call_exprs`, which derives probes from literal example calls in a task's spec text before falling back to type-hint-derived values.
- Fixed handoff summarization so infrastructure errors (API/OOM/timeout failures) are reported as "nothing was actually tried" rather than framed as rejected test-writing hypotheses for the next layer to avoid repeating (`summarize_attempt_history` in `src/layers/common.py`). Repeated identical infra errors are collapsed to one annotated entry instead of full stack traces repeated verbatim.
- Removed `scripts/build_evoeval_dataset.py` and the dataset-build step; the project now expects the 500-task EvoEval Parquet file to already exist (see `data/EVOEVAL_DATASET.md`), and `src/utils/dataset_loader.py` only reads it.

## Fairness between the proposed pipeline and the baseline

- Added a shared plateau-based stopping criterion (`PlateauTracker` in `src/layers/common.py`): a layer/iteration loop stops after `N` consecutive attempts with zero new kills (default `N=1`) instead of always spending its full fixed attempt/iteration budget. Applied identically to Layer 1, Layer 2, Layer 3, and the iterative baseline, so a cost/effectiveness comparison between them is not confounded by one arm being allowed to plateau-spend while the other always burns its full budget.
- Fixed the baseline's plateau tracking to not count infrastructure errors toward the zero-gain streak, matching the same distinction applied to the layers.
- Added `SKIP_LAYER_WHEN_ALL_PROBABLE_EQUIVALENT`: a layer is skipped entirely (no LLM call spent) when every remaining survivor is already flagged `PROBABLE_EQUIVALENT` by fuzz-probe analysis.
- Every layer and the baseline now record a `stop_reason` (`zero_survivors` / `plateau: ...` / `all_remaining_probable_equivalent` / `budget_exhausted`) in their metrics notes and handoff/metadata, so the stopping-reason breakdown across a whole run is directly reportable.

## Prompt content

- `build_handoff_summary`: `survivor_sample` now includes at most one mutant per surviving cluster (its most central member), excludes clusters already represented in the current round's `targets`, and drops `behavior_signature` when it's constant/uninformative across the sample.
- `_target_dossier` / `_probe_evidence`: dropped internal bookkeeping fields (`cluster_id`, `original_cluster_size`, `surviving_cluster_size`, `centrality`, `information_score`, `source_location`) from what is sent to the LLM -- these remain available in logged JSON artifacts for pipeline-internal analysis. Collapsed per-target probe evidence to a single summary line when every probe shows no divergence, instead of repeating the same negative result per target.
- Baseline's `_mutant_full_block`: removed `original_cluster_id` and `behavior_signature` (inconsistent with the baseline's own "no clustering" design), and added exact-diff deduplication so mutants with an identical diff aren't listed as separate blocks.

## Output structure

- Split pipeline output into a curated `report/` tier (aggregate metrics, comparison tables, statistics, figures, final per-task test suites under `generated_tests/`, run manifest) and a full diagnostic `raw/` tier (per-problem mutant/cluster dumps, per-layer directories, benchmark results, and the full LLM prompt/response log). `generate_statistical_report` now accepts a separate `raw_dir` to read per-problem/metrics data from.
- Added `VERBOSE_ARTIFACTS` (default off) controlling whether `raw/llm_responses/` -- the largest, least-needed-day-to-day artifact -- gets written at all; exposed as `--verbose-artifacts`.

## Execution / chunking

- Added `--index-start`/`--index-end` CLI flags (wired to the existing `config.PROBLEM_INDICES` / `select_problems(indices=...)` path) so a large dataset can be run in resumable index-range shards across multiple sessions.
- Added `scripts/merge_shards.py`, which combines multiple shard result directories into one unified `report/`+`raw/` output by reusing the pipeline's own `_aggregate_summaries`/`_aggregate_comparison_rows` functions and `generate_statistical_report`, rather than reimplementing aggregation logic.
- Added per-layer `--layer{1,2,3}-plateau-patience` / `--layer{1,2,3}-stop-on-plateau` CLI flags.

