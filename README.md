# Claus-Test: Cost-Aware EvoEval Mutation-Guided Test Generation

This release finalizes **EvoEval** as the primary evaluation benchmark for Claus-Test. The selected suite contains the five semantic-altering subsets—Difficult, Creative, Subtle, Combine, and Tool-use—for a total of **500 HumanEval-compatible Python tasks**.

The pipeline remains compatible with HumanEval, BigCodeBench, and generic tabular datasets, but all recommended research commands, sampling rules, metadata, statistics, and notebook cells now target EvoEval.

## Finalized dataset

The project combines these official Hugging Face repositories:

- `evoeval/EvoEval_difficult`
- `evoeval/EvoEval_creative`
- `evoeval/EvoEval_subtle`
- `evoeval/EvoEval_combine`
- `evoeval/EvoEval_tool_use`

Each repository provides 100 tasks with the HumanEval-style fields `task_id`, `prompt`, `canonical_solution`, `entry_point`, and `test`.

The two semantic-preserving prompt-paraphrase subsets are intentionally excluded from the primary dataset because they do not add independent program implementations for AST mutation analysis.

The 500-task Parquet file (`EvoEval_semantic_500.parquet`) is already built and
is expected to be present under `data/evoeval/` (or pointed to via
`CLUSE_DATASET`) -- see `data/EVOEVAL_DATASET.md` for the schema and where the
loader looks for it. This project no longer includes a dataset-build step;
`src/utils/dataset_loader.py` only reads the pre-built file.

## Direct loading without a local file

The CLI can load all five subsets directly:

```bash
python run_pipeline.py \
  --evoeval-semantic \
  --hf-split test \
  --dataset-type evoeval
```

Using the built Parquet file is recommended for repeated experiments because it avoids downloading and recombining the dataset on every run.

## Balanced sampling

EvoEval contains five subsets with different task characteristics. Recommended pilot and 10% evaluations use subset-stratified sampling:

```bash
--percent 0.10 \
--sample-mode stratified \
--stratify-by dataset_subset \
--seed 42
```

For the 500-task suite, this selects exactly 50 tasks: 10 from each subset. The selected rows are saved to:

```text
<results>/selected_tasks.csv
<results>/selected_tasks.json
```

## Fixed-cluster, high-effort generation

- Mutants are clustered **once**, before Layer 1.
- Layers 2 and 3 never recluster. They reuse the original cluster IDs and work only with surviving members.
- If the original representative is killed, the most informative live member of that same cluster becomes the prompt target.
- **Layer 1:** all live cluster targets are supplied jointly; at most 3 task-level calls.
- **Layer 2:** all surviving original clusters are refined jointly; at most 2 calls by default.
- **Layer 3:** final escalation is invoked only if survivors remain.
- If any layer kills all mutants, later layers are skipped automatically.
- OpenAI calls use `reasoning.effort=high` and low visible verbosity.
- `--layerN-max-tokens 0` and `--baseline-max-tokens 0` remove the project-imposed output-token ceiling.
- A generated test is accepted only if it passes on the canonical implementation and kills at least one active mutant.
- Response extraction is AST-aware and keeps the exact `check(candidate)` function even when a model adds fences or trailing prose.

## Recommended model assignment

The supplied notebook uses:

- Layer 1: `Qwen/Qwen2.5-Coder-1.5B-Instruct` locally;
- Layer 2: `gpt-5-nano`;
- Layer 3: `gpt-5-mini`;
- Baseline: `gpt-5-mini`.

This preserves the research design: a no-API-cost local first layer, a low-cost API refinement layer, and a stronger final model that is also used by the baseline.

Before a paid run, verify model IDs against the models available to your OpenAI project. Do not place angle brackets around model names.

## Recommended 10% experiment

```bash
python run_pipeline.py \
  --dataset /kaggle/working/evoeval_semantic_500/EvoEval_semantic_500.parquet \
  --dataset-type evoeval \
  --run-name evoeval_10pct \
  --percent 0.10 \
  --sample-mode stratified \
  --stratify-by dataset_subset \
  --seed 42 \
  --results /kaggle/working/cluse_evoeval_10pct \
  --max-layers 3 \
  --max-mutants 25 \
  --max-probes 6 \
  --layer1-provider hf \
  --layer1-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer1-max-attempts 3 \
  --layer1-max-tokens 0 \
  --layer2-provider openai \
  --layer2-model gpt-5-nano \
  --layer2-fallback-models "" \
  --layer2-max-attempts 2 \
  --layer2-max-tokens 0 \
  --layer3-provider openai \
  --layer3-model gpt-5-mini \
  --layer3-fallback-models "" \
  --layer3-max-attempts 1 \
  --layer3-max-tokens 0 \
  --run-baseline \
  --baseline-provider openai \
  --baseline-model gpt-5-mini \
  --baseline-fallback-models "" \
  --baseline-max-iterations 10 \
  --baseline-max-tokens 0 \
  --openai-reasoning-effort high \
  --openai-text-verbosity low \
  --baseline-plateau-patience 2 \
  --baseline-stop-on-plateau 0 \
  --require-productive-test 1 \
  --display-llm-responses 1 \
  --display-first-problem-only 1 \
  --display-compact-call-summary 1 \
  --log-full-llm-io 0 \
  --save-llm-responses 1 \
  --generate-statistics 1 \
  --bootstrap-samples 5000 \
  --figure-pdf 1 \
  --zip-output
```

## EvoEval-aware statistical outputs

The release retains the paper-aligned effectiveness, agreement, efficiency, runtime, operator, layer, and baseline-fairness outputs. It additionally writes:

```text
statistics/evoeval_subset_summary.csv
statistics/evoeval_subset_paired_comparison.csv
figures/evoeval_subset_effectiveness.png
figures/evoeval_subset_token_efficiency.png
```

Because several evolved tasks share a HumanEval parent, `paired_comparison_summary.csv` includes a **HumanEval-parent-clustered bootstrap confidence interval** in addition to the ordinary task-level interval.

The principal reported measures should remain separate:

- macro and micro mutation score;
- official-test kill-set accuracy, precision, recall, and F1;
- paid and total tokens;
- estimated API cost;
- tokens/cost/runtime per killed mutant;
- marginal kills and marginal efficiency by layer;
- fixed-iteration, first-final-score, plateau, and matched-budget baseline comparisons.

## Validation

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

Use `--mock` only for integration validation. Mock results are not research findings.
