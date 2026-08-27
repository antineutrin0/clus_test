# Kaggle Quick Start

## 1. Sync the project via GitHub (recommended)

Rather than re-uploading a zipped Kaggle Dataset every time the code
changes, clone/pull the project from GitHub inside the notebook:

```python
import os
if os.path.exists("/kaggle/working/CLUSE-Test-EvoEval"):
    !cd /kaggle/working/CLUSE-Test-EvoEval && git pull
else:
    !git clone https://github.com/<you>/CLUSE-Test-EvoEval.git /kaggle/working/CLUSE-Test-EvoEval
%cd /kaggle/working/CLUSE-Test-EvoEval
```

Use a private repo with a token in Kaggle Secrets if the code isn't public.
This gives real version history tied to each Kaggle run, and lets you pull
a one-line fix without re-uploading a Dataset. A Kaggle Dataset upload is
still fine for a final, frozen snapshot you want archived alongside a
specific set of results -- just re-version the same Dataset rather than
creating a new one each time, so the attached path stays stable.

## 2. Enable Internet

Turn Kaggle Internet **On** to `git pull`/`git clone`, install packages, and
reach the OpenAI API from Layer 2/3 and the baseline.

## 3. Configure secrets

Create and enable in Kaggle's Secrets/Environment Variables panel (Add-ons -> Secrets):

```text
OPENAI_API_KEY
```

Never hardcode this or commit a `.env` file to the repo -- check `.gitignore`
covers it.

## 4. Provide the dataset

The 500-task EvoEval Parquet file is already built (see
`data/EVOEVAL_DATASET.md`) -- this project does not include a dataset-build
step. Either:

- point `CLUSE_DATASET` at the file's path, or
- upload the Parquet file itself as its own small Kaggle Dataset (separate
  from the code) and attach it via **Add Input**, so you're not rebuilding
  or re-uploading it alongside every code change.

## 5. Run the smoke test first

Before spending any API budget, run the mock-LLM smoke test and the unit
suite inside Kaggle itself -- this catches environment/package drift (a
different installed version than wherever you last tested) before it costs
real money:

```bash
python -m unittest discover -s tests -v
python run_pipeline.py --dataset <path> --dataset-type evoeval --mock \
  --results /kaggle/working/smoke_results --limit 1
```

This checks schema normalization, AST mutation, clustering, batched layers,
statistics, figures, and the report/raw output split without API charges.

## 6. Run in resumable index-range shards

A full 500-task run in one sitting can exceed Kaggle's session limits.
Running in shards makes each one a self-contained, resumable unit -- if a
shard crashes partway through, only that shard needs rerunning:

```bash
# Session 1
python run_pipeline.py --evoeval-semantic --hf-split test --dataset-type evoeval \
  --index-start 0 --index-end 100 --results /kaggle/working/results_shard0 \
  --layer1-provider hf --layer1-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer2-provider openai --layer2-model gpt-5-nano \
  --layer3-provider openai --layer3-model gpt-5-mini \
  --run-baseline --baseline-provider openai --baseline-model gpt-5-mini

# Session 2 (repeat with 100-200, 200-300, 300-400, 400-500)
python run_pipeline.py --evoeval-semantic --hf-split test --dataset-type evoeval \
  --index-start 100 --index-end 200 --results /kaggle/working/results_shard1 \
  ...
```

Shard boundaries can follow the EvoEval subsets directly (5 shards of 100:
difficult/creative/subtle/combine/tool_use), which lines up with the
existing stratified sampling design and lets you inspect one subset's
results before the others finish.

Save each Kaggle notebook version ("Save Version" -> "Save & Run All") after
a shard completes -- that's your audit trail for exactly what a given run's
environment/package state was, since GitHub tracks code but not that.

## 7. Combine shard results

```bash
python scripts/merge_shards.py \
  --shards /kaggle/working/results_shard0 /kaggle/working/results_shard1 \
           /kaggle/working/results_shard2 /kaggle/working/results_shard3 \
           /kaggle/working/results_shard4 \
  --output /kaggle/working/results_full
```

This reuses the pipeline's own aggregation functions (it does not
reimplement them), so the combined `report/` is produced by the same code
path a single full run would have used. Confirm the combined
`report/run_manifest.json` shows `selected_problems: 500` and
`subset_counts` totalling 100 per subset before treating results as final.

## 8. Output

Research artifacts are written under the chosen `--results` directory,
split into two tiers (see `README.md` / `VALIDATION.md` for the rationale):

```text
<results>/report/
  aggregate_summary.csv
  aggregate_metrics.json
  selected_tasks.csv
  comparison/
  statistics/
  figures/
  generated_tests/*.py      # final accepted test suite per task
  run_manifest.json

<results>/raw/
  <task_id>/                # per-problem mutants, clusters, normalized_task.json
  layer1/ layer2/ layer3/
  baseline_iterative/ baseline_metrics/
  metrics/
  benchmark/
  llm_responses/            # only written with --verbose-artifacts 1
```

`--zip-output` creates a downloadable ZIP of the whole `--results` directory
automatically.

## 9. Debugging a specific run

For a run whose output you need to inspect closely (a task's score looks
wrong, or you're checking exactly what a model was shown), pass
`--verbose-artifacts 1` to also write the full per-attempt LLM
prompt/response log under `raw/llm_responses/`. Leave this off (the
default) for the pilot/full runs you intend to cite -- it's the largest
artifact by far and isn't needed day to day.

## 10. Pipeline settings worth knowing about

```text
--layer1-max-tokens 0 / --layer2-max-tokens 0 / --layer3-max-tokens 0 / --baseline-max-tokens 0
    A value of 0 removes the project-level output-token ceiling.

--openai-reasoning-effort high --openai-text-verbosity low
    Recommended defaults for Layer 2/3/baseline OpenAI calls.

--layer{1,2,3}-plateau-patience / --layer{1,2,3}-stop-on-plateau
--baseline-plateau-patience / --baseline-stop-on-plateau
    All default to "stop after 1 consecutive zero-new-kill attempt" rather
    than always spending the full fixed attempt/iteration budget. This
    criterion is shared code (PlateauTracker) applied identically to every
    layer and the baseline, so cost/effectiveness comparisons between them
    are not confounded by asymmetric stopping rules. Per-task metrics
    record which stop_reason applied (zero_survivors / plateau /
    all_remaining_probable_equivalent / budget_exhausted).
```

The initial cluster partition is computed once and reused across Layer 2/3
without reclustering; later layers are skipped entirely (no LLM call spent)
when no mutants survive, or when every remaining survivor is already
flagged `PROBABLE_EQUIVALENT` by fuzz-probe analysis.
