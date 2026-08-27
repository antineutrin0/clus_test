#!/usr/bin/env bash
set -euo pipefail

EXTRA_ARGS=${EXTRA_ARGS:-}
COMMON=(
  --evoeval-semantic --hf-split test
  --dataset-type evoeval --sample-mode stratified --stratify-by dataset_subset --seed 42
  --max-layers 3 --max-mutants 25 --max-probes 6
  --layer1-provider hf --layer1-model Qwen/Qwen2.5-Coder-1.5B-Instruct --layer1-max-attempts 3 --layer1-max-tokens 0
  --layer2-provider openai --layer2-model "${LAYER2_MODEL:-gpt-5-nano}" --layer2-fallback-models "" --layer2-max-attempts 2 --layer2-max-tokens 0
  --layer3-provider openai --layer3-model "${LAYER3_MODEL:-gpt-5-mini}" --layer3-fallback-models "" --layer3-max-attempts 1 --layer3-max-tokens 0
  --run-baseline --baseline-provider openai --baseline-model "${BASELINE_MODEL:-gpt-5-mini}" --baseline-fallback-models ""
  --baseline-max-iterations 10 --baseline-max-tokens 0 --openai-reasoning-effort high --openai-text-verbosity low
  --baseline-plateau-patience 1 --baseline-stop-on-plateau 1
  --layer1-plateau-patience 1 --layer1-stop-on-plateau 1
  --layer2-plateau-patience 1 --layer2-stop-on-plateau 1
  --layer3-plateau-patience 1 --layer3-stop-on-plateau 1
  --require-productive-test 1 --skip-missing-libraries 1
  --display-llm-responses 1 --display-first-problem-only 1 --display-compact-call-summary 1 --log-full-llm-io 0
  --verbose-artifacts 0 --generate-statistics 1 --bootstrap-samples 5000 --figure-pdf 1 --zip-output
)

python run_pipeline.py "${COMMON[@]}" \
  --run-name evoeval_10pct --percent 0.10 --results /kaggle/working/cluse_evoeval_10pct $EXTRA_ARGS

python run_pipeline.py "${COMMON[@]}" \
  --run-name evoeval_full --percent 1.0 --results /kaggle/working/cluse_evoeval_full $EXTRA_ARGS
