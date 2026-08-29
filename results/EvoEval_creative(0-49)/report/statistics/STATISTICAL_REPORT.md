# Claus-Test Evaluation Report

This report keeps mutation effectiveness, official-test agreement, token/API efficiency, runtime, and baseline fairness separate.
Raw macro and micro scores are primary; conservative equivalent-adjusted scores exclude only statically identified equivalent mutants.
The baseline is reported under its fixed iteration budget, its first final-score point, its first plateau, and matched proposed-token budgets.

## Main summary

See `evaluation_summary.json` for confidence intervals, paired tests, effect sizes, and aggregate values.

## Saved figures

- `../figures/accuracy_comparison.png`
- `../figures/baseline_fairness.png`
- `../figures/call_validation_outcomes.png`
- `../figures/cluster_compression.png`
- `../figures/effectiveness_token_efficiency.png`
- `../figures/evoeval_subset_effectiveness.png`
- `../figures/evoeval_subset_token_efficiency.png`
- `../figures/layer_cost_efficiency.png`
- `../figures/layer_marginal_kills.png`
- `../figures/llm_usage_by_provider_model.png`
- `../figures/mutation_score_comparison.png`
- `../figures/operator_kill_rate.png`
- `../figures/paired_score_delta.png`
- `../figures/prompt_compression.png`
- `../figures/runtime_breakdown.png`
- `../figures/token_cost_comparison.png`

## Saved tables

- `aggregate_score_conventions.csv`
- `baseline_fairness_per_task.csv`
- `baseline_iteration_history.csv`
- `call_validation_outcomes.csv`
- `cluster_compression_per_task.csv`
- `conservative_equivalent_mutants.csv`
- `effectiveness_summary.csv`
- `efficiency_summary.csv`
- `evoeval_subset_paired_comparison.csv`
- `evoeval_subset_summary.csv`
- `layer_contribution_summary.csv`
- `llm_call_level_metrics.csv`
- `llm_usage_summary.csv`
- `official_agreement_summary.csv`
- `operator_difficulty_summary.csv`
- `paired_comparison_summary.csv`
- `runtime_breakdown_per_task.csv`
