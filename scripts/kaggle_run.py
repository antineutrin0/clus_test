"""Kaggle convenience runner for the cost-aware batched Claus-Test pipeline."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "10pct", "full", "both"], default="10pct")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dataset", default=os.environ.get("CLUSE_DATASET", ""))
    parser.add_argument("--hf-dataset", default=os.environ.get("CLUSE_HF_DATASET", ""))
    parser.add_argument("--hf-split", default=os.environ.get("CLUSE_HF_SPLIT", "test"))
    parser.add_argument("--dataset-type", choices=["auto", "evoeval", "humaneval", "bigcodebench", "generic"], default="evoeval")
    parser.add_argument("--evoeval-semantic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-layers", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--max-mutants", type=int, default=25)
    parser.add_argument("--max-probes", type=int, default=6)
    parser.add_argument("--layer1-provider", default=os.environ.get("LAYER1_PROVIDER", "hf"))
    parser.add_argument("--layer1-model", default=os.environ.get("LAYER1_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"))
    parser.add_argument("--layer2-provider", default=os.environ.get("LAYER2_PROVIDER", "openai"))
    parser.add_argument("--layer2-model", default=os.environ.get("LAYER2_MODEL", "gpt-5-nano"))
    parser.add_argument("--layer3-provider", default=os.environ.get("LAYER3_PROVIDER", "openai"))
    parser.add_argument("--layer3-model", default=os.environ.get("LAYER3_MODEL", "gpt-5-mini"))
    parser.add_argument("--baseline-provider", default=os.environ.get("BASELINE_PROVIDER", "openai"))
    parser.add_argument("--baseline-model", default=os.environ.get("BASELINE_MODEL", "gpt-5-mini"))
    parser.add_argument("--run-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-max-iterations", type=int, default=10)
    parser.add_argument("--layer1-max-attempts", type=int, default=3)
    parser.add_argument("--layer2-max-attempts", type=int, default=2)
    parser.add_argument("--layer3-max-attempts", type=int, default=1)
    parser.add_argument("--display-llm-responses", type=int, choices=[0, 1], default=1)
    parser.add_argument("--display-first-problem-only", type=int, choices=[0, 1], default=1)
    parser.add_argument("--verbose-artifacts", type=int, choices=[0, 1], default=0, help="1 also saves the full per-attempt LLM prompt/response dump (raw/llm_responses/); off by default")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--figure-pdf", type=int, choices=[0, 1], default=1)
    parser.add_argument("--openai-reasoning-effort", choices=["minimal", "low", "medium", "high"], default="high")
    parser.add_argument("--openai-text-verbosity", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--zip-output", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()


    stages = ["10pct", "full"] if args.stage == "both" else [args.stage]
    root = Path.cwd()
    for stage in stages:
        percent = {"smoke": "0.002", "10pct": "0.10", "full": "1.0"}[stage]
        results = f"/kaggle/working/cluse_results_{stage}"
        cmd = [
            sys.executable, str(root / "run_pipeline.py"),
            "--dataset-type", args.dataset_type,
            "--run-name", f"benchmark_{stage}",
            "--percent", percent,
            "--sample-mode", "stratified",
            "--stratify-by", "dataset_subset",
            "--seed", "42",
            "--results", results,
            "--max-layers", str(args.max_layers),
            "--max-mutants", str(args.max_mutants),
            "--max-probes", str(args.max_probes),
            "--layer1-provider", args.layer1_provider,
            "--layer1-model", args.layer1_model,
            "--layer1-max-attempts", str(args.layer1_max_attempts),
            "--layer1-max-tokens", "0",
            "--layer2-provider", args.layer2_provider,
            "--layer2-model", args.layer2_model,
            "--layer2-fallback-models", "",
            "--layer2-max-attempts", str(args.layer2_max_attempts),
            "--layer2-max-tokens", "0",
            "--layer3-provider", args.layer3_provider,
            "--layer3-model", args.layer3_model,
            "--layer3-fallback-models", "",
            "--layer3-max-attempts", str(args.layer3_max_attempts),
            "--layer3-max-tokens", "0",
            "--baseline-provider", args.baseline_provider,
            "--baseline-model", args.baseline_model,
            "--baseline-fallback-models", "",
            "--baseline-max-iterations", str(args.baseline_max_iterations),
            "--baseline-max-tokens", "0",
            "--openai-reasoning-effort", args.openai_reasoning_effort,
            "--openai-text-verbosity", args.openai_text_verbosity,
            "--baseline-plateau-patience", "1",
            "--baseline-stop-on-plateau", "1",
            "--layer1-plateau-patience", "1", "--layer1-stop-on-plateau", "1",
            "--layer2-plateau-patience", "1", "--layer2-stop-on-plateau", "1",
            "--layer3-plateau-patience", "1", "--layer3-stop-on-plateau", "1",
            "--require-productive-test", "1",
            "--skip-missing-libraries", "1",
            "--display-llm-responses", str(args.display_llm_responses),
            "--display-first-problem-only", str(args.display_first_problem_only),
            "--display-compact-call-summary", "1",
            "--log-full-llm-io", "0",
            "--verbose-artifacts", str(args.verbose_artifacts),
            "--generate-statistics", "1",
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--figure-pdf", str(args.figure_pdf),
        ]
        if args.evoeval_semantic and not args.dataset and not args.hf_dataset:
            cmd.extend(["--evoeval-semantic", "--hf-split", args.hf_split])
        elif args.hf_dataset:
            cmd.extend(["--hf-dataset", args.hf_dataset, "--hf-split", args.hf_split])
        else:
            cmd.extend(["--dataset", args.dataset])
        if args.run_baseline:
            cmd.append("--run-baseline")
        if args.mock:
            cmd.append("--mock")
        if args.zip_output:
            cmd.append("--zip-output")
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        print(f"Finished {stage}; results: {results}", flush=True)


if __name__ == "__main__":
    main()
