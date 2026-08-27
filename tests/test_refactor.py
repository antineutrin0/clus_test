from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.clustering.representative_selector import RepresentativeStack
from src.layers.common import (
    evaluate_single_generated_test,
    extract_check_function,
    validate_check_contract,
)
from src.layers.layer1_generator import Layer1Generator
from src.llm.llm_client import LLMResponse, OpenAIClient, create_client
from src.llm.prompt_builder import build_layer1_batch_prompt
from src.mutation.mutation_engine import Mutant, generate_mutants, mark_obvious_equivalents
from src.pipeline import CLUSEPipeline, _surviving_clusters_from_initial
from src.utils import config
from src.utils.dataset_loader import Problem, load_dataset, select_problems


SOURCE = '''def task_func(x: int):
    """Return absolute value for negatives; otherwise increment by one."""
    if x < 0:
        return -x
    return x + 1
'''


class FakeClient:
    provider_name = "mock"
    model_name = "fake-batched"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        text = '''def check(candidate):
    assert candidate(-1) == 1
    assert candidate(0) == 1
    assert candidate(2) == 3
'''
        return LLMResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=len(text.split()),
            total_tokens=max(1, len(prompt.split())) + len(text.split()),
        )


class RefactorTests(unittest.TestCase):
    def test_contract_rejects_vacuous_test(self):
        ok, reason, _ = validate_check_contract("def check(candidate):\n    assert candidate is not None\n")
        self.assertFalse(ok)
        self.assertIn("never calls candidate", reason)

    def test_productive_validation_rejects_zero_kill(self):
        mutants = generate_mutants("Task/0", SOURCE, max_mutants=5)
        ok, _, _, reason, _ = evaluate_single_generated_test(
            "def check(candidate):\n    assert candidate(2) == candidate(2)\n",
            SOURCE,
            "task_func",
            mutants,
            require_kill=True,
        )
        self.assertFalse(ok)
        self.assertIn("unproductive", reason)

    def test_layer1_batches_all_representatives_and_limits_calls(self):
        mutants = generate_mutants("Task/1", SOURCE, max_mutants=8)
        self.assertGreaterEqual(len(mutants), 2)
        stacks = []
        for cluster_id, mutant in enumerate(mutants[:3]):
            mutant.cluster_id = cluster_id
            mutant.information_score = 1.0
            mutant.centrality = 1.0
            stacks.append(RepresentativeStack(cluster_id=cluster_id, representatives=[mutant]))
        for mutant in mutants[3:]:
            mutant.cluster_id = 0
            stacks[0].non_representatives.append(mutant)

        client = FakeClient()
        old_attempts = config.LAYER1_MAX_REFINEMENT
        try:
            config.LAYER1_MAX_REFINEMENT = 3
            layer = Layer1Generator(
                "Task/1",
                SOURCE,
                "task_func",
                "Return absolute value for negatives; otherwise increment by one.",
                output_dir=Path(tempfile.mkdtemp()) / "layer1",
                llm=client,
                probe_exprs=["candidate(-1)", "candidate(0)"],
                probe_outcomes=[("VALUE", "1"), ("VALUE", "1")],
                task_metadata={"dataset_name": "generic"},
            )
            _, _, metrics = layer.run(stacks, mutants, [])
        finally:
            config.LAYER1_MAX_REFINEMENT = old_attempts

        self.assertLessEqual(len(client.prompts), 3)
        self.assertEqual(metrics.llm_calls, len(client.prompts))
        first = client.prompts[0]
        for stack in stacks:
            self.assertIn(stack.representatives[0].mutant_id, first)
        self.assertIn("<targets>", first)
        self.assertIn("canonical_value", first)

    def test_prompt_uses_single_task_spec_and_targets_near_tail(self):
        mutants = generate_mutants("Task/2", SOURCE, max_mutants=3)
        for index, mutant in enumerate(mutants):
            mutant.cluster_id = index
        spec = "UNIQUE_SPEC_TOKEN: return the required numeric result."
        prompt = build_layer1_batch_prompt(
            targets=mutants,
            source_code=SOURCE,
            entry_point="task_func",
            prompt_text=spec,
            cluster_contexts={i: {"cluster_size": 1, "surviving_cluster_size": 1} for i in range(len(mutants))},
            probe_exprs=["candidate(0)"],
            probe_outcomes=[("VALUE", "1")],
            attempt=1,
        )
        self.assertEqual(prompt.count("UNIQUE_SPEC_TOKEN"), 1)
        # Prompt sections now use XML-style tags rather than all-caps headers
        # (see CHANGES.md prompt-format refactor). Assert on the tags that
        # actually appear, and on the same ordering intent as before: targets
        # come after the canonical implementation, and the output contract
        # comes after targets.
        self.assertGreater(
            prompt.rfind("<targets>"),
            prompt.rfind("<canonical_implementation>"),
        )
        self.assertGreater(
            prompt.rfind("<output_contract>"),
            prompt.rfind("<targets>"),
        )


    def test_extract_check_ignores_fences_and_trailing_prose(self):
        response = """Here is the test:
```python
def check(candidate):
    assert candidate(2) == 4
```
This test covers the boundary.
"""
        extracted = extract_check_function(response)
        self.assertEqual(
            extracted,
            "def check(candidate):\n    assert candidate(2) == 4",
        )

    def test_fixed_initial_clusters_are_filtered_not_recomputed(self):
        mutants = generate_mutants("Task/fixed", SOURCE, max_mutants=5)
        self.assertGreaterEqual(len(mutants), 3)
        initial = {0: mutants[:2], 1: mutants[2:]}
        for cluster_id, members in initial.items():
            for member in members:
                member.cluster_id = cluster_id
        surviving = [mutants[1], mutants[-1]]
        filtered = _surviving_clusters_from_initial(initial, surviving)
        self.assertEqual(set(filtered), {0, 1})
        self.assertEqual([m.mutant_id for m in filtered[0]], [mutants[1].mutant_id])
        self.assertEqual([m.mutant_id for m in filtered[1]], [mutants[-1].mutant_id])
        self.assertEqual(filtered[0][0].cluster_id, 0)
        self.assertEqual(filtered[1][0].cluster_id, 1)

    def test_openai_uses_high_effort_without_project_output_cap(self):
        from types import SimpleNamespace

        captured = {}

        class Responses:
            def create(self, **kwargs):
                captured.update(kwargs)
                usage = SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=2),
                )
                return SimpleNamespace(output_text="def check(candidate):\n    assert candidate(1) == 1", usage=usage)

        fake_client = SimpleNamespace(responses=Responses())
        client = OpenAIClient("gpt-5-mini", 0)
        client._client = lambda: fake_client  # type: ignore[method-assign]
        old_effort = config.OPENAI_REASONING_EFFORT
        old_verbosity = config.OPENAI_TEXT_VERBOSITY
        try:
            config.OPENAI_REASONING_EFFORT = "high"
            config.OPENAI_TEXT_VERBOSITY = "low"
            response = client.generate("test prompt")
        finally:
            config.OPENAI_REASONING_EFFORT = old_effort
            config.OPENAI_TEXT_VERBOSITY = old_verbosity
        self.assertEqual(captured["reasoning"], {"effort": "high"})
        self.assertEqual(captured["text"], {"verbosity": "low"})
        self.assertNotIn("max_output_tokens", captured)
        self.assertEqual(response.thoughts_tokens, 2)

    def test_model_validation_rejects_placeholders_and_cross_provider_fallbacks(self):
        with self.assertRaises(ValueError):
            create_client("openai", "<gpt-5-mini>", 100)
        with self.assertRaises(ValueError):
            create_client("openai", "gpt-5-mini", 100, ["gemini-2.5-flash"])

    def test_conservative_equivalent_detector(self):
        source = "def task_func(a, b):\n    return min(a, b)\n"
        mutants = generate_mutants("Task/3", source, max_mutants=20)
        mark_obvious_equivalents(source, mutants)
        equivalent = [m for m in mutants if m.equivalence_status == "STATIC_EQUIVALENT"]
        self.assertTrue(any(m.operator == "ARGUMENT_SWAP" for m in equivalent))


    def test_evoeval_normalization_uses_unique_subset_id(self):
        import json
        temp = Path(tempfile.mkdtemp()) / "evo.jsonl"
        row = {
            "task_id": "EvoEval/7",
            "prompt": "def task_func(x):\n    \"\"\"Return x.\"\"\"\n",
            "canonical_solution": "    return x\n",
            "entry_point": "task_func",
            "test": "def check(candidate):\n    assert candidate(1) == 1\n",
            "evoeval_subset": "subtle",
        }
        temp.write_text(json.dumps(row) + "\n", encoding="utf-8")
        problem = load_dataset(temp, dataset_type="evoeval")[0]
        self.assertEqual(problem.task_id, "EvoEval_subtle/7")
        self.assertEqual(problem.dataset_subset, "subtle")
        self.assertEqual(problem.parent_task_id, "HumanEval/7")
        self.assertEqual(problem.source_task_id, "EvoEval/7")

    def test_stratified_evoeval_sampling_is_balanced(self):
        problems = []
        for subset in ("difficult", "creative", "subtle", "combine", "tool_use"):
            for index in range(20):
                problems.append(Problem(
                    task_id=f"EvoEval_{subset}/{index}",
                    source_prompt="def task_func(x):\n    return x\n",
                    canonical_solution="",
                    official_test="def check(candidate):\n    assert candidate(1) == 1\n",
                    entry_point="task_func",
                    dataset_name="evoeval",
                    dataset_subset=subset,
                    parent_task_id=f"HumanEval/{index}",
                ))
        selected = select_problems(
            problems, percent=0.5, sample_mode="stratified", seed=42, stratify_by="dataset_subset"
        )
        counts = {subset: sum(p.dataset_subset == subset for p in selected) for subset in {p.dataset_subset for p in problems}}
        self.assertEqual(len(selected), 50)
        self.assertEqual(set(counts.values()), {10})


    def test_layer2_reuses_initial_clusters_and_skips_layer3_after_success(self):
        class InvalidClient:
            provider_name = "mock"
            model_name = "invalid-l1"

            def __init__(self):
                self.prompts = []

            def generate(self, prompt: str) -> LLMResponse:
                self.prompts.append(prompt)
                return LLMResponse(
                    text="def check(candidate):\n    assert candidate is not None",
                    model=self.model_name, provider=self.provider_name, total_tokens=10,
                )

        class BombClient:
            provider_name = "mock"
            model_name = "must-not-run"

            def __init__(self):
                self.calls = 0

            def generate(self, prompt: str):
                self.calls += 1
                raise AssertionError("Layer 3 should have been skipped")

        results = Path(tempfile.mkdtemp()) / "results"
        problem = Problem(
            task_id="Task/reuse",
            source_prompt=SOURCE,
            canonical_solution="",
            official_test=(
                "def check(candidate):\n"
                "    assert candidate(-1) == 1\n"
                "    assert candidate(0) == 1\n"
                "    assert candidate(2) == 3\n"
            ),
            entry_point="task_func",
            dataset_name="generic",
        )
        pipeline = CLUSEPipeline(
            results_dir=results, max_layers=3, max_mutants=5, max_probes=2,
            run_baseline=False, generate_statistics=False,
            layer_providers={1: "mock", 2: "mock", 3: "mock"},
            layer_models={1: "mock-l1", 2: "mock-l2", 3: "mock-l3"},
        )
        l1 = InvalidClient()
        l2 = FakeClient()
        l3 = BombClient()
        pipeline.layer_clients = {1: l1, 2: l2, 3: l3}
        old_l1 = config.LAYER1_MAX_REFINEMENT
        old_l2 = config.LAYER2_MAX_REFINEMENT
        try:
            config.LAYER1_MAX_REFINEMENT = 1
            config.LAYER2_MAX_REFINEMENT = 1
            tracker = pipeline.run_problem(problem)
        finally:
            config.LAYER1_MAX_REFINEMENT = old_l1
            config.LAYER2_MAX_REFINEMENT = old_l2
        self.assertEqual(len(l2.prompts), 1)
        self.assertEqual(l3.calls, 0)
        self.assertIn("Fixed initial partition, reused unmodified", l2.prompts[0])
        self.assertTrue(all(row["cluster_assignment_reused"] for row in tracker.metadata["layer2_clusters"]))
        self.assertEqual(tracker.metadata.get("layer3_skipped_reason"), "all_mutants_killed_before_layer3")

    def test_missing_library_preflight(self):
        problem = Problem(
            task_id="Task/env",
            source_prompt="def task_func(x):\n    return x\n",
            canonical_solution="",
            official_test="def check(candidate):\n    assert candidate(1) == 1\n",
            entry_point="task_func",
            libs=["definitely_missing_cluse_test_package_xyz"],
        )
        self.assertEqual(
            problem.missing_libraries(),
            ["definitely_missing_cluse_test_package_xyz"],
        )


if __name__ == "__main__":
    unittest.main()
