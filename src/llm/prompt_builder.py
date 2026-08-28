"""Standardized prompts for mutation-guided test generation.

The four prompt families share the same evidence order while preserving their
separate responsibilities:

* Layer 1 performs joint, low-cost generation over the initial representatives.
* Layer 2 refines only the surviving members of those same initial clusters.
* Layer 3 performs final escalation over the remaining hard clusters.
* The baseline receives every surviving mutant without cluster compression.

Per project decision, no artificial input/output token budget is enforced in
any of these prompts: full source, full dossiers, and full evidence are
always passed through. The only response constraint is the executable
``check(candidate)`` contract. Helper truncation utilities (``_bounded`` and
friends) are retained for optional reuse elsewhere, but the prompt builders
in this file no longer call them with a restrictive character budget.

All four prompt builders use the same rule-based structure: an explicit
<role>, background <task_information>/<canonical_implementation>/evidence
sections, a numbered, atomic <rules> block, explicit <reasoning_steps>, a
worked <good_example>/<bad_example> pair, and a final <output_contract>.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
from collections import Counter
from typing import Dict, List, Optional, Sequence

from src.mutation.mutation_engine import Mutant


# ---------------------------------------------------------------------------
# Shared rule text
# ---------------------------------------------------------------------------
# Kept as a single source of truth so the three escalation layers cannot
# silently drift out of sync on shared wording. Each layer extends this base
# with its own additional, layer-specific rules and renumbers the full list
# when it renders its final prompt.

_BASE_RULES: List[str] = [
    "Define exactly one top-level function named `check` with signature "
    "`def check(candidate):`.",
    "Every rule in this list is mandatory. Do not skip a rule because it "
    "seems inconvenient for a particular target.",
    "Call `candidate(...)` with concrete, deterministic, in-domain inputs "
    "only. Never invent inputs outside the domain implied by the "
    "specification.",
    "Every `assert` must depend on the actual return value of a "
    "`candidate(...)` call made inside this function. Never assert a fact "
    "that would be true regardless of what `candidate` returns -- that is a "
    "vacuous test and is a hard failure even if it technically passes.",
    "Do not guess exact expected outputs. Derive expected values either from "
    "the specification's stated semantics, or from a "
    "`<canonical_probe_oracles>` entry. If neither gives you a value for an "
    "input you want to use, pick a different input that a probe oracle or "
    "the specification does cover.",
    "If a target's `probe_evidence` is empty or shows only \"same\" "
    "outcomes, no existing probe distinguishes it yet -- do not assume the "
    "mutant's behavior; derive a new distinguishing input directly from "
    "comparing `original_code` and `mutated_code` in that target's dossier.",
]

_SHARED_TAIL_RULES: List[str] = [
    "Deterministic standard-library tools are allowed where the task "
    "requires them: `random.seed`, `unittest.mock.patch`, `tempfile`, "
    "`math`, context managers.",
    "Do not define `unittest.TestCase` or `pytest` classes/fixtures.",
    "Do not inspect `candidate`'s source, AST, bytecode, `__globals__`, or "
    "object identity. Test only through its input/output behavior.",
    "Do not access the network or leave persistent external state (files, "
    "sockets, temp dirs not cleaned up).",
]

_OUTPUT_FORMAT_RULE = (
    "Output only the raw Python function. No Markdown fences, no prose, no "
    "`print`, no trailing `return True`, no truncation or abbreviation."
)

_OUTPUT_CONTRACT_BLOCK = """<output_contract>
Return one complete Python function with this exact signature:

def check(candidate):
    ...

Follow every rule in <rules>. Output only the function -- no fences, no
explanation, no partial code.
</output_contract>
"""

# ---------------------------------------------------------------------------
# Compact rule/handoff format for Layer 2 and Layer 3 only.
# ---------------------------------------------------------------------------
# Layer 1 keeps the full scaffolded prompt (verbose rules, a worked
# good/bad example pair, a numbered reasoning walkthrough) because it targets
# the weakest, cheapest model in the pipeline, which benefits most from
# explicit scaffolding. Layer 2 and Layer 3 use stronger reasoning models and
# already receive real prior attempts (not synthetic examples) via the
# handoff, so the same scaffolding there was mostly redundant token cost
# rather than signal the model used. These layers use a denser rule set, a
# 3-line reasoning checklist instead of the full walkthrough, and drop the
# synthetic worked examples entirely -- the compact handoff below supplies a
# real accepted/failed example from this exact task instead.

_COMPACT_CORE_RULES: List[str] = [
    "Define exactly one `def check(candidate):`.",
    "Call `candidate(...)` only with concrete, in-domain inputs.",
    "Base every `assert` on a real `candidate(...)` return value -- no vacuous asserts.",
    "Never guess an expected output; use the spec or a probe's `canonical_value`.",
    "If a target's `probe_evidence` is empty or all \"same\", derive a new input from its `exact_diff`.",
    "Skip `TestCase`/`pytest` fixtures and inspecting `candidate` internals (AST, `__globals__`).",
    "Avoid network/filesystem side effects; `random.seed`, `mock.patch`, `tempfile`, `math` are fine.",
    "Pass on the canonical implementation; include only targets you're confident about.",
    "Don't reproduce a call/assertion already in `<accepted_test_digest>`.",
]

_COMPACT_REASONING_STEPS = """<reasoning_steps>
Silently, before writing code:
1. Read the handoff; note what already failed per target.
2. Per target, find one untested input where canonical and mutant diverge,
   sourced from the spec or a probe value -- never guessed.
3. Drop any target you can't back with a concrete input.
Output only the final function.
</reasoning_steps>
"""


def _fmt_call_list(calls: Sequence[str], limit: int = 3) -> str:
    calls = list(calls)[:limit]
    return "; ".join(calls) if calls else "(no candidate calls recorded)"


def _compact_handoff_block(handoff: Optional[Dict], max_attempts_shown: int = 3, max_sample_shown: int = 6) -> str:
    """Render a layer's handoff as short, information-dense prose instead of
    an indented JSON dump.

    Carries the same facts the JSON form carried -- what was tried, what
    happened, why the loop stopped, what's still alive -- in a fraction of
    the tokens, and is more directly readable by the model than parsing a
    nested object. Real attempted candidate calls/assertions from this exact
    task are shown, which serves as a concrete, task-grounded example in
    place of the synthetic good/bad examples the full-scaffold layers use.
    """
    handoff = handoff or {}
    lines: List[str] = []

    attempt_count = handoff.get("attempt_count", 0)
    productive = handoff.get("productive_attempts", 0)
    stop_reason = handoff.get("stop_reason", "n/a")
    lines.append(
        f"{attempt_count} attempt(s) so far, {productive} productive, stopped: {stop_reason}."
    )

    infra = handoff.get("infrastructure_errors")
    if infra:
        lines.append(
            f"{infra['count']} attempt(s) failed from an infrastructure error, not a rejected "
            f"strategy -- no test was actually evaluated in those: {infra['first_error'][:140]}"
        )

    attempts = handoff.get("attempts", [])
    shown = [a for a in attempts if a.get("status") in ("PRODUCTIVE", "VALID_ZERO_KILL", "REJECTED")][-max_attempts_shown:]
    for a in shown:
        calls = _fmt_call_list(a.get("candidate_calls", []))
        lines.append(
            f"- attempt {a.get('attempt', '?')} [{a.get('status', '?')}, "
            f"+{a.get('new_kills', 0)} new kills]: {calls}"
        )

    surviving_count = handoff.get("surviving_count", 0)
    op_counts = handoff.get("surviving_operator_counts", {})
    op_str = ", ".join(f"{op}:{n}" for op, n in op_counts.items()) or "none"
    lines.append(f"{surviving_count} mutant(s) still surviving overall ({op_str}).")

    sample = handoff.get("survivor_sample", [])[:max_sample_shown]
    if sample:
        lines.append("Other still-live clusters not in this round's targets:")
        for s in sample:
            lines.append(
                f"  - {s['id']} [{s['operator']}] {s['change']} "
                f"(equiv={s.get('equivalence_status', 'UNKNOWN')})"
            )

    prev = handoff.get("previous_layer_summary")
    if prev:
        lines.append(
            f"Before this: {prev.get('source_layer', '?')} ran "
            f"{prev.get('attempt_count', 0)} attempt(s), "
            f"{prev.get('surviving_count_before_layer2', prev.get('surviving_count_before_layer3', '?'))} "
            f"survivor(s) handed forward."
        )

    return "\n".join(lines)


def _render_rules(rules: Sequence[str]) -> str:
    lines = [f"R{i + 1}. {text}" for i, text in enumerate(rules)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generic text/AST utilities (unchanged behavior; still usable elsewhere,
# e.g. for logging, storage budgets, or any future caller that does want a
# character cap).
# ---------------------------------------------------------------------------

def _bounded(text: str, max_chars: int, *, keep_tail: bool = False) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n...<input context compressed>...\n"
    available = max(0, max_chars - len(marker))
    if keep_tail:
        return marker + text[-available:]
    head = int(available * 0.68)
    tail = available - head
    return text[:head] + marker + text[-tail:]


class _DocstringStripper(ast.NodeTransformer):
    @staticmethod
    def _strip(body: list[ast.stmt]) -> list[ast.stmt]:
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return body[1:]
        return body

    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        node.body = self._strip(node.body)
        self.generic_visit(node)
        return node

    # type: ignore[override]
    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.body = self._strip(node.body) or [ast.Pass()]
        self.generic_visit(node)
        return node

    # type: ignore[override]
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        node.body = self._strip(node.body) or [ast.Pass()]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):  # type: ignore[override]
        node.body = self._strip(node.body) or [ast.Pass()]
        self.generic_visit(node)
        return node


def compact_source(source_code: str, entry_point: str, max_chars: int = 0) -> str:
    """Normalize source (strip docstrings) via AST round-trip.

    ``max_chars`` defaults to 0, meaning no budget is applied -- the full
    normalized source is always returned. A positive value is still honored
    for any caller outside the prompt builders that wants a cap.
    """
    try:
        tree = ast.parse(source_code)
        tree = _DocstringStripper().visit(tree)
        ast.fix_missing_locations(tree)
        text = ast.unparse(tree)
    except Exception:
        text = source_code

    if max_chars <= 0 or len(text) <= max_chars:
        return text

    try:
        tree = ast.parse(text)
        imports: list[ast.stmt] = []
        target: list[ast.stmt] = []
        helpers: list[ast.stmt] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
                target.append(node)
            else:
                helpers.append(node)
        pieces: list[str] = []
        for node in imports + helpers + target:
            rendered = ast.unparse(node)
            if pieces and sum(len(piece) + 2 for piece in pieces) + len(rendered) > max_chars:
                continue
            pieces.append(rendered)
        if pieces:
            return _bounded("\n\n".join(pieces), max_chars, keep_tail=True)
    except Exception:
        pass
    return _bounded(text, max_chars, keep_tail=True)


def mutation_diff(original_source: str, mutant: Mutant, max_chars: int = 0) -> str:
    """Full unified diff between canonical and mutant source.

    ``n=3`` context lines are used (widened from the previous ``n=1``) so a
    multi-statement mutation is not silently reduced to a single line of
    context. ``max_chars`` defaults to 0 (no truncation).
    """
    try:
        original = ast.unparse(ast.parse(original_source)).splitlines()
    except Exception:
        original = original_source.splitlines()
    try:
        changed = ast.unparse(ast.parse(mutant.mutated_source)).splitlines()
    except Exception:
        changed = (mutant.mutated_source or "").splitlines()
    diff = list(
        difflib.unified_diff(
            original,
            changed,
            fromfile="canonical",
            tofile=mutant.mutant_id,
            n=3,
            lineterm="",
        )
    )
    rendered = "\n".join(diff) if diff else mutant.description
    if max_chars <= 0:
        return rendered
    return _bounded(rendered, max_chars, keep_tail=True)


def _changed_statement_pair(original_source: str, mutant: Mutant) -> tuple[str, str]:
    """Return the full set of removed/added lines, not just the first pair.

    Multiple removed lines and multiple added lines are each joined so a
    multi-statement mutation is represented completely rather than being
    reduced to only its first changed line.
    """
    diff = mutation_diff(original_source, mutant).splitlines()
    removed = [line[1:].strip() for line in diff if line.startswith(
        "-") and not line.startswith("---")]
    added = [line[1:].strip() for line in diff if line.startswith("+")
             and not line.startswith("+++")]
    return ("\n".join(removed) if removed else "", "\n".join(added) if added else "")


def _probe_evidence(
    mutant: Mutant,
    probe_exprs: Sequence[str],
    probe_outcomes: Optional[Sequence[Sequence[str]]] = None,
) -> List[Dict[str, object]]:
    """Report per-probe evidence, but only when it says something.

    A mutant that is behaviorally silent on every existing probe (the common
    profile of a survivor reaching Layer 2/3) still needs *a* signal that
    probing was attempted and found nothing -- an empty list would be
    ambiguous with "never checked." But repeating that same "all probes say
    same" fact once per probe, per target, across a whole batch is pure
    duplication with no added information. So: if every probe outcome for
    this mutant is "same" (no divergence found), collapse to a single
    negative-evidence marker instead of one entry per probe.
    """
    labels = {0: "same", 1: "different", 2: "timeout/error"}
    states = list(mutant.behavior_signature or [])
    if states and all(labels.get(s, s) == "same" for s in states):
        return [{"note": f"no divergence across {len(states)} existing probes; derive a new distinguishing input from the diff instead"}]

    evidence: List[Dict[str, object]] = []
    for index, state in enumerate(states):
        item: Dict[str, object] = {
            "probe": probe_exprs[index] if index < len(probe_exprs) else f"probe_{index}",
            "mutant_outcome": labels.get(state, str(state)),
        }
        if probe_outcomes and index < len(probe_outcomes):
            outcome = list(probe_outcomes[index])
            if outcome:
                item["canonical_status"] = outcome[0]
            if len(outcome) > 1 and outcome[1]:
                item["canonical_value"] = str(outcome[1])
        evidence.append(item)
    return evidence


def _target_dossier(
    mutant: Mutant,
    original_source: str,
    probe_exprs: Sequence[str],
    cluster_context: Optional[Dict],
    probe_outcomes: Optional[Sequence[Sequence[str]]] = None,
    *,
    compact: bool = False,
) -> Dict[str, object]:
    """Build the per-target payload actually sent to the LLM.

    Deliberately excludes internal pipeline bookkeeping the model does not
    need to write a correct test -- cluster sizes, centrality, and
    information_score drive *our* representative-selection math, not the
    model's reasoning about a diff. Those remain available in logged
    artifacts (see mutation_engine's own JSON dumps) for our own analysis;
    they are cut here purely to reduce prompt size without reducing what the
    model can actually use. `source_location` is also dropped: `exact_diff`
    already shows the location in context.

    ``compact=True`` (Layer 2/Layer 3 only -- both API-metered, cost-
    sensitive stages) sends `exact_diff` alone instead of the separate
    `original_code`/`mutated_code` fields: a unified diff already shows both
    the removed and added statements in one place against the full source
    already given in `<canonical_implementation>`, so carrying both
    representations is pure duplication for a paid call. It also drops an
    empty `equivalence_reason` rather than sending an empty string. Layer 1
    (local, free) keeps `original_code`/`mutated_code` instead since the
    redundancy costs nothing there and its weaker model benefits more from
    the extra explicit framing.
    """
    payload: Dict[str, object] = {
        "representative_id": mutant.mutant_id,
        "operator": mutant.operator,
        "change_description": mutant.description,
    }
    if compact:
        payload["exact_diff"] = mutation_diff(original_source, mutant)
    else:
        original_code, mutated_code = _changed_statement_pair(
            original_source, mutant)
        payload["original_code"] = original_code
        payload["mutated_code"] = mutated_code
    payload["probe_evidence"] = _probe_evidence(mutant, probe_exprs, probe_outcomes)
    payload["equivalence_status"] = getattr(mutant, "equivalence_status", "UNKNOWN")
    reason = getattr(mutant, "equivalence_reason", "")
    if reason or not compact:
        payload["equivalence_reason"] = reason
    return payload


def _dossiers(
    targets: Sequence[Mutant],
    original_source: str,
    probe_exprs: Sequence[str],
    cluster_contexts: Dict[int, Dict],
    probe_outcomes: Optional[Sequence[Sequence[str]]] = None,
    *,
    compact: bool = False,
) -> str:
    payload = [
        _target_dossier(
            target,
            original_source,
            probe_exprs,
            cluster_contexts.get(int(target.cluster_id)
                                 ) if target.cluster_id is not None else None,
            probe_outcomes,
            compact=compact,
        )
        for target in targets
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def compact_test_digest(previous_tests: Sequence[str], max_tests: int = 0, max_chars: int = 0) -> str:
    """Structured summary of previously accepted tests.

    ``max_tests`` <= 0 means include every accepted test, not just the most
    recent N. ``max_chars`` <= 0 means no character budget is applied.
    """
    tests = list(previous_tests) if max_tests <= 0 else list(
        previous_tests)[-max_tests:]
    records: List[Dict[str, object]] = []
    for test in tests:
        assertions: List[str] = []
        calls: List[str] = []
        try:
            tree = ast.parse(test)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    assertions.append(ast.unparse(node.test))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "candidate":
                    calls.append(ast.unparse(node))
        except Exception:
            pass
        records.append({
            "sha1": hashlib.sha1(test.encode("utf-8")).hexdigest()[:10],
            "candidate_calls": calls,
            "assertions": assertions,
        })
    rendered = json.dumps(records, indent=2)
    if max_chars <= 0:
        return rendered
    return _bounded(rendered, max_chars)


def build_handoff_summary(
    *,
    layer: str,
    attempts: Sequence[Dict],
    surviving_mutants: Sequence[Mutant],
    previous_tests: Sequence[str],
    target_mutant_ids: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Aggregate handoff passed between layers.

    Two deliberate compressions versus a naive "pass everything through":

    1. Attempt history is run through ``summarize_attempt_history`` so that
       infrastructure failures (API errors, OOM, timeouts) are reported as
       what they are -- "nothing was actually tried" -- rather than framed
       as rejected test-writing hypotheses for the next layer to avoid
       repeating. Repeated identical infra errors are collapsed to one
       annotated entry instead of N verbatim stack traces.
    2. ``survivor_sample`` includes at most one mutant per surviving
       *cluster* (its most central member) rather than every survivor. The
       ``targets`` list already carries one representative per cluster this
       round; survivor_sample's job is only to show the next layer which
       *other* clusters/behaviors still exist, so within-cluster duplicates
       (which, before probe-domain fixes, were often literally identical)
       add token cost with no decision value. Pass ``target_mutant_ids`` so
       clusters already fully represented in this round's targets are
       excluded from the sample entirely, avoiding cross-section duplication
       within the same prompt.
    """
    from src.layers.common import summarize_attempt_history  # local import avoids a cycle

    handoff = summarize_attempt_history(attempts, source_layer=layer)

    operator_counts = Counter(mutant.operator for mutant in surviving_mutants)
    target_ids = set(target_mutant_ids or [])

    by_cluster: Dict[object, List[Mutant]] = {}
    for mutant in surviving_mutants:
        by_cluster.setdefault(mutant.cluster_id, []).append(mutant)

    sample: List[Mutant] = []
    for cluster_id, members in by_cluster.items():
        if any(m.mutant_id in target_ids for m in members):
            continue  # this cluster is already shown in <targets> this round
        best = max(members, key=lambda m: float(m.centrality or 0.0))
        sample.append(best)
    sample.sort(key=lambda m: float(m.centrality or 0.0), reverse=True)

    # behavior_signature is only worth showing when it actually varies across
    # the sampled clusters -- a constant/zero signature everywhere (as
    # happens when probes don't distinguish anything for this task) carries
    # no information and is dropped rather than repeated once per entry.
    signatures = [tuple(m.behavior_signature or []) for m in sample]
    signature_is_informative = len(set(signatures)) > 1

    survivor_sample = []
    for mutant in sample:
        entry = {
            "id": mutant.mutant_id,
            "operator": mutant.operator,
            "change": mutant.description,
            "equivalence_status": getattr(mutant, "equivalence_status", "UNKNOWN"),
        }
        if signature_is_informative:
            entry["behavior_signature"] = mutant.behavior_signature
        survivor_sample.append(entry)

    handoff.update({
        "surviving_count": len(surviving_mutants),
        "surviving_original_cluster_ids": sorted({m.cluster_id for m in surviving_mutants if m.cluster_id is not None}),
        "surviving_operator_counts": dict(sorted(operator_counts.items())),
        "survivor_sample": survivor_sample,
        "accepted_test_digest": json.loads(compact_test_digest(previous_tests) or "[]"),
        "cluster_assignment_reused": True,
    })
    return handoff


def _task_header(*, prompt_text: str, entry_point: str, task_metadata: Optional[Dict]) -> str:
    metadata = task_metadata or {}
    libraries = metadata.get("libraries") or metadata.get("libs") or []
    return (
        f"Problem ID: {metadata.get('task_id', metadata.get('source_task_id', 'unknown'))}\n"
        f"Dataset: {metadata.get('dataset_name', 'unknown')}\n"
        f"Dataset subset: {metadata.get('dataset_subset', '') or 'not specified'}\n"
        f"Entry point: {entry_point}\n"
        f"Available/required libraries: {', '.join(map(str, libraries)) if libraries else 'standard environment'}\n\n"
        f"Natural-language specification:\n{prompt_text}"
    )


def _probe_oracles(
    probe_exprs: Sequence[str],
    probe_outcomes: Optional[Sequence[Sequence[str]]],
) -> str:
    """Every probe/oracle pair, not just the first N."""
    rows: List[Dict[str, object]] = []
    for index, expr in enumerate(probe_exprs):
        row: Dict[str, object] = {"probe": expr}
        if probe_outcomes and index < len(probe_outcomes):
            outcome = list(probe_outcomes[index])
            row["canonical_status"] = outcome[0] if outcome else "UNKNOWN"
            if len(outcome) > 1 and outcome[1]:
                row["canonical_value"] = str(outcome[1])
        rows.append(row)
    return json.dumps(rows, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Layer 1: local lightweight generation
# ---------------------------------------------------------------------------

_LAYER1_SCORED_RULES: List[str] = [
    "[SCORE 100 -- NEVER VIOLATE, checked mechanically before anything else] "
    "If <seed_test> is non-empty, your output MUST begin with every line of "
    "<seed_test> reproduced verbatim, character-for-character, in the same "
    "order. Do not delete, reorder, reword, renumber, or 're-derive' a seed "
    "line -- not even if you believe a different value is more correct. The "
    "seed lines are already proven correct against the canonical "
    "implementation; you are not able to verify that independently, so any "
    "seed line you change is a wrong change by definition.",
    "[SCORE 100 -- NEVER VIOLATE] Only ADD new `assert candidate(...) == ...` "
    "statements after the reproduced seed lines. Never edit, wrap, or remove "
    "a seed line to make room for a new one -- appending is the only "
    "permitted operation on the seed.",
    "[SCORE 90] Every new `assert` you add must depend on the actual return "
    "value of a `candidate(...)` call you make inside this same function. An "
    "assertion that would hold no matter what `candidate` returns (e.g. "
    "`assert isinstance(result, bool)`, `assert 1 == 1`) is a hard failure "
    "even if it never raises.",
    "[SCORE 90] Never guess or reconstruct an expected value from memory. "
    "Every expected value in a new assertion must be either copied "
    "character-for-character from a `<canonical_probe_oracles>` entry, or be "
    "a value you can derive with certainty from the specification's stated "
    "semantics. If you cannot ground a value either way, do not write that "
    "assertion -- pick a different input you can ground instead.",
    "[SCORE 80] For each target whose `probe_evidence` shows no divergence "
    "yet, derive one new input by comparing `original_code` and "
    "`mutated_code` in its dossier: pick the smallest concrete input where "
    "the two expressions evaluate differently. Only include the target if "
    "you can also ground the canonical expected output for that input per "
    "the rule above.",
    "[SCORE 70] Prefer several compatible targets in one function, but only "
    "when you are equally confident about every one you include. Do not add "
    "a low-confidence target just to raise coverage -- one wrong assumption "
    "invalidates every assertion after the seed.",
    "[SCORE 60] Do not repeat an input/assertion combination already marked "
    "as attempted and failed in <attempt_context>. Correct the specific "
    "failure instead of reproducing it.",
]

_LAYER1_OUTPUT_FORMAT_RULE = (
    "[SCORE 100 -- NEVER VIOLATE] Output only the raw Python function body: "
    "the reproduced <seed_test> lines followed by your new lines, nothing "
    "else. No Markdown fences, no prose, no comments explaining your "
    "reasoning, no `print`, no trailing `return`, no truncation."
)


def _layer1_seed_block(seed_test: str) -> str:
    seed_test = (seed_test or "").strip()
    if not seed_test:
        return (
            "<seed_test>\n"
            "(none available -- no canonical probe produced a usable VALUE "
            "for this problem.) Write the complete function yourself, "
            "following every rule below exactly as if it were rule 1 and 2.\n"
            "</seed_test>\n"
        )
    return f"""<seed_test>
This is a guaranteed-correct partial test, built mechanically from already-
verified <canonical_probe_oracles> entries -- no model produced these lines,
so they are not in question. You MUST reproduce every line below verbatim as
the start of your output, then append your own new assertions after it. Do
not modify anything inside this block.

```python
{seed_test}
```
</seed_test>
"""


def build_layer1_batch_prompt(
    *,
    targets: Sequence[Mutant],
    source_code: str,
    entry_point: str,
    prompt_text: str,
    cluster_contexts: Dict[int, Dict],
    probe_exprs: Sequence[str],
    probe_outcomes: Optional[Sequence[Sequence[str]]] = None,
    attempt: int,
    feedback: Optional[Dict] = None,
    task_metadata: Optional[Dict] = None,
    seed_test: str = "",
) -> str:
    task_header = _task_header(
        prompt_text=prompt_text, entry_point=entry_point, task_metadata=task_metadata)
    full_source = compact_source(source_code, entry_point)
    dossiers = _dossiers(targets, source_code, probe_exprs,
                         cluster_contexts, probe_outcomes)
    probe_oracle_json = _probe_oracles(probe_exprs, probe_outcomes)
    feedback_json = json.dumps(
        feedback or {"status": "INITIAL_ATTEMPT"}, indent=2)
    seed_block = _layer1_seed_block(seed_test)

    rules = _render_rules(
        _LAYER1_SCORED_RULES + [_LAYER1_OUTPUT_FORMAT_RULE])

    return f"""<role>
You are Layer 1, the first-stage mutation-guided unit-test generator in
Claus-Test. You analyze all supplied mutant representatives together and
return exactly one Python test function.
</role>

<objective>
Extend the guaranteed-correct <seed_test> below with new assertions that:
1. Still pass on the canonical implementation (the seed lines already do; any
   new line you add must too).
2. Kill as many of the currently surviving mutant representatives below as
   you can, using real evidence rather than assumption.
A response that reproduces the seed but adds nothing useful is accepted but
wastes this attempt; a response that alters the seed or invents an ungrounded
value is rejected outright regardless of anything else it gets right. Rules
below are tagged with a priority score -- rules scored 100 are mechanically
checked and override every other consideration.
</objective>

<task_information>
{task_header}
</task_information>

<canonical_implementation>
```python
{full_source}
```
</canonical_implementation>

<canonical_probe_oracles>
These are trusted, verified outputs of the canonical implementation. Treat
every value here as ground truth. Do not contradict them.

{probe_oracle_json}
</canonical_probe_oracles>

{seed_block}

<attempt_context>
Attempt number: {attempt}
Previous-attempt feedback:
{feedback_json}
</attempt_context>

<targets>
The initial cluster partition is fixed and reused across every layer. Each
record below is either the original cluster representative or the most
informative surviving member of that same cluster.

{dossiers}
</targets>

<rules>
{rules}
</rules>

<reasoning_steps>
Work through these silently before writing any code. Do not show this
reasoning in your output -- only the final function.
1. Copy <seed_test> verbatim as your starting point (or start fresh only if
   it says "none available").
2. For each target, read `original_code` vs `mutated_code` and state to
   yourself, precisely, what single input property would make the two diverge.
3. Check whether a `<canonical_probe_oracles>` entry already covers an input
   with that property. If yes, reuse its value as ground truth, copied
   verbatim.
4. If no oracle covers it, derive the expected canonical output only from the
   specification's stated behavior -- never invent one.
5. Decide, per target, whether you are confident enough to include it. Drop
   targets you are not confident about rather than guessing.
6. Draft each new assertion and check it individually: does its truth value
   depend on candidate's actual return? If not, discard or rewrite it.
7. Re-read the full function once against every rule in <rules>, seed lines
   first and unmodified, before finalizing.
</reasoning_steps>

<good_example>
# <seed_test> contained:
#   assert candidate([]) == 'Shangri-La not found'
# Target: mutated a boundary check from `n < 0` to `n <= 0`.
# Spec says the function should treat 0 as valid, non-negative input.
def check(candidate):
    assert candidate([]) == 'Shangri-La not found'  # seed line, reproduced verbatim and unmodified
    assert candidate(0) == expected_result_for_zero  # new line: value taken from a probe oracle entry or the spec, never invented
    assert candidate(-1) == expected_result_for_negative_input  # new line, same standard
</good_example>
<why_good>
The seed line is untouched. Every new assertion depends on candidate's
actual return value, 0 is exactly the boundary the mutation altered, and
every literal was either copied from the seed or grounded the same way the
seed's was -- never reconstructed from memory. Retyping a value from memory
instead of copying it is how a technically-reasonable test still gets
rejected: even one wrong character fails the whole function against the
canonical implementation.
</why_good>

<bad_example>
def check(candidate):
    assert candidate([]) == 'nothing found'  # seed line REWRITTEN -- forbidden even if this looks more natural
    result = candidate(5)
    assert isinstance(result, int)  # true regardless of candidate's logic -- vacuous
    assert 1 == 1  # does not depend on candidate at all -- vacuous
</bad_example>
<why_bad>
The first line silently altered a seed assertion -- a SCORE 100 violation on
its own, rejected regardless of anything else in the function. The last two
assertions are vacuous: neither can ever fail against a mutant that returns
any integer at all. This test would be accepted by a harness that only
checks for uncaught exceptions, but it exercises nothing and breaks the one
guarantee the seed was providing.
</why_bad>

{_OUTPUT_CONTRACT_BLOCK}"""


# ---------------------------------------------------------------------------
# Layer 2: API-based refinement
# ---------------------------------------------------------------------------

def build_layer2_batch_prompt(
    *,
    targets: Sequence[Mutant],
    source_code: str,
    entry_point: str,
    prompt_text: str,
    cluster_contexts: Dict[int, Dict],
    probe_exprs: Sequence[str],
    probe_outcomes: Optional[Sequence[Sequence[str]]] = None,
    previous_tests: Sequence[str],
    handoff: Optional[Dict],
    attempt: int,
    feedback: Optional[Dict] = None,
    task_metadata: Optional[Dict] = None,
) -> str:
    task_header = _task_header(
        prompt_text=prompt_text, entry_point=entry_point, task_metadata=task_metadata)
    full_source = compact_source(source_code, entry_point)
    dossiers = _dossiers(targets, source_code, probe_exprs,
                         cluster_contexts, probe_outcomes, compact=True)
    handoff_text = _compact_handoff_block(handoff)
    accepted_digest_json = compact_test_digest(previous_tests)
    feedback_json = json.dumps(
        feedback or {"status": "INITIAL_REFINEMENT_ATTEMPT"}, indent=2)

    rules = _render_rules(_COMPACT_CORE_RULES + [
        "Check <layer1_handoff> first; your input must differ from what already failed there.",
        _OUTPUT_FORMAT_RULE,
    ])

    return f"""<role>
Layer 2 (API refinement) in Claus-Test. Layer 1 already ran and failed to
kill the targets below -- same fixed initial clusters, live members only.
Return exactly one additional Python test function.
</role>

<objective>
Kill as many surviving targets below as you can, without repeating a
strategy <layer1_handoff> shows already failed. Use the handoff, the
accepted-test digest, and each target's `exact_diff`/`probe_evidence`
together -- ground truth values live per target, not in a separate block.
</objective>

<task_information>
{task_header}
</task_information>

<canonical_implementation>
```python
{full_source}
```
</canonical_implementation>

<layer1_handoff>
{handoff_text}
</layer1_handoff>

<accepted_test_digest>
Already accepted -- do not reproduce these exact calls/assertions.
{accepted_digest_json}
</accepted_test_digest>

<attempt_context>
Refinement attempt {attempt}. Feedback: {feedback_json}
</attempt_context>

<targets>
Fixed initial partition, reused unmodified. A killed representative's
dossier below is its cluster's next-most-informative live member instead.

{dossiers}
</targets>

<rules>
{rules}
</rules>

{_COMPACT_REASONING_STEPS}
{_OUTPUT_CONTRACT_BLOCK}"""


# ---------------------------------------------------------------------------
# Layer 3: final escalation
# ---------------------------------------------------------------------------

def build_layer3_batch_prompt(
    *,
    targets: Sequence[Mutant],
    source_code: str,
    entry_point: str,
    prompt_text: str,
    cluster_contexts: Dict[int, Dict],
    probe_exprs: Sequence[str],
    probe_outcomes: Optional[Sequence[Sequence[str]]] = None,
    previous_tests: Sequence[str],
    handoff: Optional[Dict],
    task_metadata: Optional[Dict] = None,
) -> str:
    task_header = _task_header(
        prompt_text=prompt_text, entry_point=entry_point, task_metadata=task_metadata)
    full_source = compact_source(source_code, entry_point)
    dossiers = _dossiers(targets, source_code, probe_exprs,
                         cluster_contexts, probe_outcomes, compact=True)
    handoff_text = _compact_handoff_block(handoff)
    accepted_digest_json = compact_test_digest(previous_tests)

    rules = _render_rules(_COMPACT_CORE_RULES + [
        "Check <pipeline_handoff>; your input must differ from every prior attempt shown there.",
        "Weigh `equivalence_status`/`equivalence_reason` against `exact_diff`; "
        "omit a target rather than guess.",
        "Accept non-value kill signals too: an exception, a timeout, or "
        "divergence under a seeded `random.seed` or a mock.",
        _OUTPUT_FORMAT_RULE,
    ])

    return f"""<role>
Layer 3 (final escalation) in Claus-Test -- most capable, most expensive
stage, reserved for mutants two cheaper layers could not resolve. Return
exactly one Python test function.
</role>

<objective>
For each target: either find a concrete input where canonical and mutant
provably diverge and kill it, or decide it is very likely equivalent and
omit it. A function killing fewer targets correctly beats one with a single
guessed assertion -- a wrong guess forfeits the whole function's credit.
</objective>

<task_information>
{task_header}
</task_information>

<canonical_implementation>
```python
{full_source}
```
</canonical_implementation>

<pipeline_handoff>
{handoff_text}
</pipeline_handoff>

<accepted_test_digest>
Already accepted -- do not reproduce these exact calls/assertions.
{accepted_digest_json}
</accepted_test_digest>

<targets>
Hardest survivors -- two cheaper layers already failed. `equivalence_status`
is a strong prior, not a guarantee: confirm or challenge it against the diff
and probe evidence rather than accepting or dismissing it blindly.

{dossiers}
</targets>

<rules>
{rules}
</rules>

{_COMPACT_REASONING_STEPS}
{_OUTPUT_CONTRACT_BLOCK}"""


# ---------------------------------------------------------------------------
# Backward-compatible single-target wrappers.
# ---------------------------------------------------------------------------

def build_layer1_prompt(mutant: Mutant, source_code: str, entry_point: str,
                        prompt_text: str = "", cluster_context: Optional[Dict] = None) -> str:
    contexts = {int(mutant.cluster_id or 0): cluster_context or {}}
    return build_layer1_batch_prompt(
        targets=[mutant], source_code=source_code, entry_point=entry_point,
        prompt_text=prompt_text, cluster_contexts=contexts, probe_exprs=[], attempt=1,
    )


def build_layer2_prompt(mutant: Mutant, source_code: str, entry_point: str,
                        prompt_text: str, previous_tests: List[str], failure_context: Dict,
                        cluster_context: Optional[Dict] = None) -> str:
    contexts = {int(mutant.cluster_id or 0): cluster_context or {}}
    return build_layer2_batch_prompt(
        targets=[mutant], source_code=source_code, entry_point=entry_point,
        prompt_text=prompt_text, cluster_contexts=contexts, probe_exprs=[],
        previous_tests=previous_tests, handoff=failure_context, attempt=1,
    )


def build_layer3_prompt(mutant: Mutant, source_code: str, entry_point: str,
                        prompt_text: str, previous_tests: List[str], failure_context: Dict,
                        cluster_context: Optional[Dict] = None) -> str:
    contexts = {int(mutant.cluster_id or 0): cluster_context or {}}
    return build_layer3_batch_prompt(
        targets=[mutant], source_code=source_code, entry_point=entry_point,
        prompt_text=prompt_text, cluster_contexts=contexts, probe_exprs=[],
        previous_tests=previous_tests, handoff=failure_context,
    )


def _mutant_full_block(
    mutant: Mutant,
    original_source: str,
    probe_exprs: Sequence[str],
    probe_outcomes: Optional[Sequence[Sequence[str]]] = None,
) -> str:
    """Per-mutant payload for the no-clustering iterative baseline.

    Deliberately omits `original_cluster_id`: the baseline's whole premise
    (stated in its own prompt) is that every mutant is considered
    independently, with no clustering or representative compression --
    including cluster metadata here would be a conceptual leak of the
    proposed pipeline's structure into what is supposed to be a
    clustering-free control condition.

    Same shape as the proposed pipeline's compact Layer 2/3 dossier now:
    `exact_diff` alone (not also `original_code`/`mutated_code`) plus
    `probe_evidence` per mutant. Keeping the per-target field set identical
    across baseline and proposed pipeline is what makes the comparison
    isolate clustering as the actual variable under test, rather than one
    arm also carrying an incidentally larger or smaller per-target payload.
    """
    payload = {
        "mutant_id": mutant.mutant_id,
        "operator": mutant.operator,
        "change_description": mutant.description,
        "exact_diff": mutation_diff(original_source, mutant),
        "probe_evidence": _probe_evidence(mutant, probe_exprs, probe_outcomes),
        "equivalence_status": getattr(mutant, "equivalence_status", "UNKNOWN"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Iterative full-mutant baseline (no clustering/representative compression)
# ---------------------------------------------------------------------------

def build_iterative_baseline_prompt(
    source_code: str,
    entry_point: str,
    prompt_text: str,
    surviving_mutants: List[Mutant],
    previous_tests: List[str],
    iteration: int,
    max_iterations: int,
    feedback: Optional[Dict] = None,
    probe_exprs: Optional[Sequence[str]] = None,
    probe_outcomes: Optional[Sequence[Sequence[str]]] = None,
) -> str:
    full_source = compact_source(source_code, entry_point)
    probe_exprs = list(probe_exprs or [])

    # Dedupe literally-identical diffs (e.g. the same operator applied to
    # equivalent AST copies) before rendering. This does not cluster by
    # similarity -- it only removes exact duplicate blocks that add zero
    # marginal signal regardless of whether clustering exists at all, so it
    # doesn't inject any of the proposed pipeline's semantic machinery into
    # this no-clustering baseline.
    seen_diffs: set[str] = set()
    deduped: List[Mutant] = []
    for mutant in surviving_mutants:
        diff = mutation_diff(source_code, mutant)
        if diff in seen_diffs:
            continue
        seen_diffs.add(diff)
        deduped.append(mutant)

    mutant_blocks = "\n\n".join(_mutant_full_block(
        mutant, source_code, probe_exprs, probe_outcomes) for mutant in deduped)
    accepted_digest_json = compact_test_digest(previous_tests)
    feedback_json = json.dumps(
        feedback or {"status": "INITIAL_ITERATION"}, indent=2)

    rules = _render_rules(_COMPACT_CORE_RULES + [
        "Weigh every surviving mutant below -- no clustering, all are independent.",
        "Prefer one input that kills several mutants over a single-mutant kill, without guessing.",
        "Don't repeat a strategy already marked failed in <iteration_context>.",
        _OUTPUT_FORMAT_RULE,
    ])

    return f"""<role>
You are the iterative full-mutant baseline generator. No clustering or
representative compression is used -- every currently surviving mutant is
presented to you independently. You return exactly one additional Python
test function.
</role>

<objective>
Write a test that:
1. Passes on the canonical implementation.
2. Kills as many of the currently surviving mutants below as you can.
3. Does not duplicate an accepted test or repeat a strategy already marked
   as failed in a previous iteration.
</objective>

<task_information>
Entry point: {entry_point}

Natural-language specification:
{prompt_text}
</task_information>

<canonical_implementation>
```python
{full_source}
```
</canonical_implementation>

<iteration_context>
Baseline iteration: {iteration} of {max_iterations}.
Previous-iteration feedback:
{feedback_json}
</iteration_context>

<accepted_test_digest>
Tests already accepted into the suite. Do not reproduce these exact
candidate-call / assertion combinations.

{accepted_digest_json}
</accepted_test_digest>

<surviving_mutants count="{len(deduped)}">
Every survivor is listed independently below -- there is no cluster
representative compression at this stage. (Mutants with a diff identical to
one already shown have been deduplicated; this removes exact duplicates
only, not similar-but-distinct mutants.)

{mutant_blocks}
</surviving_mutants>

<rules>
{rules}
</rules>

{_COMPACT_REASONING_STEPS}
<good_example>
# Two surviving mutants both change how a boundary value of 0 is handled in
# unrelated branches; the specification states 0 is a valid, defined input.
def check(candidate):
    assert candidate(0) == expected_result_for_zero  # value from the specification
</good_example>
<why_good>
One well-chosen input kills two independent mutants at once, and the
expected value is taken directly from the specification rather than guessed.
</why_good>

<bad_example>
def check(candidate):
    result = candidate(0)
    assert result == result  # always true regardless of candidate -- vacuous
</bad_example>
<why_bad>
This assertion can never fail for any implementation, correct or mutated. It
would be accepted by a harness that only checks for uncaught exceptions but
kills nothing.
</why_bad>

{_OUTPUT_CONTRACT_BLOCK}"""