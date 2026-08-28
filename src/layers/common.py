"""Shared helpers for Claus-Test generation layers."""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.clustering.representative_selector import RepresentativeStack
from src.mutation.mutation_engine import Mutant, run_suite_against_mutants, verify_no_false_positives


def extract_check_function(text: str) -> str:
    """Extract exactly the first top-level ``check(candidate)`` function.

    Models occasionally add a code fence, a preamble, or trailing prose.  The
    previous extractor returned everything after ``def check`` and could turn a
    valid function into a syntax error because of trailing explanation.  This
    version uses Python AST line information and progressively trims invalid
    trailing content when necessary.
    """
    cleaned = re.sub(r"```(?:python)?\s*", "", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned).strip()
    match = re.search(r"(?m)^\s*def\s+check\s*\(\s*candidate\s*\)\s*:", cleaned)
    if not match:
        return ""
    candidate_text = cleaned[match.start():].lstrip()

    def _extract(parseable: str) -> str:
        try:
            tree = ast.parse(parseable)
        except SyntaxError:
            return ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check":
                positional = list(node.args.posonlyargs) + list(node.args.args)
                if len(positional) != 1 or positional[0].arg != "candidate":
                    return ""
                lines = parseable.splitlines()
                end = getattr(node, "end_lineno", None) or len(lines)
                return "\n".join(lines[node.lineno - 1:end]).strip()
        return ""

    exact = _extract(candidate_text)
    if exact:
        return exact

    lines = candidate_text.splitlines()
    for end in range(len(lines) - 1, 0, -1):
        exact = _extract("\n".join(lines[:end]))
        if exact:
            return exact
    return ""


def summarize_generated_test(test_code: str) -> Dict[str, object]:
    """Return deterministic response details for repair prompts and logs."""
    summary: Dict[str, object] = {"candidate_calls": [], "assertions": []}
    if not test_code.strip():
        return summary
    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        summary["syntax_error"] = str(exc)
        return summary
    calls: List[str] = []
    assertions: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "candidate":
            calls.append(ast.unparse(node))
        if isinstance(node, ast.Assert):
            assertions.append(ast.unparse(node.test))
    summary["candidate_calls"] = calls[:12]
    summary["assertions"] = assertions[:12]
    return summary


def validation_failure_category(error: str, test_code: str = "") -> str:
    text = (error or "").lower()
    if not test_code.strip() or "no check(candidate)" in text:
        return "NO_CHECK_FUNCTION_EXTRACTED"
    if "syntax error" in text:
        return "SYNTAX_ERROR"
    if "never calls candidate" in text:
        return "NO_CANDIDATE_CALL"
    if "contains no assert" in text:
        return "NO_ASSERTION"
    if "unproductive" in text or "zero active mutants" in text:
        return "VALID_ZERO_KILL"
    if "timeout" in text:
        return "CANONICAL_TIMEOUT"
    if "forbidden" in text or "introspection" in text:
        return "FORBIDDEN_INTROSPECTION"
    if error:
        return "CANONICAL_EXECUTION_FAILURE"
    return "NONE"


def validate_check_contract(test_code: str) -> Tuple[bool, str, Dict[str, int]]:
    """Reject malformed or vacuous generated tests before subprocess execution.

    Passing on the canonical implementation is not sufficient: accepted tests must
    actually call ``candidate`` and contain an assertion.  This closes the
    validated-but-vacuous failure mode identified in the paper draft.
    """
    if not test_code.strip():
        return False, "no check(candidate) function extracted", {"candidate_calls": 0, "asserts": 0}
    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}", {"candidate_calls": 0, "asserts": 0}

    checks = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check"]
    if len(checks) != 1:
        return False, f"expected exactly one top-level check function, found {len(checks)}", {"candidate_calls": 0, "asserts": 0}
    check = checks[0]
    positional = list(check.args.posonlyargs) + list(check.args.args)
    if len(positional) != 1 or positional[0].arg != "candidate" or check.args.vararg or check.args.kwarg:
        return False, "check must have exactly one positional parameter named candidate", {"candidate_calls": 0, "asserts": 0}

    candidate_calls = 0
    asserts = 0
    forbidden_introspection = 0
    for node in ast.walk(check):
        if isinstance(node, ast.Assert):
            asserts += 1
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "candidate":
                candidate_calls += 1
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile"}:
                forbidden_introspection += 1
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "candidate":
            if node.attr in {"__code__", "__dict__", "__globals__", "__module__"}:
                forbidden_introspection += 1

    details = {
        "candidate_calls": candidate_calls,
        "asserts": asserts,
        "forbidden_introspection": forbidden_introspection,
    }
    if candidate_calls == 0:
        return False, "vacuous test: check never calls candidate", details
    if asserts == 0:
        return False, "vacuous test: check contains no assert", details
    if forbidden_introspection:
        return False, "test uses forbidden evaluation or candidate introspection", details
    return True, "", details


def mutant_copies(mutants: Sequence[Mutant]) -> List[Mutant]:
    return [Mutant.from_dict(m.to_dict()) for m in mutants]


def flatten_stacks(stacks: Sequence[RepresentativeStack]) -> List[Mutant]:
    return [rep for stack in stacks for rep in stack.representatives]


def choose_dynamic_targets(stacks: Sequence[RepresentativeStack], active_mutants: Sequence[Mutant]) -> List[Mutant]:
    """Choose one live target per cluster without another clustering/model call."""
    active_by_id = {m.mutant_id: m for m in active_mutants}
    active_by_cluster: Dict[int, List[Mutant]] = defaultdict(list)
    for mutant in active_mutants:
        if mutant.cluster_id is not None:
            active_by_cluster[int(mutant.cluster_id)].append(mutant)

    targets: List[Mutant] = []
    for stack in stacks:
        live_rep = next((active_by_id.get(rep.mutant_id) for rep in stack.representatives if rep.mutant_id in active_by_id), None)
        if live_rep is not None:
            targets.append(live_rep)
            continue
        candidates = active_by_cluster.get(int(stack.cluster_id), [])
        if not candidates:
            continue
        targets.append(
            max(
                candidates,
                key=lambda m: (
                    float(m.information_score or 0.0),
                    float(m.centrality or 0.0),
                    sum(1 for value in (m.behavior_signature or []) if value != 0),
                ),
            )
        )
    return targets


def cluster_context_from_stack(stack: RepresentativeStack, active_mutants: Optional[Sequence[Mutant]] = None) -> Dict:
    all_members = list(stack.representatives) + list(stack.non_representatives)
    active_ids = {m.mutant_id for m in (active_mutants or all_members)}
    active_members = [m for m in all_members if m.mutant_id in active_ids]
    return {
        "cluster_id": stack.cluster_id,
        "strategy": stack.strategy,
        "cluster_size": len(all_members),
        "surviving_cluster_size": len(active_members),
        "representative_ids": [m.mutant_id for m in stack.representatives],
        "operator_counts": dict(sorted(Counter(m.operator for m in active_members).items())),
        "behaviorally_distinct_count": len({tuple(m.behavior_signature or []) for m in active_members}),
    }


def build_cluster_contexts(
    stacks: Sequence[RepresentativeStack],
    active_mutants: Optional[Sequence[Mutant]] = None,
) -> Dict[int, Dict]:
    return {int(stack.cluster_id): cluster_context_from_stack(stack, active_mutants) for stack in stacks}


def compact_mutant_summary(mutant: Mutant) -> Dict:
    return {
        "mutant_id": mutant.mutant_id,
        "cluster_id": mutant.cluster_id,
        "operator": mutant.operator,
        "line": mutant.line_number,
        "change": mutant.description,
        "behavior_signature": mutant.behavior_signature,
        "centrality": round(float(mutant.centrality or 0.0), 4),
        "information_score": round(float(mutant.information_score or 0.0), 4),
        "status": mutant.status,
        "equivalence_status": getattr(mutant, "equivalence_status", "UNKNOWN"),
    }


def build_failure_context(
    mutant: Mutant,
    evaluated_mutants: Sequence[Mutant],
    prior_test_count: int,
    survivor_summary_limit: int = 10,
) -> Dict:
    same_cluster = [m for m in evaluated_mutants if m.cluster_id == mutant.cluster_id]
    survived = [m for m in same_cluster if not m.is_killed]
    ranked = sorted(
        survived,
        key=lambda m: (float(m.information_score or 0.0), float(m.centrality or 0.0)),
        reverse=True,
    )[: max(0, survivor_summary_limit)]
    return {
        "target_mutant_id": mutant.mutant_id,
        "cluster_id": mutant.cluster_id,
        "prior_test_count": prior_test_count,
        "cluster_size": len(same_cluster),
        "cluster_survived_count": len(survived),
        "cluster_survived_operator_counts": dict(sorted(Counter(m.operator for m in survived).items())),
        "surviving_mutant_summaries": [compact_mutant_summary(m) for m in ranked],
    }


def _oracle_value_map(
    probe_exprs: Sequence[str],
    probe_outcomes: Optional[Sequence[Sequence[str]]],
) -> Dict[str, ast.AST]:
    """Map ``repr(call_args_tuple) -> AST literal node`` for every probe whose
    canonical outcome was a clean, verified ``VALUE`` (never an error or
    timeout, which cannot be encoded as an ``==`` literal)."""
    mapping: Dict[str, ast.AST] = {}
    if not probe_exprs or not probe_outcomes:
        return mapping
    for expr, outcome in zip(probe_exprs, probe_outcomes):
        outcome = list(outcome or [])
        if len(outcome) < 2 or outcome[0] != "VALUE" or not outcome[1]:
            continue
        match = re.match(r"^\s*candidate\((.*)\)\s*$", expr.strip(), re.S)
        if not match:
            continue
        try:
            args_key = repr(ast.literal_eval(f"({match.group(1)},)"))
            oracle_node = ast.parse(str(outcome[1]), mode="eval").body
        except Exception:
            continue
        mapping[args_key] = oracle_node
    return mapping


def _call_args_key(call: ast.Call) -> Optional[str]:
    if not isinstance(call.func, ast.Name) or call.func.id != "candidate" or call.keywords:
        return None
    try:
        return repr(ast.literal_eval(ast.Expression(body=ast.Tuple(elts=list(call.args), ctx=ast.Load()))))
    except Exception:
        return None


def reconcile_literal_oracle_assertions(
    test_code: str,
    probe_exprs: Sequence[str],
    probe_outcomes: Optional[Sequence[Sequence[str]]],
) -> str:
    """Rewrite ``assert candidate(<args>) == <literal>`` (either operand
    order, and through a simple local ``result = candidate(<args>)``
    assignment) so the literal side exactly matches the verified canonical
    output, whenever ``<args>`` matches a ``<canonical_probe_oracles>`` entry.

    Small/cheap models are prone to hallucinating a *plausible-looking* but
    wrong literal (wrong case, punctuation, or wording) even when the exact
    correct value was handed to them verbatim in the oracle block -- this is
    the dominant real-world rejection cause for Layer 1's local model, not a
    bug in the sanity-check gate itself. Rather than relying only on prompt
    instructions to stop that, this removes the need for the model to
    reproduce the literal by hand for any input we already have verified
    ground truth for. It never invents a value: only literals whose call
    arguments exactly match a probe actually executed against the canonical
    implementation are substituted, and only in place of another literal
    (never a variable or expression), so this can only fix a wrong constant,
    never change the test's logic or which inputs it exercises.
    """
    oracle_map = _oracle_value_map(probe_exprs, probe_outcomes)
    if not oracle_map or not test_code.strip():
        return test_code
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return test_code

    changed = False
    for func in tree.body:
        if not (isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and func.name == "check"):
            continue
        local_calls: Dict[str, str] = {}
        for stmt in ast.walk(func):
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                key = _call_args_key(stmt.value)
                if key is not None:
                    local_calls[stmt.targets[0].id] = key
            if not (isinstance(stmt, ast.Assert) and isinstance(stmt.test, ast.Compare)
                    and len(stmt.test.ops) == 1 and isinstance(stmt.test.ops[0], ast.Eq)):
                continue
            left, right = stmt.test.left, stmt.test.comparators[0]
            for call_side, literal_side, call_is_left in ((left, right, True), (right, left, False)):
                key = None
                if isinstance(call_side, ast.Call):
                    key = _call_args_key(call_side)
                elif isinstance(call_side, ast.Name) and call_side.id in local_calls:
                    key = local_calls[call_side.id]
                if key is None or key not in oracle_map:
                    continue
                if not isinstance(literal_side, (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.UnaryOp)):
                    continue
                if call_is_left:
                    stmt.test.comparators[0] = oracle_map[key]
                else:
                    stmt.test.left = oracle_map[key]
                changed = True
                break

    if not changed:
        return test_code
    try:
        return ast.unparse(tree)
    except Exception:
        return test_code


def evaluate_single_generated_test(
    test_code: str,
    correct_source: str,
    entry_point: str,
    all_mutants: Sequence[Mutant],
    *,
    require_kill: bool = False,
) -> Tuple[bool, int, int, str, List[str]]:
    """Validate and evaluate one generated test against the supplied active mutants.

    ``require_kill=True`` makes acceptance productive rather than merely valid.
    This is used by all proposed layers and the baseline.
    """
    contract_ok, contract_error, _ = validate_check_contract(test_code)
    if not contract_ok:
        return False, 0, len(all_mutants), contract_error, []

    sanity = verify_no_false_positives([test_code], correct_source, entry_point)
    if not sanity["all_passed"]:
        return False, 0, len(all_mutants), "; ".join(sanity["failures"][:2]), []

    evaluated = run_suite_against_mutants([test_code], mutant_copies(all_mutants), entry_point)
    killed_ids = [m.mutant_id for m in evaluated if m.is_killed]
    killed = len(killed_ids)
    if require_kill and killed == 0:
        return False, 0, len(all_mutants), "valid but unproductive: killed zero active mutants", []
    return True, killed, len(all_mutants) - killed, "", killed_ids


def apply_cumulative_kills(all_mutants: Sequence[Mutant], killed_ids: Iterable[str], kill_label: str) -> List[Mutant]:
    killed_set = set(killed_ids)
    for mutant in all_mutants:
        if mutant.mutant_id in killed_set:
            mutant.status = "KILLED"
            if kill_label and kill_label not in mutant.kill_tests:
                mutant.kill_tests.append(kill_label)
        else:
            mutant.status = "SURVIVED"
    return list(all_mutants)


class PlateauTracker:
    """Shared stop-on-plateau criterion for every layer and the baseline.

    Replaces "always spend the full fixed attempt/iteration budget" with
    "stop once `patience` consecutive attempts produce zero *new* kills."
    Used identically by Layer 1/2/3 and the iterative baseline so a
    cost/effectiveness comparison between them is not confounded by one arm
    being allowed to plateau-spend while the other is not.

    `patience=1` (the default everywhere) stops immediately after the first
    zero-new-kill attempt -- the most aggressive, cheapest setting. A higher
    patience tolerates one or more non-improving attempts before giving up,
    which may suit a loop that searches a larger space per call (e.g. the
    baseline, which sees every surviving mutant each iteration, not just a
    small representative batch).
    """

    def __init__(self, patience: int = 1, enabled: bool = True):
        self.patience = max(1, int(patience))
        self.enabled = enabled
        self._zero_kill_streak = 0
        self.stop_reason: Optional[str] = None

    def record(self, new_kills: int) -> None:
        if new_kills > 0:
            self._zero_kill_streak = 0
        else:
            self._zero_kill_streak += 1

    def should_stop(self) -> bool:
        if not self.enabled:
            return False
        if self._zero_kill_streak >= self.patience:
            self.stop_reason = f"plateau: {self._zero_kill_streak} consecutive attempt(s) with zero new kills"
            return True
        return False


def all_probable_equivalent(mutants: Sequence[Mutant]) -> bool:
    """True when every mutant in the batch is already flagged as probably
    equivalent by upstream fuzz-probe analysis.

    Used to skip an LLM call entirely rather than pay for a very-unlikely
    kill against a batch already believed equivalent -- a stopping decision
    based on what's left to test, not on how many attempts have run.
    """
    if not mutants:
        return False
    return all(getattr(m, "equivalence_status", "UNKNOWN") == "PROBABLE_EQUIVALENT" for m in mutants)


def summarize_attempt_history(attempt_history: Sequence[Dict], source_layer: str) -> Dict:
    """Build a compact, honest handoff summary from a layer's raw attempt log.

    Two fixes over passing `attempt_history` straight through into the next
    layer's prompt:
    1. Distinguishes genuine failed test-strategy attempts from
       infrastructure errors (LLM API/OOM/timeout failures). Infrastructure
       failures are not "hypotheses already known to be wrong" -- nothing
       was actually tried -- so they must not be framed as prior reasoning
       failures for the next layer to avoid repeating.
    2. Collapses repeated *identical* infrastructure errors into one
       annotated entry instead of repeating full stack traces verbatim,
       which previously bloated prompts without adding decision-relevant
       information.
    """
    strategy_attempts: List[Dict] = []
    infra_errors: List[str] = []
    productive_count = 0
    for row in attempt_history:
        category = row.get("failure_category", "")
        if category == "LLM_API_ERROR" or row.get("status") == "ERROR":
            infra_errors.append(row.get("error", "unknown error"))
            continue
        strategy_attempts.append(row)
        if row.get("status") == "PRODUCTIVE":
            productive_count += 1

    infra_error_summary = None
    if infra_errors:
        first = infra_errors[0]
        infra_error_summary = {
            "count": len(infra_errors),
            "note": (
                "These attempts failed due to infrastructure errors (API/timeout/"
                "OOM), not a rejected test strategy -- no test-writing hypothesis "
                "was actually evaluated in these attempts."
            ),
            "first_error": first[:300],
        }

    return {
        "source_layer": source_layer,
        "attempt_count": len(strategy_attempts),
        "productive_attempts": productive_count,
        "attempts": strategy_attempts,
        "infrastructure_errors": infra_error_summary,
    }


def compute_cluster_kill_consistency(evaluated_mutants: Sequence[Mutant], target_representatives: Sequence[Mutant]) -> float:
    if not evaluated_mutants or not target_representatives:
        return 0.0
    by_cluster: Dict[int, List[Mutant]] = defaultdict(list)
    for mutant in evaluated_mutants:
        if mutant.cluster_id is not None:
            by_cluster[int(mutant.cluster_id)].append(mutant)
    status_by_id = {m.mutant_id: m.is_killed for m in evaluated_mutants}
    scores: List[float] = []
    for rep in target_representatives:
        if rep.cluster_id is None or rep.mutant_id not in status_by_id:
            continue
        members = by_cluster.get(int(rep.cluster_id), [])
        if not members:
            continue
        rep_status = status_by_id[rep.mutant_id]
        scores.append(sum(1 for m in members if m.is_killed == rep_status) / len(members))
    return round(sum(scores) / len(scores), 4) if scores else 0.0