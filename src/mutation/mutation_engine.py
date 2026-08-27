"""
mutation_engine.py — Python AST mutation engine for HumanEval.

The engine generates realistic single-point mutants from a complete Python PUT
(program under test), executes HumanEval-style tests in isolated subprocesses,
and can build cheap behavioral signatures from probe executions.

Implemented mutation families include comparison/boundary changes, boolean
changes, arithmetic changes, return-value changes, membership/identity flips,
index/range off-by-one changes, and call-argument swaps.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.utils.config import MAX_MUTANTS_PER_PROBLEM, MUTATION_TIMEOUT_SEC, PROBE_TIMEOUT_SEC, PYTHON_EXECUTABLE
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Mutant:
    mutant_id: str
    problem_id: str
    operator: str
    description: str
    line_number: int
    mutated_source: str
    ast_node_type: str = ""
    parent_node_type: str = ""
    status: str = "PENDING"
    kill_tests: List[str] = field(default_factory=list)
    behavior_signature: List[int] = field(default_factory=list)
    cluster_id: Optional[int] = None
    centrality: Optional[float] = None
    information_score: Optional[float] = None
    equivalence_status: str = "UNKNOWN"
    equivalence_reason: str = ""

    @property
    def is_killed(self) -> bool:
        return self.status == "KILLED"

    @property
    def is_survived(self) -> bool:
        return self.status == "SURVIVED"

    @property
    def is_no_coverage(self) -> bool:
        return self.status in {"TIMEOUT", "ERROR"}

    def to_dict(self) -> Dict:
        return {
            "mutant_id": self.mutant_id,
            "problem_id": self.problem_id,
            "operator": self.operator,
            "description": self.description,
            "line_number": self.line_number,
            "mutated_source": self.mutated_source,
            "ast_node_type": self.ast_node_type,
            "parent_node_type": self.parent_node_type,
            "status": self.status,
            "kill_tests": self.kill_tests,
            "behavior_signature": self.behavior_signature,
            "cluster_id": self.cluster_id,
            "centrality": self.centrality,
            "information_score": self.information_score,
            "equivalence_status": self.equivalence_status,
            "equivalence_reason": self.equivalence_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Mutant":
        return cls(
            mutant_id=d["mutant_id"],
            problem_id=d["problem_id"],
            operator=d["operator"],
            description=d["description"],
            line_number=d["line_number"],
            mutated_source=d["mutated_source"],
            ast_node_type=d.get("ast_node_type", ""),
            parent_node_type=d.get("parent_node_type", ""),
            status=d.get("status", "PENDING"),
            kill_tests=list(d.get("kill_tests", [])),
            behavior_signature=list(d.get("behavior_signature", [])),
            cluster_id=d.get("cluster_id"),
            centrality=d.get("centrality"),
            information_score=d.get("information_score"),
            equivalence_status=d.get("equivalence_status", "UNKNOWN"),
            equivalence_reason=d.get("equivalence_reason", ""),
        )


@dataclass
class MutationCandidate:
    operator: str
    line: int
    description: str
    node_type: str
    parent_type: str
    ordinal: int
    apply_kind: str
    attr: str = ""
    index: Optional[int] = None
    new_value: object = None


class _ParentAnnotator(ast.NodeVisitor):
    def visit(self, node: ast.AST) -> None:  # type: ignore[override]
        for child in ast.iter_child_nodes(node):
            setattr(child, "_parent", node)
        super().visit(node)


def _parent_type(node: ast.AST) -> str:
    return type(getattr(node, "_parent", None)).__name__ if getattr(node, "_parent", None) else "Module"


_COMPARISON_FLIPS = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.Gt,
    ast.LtE: ast.GtE,
    ast.Gt: ast.Lt,
    ast.GtE: ast.LtE,
}

_BOUNDARY_SHIFTS = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
}

_ARITHMETIC_FLIPS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Div,
    ast.Mod: ast.Mult,
    ast.Pow: ast.Mult,
}

_BOOL_FLIPS = {ast.And: ast.Or, ast.Or: ast.And}
_MEMBERSHIP_FLIPS = {ast.In: ast.NotIn, ast.NotIn: ast.In}
_IDENTITY_FLIPS = {ast.Is: ast.IsNot, ast.IsNot: ast.Is}


class _MutationCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.candidates: List[MutationCandidate] = []

    def _add(self, operator: str, node: ast.AST, description: str, apply_kind: str,
             attr: str = "", index: Optional[int] = None, new_value: object = None) -> None:
        self.candidates.append(
            MutationCandidate(
                operator=operator,
                line=int(getattr(node, "lineno", 0) or 0),
                description=description,
                node_type=type(node).__name__,
                parent_type=_parent_type(node),
                ordinal=len(self.candidates),
                apply_kind=apply_kind,
                attr=attr,
                index=index,
                new_value=new_value,
            )
        )

    def visit_Compare(self, node: ast.Compare) -> None:
        for i, op in enumerate(node.ops):
            op_type = type(op)
            if op_type in _COMPARISON_FLIPS:
                self._add(
                    "COMPARISON_FLIP", node,
                    f"{op_type.__name__} -> {_COMPARISON_FLIPS[op_type].__name__}",
                    "set_list_item", "ops", i, _COMPARISON_FLIPS[op_type](),
                )
            if op_type in _BOUNDARY_SHIFTS:
                self._add(
                    "BOUNDARY_SHIFT", node,
                    f"{op_type.__name__} -> {_BOUNDARY_SHIFTS[op_type].__name__}",
                    "set_list_item", "ops", i, _BOUNDARY_SHIFTS[op_type](),
                )
            if op_type in _MEMBERSHIP_FLIPS:
                self._add(
                    "MEMBERSHIP_FLIP", node,
                    f"{op_type.__name__} -> {_MEMBERSHIP_FLIPS[op_type].__name__}",
                    "set_list_item", "ops", i, _MEMBERSHIP_FLIPS[op_type](),
                )
            if op_type in _IDENTITY_FLIPS:
                self._add(
                    "IDENTITY_FLIP", node,
                    f"{op_type.__name__} -> {_IDENTITY_FLIPS[op_type].__name__}",
                    "set_list_item", "ops", i, _IDENTITY_FLIPS[op_type](),
                )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        op_type = type(node.op)
        if op_type in _ARITHMETIC_FLIPS:
            self._add(
                "ARITHMETIC_FLIP", node,
                f"{op_type.__name__} -> {_ARITHMETIC_FLIPS[op_type].__name__}",
                "set_attr", "op", None, _ARITHMETIC_FLIPS[op_type](),
            )
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        op_type = type(node.op)
        if op_type in _BOOL_FLIPS:
            self._add(
                "BOOLEAN_FLIP", node,
                f"{op_type.__name__} -> {_BOOL_FLIPS[op_type].__name__}",
                "set_attr", "op", None, _BOOL_FLIPS[op_type](),
            )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool):
            self._add("BOOLEAN_FLIP", node, f"{node.value} -> {not node.value}",
                      "set_attr", "value", None, not node.value)
        elif isinstance(node.value, int) and not isinstance(node.value, bool):
            for delta in (1, -1):
                self._add("CONSTANT_CHANGE", node, f"constant {node.value} -> {node.value + delta}",
                          "set_attr", "value", None, node.value + delta)
            if node.value not in (0, 1, -1):
                self._add("CONSTANT_CHANGE", node, f"constant {node.value} -> 0",
                          "set_attr", "value", None, 0)
        elif isinstance(node.value, float):
            for delta in (1.0, -1.0):
                self._add("CONSTANT_CHANGE", node, f"constant {node.value} -> {node.value + delta}",
                          "set_attr", "value", None, node.value + delta)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self._add("RETURN_VALUE_CHANGE", node, "return <expr> -> return None",
                      "set_attr", "value", None, ast.Constant(value=None))
            self._add("RETURN_VALUE_CHANGE", node, "return <expr> -> return False",
                      "set_attr", "value", None, ast.Constant(value=False))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._add("NEGATE_CONDITION", node, "if <cond> -> if not (<cond>)",
                  "set_attr", "test", None, ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(node.test)))
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._add("NEGATE_CONDITION", node, "while <cond> -> while not (<cond>)",
                  "set_attr", "test", None, ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(node.test)))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            for delta in (1, -1):
                self._add("INDEX_BOUNDARY", node, f"index {node.slice.value} -> {node.slice.value + delta}",
                          "set_attr", "slice", None, ast.Constant(value=node.slice.value + delta))
        self.generic_visit(node)

    def visit_Slice(self, node: ast.Slice) -> None:
        for attr in ("lower", "upper"):
            val = getattr(node, attr)
            if isinstance(val, ast.Constant) and isinstance(val.value, int):
                self._add("INDEX_BOUNDARY", node, f"slice {attr} {val.value} -> {val.value + 1}",
                          "set_attr", attr, None, ast.Constant(value=val.value + 1))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if len(node.args) >= 2:
            new_args = list(node.args)
            new_args[0], new_args[1] = copy.deepcopy(
                new_args[1]), copy.deepcopy(new_args[0])
            self._add("ARGUMENT_SWAP", node, "swap first two call arguments",
                      "set_attr", "args", None, new_args)

        if isinstance(node.func, ast.Name) and node.func.id == "range":
            for i, arg in enumerate(node.args):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    for delta in (1, -1):
                        new_args = copy.deepcopy(node.args)
                        new_args[i] = ast.Constant(value=arg.value + delta)
                        self._add("RANGE_BOUNDARY", node, f"range arg {i}: {arg.value} -> {arg.value + delta}",
                                  "set_attr", "args", None, new_args)
        self.generic_visit(node)


def _collect_candidates(tree: ast.AST) -> List[MutationCandidate]:
    _ParentAnnotator().visit(tree)
    collector = _MutationCollector()
    collector.visit(tree)
    return collector.candidates


def _apply_mutation(tree: ast.AST, ordinal: int) -> Optional[ast.AST]:
    mutated_tree = copy.deepcopy(tree)
    candidates = _collect_candidates(mutated_tree)
    if ordinal < 0 or ordinal >= len(candidates):
        return None
    target = candidates[ordinal]

    # Re-find the node by traversing the target ordinal again. The collector stored
    # the mutation target object inside its local walk; to mutate it we need to repeat
    # the same walk and then operate on the object referenced by that target.
    # Because candidates were collected from mutated_tree, target is already live.
    live_target = target

    # The live target node is not directly stored in MutationCandidate to keep it
    # serializable, so collect a parallel map with objects now.
    object_candidates: List[Tuple[MutationCandidate, ast.AST]] = []

    class ObjectCollector(_MutationCollector):
        def _add(self, operator: str, node: ast.AST, description: str, apply_kind: str,
                 attr: str = "", index: Optional[int] = None, new_value: object = None) -> None:
            cand = MutationCandidate(
                operator=operator, line=int(getattr(node, "lineno", 0) or 0),
                description=description, node_type=type(
                    node).__name__, parent_type=_parent_type(node),
                ordinal=len(self.candidates), apply_kind=apply_kind, attr=attr, index=index,
                new_value=new_value,
            )
            self.candidates.append(cand)
            object_candidates.append((cand, node))

    _ParentAnnotator().visit(mutated_tree)
    oc = ObjectCollector()
    oc.visit(mutated_tree)
    if ordinal >= len(object_candidates):
        return None
    live_target, node = object_candidates[ordinal]

    if live_target.apply_kind == "set_list_item":
        seq = getattr(node, live_target.attr)
        if live_target.index is None or live_target.index >= len(seq):
            return None
        seq[live_target.index] = live_target.new_value
    elif live_target.apply_kind == "set_attr":
        setattr(node, live_target.attr, live_target.new_value)
    else:
        return None

    ast.fix_missing_locations(mutated_tree)
    return mutated_tree


def generate_mutants(problem_id: str, source_code: str, max_mutants: int = MAX_MUTANTS_PER_PROBLEM) -> List[Mutant]:
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        log.error("Cannot parse source for %s: %s", problem_id, e)
        return []

    candidates = _collect_candidates(tree)
    if not candidates:
        return []

    if len(candidates) > max_mutants:
        step = len(candidates) / max_mutants
        selected_indices = [int(i * step) for i in range(max_mutants)]
    else:
        selected_indices = list(range(len(candidates)))

    mutants: List[Mutant] = []
    seen_sources = set()
    safe_problem_id = problem_id.replace("/", "_")

    for out_idx, ordinal in enumerate(selected_indices):
        cand = candidates[ordinal]
        mutated_tree = _apply_mutation(tree, ordinal)
        if mutated_tree is None:
            continue
        try:
            mutated_source = ast.unparse(mutated_tree)
            compile(mutated_source, "<mutant>", "exec")
        except Exception:
            continue

        source_hash = hashlib.sha1(
            mutated_source.encode("utf-8")).hexdigest()[:10]
        if source_hash in seen_sources:
            continue
        seen_sources.add(source_hash)

        mutant_id = f"{safe_problem_id}:m{out_idx:03d}:{cand.operator}:{source_hash}"
        mutants.append(
            Mutant(
                mutant_id=mutant_id,
                problem_id=problem_id,
                operator=cand.operator,
                description=cand.description,
                line_number=cand.line,
                mutated_source=mutated_source,
                ast_node_type=cand.node_type,
                parent_node_type=cand.parent_type,
            )
        )

    log.info("%s: generated %d mutants from %d candidate sites",
             problem_id, len(mutants), len(candidates))
    return mutants


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _is_first_two_args_swapped(original: ast.Call, changed: ast.Call) -> bool:
    if _call_name(original) != _call_name(changed):
        return False
    if len(original.args) < 2 or len(changed.args) != len(original.args):
        return False
    original_args = [ast.dump(arg, include_attributes=False)
                     for arg in original.args]
    changed_args = [ast.dump(arg, include_attributes=False)
                    for arg in changed.args]
    original_keywords = [(kw.arg, ast.dump(
        kw.value, include_attributes=False)) for kw in original.keywords]
    changed_keywords = [(kw.arg, ast.dump(
        kw.value, include_attributes=False)) for kw in changed.keywords]
    return (
        changed_args[0] == original_args[1]
        and changed_args[1] == original_args[0]
        and changed_args[2:] == original_args[2:]
        and original_keywords == changed_keywords
    )


def mark_obvious_equivalents(correct_source: str, mutants: List[Mutant]) -> List[Mutant]:
    """Conservatively mark only statically obvious equivalent mutants.

    This is intentionally narrower than a general equivalence detector.  The
    adjusted score excludes only argument swaps in known commutative built-ins
    (min/max and math.gcd/lcm).  All other survivors remain in the denominator
    unless a future analysis supplies stronger evidence.
    """
    try:
        original_tree = ast.parse(correct_source)
        original_calls = [node for node in ast.walk(
            original_tree) if isinstance(node, ast.Call)]
    except Exception:
        return mutants

    commutative_calls = {"min", "max", "math.gcd", "math.lcm"}
    for mutant in mutants:
        mutant.equivalence_status = "UNKNOWN"
        mutant.equivalence_reason = ""
        if mutant.operator != "ARGUMENT_SWAP":
            continue
        try:
            changed_tree = ast.parse(mutant.mutated_source)
            changed_calls = [node for node in ast.walk(
                changed_tree) if isinstance(node, ast.Call)]
        except Exception:
            continue
        if len(original_calls) != len(changed_calls):
            continue
        differences = [
            (before, after)
            for before, after in zip(original_calls, changed_calls)
            if ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False)
        ]
        if len(differences) != 1:
            continue
        before, after = differences[0]
        call_name = _call_name(before)
        if call_name in commutative_calls and _is_first_two_args_swapped(before, after):
            mutant.equivalence_status = "STATIC_EQUIVALENT"
            mutant.equivalence_reason = f"first two arguments swapped in commutative call {call_name}"
    return mutants


_RUNNER_TEMPLATE = r'''
import signal
import math
import itertools
import functools
import collections
from typing import *

class _Timeout(Exception):
    pass

def _handler(signum, frame):
    raise _Timeout()

signal.signal(signal.SIGALRM, _handler)
signal.alarm({timeout})

try:
{indented_source}
    candidate = {entry_point}

{indented_test}

    check(candidate)

    print("__CLUSE_RESULT__:PASS")

except _Timeout:
    print("__CLUSE_RESULT__:TIMEOUT")

except AssertionError as e:
    msg = str(e)

    if not msg:
        tb = e.__traceback__
        while tb.tb_next:
            tb = tb.tb_next

        frame = tb.tb_frame

        locs = {{
            k: v
            for k, v in frame.f_locals.items()
            if k not in ("candidate",)
        }}

        msg = (
            f"assertion failed (no message); "
            f"locals at failure: {{locs!r}}"
        )

    print("__CLUSE_RESULT__:FAIL:" + msg[:500])

except Exception as e:
    print(
        "__CLUSE_RESULT__:ERROR:"
        + type(e).__name__
        + ":"
        + str(e)[:500]
    )
'''


def _run_subprocess_script(script_path: str, hard_timeout: float) -> subprocess.CompletedProcess:
    """Run a temporary Python script with robust process-group cleanup.

    On Kaggle, generated tests or mutants may hang in ways where simple subprocess.run
    can leave work behind. Popen + start_new_session lets us kill the whole process group
    on timeout instead of only waiting for the parent process.
    """
    cmd = [PYTHON_EXECUTABLE or sys.executable, script_path]
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "start_new_session": os.name != "nt",
    }
    proc = subprocess.Popen(cmd, **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=hard_timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            proc.kill()
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            cmd, hard_timeout, output=stdout, stderr=stderr)


def _run_python_script(script: str, timeout: int) -> Dict[str, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name
    try:
        result = _run_subprocess_script(script_path, hard_timeout=timeout + 5)
        output = result.stdout.strip()
        if "__CLUSE_RESULT__:PASS" in output:
            return {"result": "PASS", "detail": ""}
        if "__CLUSE_RESULT__:FAIL:" in output:
            return {"result": "FAIL", "detail": output.split("__CLUSE_RESULT__:FAIL:", 1)[-1]}
        if "__CLUSE_RESULT__:TIMEOUT" in output:
            return {"result": "TIMEOUT", "detail": ""}
        if "__CLUSE_RESULT__:ERROR:" in output:
            return {"result": "ERROR", "detail": output.split("__CLUSE_RESULT__:ERROR:", 1)[-1]}
        return {"result": "ERROR", "detail": "no result marker; stderr=" + result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"result": "TIMEOUT", "detail": "subprocess hard timeout"}
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def _run_test_against_source(source_code: str, entry_point: str, test_code: str,
                             timeout: int = MUTATION_TIMEOUT_SEC) -> Dict[str, str]:
    script = _RUNNER_TEMPLATE.format(
        timeout=timeout,
        indented_source=textwrap.indent(source_code, "    "),
        entry_point=entry_point,
        indented_test=textwrap.indent(test_code, "    "),
    )
    return _run_python_script(script, timeout)


def _rename_check_function(test_code: str, index: int) -> str:
    import re

    return re.sub(r"def\s+check\s*\(\s*candidate\s*\)\s*:", f"def __cluse_check_{index}(candidate):", test_code, count=1)


_BATCH_RUNNER_TEMPLATE = r'''
import signal
import math
import itertools
import functools
import collections
from typing import *

class _Timeout(Exception):
    pass

def _handler(signum, frame):
    raise _Timeout()

signal.signal(signal.SIGALRM, _handler)
signal.alarm({timeout})

try:
{indented_source}
    candidate = {entry_point}

{indented_tests}

    __cluse_results = []

    for __cluse_i, __cluse_fn in enumerate(__cluse_checks):
        try:
            __cluse_fn(candidate)
            __cluse_results.append((__cluse_i, "PASS", ""))
        except AssertionError as e:
            msg = str(e)
            if not msg:
                tb = e.__traceback__
                while tb.tb_next:
                    tb = tb.tb_next
                frame = tb.tb_frame
                locs = {{
                    k: v
                    for k, v in frame.f_locals.items()
                    if k not in ("candidate",)
                }}
                msg = (
                    f"assertion failed (no message); "
                    f"locals at failure: {{locs!r}}"
                )

            __cluse_results.append((__cluse_i, "FAIL", msg[:500]))

        except Exception as e:
            __cluse_results.append(
                (
                    __cluse_i,
                    "ERROR",
                    type(e).__name__ + ":" + str(e)[:300],
                )
            )

    print("__CLUSE_BATCH__:" + repr(__cluse_results))

except _Timeout:
    print("__CLUSE_BATCH_TIMEOUT__")

except Exception as e:
    print(
        "__CLUSE_BATCH_ERROR__:"
        + type(e).__name__
        + ":"
        + str(e)[:500]
    )
'''


def _run_tests_against_source_batch(source_code: str, entry_point: str, test_snippets: List[str],
                                    timeout: int = MUTATION_TIMEOUT_SEC) -> List[Dict[str, str]]:
    # Run many HumanEval-style check(candidate) snippets in one subprocess.
    # This is much faster on Kaggle than spawning one Python process per mutant per test.
    if not test_snippets:
        return []

    renamed = []
    check_names = []
    for i, test in enumerate(test_snippets):
        renamed_test = _rename_check_function(test, i)
        if f"__cluse_check_{i}" not in renamed_test:
            return [{"result": "ERROR", "detail": "missing check(candidate) function"} for _ in test_snippets]
        renamed.append(renamed_test)
        check_names.append(f"__cluse_check_{i}")

    checks_line = "    __cluse_checks = [" + ", ".join(check_names) + "]\n"
    indented_tests = textwrap.indent(
        "\n\n".join(renamed), "    ") + "\n" + checks_line
    script = _BATCH_RUNNER_TEMPLATE.format(
        timeout=max(timeout, 1),
        indented_source=textwrap.indent(source_code, "    "),
        entry_point=entry_point,
        indented_tests=indented_tests,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name
    try:
        hard_timeout = max(
            timeout + 5, min(timeout * max(len(test_snippets), 1) + 3, 30))
        result = _run_subprocess_script(script_path, hard_timeout=hard_timeout)
        out = result.stdout.strip()
        if "__CLUSE_BATCH__:" in out:
            payload = out.split("__CLUSE_BATCH__:", 1)[-1].splitlines()[0]
            try:
                import ast as _ast
                rows = _ast.literal_eval(payload)
                by_index = {int(i): {"result": str(r), "detail": str(d)}
                            for i, r, d in rows}
                return [by_index.get(i, {"result": "ERROR", "detail": "missing batch row"}) for i in range(len(test_snippets))]
            except Exception as e:
                return [{"result": "ERROR", "detail": f"cannot parse batch result: {e}"} for _ in test_snippets]
        if "__CLUSE_BATCH_TIMEOUT__" in out:
            return [{"result": "TIMEOUT", "detail": "batch timeout"} for _ in test_snippets]
        if "__CLUSE_BATCH_ERROR__:" in out:
            detail = out.split("__CLUSE_BATCH_ERROR__:", 1)[-1]
            return [{"result": "ERROR", "detail": detail} for _ in test_snippets]
        return [{"result": "ERROR", "detail": "no batch marker; stderr=" + result.stderr[-500:]} for _ in test_snippets]
    except subprocess.TimeoutExpired:
        return [{"result": "TIMEOUT", "detail": "subprocess hard timeout"} for _ in test_snippets]
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def run_suite_against_mutants(test_snippets: List[str], mutants: List[Mutant], entry_point: str,
                              timeout: int = MUTATION_TIMEOUT_SEC) -> List[Mutant]:
    if not test_snippets:
        for m in mutants:
            m.status = "SURVIVED"
            m.kill_tests = []
        return mutants

    for m in mutants:
        outcomes = _run_tests_against_source_batch(
            m.mutated_source, entry_point, test_snippets, timeout=timeout)
        killed_by = [f"test_{i}:{o['result']}" for i, o in enumerate(
            outcomes) if o["result"] in {"FAIL", "ERROR", "TIMEOUT"}]
        m.status = "KILLED" if killed_by else "SURVIVED"
        m.kill_tests = killed_by
    return mutants


def verify_no_false_positives(test_snippets: List[str], correct_source: str, entry_point: str,
                              timeout: int = MUTATION_TIMEOUT_SEC) -> Dict[str, object]:
    failures: List[str] = []
    outcomes = _run_tests_against_source_batch(
        correct_source, entry_point, test_snippets, timeout=timeout)
    for i, outcome in enumerate(outcomes):
        if outcome["result"] != "PASS":
            failures.append(
                f"test_{i}: {outcome['result']} - {outcome['detail']}")
    return {"all_passed": len(failures) == 0, "failures": failures}


_PROBE_TEMPLATE = r'''
import signal
import json
import math
import itertools
import functools
import collections
from typing import *

class _Timeout(Exception):
    pass

def _handler(signum, frame):
    raise _Timeout()

signal.signal(signal.SIGALRM, _handler)
signal.alarm({timeout})

try:
{indented_source}

    candidate = {entry_point}
    value = {probe_expr}
    print("__CLUSE_PROBE__:VALUE:" + repr(value)[:1000])
except _Timeout:
    print("__CLUSE_PROBE__:TIMEOUT")
except Exception as e:
    print("__CLUSE_PROBE__:ERROR:" + type(e).__name__ + ":" + str(e)[:300])
'''


def run_probe_against_source(source_code: str, entry_point: str, probe_expr: str,
                             timeout: int = PROBE_TIMEOUT_SEC) -> Tuple[str, str]:
    script = _PROBE_TEMPLATE.format(
        timeout=timeout,
        indented_source=textwrap.indent(source_code, "    "),
        entry_point=entry_point,
        probe_expr=probe_expr,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name
    try:
        result = _run_subprocess_script(script_path, hard_timeout=timeout + 5)
        out = result.stdout.strip()
        if "__CLUSE_PROBE__:VALUE:" in out:
            return "VALUE", out.split("__CLUSE_PROBE__:VALUE:", 1)[-1]
        if "__CLUSE_PROBE__:TIMEOUT" in out:
            return "TIMEOUT", ""
        if "__CLUSE_PROBE__:ERROR:" in out:
            return "ERROR", out.split("__CLUSE_PROBE__:ERROR:", 1)[-1]
        return "ERROR", result.stderr[-300:]
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def attach_behavior_signatures(correct_source: str, entry_point: str, mutants: List[Mutant],
                               probe_exprs: List[str], timeout: int = PROBE_TIMEOUT_SEC) -> List[Tuple[str, str]]:
    """
    For each mutant, create a vector over probe expressions:
        0 = same as original, 1 = different value/error behavior, 2 = mutant timed out.
    This is an unsupervised signal used for clustering and representative selection.
    """
    if not probe_exprs:
        for m in mutants:
            m.behavior_signature = []
        return []

    original = [run_probe_against_source(
        correct_source, entry_point, expr, timeout) for expr in probe_exprs]
    for mutant in mutants:
        signature: List[int] = []
        for expr, orig_outcome in zip(probe_exprs, original):
            mut_outcome = run_probe_against_source(
                mutant.mutated_source, entry_point, expr, timeout)
            if mut_outcome[0] == "TIMEOUT":
                signature.append(2)
            elif mut_outcome == orig_outcome:
                signature.append(0)
            else:
                signature.append(1)
        mutant.behavior_signature = signature
    return original


def save_mutants(mutants: List[Mutant], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in mutants], f, indent=2)
    log.info("Saved %d mutants -> %s", len(mutants), path)


def load_mutants(path: Path) -> List[Mutant]:
    with open(path, encoding="utf-8") as f:
        return [Mutant.from_dict(d) for d in json.load(f)]
