"""Lightweight probe generation for behavior-aware mutant clustering."""

from __future__ import annotations

import ast
import re
from typing import List

from src.utils.config import MAX_PROBES_PER_PROBLEM


def _split_params(params: str) -> List[str]:
    """Split a comma-separated argument/parameter list, respecting nested
    brackets *and* string literals -- a comma inside a quoted string (e.g.
    the `"1,234"` in an example call) must not be treated as an argument
    separator."""
    parts: List[str] = []
    current = ""
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(params):
        ch = params[i]
        if quote:
            current += ch
            if ch == "\\" and i + 1 < len(params):
                # preserve an escaped character verbatim without toggling quote state
                i += 1
                current += params[i]
            elif ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            current += ch
        elif ch in "[(":
            depth += 1
            current += ch
        elif ch in ")]":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
        i += 1
    if current.strip():
        parts.append(current.strip())
    return parts


def extract_param_types(prompt: str, entry_point: str) -> List[str]:
    match = re.search(r"def\s+" + re.escape(entry_point) + r"\s*\(([^)]*)\)", prompt)
    if not match:
        return []
    raw_params = _split_params(match.group(1).strip())
    types: List[str] = []
    for p in raw_params:
        if not p:
            continue
        if ":" in p:
            t = p.split(":", 1)[1].split("=", 1)[0].strip()
        else:
            t = "Any"
        types.append(t)
    return types


def _values_for_type(type_hint: str) -> List[str]:
    t = type_hint.replace(" ", "").lower()
    if "list" in t or t.startswith("sequence") or t.startswith("iterable"):
        if "str" in t:
            return ['[]', '["a"]', '["a", "b"]']
        if "float" in t:
            return ['[]', '[0.0]', '[0.0, 1.0, -1.0]']
        return ['[]', '[0]', '[0, 1, -1]', '[1, 2, 3]']
    if "tuple" in t:
        return ['()', '(0,)', '(0, 1)']
    if "dict" in t:
        return ['{}', '{"a": 1}', '{0: 0, 1: 1}']
    if "str" in t:
        return ['""', '"a"', '"abc"', '"aba"']
    if "bool" in t:
        return ['False', 'True']
    if "float" in t:
        return ['0.0', '1.0', '-1.0', '0.5']
    if "int" in t:
        return ['0', '1', '-1', '2']
    # HumanEval often omits hints; these cover common algorithmic functions.
    return ['0', '1', '"a"', '[0, 1]']


def _replace_entry_call(expr: str, entry_point: str) -> str:
    # Convert doctest expressions like foo(1) to candidate(1).
    return re.sub(r"\b" + re.escape(entry_point) + r"\s*\(", "candidate(", expr, count=1)


def extract_doctest_probe_exprs(prompt: str, entry_point: str) -> List[str]:
    probes: List[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(">>>"):
            expr = stripped[3:].strip()
            if entry_point in expr:
                probes.append(_replace_entry_call(expr, entry_point))
    return probes


def _find_balanced_call_args(text: str, start_paren: int) -> str | None:
    """Given the index of an opening '(' in text, return the substring of
    its arguments up to the matching ')', respecting nested brackets."""
    depth = 0
    for i in range(start_paren, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start_paren + 1:i]
    return None


def extract_example_call_exprs(prompt: str, entry_point: str) -> List[str]:
    """Find literal `entry_point(...)` calls anywhere in the spec text,
    excluding `>>>` doctest lines.

    Many task specs give worked examples inline (not as `>>>` doctests), e.g.
    `compare_happiness("1,234", "5,678") is "5,678"`. Extracting these
    directly grounds probes in inputs the spec author actually chose for
    this task's real domain -- e.g. numeral strings, not generic ints -- and
    fixes the case where type-hint-based generation falls back to a
    domain-mismatched default (see `_values_for_type`'s fallback) because
    the signature carries no type hints at all.

    Doctest lines are excluded here because `extract_doctest_probe_exprs`
    already turns them into probes (as an `expr == expected` string). Without
    this exclusion, the exact same call inside a doctest line -- e.g.
    `>>> foo([1, 2]) == {'a': 1}` -- would be captured a second time here as
    the bare call `candidate([1, 2])`, producing two probes over one real
    input: one carrying the expected value as an equality check, the other
    as a raw call. `build_probe_exprs` and `_probe_evidence` then show both
    to the model as if they were independent evidence, which is pure
    duplication, not two distinguishing inputs.

    Only calls whose arguments parse as Python literals are kept, so this
    never risks injecting non-deterministic or unsafe expressions as probes.
    """
    doctest_spans: List[tuple[int, int]] = []
    offset = 0
    for line in prompt.splitlines(keepends=True):
        if line.strip().startswith(">>>"):
            doctest_spans.append((offset, offset + len(line)))
        offset += len(line)

    def _in_doctest_line(pos: int) -> bool:
        return any(start <= pos < end for start, end in doctest_spans)

    probes: List[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"\b" + re.escape(entry_point) + r"\s*\(")
    for match in pattern.finditer(prompt):
        if _in_doctest_line(match.start()):
            continue
        args_text = _find_balanced_call_args(prompt, match.end() - 1)
        if args_text is None:
            continue
        try:
            parsed = ast.literal_eval(f"({args_text},)")
        except Exception:
            continue
        # literal_eval wraps a single arg without a trailing comma awkwardly;
        # normalize by re-splitting the original literal argument text so
        # commas inside nested brackets/strings are respected.
        try:
            call_args = _split_params(args_text)
            for a in call_args:
                ast.literal_eval(a)  # validate each arg is a pure literal
        except Exception:
            continue
        expr = f"candidate({args_text})"
        if expr not in seen:
            seen.add(expr)
            probes.append(expr)
    return probes


def _infer_types_from_examples(prompt: str, entry_point: str, arity_hint: int = 0) -> List[str]:
    """When the signature has no type hints, infer a plausible type per
    parameter position from any literal example call found in the spec text.

    This is what prevents the domain-mismatch failure mode where an
    untyped, string-based function (e.g. `def f(a, b):` used with
    `f("1,234", "5,678")` in the spec) got generic int-heavy fallback probes
    like `candidate(0, 0)` that error out before ever reaching the mutated
    code, making probe-derived signals (information_score, centrality)
    uniformly zero for the whole task.
    """
    pattern = re.compile(r"\b" + re.escape(entry_point) + r"\s*\(")
    for match in pattern.finditer(prompt):
        args_text = _find_balanced_call_args(prompt, match.end() - 1)
        if args_text is None:
            continue
        try:
            parts = _split_params(args_text)
            values = [ast.literal_eval(p) for p in parts]
        except Exception:
            continue
        if arity_hint and len(values) != arity_hint:
            continue
        types: List[str] = []
        for v in values:
            if isinstance(v, bool):
                types.append("bool")
            elif isinstance(v, str):
                types.append("str")
            elif isinstance(v, int):
                types.append("int")
            elif isinstance(v, float):
                types.append("float")
            elif isinstance(v, list):
                types.append("List[str]" if v and isinstance(v[0], str) else "list")
            elif isinstance(v, tuple):
                types.append("tuple")
            elif isinstance(v, dict):
                types.append("dict")
            else:
                types.append("Any")
        if types:
            return types
    return []


def generate_type_probe_exprs(prompt: str, entry_point: str, max_probes: int = MAX_PROBES_PER_PROBLEM) -> List[str]:
    param_types = extract_param_types(prompt, entry_point)
    if not param_types:
        return []
    # If the signature omits type hints ("Any" everywhere), try to recover
    # real types from a literal example call in the spec before falling back
    # to the generic (int-heavy) default -- see _infer_types_from_examples.
    if all(t == "Any" for t in param_types):
        inferred = _infer_types_from_examples(prompt, entry_point, arity_hint=len(param_types))
        if inferred and len(inferred) == len(param_types):
            param_types = inferred
    value_lists = [_values_for_type(t) for t in param_types]

    probes: List[str] = []
    # Coordinate-wise examples avoid combinatorial explosion but still exercise
    # zeros, boundaries, empty containers, and small non-empty containers.
    longest = max(len(v) for v in value_lists)
    for i in range(longest):
        args = [values[min(i, len(values) - 1)] for values in value_lists]
        probes.append("candidate(" + ", ".join(args) + ")")
        if len(probes) >= max_probes:
            break
    return probes


def build_probe_exprs(prompt: str, entry_point: str, max_probes: int = MAX_PROBES_PER_PROBLEM) -> List[str]:
    """Priority order: doctest examples > inline literal spec examples >
    type-hint-derived values. Doctests and inline examples are inputs the
    spec's own author chose as representative of the task's real domain, so
    they take precedence over generically generated values, which are only
    a fallback for when the spec gives no worked examples at all.
    """
    probes: List[str] = []
    for expr in extract_doctest_probe_exprs(prompt, entry_point):
        if expr not in probes:
            probes.append(expr)
        if len(probes) >= max_probes:
            return probes
    for expr in extract_example_call_exprs(prompt, entry_point):
        if expr not in probes:
            probes.append(expr)
        if len(probes) >= max_probes:
            return probes
    for expr in generate_type_probe_exprs(prompt, entry_point, max_probes=max_probes):
        if expr not in probes:
            probes.append(expr)
        if len(probes) >= max_probes:
            break
    return probes
