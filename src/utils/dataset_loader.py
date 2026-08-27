"""Dataset loading and normalization for CLUSE-Test.

The pipeline consumes a normalized :class:`Problem` object.  Adapters support
HumanEval, EvoEval, BigCodeBench, and compatible CSV/Parquet/JSON/JSONL files.
EvoEval is the primary benchmark for the finalized research configuration.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.utils.logger import get_logger

log = get_logger(__name__)

EVOEVAL_SEMANTIC_SUBSETS: tuple[str, ...] = (
    "EvoEval_difficult",
    "EvoEval_creative",
    "EvoEval_subtle",
    "EvoEval_combine",
    "EvoEval_tool_use",
)
EVOEVAL_HF_PREFIX = "evoeval"


def _clean(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
    except Exception:
        pass
    return str(value)


def _jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except Exception:
                continue
    return default


def _contains_check_function(test_source: str) -> bool:
    return bool(re.search(r"def\s+check\s*\(\s*candidate\s*\)\s*:", test_source))


def adapt_reference_test(test_source: str, entry_point: str) -> str:
    """Convert reference tests to the runner's ``check(candidate)`` contract.

    HumanEval and EvoEval already provide ``check(candidate)`` and are returned
    unchanged. BigCodeBench commonly provides ``unittest.TestCase`` classes;
    those tests are executed in a namespace where the entry point is rebound to
    ``candidate``.
    """
    test_source = _clean(test_source).strip()
    if not test_source:
        return "def check(candidate):\n    return None\n"
    if _contains_check_function(test_source):
        return test_source + ("\n" if not test_source.endswith("\n") else "")

    quoted = repr(test_source)
    entry_quoted = repr(entry_point)
    return f'''def check(candidate):
    import unittest
    __cluse_test_source = {quoted}
    __cluse_namespace = {{
        "__name__": "__cluse_reference_tests__",
        {entry_quoted}: candidate,
    }}
    exec(__cluse_test_source, __cluse_namespace, __cluse_namespace)
    __cluse_cases = []
    for __cluse_obj in list(__cluse_namespace.values()):
        if isinstance(__cluse_obj, type):
            try:
                if issubclass(__cluse_obj, unittest.TestCase) and __cluse_obj is not unittest.TestCase:
                    __cluse_cases.append(__cluse_obj)
            except TypeError:
                pass
    if not __cluse_cases:
        raise AssertionError("Reference test source did not define a unittest.TestCase or check(candidate).")
    __cluse_suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in __cluse_cases
    )
    __cluse_result = unittest.TestResult()
    __cluse_suite.run(__cluse_result)
    if not __cluse_result.wasSuccessful():
        __cluse_details = [str(item[1]) for item in (__cluse_result.failures + __cluse_result.errors)]
        raise AssertionError(" | ".join(__cluse_details[:5]))
'''


def _compose_source(source_prompt: str, canonical_solution: str, entry_point: str) -> str:
    prompt = _clean(source_prompt)
    solution = _clean(canonical_solution)

    if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", solution):
        return solution.rstrip() + "\n"
    if not prompt:
        return solution.rstrip() + "\n"
    if not solution:
        return prompt.rstrip() + "\n"

    first_nonempty = next((line for line in solution.splitlines() if line.strip()), "")
    if (
        re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", prompt)
        and first_nonempty
        and not first_nonempty.startswith((" ", "\t"))
    ):
        solution = textwrap.indent(solution, "    ")

    separator = "" if prompt.endswith("\n") else "\n"
    return (prompt + separator + solution).rstrip() + "\n"


def normalize_evoeval_subset(value: Any) -> str:
    text = _clean(value).strip()
    if not text:
        return ""
    lower = text.lower().replace("-", "_").replace(" ", "_")
    if lower.startswith("evoeval_"):
        lower = lower[len("evoeval_"):]
    aliases = {
        "tooluse": "tool_use",
        "tool": "tool_use",
        "difficult": "difficult",
        "creative": "creative",
        "subtle": "subtle",
        "combine": "combine",
        "tool_use": "tool_use",
    }
    return aliases.get(lower, lower)


def evoeval_unique_task_id(source_task_id: str, subset: str) -> str:
    subset = normalize_evoeval_subset(subset) or "unknown"
    source = _clean(source_task_id).strip() or "EvoEval/unknown"
    suffix = source.split("/", 1)[1] if "/" in source else source
    return f"EvoEval_{subset}/{suffix}"


def infer_humaneval_parent(task_id: str) -> str:
    text = _clean(task_id)
    match = re.search(r"(?:EvoEval|HumanEval)[_A-Za-z-]*/(\d+)$", text)
    if not match:
        match = re.search(r"/(\d+)$", text)
    return f"HumanEval/{match.group(1)}" if match else ""


@dataclass
class Problem:
    """Normalized coding-benchmark task used by the full pipeline."""

    task_id: str
    source_prompt: str
    canonical_solution: str
    official_test: str
    entry_point: str
    instruction_prompt: str = ""
    code_prompt: str = ""
    complete_prompt: str = ""
    doc_struct: Dict[str, Any] = field(default_factory=dict)
    libs: List[str] = field(default_factory=list)
    dataset_name: str = "unknown"
    dataset_subset: str = ""
    source_task_id: str = ""
    parent_task_id: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def prompt(self) -> str:
        return self.source_prompt

    @property
    def prompt_text(self) -> str:
        """Compact natural-language specification supplied to LLM layers."""
        if self.instruction_prompt.strip():
            return self.instruction_prompt.strip()
        try:
            tree = ast.parse(self.source_prompt)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == self.entry_point:
                    doc = ast.get_docstring(node, clean=True)
                    if doc:
                        return doc.strip()
        except Exception:
            pass
        return self.source_prompt.strip()

    @property
    def safe_id(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.task_id).strip("_") or "task"

    @property
    def complete_source(self) -> str:
        return _compose_source(self.source_prompt, self.canonical_solution, self.entry_point)

    def guess_param_types(self) -> List[str]:
        match = re.search(r"def\s+" + re.escape(self.entry_point) + r"\s*\(([^)]*)\)", self.source_prompt)
        if not match:
            return []
        params = match.group(1).strip()
        if not params:
            return []
        parts: List[str] = []
        depth = 0
        current = ""
        for ch in params:
            if ch in "[(":
                depth += 1
            elif ch in ")]":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())
        return [p.split(":", 1)[1].split("=", 1)[0].strip() if ":" in p else "Any" for p in parts]

    def missing_libraries(self) -> List[str]:
        """Return declared import modules unavailable in the current runtime."""
        missing: List[str] = []
        aliases = {"sklearn": "sklearn", "cv2": "cv2", "PIL": "PIL", "yaml": "yaml"}
        for library in self.libs:
            name = str(library).strip()
            if not name:
                continue
            root = aliases.get(name, name).split(".", 1)[0]
            try:
                available = importlib.util.find_spec(root) is not None
            except (ImportError, AttributeError, ValueError):
                available = False
            if not available:
                missing.append(name)
        return sorted(set(missing))

    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_subset": self.dataset_subset,
            "source_task_id": self.source_task_id or self.task_id,
            "parent_task_id": self.parent_task_id,
            "has_instruction_prompt": bool(self.instruction_prompt),
            "has_code_prompt": bool(self.code_prompt),
            "has_doc_struct": bool(self.doc_struct),
            "libraries": list(self.libs),
        }


def _infer_dataset_name(task_id: str, columns: Iterable[str], requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    cols = set(columns)
    lower_id = task_id.lower()
    if "evoeval_subset" in cols or lower_id.startswith("evoeval/") or lower_id.startswith("evoeval_"):
        return "evoeval"
    if "complete_prompt" in cols or "instruct_prompt" in cols or lower_id.startswith("bigcodebench/"):
        return "bigcodebench"
    if lower_id.startswith("humaneval/") or "prompt" in cols:
        return "humaneval"
    return "generic"


def _row_to_problem(row: Dict[str, Any], dataset_type: str = "auto") -> Problem:
    columns = set(row)
    raw_task_id = _clean(row.get("task_id") or row.get("id") or row.get("problem_id"))
    if not raw_task_id:
        raise ValueError("Dataset row is missing task_id/id/problem_id")

    entry_point = _clean(row.get("entry_point") or row.get("function_name") or row.get("target"))
    complete_prompt = _clean(row.get("complete_prompt"))
    code_prompt = _clean(row.get("code_prompt"))
    source_prompt = _clean(row.get("prompt")) or complete_prompt or code_prompt
    canonical_solution = _clean(row.get("canonical_solution") or row.get("solution") or row.get("reference_solution"))
    raw_test = _clean(row.get("test") or row.get("tests") or row.get("official_test"))

    if not entry_point:
        match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", source_prompt or canonical_solution)
        if match:
            entry_point = match.group(1)
    if not entry_point:
        raise ValueError(f"Task {raw_task_id} is missing entry_point and no function definition could be inferred")
    if not source_prompt and not canonical_solution:
        raise ValueError(f"Task {raw_task_id} has neither a source prompt nor canonical solution")

    name = _infer_dataset_name(raw_task_id, columns, dataset_type)
    subset = normalize_evoeval_subset(
        row.get("evoeval_subset") or row.get("dataset_subset") or row.get("subset")
    )
    source_task_id = _clean(row.get("source_task_id")) or raw_task_id
    task_id = raw_task_id
    parent_task_id = _clean(row.get("parent_task_id"))
    if name == "evoeval":
        if not subset:
            match = re.match(r"EvoEval_([^/]+)/", raw_task_id, flags=re.IGNORECASE)
            subset = normalize_evoeval_subset(match.group(1)) if match else "unknown"
        if raw_task_id.lower().startswith("evoeval/"):
            task_id = evoeval_unique_task_id(raw_task_id, subset)
        if not parent_task_id:
            parent_task_id = infer_humaneval_parent(source_task_id)

    return Problem(
        task_id=task_id,
        source_prompt=source_prompt,
        canonical_solution=canonical_solution,
        official_test=adapt_reference_test(raw_test, entry_point),
        entry_point=entry_point,
        instruction_prompt=_clean(row.get("instruct_prompt") or row.get("instruction") or row.get("description")),
        code_prompt=code_prompt,
        complete_prompt=complete_prompt,
        doc_struct=_jsonish(row.get("doc_struct"), {}),
        libs=list(_jsonish(row.get("libs"), [])),
        dataset_name=name,
        dataset_subset=subset,
        source_task_id=source_task_id,
        parent_task_id=parent_task_id,
        raw=row,
    )


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL line {line_no} is not an object")
                    rows.append(value)
        return rows
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, dict):
            value = value.get("data") or value.get("tasks") or value.get("problems") or [value]
        if not isinstance(value, list):
            raise ValueError("JSON dataset must contain a list of task objects")
        return [dict(x) for x in value]

    import pandas as pd

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError("Reading parquet requires pyarrow or fastparquet. Run: pip install pyarrow") from exc
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}. Use parquet, csv, json, or jsonl.")
    return df.to_dict(orient="records")


def _normalize_rows(rows: Sequence[Dict[str, Any]], dataset_type: str, source_label: str) -> List[Problem]:
    problems: List[Problem] = []
    errors: List[str] = []
    for index, row in enumerate(rows):
        try:
            problems.append(_row_to_problem(dict(row), dataset_type=dataset_type))
        except Exception as exc:
            errors.append(f"row {index}: {type(exc).__name__}: {exc}")
    if errors:
        raise ValueError(f"Failed to normalize {len(errors)}/{len(rows)} rows from {source_label}. {' | '.join(errors[:5])}")
    duplicate_ids = sorted({p.task_id for p in problems if sum(q.task_id == p.task_id for q in problems) > 1})
    if duplicate_ids:
        raise ValueError(f"Normalized task IDs are not unique in {source_label}: {duplicate_ids[:5]}")
    return problems


def load_huggingface_dataset(dataset_id: str, split: str = "test", dataset_type: str = "auto") -> List[Problem]:
    """Load one Hugging Face dataset directly into normalized tasks."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise ImportError("Direct Hugging Face loading requires: pip install datasets") from exc
    dataset = hf_load_dataset(dataset_id, split=split)
    rows = [dict(row) for row in dataset]
    if dataset_id.lower().startswith("evoeval/evoeval_"):
        subset = normalize_evoeval_subset(dataset_id.rsplit("/", 1)[-1])
        for row in rows:
            source_id = _clean(row.get("task_id"))
            row["source_task_id"] = source_id
            row["task_id"] = evoeval_unique_task_id(source_id, subset)
            row["evoeval_subset"] = subset
            row["parent_task_id"] = infer_humaneval_parent(source_id)
        dataset_type = "evoeval"
    problems = _normalize_rows(rows, dataset_type, f"Hugging Face {dataset_id}[{split}]")
    names = sorted({p.dataset_name for p in problems})
    log.info("Loaded %d normalized problems from Hugging Face %s[%s] (dataset=%s)", len(problems), dataset_id, split, ",".join(names))
    return problems


def load_evoeval_semantic_dataset(
    split: str = "test",
    subsets: Optional[Sequence[str]] = None,
) -> List[Problem]:
    """Load and combine the five semantic-altering EvoEval subsets (500 tasks).

    The original ``task_id`` values repeat across subsets.  Each row is therefore
    annotated with a unique subset-qualified ID while preserving ``source_task_id``
    and an inferred ``parent_task_id`` for parent-clustered analysis.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise ImportError("Direct Hugging Face loading requires: pip install datasets") from exc

    requested = list(subsets or EVOEVAL_SEMANTIC_SUBSETS)
    rows: List[Dict[str, Any]] = []
    for subset_repo in requested:
        repo_name = subset_repo if subset_repo.startswith("EvoEval_") else f"EvoEval_{normalize_evoeval_subset(subset_repo)}"
        dataset_id = f"{EVOEVAL_HF_PREFIX}/{repo_name}"
        ds = hf_load_dataset(dataset_id, split=split)
        subset = normalize_evoeval_subset(repo_name)
        for item in ds:
            row = dict(item)
            source_id = _clean(row.get("task_id"))
            row["source_task_id"] = source_id
            row["task_id"] = evoeval_unique_task_id(source_id, subset)
            row["evoeval_subset"] = subset
            row["parent_task_id"] = infer_humaneval_parent(source_id)
            rows.append(row)
        log.info("Loaded EvoEval subset %s with %d tasks", subset, len(ds))

    problems = _normalize_rows(rows, "evoeval", "EvoEval semantic suite")
    subset_counts: Dict[str, int] = defaultdict(int)
    for problem in problems:
        subset_counts[problem.dataset_subset] += 1
    log.info("Loaded %d EvoEval semantic tasks across subsets: %s", len(problems), dict(sorted(subset_counts.items())))
    return problems


def load_dataset(dataset_path: Path, dataset_type: str = "auto") -> List[Problem]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    rows = _read_rows(dataset_path)
    problems = _normalize_rows(rows, dataset_type, str(dataset_path))
    names = sorted({p.dataset_name for p in problems})
    log.info("Loaded %d normalized problems from %s (dataset=%s)", len(problems), dataset_path, ",".join(names))
    return problems


def _stratified_indices(
    all_problems: Sequence[Problem],
    limit: int,
    seed: int,
    stratify_by: str,
) -> List[int]:
    import random

    groups: Dict[str, List[int]] = defaultdict(list)
    for index, problem in enumerate(all_problems):
        value = getattr(problem, stratify_by, "") or "unknown"
        groups[str(value)].append(index)
    keys = sorted(groups)
    if not keys:
        return list(range(min(limit, len(all_problems))))

    rng = random.Random(seed)
    for key in keys:
        rng.shuffle(groups[key])

    allocation = {key: 0 for key in keys}
    base, remainder = divmod(limit, len(keys))
    for position, key in enumerate(keys):
        allocation[key] = min(len(groups[key]), base + (1 if position < remainder else 0))

    assigned = sum(allocation.values())
    while assigned < limit:
        changed = False
        for key in keys:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                assigned += 1
                changed = True
                if assigned >= limit:
                    break
        if not changed:
            break

    selected: List[int] = []
    for key in keys:
        selected.extend(groups[key][: allocation[key]])
    return sorted(selected)


def select_problems(
    all_problems: List[Problem],
    indices: Optional[List[int]] = None,
    limit: Optional[int] = None,
    percent: float = 0.0,
    sample_mode: str = "first",
    seed: int = 42,
    stratify_by: str = "dataset_subset",
) -> List[Problem]:
    """Select a deterministic first, random, or stratified subset."""
    import random

    if indices is not None:
        return [all_problems[i] for i in indices if 0 <= i < len(all_problems)]
    n = len(all_problems)
    if percent and percent > 0:
        limit = max(1, int(math.ceil(n * min(percent, 1.0))))
    elif limit is None:
        return list(all_problems)
    limit = min(int(limit), n)

    if sample_mode == "stratified":
        selected_indices = _stratified_indices(all_problems, limit, seed, stratify_by)
        return [all_problems[i] for i in selected_indices]
    if sample_mode == "random":
        rng = random.Random(seed)
        return [all_problems[i] for i in sorted(rng.sample(range(n), limit))]
    return all_problems[:limit]
