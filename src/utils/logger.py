"""
logger.py — Structured logging for CLUSE-Test.
Compatible with Python 3.8+.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_logger(name, log_dir=None):
    # type: (str, Optional[Path]) -> logging.Logger
    """
    Return a configured logger. Safe to call multiple times — handlers
    are not duplicated.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File — DEBUG and above, only when log_dir given
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            str(log_dir / ("cluse_test_" + ts + ".log")),
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def should_trace_problem(problem_id: str) -> bool:
    """Return True only for the selected research-example problem."""
    import os
    target = os.environ.get("TRACE_PROBLEM_ID", "").strip()
    return bool(target) and problem_id == target


def save_llm_trace(results_dir, problem_id, layer, prompt, response, metadata=None):
    """Save detailed LLM interaction only for TRACE_PROBLEM_ID."""
    if not should_trace_problem(problem_id):
        return
    from pathlib import Path
    import json
    d = Path(results_dir) / "traces" / str(problem_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{layer}_prompt.txt").write_text(prompt or "", encoding="utf-8")
    (d / f"{layer}_response.txt").write_text(response or "", encoding="utf-8")
    if metadata:
        (d / f"{layer}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
