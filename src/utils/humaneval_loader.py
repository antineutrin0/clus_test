"""Backward-compatible imports for the dataset-agnostic loader.

New code should import from :mod:`src.utils.dataset_loader`.
"""
from src.utils.dataset_loader import Problem, adapt_reference_test, load_dataset, select_problems


def load_humaneval(parquet_path):
    """Compatibility alias; auto-detects HumanEval or BigCodeBench schemas."""
    return load_dataset(parquet_path, dataset_type="auto")


__all__ = ["Problem", "adapt_reference_test", "load_dataset", "load_humaneval", "select_problems"]
