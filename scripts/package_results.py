"""Zip a CLUSE-Test results directory, including figures and response artifacts."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()
results = args.results.resolve()
if not results.exists():
    raise FileNotFoundError(results)
base = (args.output or results).with_suffix("")
archive = shutil.make_archive(str(base), "zip", root_dir=results)
print(archive)
