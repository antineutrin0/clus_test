# Evaluation dataset

The pipeline consumes a single pre-built Parquet file of the **500 semantic-altering
EvoEval tasks**, already generated and included as part of this project's data
(no build step is required or included in this repository):

- `EvoEval_difficult` (100 tasks)
- `EvoEval_creative` (100 tasks)
- `EvoEval_subtle` (100 tasks)
- `EvoEval_combine` (100 tasks)
- `EvoEval_tool_use` (100 tasks)

Schema (columns present in the Parquet file):

| column               | type   | notes                                                        |
|----------------------|--------|---------------------------------------------------------------|
| `task_id`            | object | unique, e.g. `EvoEval_difficult/0`                            |
| `prompt`             | object | HumanEval-style function signature + docstring                |
| `canonical_solution` | object | reference implementation body                                 |
| `entry_point`        | object | function name under test                                      |
| `test`               | object | original EvoEval-provided test scaffold                       |
| `source_task_id`     | object | e.g. `EvoEval/0` (100 unique values -- one per parent HumanEval task, shared across the 5 subset variants) |
| `evoeval_subset`     | object | one of the 5 subset names above                                |
| `dataset_name`       | object | constant: `evoeval`                                            |
| `parent_task_id`     | object | originating HumanEval task, e.g. `HumanEval/0`                 |

No missing values in any column; 500 unique `task_id`s across 100 unique
`parent_task_id`s (5 subset variants each).

## Pointing the pipeline at the file

Set the path explicitly (recommended, avoids ambiguity):

```bash
export CLUSE_DATASET=/path/to/EvoEval_semantic_500.parquet
```

or place it at `data/evoeval/EvoEval_semantic_500.parquet` relative to the
project root -- `src/utils/config.discover_dataset()` checks that path (and a
couple of Kaggle-input fallbacks) automatically if `CLUSE_DATASET` is unset.

`src/utils/dataset_loader.py` only *reads* this file (Parquet/CSV/JSON/JSONL
are all supported); it does not build or regenerate the dataset. If you need
to rebuild it from the original EvoEval Hugging Face repositories for some
other project, that logic is no longer part of this repository -- only the
already-built 500-task file is expected here.
