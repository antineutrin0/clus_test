"""Paper-aligned statistical analysis and publication figure generation.

The analysis keeps four questions separate:
1. effectiveness: raw and conservative equivalent-adjusted mutation score;
2. reference agreement: overlap with benchmark-provided tests;
3. efficiency: tokens, paid tokens, cost, calls, and runtime per new kill;
4. fairness: fixed-iteration baseline versus plateau and matched-budget views.

No composite score is manufactured from these incompatible dimensions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        log.exception("Could not read CSV %s", path)
        return pd.DataFrame()


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _num(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(np.full(len(df), default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _mean_ci(values: Sequence[float], rng: np.random.Generator, n_boot: int) -> Tuple[float, float, float]:
    arr = _finite(values)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    estimate = float(arr.mean())
    if arr.size == 1 or n_boot <= 0:
        return estimate, estimate, estimate
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return estimate, float(low), float(high)


def _clustered_mean_ci(
    values: Sequence[float],
    clusters: Sequence[object],
    rng: np.random.Generator,
    n_boot: int,
) -> Tuple[float, float, float, int]:
    """Bootstrap a mean by resampling independent cluster-level means.

    EvoEval contains several evolved tasks for each HumanEval parent. Treating
    all derived tasks as independent would make confidence intervals too narrow.
    """
    frame = pd.DataFrame({"value": values, "cluster": clusters})
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    frame["cluster"] = frame["cluster"].fillna("").astype(str)
    frame.loc[frame["cluster"].eq(""), "cluster"] = [f"row-{i}" for i in frame.index[frame["cluster"].eq("")]]
    cluster_means = frame.groupby("cluster", sort=True)["value"].mean().to_numpy(dtype=float)
    estimate = float(cluster_means.mean()) if cluster_means.size else 0.0
    if cluster_means.size <= 1 or n_boot <= 0:
        return estimate, estimate, estimate, int(cluster_means.size)
    idx = rng.integers(0, cluster_means.size, size=(n_boot, cluster_means.size))
    boot = cluster_means[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return estimate, float(low), float(high), int(cluster_means.size)


def _ratio_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    rng: np.random.Generator,
    n_boot: int,
) -> Tuple[float, float, float]:
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den >= 0)
    num, den = num[mask], den[mask]
    estimate = _safe_div(float(num.sum()), float(den.sum()))
    if num.size <= 1 or n_boot <= 0:
        return estimate, estimate, estimate
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    boot_num = num[idx].sum(axis=1)
    boot_den = den[idx].sum(axis=1)
    boot = np.divide(boot_num, boot_den, out=np.zeros_like(boot_num), where=boot_den != 0)
    low, high = np.percentile(boot, [2.5, 97.5])
    return estimate, float(low), float(high)


def _two_sided_sign_test(differences: Sequence[float], tolerance: float = 1e-12) -> Dict[str, float]:
    diff = _finite(differences)
    wins = int(np.sum(diff > tolerance))
    losses = int(np.sum(diff < -tolerance))
    ties = int(diff.size - wins - losses)
    n = wins + losses
    if n == 0:
        p_value = 1.0
    else:
        k = min(wins, losses)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        p_value = min(1.0, 2.0 * tail)
    return {"wins": wins, "ties": ties, "losses": losses, "n_non_ties": n, "p_value": float(p_value)}


def _paired_effect_sizes(differences: Sequence[float]) -> Dict[str, object]:
    diff = _finite(differences)
    nonzero = diff[np.abs(diff) > 1e-12]
    if diff.size < 2 or float(diff.std(ddof=1)) <= 1e-12:
        cohen_dz = None
        undefined = True
    else:
        cohen_dz = round(float(diff.mean() / diff.std(ddof=1)), 6)
        undefined = False
    if nonzero.size == 0:
        rank_biserial = 0.0
    else:
        ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy(dtype=float)
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = _safe_div(positive - negative, positive + negative)
    return {
        "cohen_dz": cohen_dz,
        "cohen_dz_undefined_zero_variance": undefined,
        "rank_biserial": round(float(rank_biserial), 6),
    }


def _load_metric_payloads(metrics_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(Path(metrics_dir).glob("*_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_path"] = str(path)
            rows.append(payload)
        except Exception:
            log.exception("Could not read metrics file %s", path)
    return rows


def _layer_dataframe(payloads: Sequence[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for payload in payloads:
        for layer in payload.get("layers") or []:
            rows.append({"problem_id": payload.get("problem_id"), **layer})
    return pd.DataFrame(rows)


def _call_dataframe(proposed: Sequence[Dict], baseline: Sequence[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for method, payloads in (("proposed", proposed), ("baseline", baseline)):
        for payload in payloads:
            for call in payload.get("llm_calls") or []:
                rows.append({"method": method, "problem_id": payload.get("problem_id"), **call})
    return pd.DataFrame(rows)


def _timing_dataframe(payloads: Sequence[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for payload in payloads:
        timing = payload.get("timing") or {}
        rows.append({"problem_id": payload.get("problem_id"), **timing})
    return pd.DataFrame(rows)


def _cluster_dataframe(payloads: Sequence[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for payload in payloads:
        metadata = payload.get("metadata") or {}
        clusters = metadata.get("layer1_clusters") or []
        mutants = int(metadata.get("mutant_count") or 0)
        reps = sum(len(item.get("representative_ids", [])) for item in clusters)
        rows.append({
            "problem_id": payload.get("problem_id"),
            "total_mutants": mutants,
            "num_clusters": len(clusters),
            "representatives": reps,
            "compression_ratio_mutants_per_representative": _safe_div(mutants, reps),
            "mean_cluster_size": _safe_div(mutants, len(clusters)),
            "layer1_max_calls": int(metadata.get("layer1_max_calls_per_problem") or 0),
            "layer1_actual_calls": int(metadata.get("layer1_iterations") or 0),
            "layer2_actual_calls": int(metadata.get("layer2_iterations") or 0),
            "layer3_actual_calls": int(metadata.get("layer3_iterations") or 0),
        })
    return pd.DataFrame(rows)


def _baseline_history_dataframe(payloads: Sequence[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for payload in payloads:
        metadata = payload.get("metadata") or {}
        for item in metadata.get("iteration_history") or []:
            rows.append({"problem_id": payload.get("problem_id"), **item})
    return pd.DataFrame(rows)


def _operator_dataframe(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in sorted(results_dir.glob("*/final_mutants.json")):
        try:
            mutants = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for mutant in mutants:
            status = str(mutant.get("status") or "").upper()
            rows.append({
                "problem_id": mutant.get("problem_id") or path.parent.name,
                "mutant_id": mutant.get("mutant_id"),
                "operator": mutant.get("operator") or "UNKNOWN",
                "killed": int(status == "KILLED"),
                "status": status,
                "equivalence_status": mutant.get("equivalence_status", "UNKNOWN"),
                "equivalence_reason": mutant.get("equivalence_reason", ""),
            })
    return pd.DataFrame(rows)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_figure(fig, figures_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    if config.SAVE_FIGURE_PDF:
        fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_score_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    columns = [c for c in ("proposed_score", "baseline_score", "official_score") if c in df]
    data = [_num(df, c).to_numpy() for c in columns]
    if not columns or not any(len(x) for x in data):
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, tick_labels=[c.replace("_score", "").title() for c in columns], showmeans=True)
    ax.set_ylabel("Mutation score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Task-level mutation score distribution")
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "mutation_score_comparison")


def _aggregate_score_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    total = float(_num(comparison, "total_mutants").sum())
    equivalent = float(_num(comparison, "equivalent_mutants").sum())
    adjusted_total = max(0.0, total - equivalent)
    rows: List[Dict] = []
    for label, prefix in (("Proposed", "proposed"), ("Baseline", "baseline"), ("Official", "official")):
        raw = _num(comparison, f"{prefix}_score")
        killed = _num(comparison, f"{prefix}_killed")
        adjusted = _num(comparison, f"{prefix}_equivalent_adjusted_score")
        adjusted_killed = adjusted * np.maximum(0.0, _num(comparison, "total_mutants") - _num(comparison, "equivalent_mutants"))
        rows.append({
            "method": label,
            "macro_raw": float(raw.mean()) if len(raw) else 0.0,
            "micro_raw": _safe_div(float(killed.sum()), total),
            "macro_equivalent_adjusted": float(adjusted.mean()) if len(adjusted) else 0.0,
            "micro_equivalent_adjusted": _safe_div(float(adjusted_killed.sum()), adjusted_total),
        })
    return pd.DataFrame(rows)


def _plot_accuracy_aggregate(score_df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if score_df.empty:
        return
    metrics = ["macro_raw", "micro_raw", "macro_equivalent_adjusted", "micro_equivalent_adjusted"]
    labels = ["Macro raw", "Micro raw", "Macro adjusted", "Micro adjusted"]
    x = np.arange(len(metrics), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, (_, row) in enumerate(score_df.iterrows()):
        ax.bar(x + (offset - 1) * width, [float(row[m]) for m in metrics], width, label=row["method"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mutation score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Raw and conservative equivalent-adjusted effectiveness")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "accuracy_comparison")


def _plot_paired_delta(df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if "score_delta_proposed_minus_baseline" not in df:
        return
    values = _num(df, "score_delta_proposed_minus_baseline").sort_values().to_numpy()
    if not len(values):
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(values)), values)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Tasks sorted by score difference")
    ax.set_ylabel("Proposed − baseline score")
    ax.set_title("Paired effectiveness difference")
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "paired_score_delta")


def _plot_efficiency(df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    needed = {"proposed_total_tokens", "baseline_total_tokens", "proposed_score", "baseline_score"}
    if not needed.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(_num(df, "proposed_total_tokens"), _num(df, "proposed_score"), label="Proposed", alpha=0.75)
    ax.scatter(_num(df, "baseline_total_tokens"), _num(df, "baseline_score"), label="Baseline", marker="x", alpha=0.75)
    ax.set_xlabel("Total LLM tokens per task")
    ax.set_ylabel("Mutation score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Effectiveness–token trade-off")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_figure(fig, figures_dir, "effectiveness_token_efficiency")


def _plot_token_cost(comparison: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if comparison.empty:
        return
    labels = ["Proposed", "Baseline fixed", "Baseline first-final", "Baseline plateau"]
    values = [
        float(_num(comparison, "proposed_total_tokens").sum()),
        float(_num(comparison, "baseline_total_tokens").sum()),
        float(_num(comparison, "baseline_tokens_at_first_final_score").sum()),
        float(_num(comparison, "baseline_tokens_at_plateau").sum()),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values)
    ax.set_ylabel("Aggregate LLM tokens")
    ax.set_title("Token consumption under fixed and fair stopping views")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "token_cost_comparison")


def _plot_baseline_fairness(comparison: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if comparison.empty:
        return
    labels = ["Proposed final", "Baseline fixed", "Baseline @ proposed tokens", "Baseline @ proposed paid tokens"]
    values = [
        float(_num(comparison, "proposed_score").mean()),
        float(_num(comparison, "baseline_score").mean()),
        float(_num(comparison, "baseline_score_at_matched_proposed_tokens").mean()),
        float(_num(comparison, "baseline_score_at_matched_proposed_paid_tokens").mean()),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values)
    ax.set_ylabel("Macro mutation score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline fairness: fixed-run and matched-budget comparisons")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "baseline_fairness")


def _plot_layer_gains(layer_summary: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if layer_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.bar(layer_summary["layer"], layer_summary["mean_new_kills"])
    ax.set_xlabel("Pipeline layer")
    ax.set_ylabel("Mean marginal mutant kills per task")
    ax.set_title("Marginal effectiveness of each layer")
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "layer_marginal_kills")


def _plot_layer_cost(layer_summary: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if layer_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(layer_summary["layer"], layer_summary["tokens_per_new_kill"])
    ax.set_xlabel("Pipeline layer")
    ax.set_ylabel("Tokens per new killed mutant")
    ax.set_title("Marginal token efficiency by layer")
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "layer_cost_efficiency")


def _plot_runtime_breakdown(layer_summary: pd.DataFrame, timing_df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels: List[str] = []
    values: List[float] = []
    if not timing_df.empty:
        mapping = [
            ("Mutation", "mutant_generation_sec"),
            ("Probes", "probe_generation_sec"),
            ("Signatures", "behavior_signature_sec"),
            ("Cluster/select", "clustering_sec"),
        ]
        for label, col in mapping:
            if col in timing_df:
                labels.append(label)
                values.append(float(_num(timing_df, col).sum()))
    if not layer_summary.empty:
        for _, row in layer_summary.iterrows():
            labels.append(str(row["layer"]))
            values.append(float(row["total_runtime_sec"]))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.05), 4.8))
    ax.bar(labels, values)
    ax.set_ylabel("Aggregate runtime (seconds)")
    ax.set_title("Proposed-pipeline runtime breakdown")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "runtime_breakdown")


def _plot_call_outcomes(call_summary: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if call_summary.empty:
        return
    pivot = call_summary.pivot(index="method", columns="outcome", values="calls").fillna(0)
    outcomes = [c for c in ("PRODUCTIVE", "VALID_ZERO_KILL", "REJECTED", "ERROR") if c in pivot]
    if not outcomes:
        return
    x = np.arange(len(pivot.index), dtype=float)
    width = 0.8 / len(outcomes)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, outcome in enumerate(outcomes):
        ax.bar(x + (i - (len(outcomes) - 1) / 2) * width, pivot[outcome].to_numpy(), width, label=outcome)
    ax.set_xticks(x, [str(v).title() for v in pivot.index])
    ax.set_ylabel("LLM calls")
    ax.set_title("Generated-test validation outcomes")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "call_validation_outcomes")


def _plot_prompt_compression(layer_summary: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if layer_summary.empty or "mean_prompt_chars_per_call" not in layer_summary:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(layer_summary["layer"], layer_summary["mean_prompt_chars_per_call"])
    ax.set_ylabel("Mean prompt characters per call")
    ax.set_title("Bounded context size by proposed layer")
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "prompt_compression")


def _plot_operator(operator_summary: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if operator_summary.empty:
        return
    data = operator_summary.sort_values("kill_rate")
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(data))))
    ax.barh(data["operator"], data["kill_rate"])
    ax.set_xlabel("Kill rate")
    ax.set_xlim(0, 1.05)
    ax.set_title("Mutation difficulty by operator")
    ax.grid(axis="x", alpha=0.3)
    _save_figure(fig, figures_dir, "operator_kill_rate")


def _plot_compression(cluster_df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if cluster_df.empty:
        return
    values = _num(cluster_df, "compression_ratio_mutants_per_representative")
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=min(12, max(3, int(math.sqrt(len(values))))))
    ax.set_xlabel("Mutants per representative")
    ax.set_ylabel("Tasks")
    ax.set_title("Cluster-aware feedback compression")
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "cluster_compression")


def _plot_llm_usage(summary: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    labels = summary["method"].astype(str) + "\n" + summary["provider"].astype(str) + ":" + summary["model"].astype(str)
    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 1.15), 4.8))
    ax.bar(labels, summary["total_tokens"])
    ax.set_ylabel("Total tokens")
    ax.set_title("LLM usage by method, provider, and model")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "llm_usage_by_provider_model")


def _plot_subset_effectiveness(subset_df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if subset_df.empty or "dataset_subset" not in subset_df:
        return
    methods = [m for m in ("proposed", "baseline", "official_reference") if m in set(subset_df.get("method", []))]
    subsets = sorted(subset_df["dataset_subset"].dropna().astype(str).unique())
    if not methods or not subsets:
        return
    x = np.arange(len(subsets), dtype=float)
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(8, len(subsets) * 1.45), 4.8))
    for position, method in enumerate(methods):
        frame = subset_df[subset_df["method"] == method].set_index("dataset_subset")
        values = [float(frame.loc[subset, "macro_mutation_score"]) if subset in frame.index else 0.0 for subset in subsets]
        offset = (position - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=method.replace("_", " ").title())
    ax.set_xticks(x, [s.replace("_", " ").title() for s in subsets])
    ax.set_ylabel("Macro mutation score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Effectiveness by EvoEval subset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "evoeval_subset_effectiveness")


def _plot_subset_efficiency(subset_df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if subset_df.empty or "dataset_subset" not in subset_df:
        return
    frame = subset_df[subset_df["method"].isin(["proposed", "baseline"])].copy()
    if frame.empty:
        return
    methods = [m for m in ("proposed", "baseline") if m in set(frame["method"])]
    subsets = sorted(frame["dataset_subset"].dropna().astype(str).unique())
    x = np.arange(len(subsets), dtype=float)
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(8, len(subsets) * 1.45), 4.8))
    for position, method in enumerate(methods):
        method_frame = frame[frame["method"] == method].set_index("dataset_subset")
        values = [float(method_frame.loc[subset, "tokens_per_killed_mutant"]) if subset in method_frame.index else 0.0 for subset in subsets]
        offset = (position - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=method.title())
    ax.set_xticks(x, [s.replace("_", " ").title() for s in subsets])
    ax.set_ylabel("Tokens per killed mutant")
    ax.set_title("Token efficiency by EvoEval subset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir, "evoeval_subset_token_efficiency")


def _build_subset_summary(aggregate: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    if not comparison.empty and "dataset_subset" in comparison:
        for subset, group in comparison.groupby("dataset_subset", dropna=False):
            subset = str(subset or "unknown")
            total = float(_num(group, "total_mutants").sum())
            for method in ("proposed", "baseline", "official"):
                score_col = f"{method}_score"
                killed_col = f"{method}_killed"
                if score_col not in group:
                    continue
                killed = float(_num(group, killed_col).sum())
                tokens = float(_num(group, f"{method}_total_tokens").sum()) if method != "official" else 0.0
                paid_tokens = float(_num(group, f"{method}_paid_tokens").sum()) if method != "official" else 0.0
                cost = float(_num(group, f"{method}_estimated_cost_usd").sum()) if method != "official" else 0.0
                runtime = float(_num(group, f"{method}_runtime_sec").sum()) if method != "official" else 0.0
                rows.append({
                    "dataset_subset": subset,
                    "method": "official_reference" if method == "official" else method,
                    "n_tasks": int(len(group)),
                    "total_mutants": int(total),
                    "killed_mutants": int(killed),
                    "macro_mutation_score": round(float(_num(group, score_col).mean()), 6),
                    "micro_mutation_score": round(_safe_div(killed, total), 6),
                    "total_tokens": int(tokens),
                    "total_paid_tokens": int(paid_tokens),
                    "total_cost_usd": round(cost, 8),
                    "total_runtime_sec": round(runtime, 4),
                    "tokens_per_killed_mutant": round(_safe_div(tokens, killed), 4),
                })
        return pd.DataFrame(rows)

    if not aggregate.empty and "dataset_subset" in aggregate:
        for subset, group in aggregate.groupby("dataset_subset", dropna=False):
            subset = str(subset or "unknown")
            total = float(_num(group, "total_mutants").sum())
            for method, score_col, killed_col in (
                ("proposed", "final_score", "killed_mutants"),
                ("official_reference", "official_score", "official_killed"),
            ):
                killed = float(_num(group, killed_col).sum())
                tokens = float(_num(group, "total_tokens").sum()) if method == "proposed" else 0.0
                rows.append({
                    "dataset_subset": subset, "method": method, "n_tasks": int(len(group)),
                    "total_mutants": int(total), "killed_mutants": int(killed),
                    "macro_mutation_score": round(float(_num(group, score_col).mean()), 6),
                    "micro_mutation_score": round(_safe_div(killed, total), 6),
                    "total_tokens": int(tokens),
                    "total_paid_tokens": int(_num(group, "paid_api_tokens").sum()) if method == "proposed" else 0,
                    "total_cost_usd": round(float(_num(group, "estimated_cost_usd").sum()), 8) if method == "proposed" else 0.0,
                    "total_runtime_sec": round(float(_num(group, "runtime_sec").sum()), 4) if method == "proposed" else 0.0,
                    "tokens_per_killed_mutant": round(_safe_div(tokens, killed), 4),
                })
    return pd.DataFrame(rows)


def generate_statistical_report(results_dir: Path, seed: int = 42, raw_dir: Optional[Path] = None) -> Dict:
    """Generate paper-aligned CSV tables, JSON/Markdown reports, and figures.

    `results_dir` is the curated report tier (aggregate CSVs, comparison
    summary, and where statistics/figures are written). `raw_dir` is the
    full per-problem diagnostic tier: `metrics/` (proposed-pipeline metrics,
    one `<task>_metrics.json` per problem), `metrics/baseline/` (same, for
    the non-clustering baseline), and per-problem `final_mutants.json` /
    `survived_mutants.json` -- defaults to `results_dir` for backward
    compatibility with a non-tiered layout.
    """
    results_dir = Path(results_dir)
    raw_dir = Path(raw_dir) if raw_dir is not None else results_dir
    stats_dir = results_dir / "statistics"
    figures_dir = results_dir / "figures"
    stats_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_boot = max(0, int(config.STATISTICS_BOOTSTRAP_SAMPLES))
    aggregate = _read_csv(results_dir / "aggregate_summary.csv")
    if not aggregate.empty and "skipped" in aggregate:
        skipped_mask = aggregate["skipped"].astype(str).str.lower().isin({"true", "1", "yes"})
        aggregate = aggregate.loc[~skipped_mask].copy()
    comparison = _read_csv(results_dir / "comparison" / "proposed_vs_baseline_summary.csv")
    proposed_payloads = _load_metric_payloads(raw_dir / "metrics")
    baseline_payloads = _load_metric_payloads(raw_dir / "metrics" / "baseline")
    layer_df = _layer_dataframe(proposed_payloads)
    call_df = _call_dataframe(proposed_payloads, baseline_payloads)
    timing_df = _timing_dataframe(proposed_payloads)
    cluster_df = _cluster_dataframe(proposed_payloads)
    history_df = _baseline_history_dataframe(baseline_payloads)
    operator_df = _operator_dataframe(raw_dir)

    report: Dict[str, object] = {
        "methodology": {
            "primary_effectiveness": "Raw macro and micro mutation score.",
            "equivalent_adjustment": "Conservative: excludes only mutants explicitly marked STATIC_EQUIVALENT.",
            "confidence_interval": f"Task-level bootstrap plus HumanEval-parent-clustered bootstrap for EvoEval, {n_boot} resamples, 95% percentile interval.",
            "paired_test": "Two-sided exact sign test on non-tied per-task proposed-minus-baseline score differences.",
            "effect_sizes": "Paired Cohen's dz and rank-biserial correlation.",
            "fairness": "Reports fixed-iteration baseline, first-final/plateau token use, and matched-token scores separately.",
            "warning": "Official-test agreement is an agreement measure, not semantic ground truth.",
        }
    }

    # Proposed and official effectiveness, including conservative adjustment.
    if not aggregate.empty:
        rows: List[Dict] = []
        total = _num(aggregate, "total_mutants")
        adjusted_total = np.maximum(0.0, total - _num(aggregate, "equivalent_mutants"))
        for method, score_col, killed_col, adjusted_col in (
            ("proposed", "final_score", "killed_mutants", "equivalent_adjusted_score"),
            ("official_reference", "official_score", "official_killed", "official_equivalent_adjusted_score"),
        ):
            scores = _num(aggregate, score_col)
            killed = _num(aggregate, killed_col)
            adjusted_scores = _num(aggregate, adjusted_col)
            mean, low, high = _mean_ci(scores, rng, n_boot)
            micro, micro_low, micro_high = _ratio_ci(killed, total, rng, n_boot)
            adj_mean, adj_low, adj_high = _mean_ci(adjusted_scores, rng, n_boot)
            adjusted_killed = adjusted_scores * adjusted_total
            adj_micro, adj_micro_low, adj_micro_high = _ratio_ci(adjusted_killed, adjusted_total, rng, n_boot)
            rows.append({
                "method": method,
                "n_tasks": int(len(scores)),
                "macro_raw": round(mean, 6),
                "macro_raw_ci95_low": round(low, 6),
                "macro_raw_ci95_high": round(high, 6),
                "micro_raw": round(micro, 6),
                "micro_raw_ci95_low": round(micro_low, 6),
                "micro_raw_ci95_high": round(micro_high, 6),
                "macro_equivalent_adjusted": round(adj_mean, 6),
                "macro_adjusted_ci95_low": round(adj_low, 6),
                "macro_adjusted_ci95_high": round(adj_high, 6),
                "micro_equivalent_adjusted": round(adj_micro, 6),
                "micro_adjusted_ci95_low": round(adj_micro_low, 6),
                "micro_adjusted_ci95_high": round(adj_micro_high, 6),
            })
        effect_df = pd.DataFrame(rows)
        _save_table(effect_df, stats_dir / "effectiveness_summary.csv")
        report["effectiveness"] = rows

        agreement_rows: List[Dict] = []
        for col in ("kill_agreement_accuracy", "kill_precision", "kill_recall", "kill_f1"):
            if col not in aggregate:
                continue
            mean, low, high = _mean_ci(_num(aggregate, col), rng, n_boot)
            agreement_rows.append({"metric": col, "mean": round(mean, 6), "ci95_low": round(low, 6), "ci95_high": round(high, 6)})
        if agreement_rows:
            _save_table(pd.DataFrame(agreement_rows), stats_dir / "official_agreement_summary.csv")
            report["official_test_agreement"] = agreement_rows

    # Proposed-vs-baseline paired analysis and fair efficiency views.
    if not comparison.empty and {"proposed_score", "baseline_score"}.issubset(comparison.columns):
        proposed_scores = _num(comparison, "proposed_score")
        baseline_scores = _num(comparison, "baseline_score")
        diff = (proposed_scores - baseline_scores).to_numpy()
        mean_delta, delta_low, delta_high = _mean_ci(diff, rng, n_boot)
        weights = _num(comparison, "total_mutants").to_numpy()
        paired = {
            "n_tasks": int(len(diff)),
            "mean_score_delta": round(mean_delta, 6),
            "score_delta_ci95_low": round(delta_low, 6),
            "score_delta_ci95_high": round(delta_high, 6),
            "mutant_weighted_score_delta": round(float(np.average(diff, weights=weights)) if len(diff) and weights.sum() else 0.0, 6),
            **_two_sided_sign_test(diff),
            **_paired_effect_sizes(diff),
        }
        if "parent_task_id" in comparison:
            clustered_mean, clustered_low, clustered_high, n_parents = _clustered_mean_ci(
                diff, comparison["parent_task_id"].tolist(), rng, n_boot
            )
            paired.update({
                "parent_clustered_n": n_parents,
                "parent_clustered_mean_delta": round(clustered_mean, 6),
                "parent_clustered_ci95_low": round(clustered_low, 6),
                "parent_clustered_ci95_high": round(clustered_high, 6),
            })
        _save_table(pd.DataFrame([paired]), stats_dir / "paired_comparison_summary.csv")
        report["paired_proposed_vs_baseline"] = paired

        efficiency_rows: List[Dict] = []
        for method in ("proposed", "baseline"):
            tokens = _num(comparison, f"{method}_total_tokens")
            paid_tokens = _num(comparison, f"{method}_paid_tokens")
            runtime = _num(comparison, f"{method}_runtime_sec")
            cost = _num(comparison, f"{method}_estimated_cost_usd")
            killed = _num(comparison, f"{method}_killed")
            calls = _num(comparison, f"{method}_llm_calls")
            efficiency_rows.append({
                "method": method,
                "total_tokens": int(tokens.sum()),
                "total_paid_tokens": int(paid_tokens.sum()),
                "total_llm_calls": int(calls.sum()),
                "total_runtime_sec": round(float(runtime.sum()), 4),
                "total_estimated_cost_usd": round(float(cost.sum()), 8),
                "total_killed_mutants": int(killed.sum()),
                "tokens_per_killed_mutant": round(_safe_div(float(tokens.sum()), float(killed.sum())), 4),
                "paid_tokens_per_killed_mutant": round(_safe_div(float(paid_tokens.sum()), float(killed.sum())), 4),
                "runtime_sec_per_killed_mutant": round(_safe_div(float(runtime.sum()), float(killed.sum())), 6),
                "cost_usd_per_killed_mutant": round(_safe_div(float(cost.sum()), float(killed.sum())), 8),
                "calls_per_task": round(_safe_div(float(calls.sum()), len(comparison)), 4),
            })
        efficiency_df = pd.DataFrame(efficiency_rows)
        _save_table(efficiency_df, stats_dir / "efficiency_summary.csv")
        report["efficiency"] = efficiency_rows

        fairness_cols = [
            "problem_id", "proposed_score", "baseline_score", "proposed_total_tokens", "baseline_total_tokens",
            "baseline_first_final_score_iteration", "baseline_tokens_at_first_final_score",
            "baseline_first_plateau_iteration", "baseline_tokens_at_plateau",
            "baseline_score_at_matched_proposed_tokens", "baseline_score_at_matched_proposed_paid_tokens",
        ]
        fairness_df = comparison[[c for c in fairness_cols if c in comparison]].copy()
        _save_table(fairness_df, stats_dir / "baseline_fairness_per_task.csv")
        report["baseline_fairness"] = {
            "fixed_baseline_tokens": int(_num(comparison, "baseline_total_tokens").sum()),
            "tokens_at_first_final_score": int(_num(comparison, "baseline_tokens_at_first_final_score").sum()),
            "tokens_at_first_plateau": int(_num(comparison, "baseline_tokens_at_plateau").sum()),
            "proposed_tokens": int(_num(comparison, "proposed_total_tokens").sum()),
            "macro_baseline_fixed_score": round(float(baseline_scores.mean()), 6),
            "macro_baseline_score_at_matched_proposed_tokens": round(float(_num(comparison, "baseline_score_at_matched_proposed_tokens").mean()), 6),
            "macro_baseline_score_at_matched_proposed_paid_tokens": round(float(_num(comparison, "baseline_score_at_matched_proposed_paid_tokens").mean()), 6),
        }

        score_df = _aggregate_score_rows(comparison)
        _save_table(score_df, stats_dir / "aggregate_score_conventions.csv")
        report["aggregate_score_conventions"] = score_df.to_dict(orient="records")
        _plot_score_distribution(comparison, figures_dir)
        _plot_accuracy_aggregate(score_df, figures_dir)
        _plot_paired_delta(comparison, figures_dir)
        _plot_efficiency(comparison, figures_dir)
        _plot_token_cost(comparison, figures_dir)
        _plot_baseline_fairness(comparison, figures_dir)
    elif not aggregate.empty:
        plot_df = aggregate.rename(columns={"final_score": "proposed_score"})
        _plot_score_distribution(plot_df, figures_dir)

    subset_summary = _build_subset_summary(aggregate, comparison)
    if not subset_summary.empty:
        _save_table(subset_summary, stats_dir / "evoeval_subset_summary.csv")
        report["evoeval_subset_summary"] = subset_summary.to_dict(orient="records")
        _plot_subset_effectiveness(subset_summary, figures_dir)
        _plot_subset_efficiency(subset_summary, figures_dir)

    if not comparison.empty and {"dataset_subset", "proposed_score", "baseline_score"}.issubset(comparison.columns):
        subset_paired_rows: List[Dict] = []
        for subset, group in comparison.groupby("dataset_subset", dropna=False):
            subset_diff = (_num(group, "proposed_score") - _num(group, "baseline_score")).to_numpy()
            mean, low, high = _mean_ci(subset_diff, rng, n_boot)
            subset_paired_rows.append({
                "dataset_subset": str(subset or "unknown"),
                "n_tasks": int(len(group)),
                "mean_score_delta": round(mean, 6),
                "ci95_low": round(low, 6),
                "ci95_high": round(high, 6),
                **_two_sided_sign_test(subset_diff),
                **_paired_effect_sizes(subset_diff),
            })
        subset_paired_df = pd.DataFrame(subset_paired_rows)
        _save_table(subset_paired_df, stats_dir / "evoeval_subset_paired_comparison.csv")
        report["evoeval_subset_paired_comparison"] = subset_paired_rows

    # Layer contribution and batching efficiency.
    layer_summary = pd.DataFrame()
    if not layer_df.empty:
        for col in (
            "new_kills", "killed_mutants", "total_mutants", "total_tokens", "total_time_sec",
            "estimated_cost_usd", "cluster_kill_consistency", "llm_calls", "productive_calls",
            "invalid_calls", "zero_kill_calls", "prompt_chars", "response_chars", "target_count",
        ):
            if col in layer_df:
                layer_df[col] = pd.to_numeric(layer_df[col], errors="coerce").fillna(0)
        layer_summary = layer_df.groupby("layer", as_index=False).agg(
            n_tasks=("problem_id", "nunique"),
            mean_new_kills=("new_kills", "mean"),
            total_new_kills=("new_kills", "sum"),
            mean_cumulative_score=("mutation_score", "mean"),
            total_tokens=("total_tokens", "sum"),
            total_runtime_sec=("total_time_sec", "sum"),
            total_cost_usd=("estimated_cost_usd", "sum"),
            total_llm_calls=("llm_calls", "sum"),
            productive_calls=("productive_calls", "sum"),
            invalid_calls=("invalid_calls", "sum"),
            zero_kill_calls=("zero_kill_calls", "sum"),
            total_prompt_chars=("prompt_chars", "sum"),
            total_targets=("target_count", "sum"),
            mean_cluster_kill_consistency=("cluster_kill_consistency", "mean"),
        )
        layer_summary["tokens_per_new_kill"] = np.divide(
            layer_summary["total_tokens"], layer_summary["total_new_kills"],
            out=np.zeros(len(layer_summary)), where=layer_summary["total_new_kills"].to_numpy() != 0,
        )
        layer_summary["runtime_sec_per_new_kill"] = np.divide(
            layer_summary["total_runtime_sec"], layer_summary["total_new_kills"],
            out=np.zeros(len(layer_summary)), where=layer_summary["total_new_kills"].to_numpy() != 0,
        )
        layer_summary["mean_prompt_chars_per_call"] = np.divide(
            layer_summary["total_prompt_chars"], layer_summary["total_llm_calls"],
            out=np.zeros(len(layer_summary)), where=layer_summary["total_llm_calls"].to_numpy() != 0,
        )
        layer_summary["targets_per_call"] = np.divide(
            layer_summary["total_targets"], layer_summary["total_llm_calls"],
            out=np.zeros(len(layer_summary)), where=layer_summary["total_llm_calls"].to_numpy() != 0,
        )
        layer_summary["productive_call_rate"] = np.divide(
            layer_summary["productive_calls"], layer_summary["total_llm_calls"],
            out=np.zeros(len(layer_summary)), where=layer_summary["total_llm_calls"].to_numpy() != 0,
        )
        _save_table(layer_summary, stats_dir / "layer_contribution_summary.csv")
        report["layer_contribution"] = layer_summary.to_dict(orient="records")
        _plot_layer_gains(layer_summary, figures_dir)
        _plot_layer_cost(layer_summary, figures_dir)
        _plot_prompt_compression(layer_summary, figures_dir)

    _plot_runtime_breakdown(layer_summary, timing_df, figures_dir)
    if not timing_df.empty:
        _save_table(timing_df, stats_dir / "runtime_breakdown_per_task.csv")

    if not cluster_df.empty:
        _save_table(cluster_df, stats_dir / "cluster_compression_per_task.csv")
        cluster_summary = {
            "mean_mutants_per_representative": round(float(_num(cluster_df, "compression_ratio_mutants_per_representative").mean()), 6),
            "median_mutants_per_representative": round(float(_num(cluster_df, "compression_ratio_mutants_per_representative").median()), 6),
            "total_mutants": int(_num(cluster_df, "total_mutants").sum()),
            "total_representatives": int(_num(cluster_df, "representatives").sum()),
            "global_compression_ratio": round(_safe_div(float(_num(cluster_df, "total_mutants").sum()), float(_num(cluster_df, "representatives").sum())), 6),
            "mean_layer1_calls_per_task": round(float(_num(cluster_df, "layer1_actual_calls").mean()), 6),
            "maximum_observed_layer1_calls": int(_num(cluster_df, "layer1_actual_calls").max()) if len(cluster_df) else 0,
        }
        report["cluster_compression"] = cluster_summary
        _plot_compression(cluster_df, figures_dir)

    if not operator_df.empty:
        operator_summary = operator_df.groupby("operator", as_index=False).agg(total_mutants=("killed", "size"), killed_mutants=("killed", "sum"))
        operator_summary["surviving_mutants"] = operator_summary["total_mutants"] - operator_summary["killed_mutants"]
        operator_summary["kill_rate"] = operator_summary["killed_mutants"] / operator_summary["total_mutants"]
        operator_summary = operator_summary.sort_values(["kill_rate", "total_mutants"], ascending=[True, False])
        _save_table(operator_summary, stats_dir / "operator_difficulty_summary.csv")
        report["operator_difficulty"] = operator_summary.to_dict(orient="records")
        equivalent_df = operator_df[operator_df["equivalence_status"].astype(str).eq("STATIC_EQUIVALENT")].copy()
        _save_table(equivalent_df, stats_dir / "conservative_equivalent_mutants.csv")
        report["conservative_equivalent_mutants"] = int(len(equivalent_df))
        _plot_operator(operator_summary, figures_dir)

    if not call_df.empty:
        for col in (
            "prompt_tokens", "completion_tokens", "thoughts_tokens", "total_tokens", "estimated_cost_usd",
            "latency_sec", "killed_mutants", "new_kills", "prompt_chars", "response_chars",
        ):
            if col in call_df:
                call_df[col] = pd.to_numeric(call_df[col], errors="coerce").fillna(0)
        call_df["provider"] = call_df.get("provider", pd.Series("unknown", index=call_df.index)).fillna("unknown")
        call_df["model"] = call_df.get("model", pd.Series("unknown", index=call_df.index)).fillna("unknown")
        call_df["status"] = call_df.get("status", pd.Series("UNKNOWN", index=call_df.index)).fillna("UNKNOWN")
        usage = call_df.groupby(["method", "provider", "model"], as_index=False).agg(
            calls=("problem_id", "size"),
            productive_calls=("new_kills", lambda x: int((x > 0).sum())),
            total_tokens=("total_tokens", "sum"),
            total_prompt_tokens=("prompt_tokens", "sum"),
            total_completion_tokens=("completion_tokens", "sum"),
            total_cost_usd=("estimated_cost_usd", "sum"),
            total_latency_sec=("latency_sec", "sum"),
            total_new_kills=("new_kills", "sum"),
            total_prompt_chars=("prompt_chars", "sum"),
        )
        usage["productive_call_rate"] = usage["productive_calls"] / usage["calls"]
        usage["tokens_per_new_kill"] = np.divide(
            usage["total_tokens"], usage["total_new_kills"],
            out=np.zeros(len(usage)), where=usage["total_new_kills"].to_numpy() != 0,
        )
        _save_table(usage, stats_dir / "llm_usage_summary.csv")
        report["llm_usage"] = usage.to_dict(orient="records")
        _plot_llm_usage(usage, figures_dir)

        outcome_summary = call_df.groupby(["method", "status"], as_index=False).size().rename(columns={"status": "outcome", "size": "calls"})
        _save_table(outcome_summary, stats_dir / "call_validation_outcomes.csv")
        report["call_validation_outcomes"] = outcome_summary.to_dict(orient="records")
        _plot_call_outcomes(outcome_summary, figures_dir)

        call_artifacts = call_df[[c for c in (
            "method", "problem_id", "layer", "attempt", "provider", "model", "status", "target_count",
            "new_kills", "cumulative_score", "prompt_tokens", "completion_tokens", "total_tokens",
            "prompt_chars", "response_chars", "latency_sec", "estimated_cost_usd", "validation_reason",
        ) if c in call_df]].copy()
        _save_table(call_artifacts, stats_dir / "llm_call_level_metrics.csv")

    if not history_df.empty:
        _save_table(history_df, stats_dir / "baseline_iteration_history.csv")
        report["baseline_iteration_rows"] = int(len(history_df))

    report["artifacts"] = {
        "statistics_directory": str(stats_dir),
        "figures_directory": str(figures_dir),
        "figure_files": sorted(p.name for p in figures_dir.glob("*")),
        "table_files": sorted(p.name for p in stats_dir.glob("*.csv")),
    }
    (stats_dir / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )

    markdown = [
        "# Claus-Test Evaluation Report",
        "",
        "This report keeps mutation effectiveness, official-test agreement, token/API efficiency, runtime, and baseline fairness separate.",
        "Raw macro and micro scores are primary; conservative equivalent-adjusted scores exclude only statically identified equivalent mutants.",
        "The baseline is reported under its fixed iteration budget, its first final-score point, its first plateau, and matched proposed-token budgets.",
        "",
        "## Main summary",
        "",
        "See `evaluation_summary.json` for confidence intervals, paired tests, effect sizes, and aggregate values.",
        "",
        "## Saved figures",
        "",
    ]
    markdown.extend(f"- `../figures/{name}`" for name in report["artifacts"]["figure_files"] if str(name).endswith(".png"))
    markdown.extend(["", "## Saved tables", ""])
    markdown.extend(f"- `{name}`" for name in report["artifacts"]["table_files"])
    (stats_dir / "STATISTICAL_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    log.info("Saved statistical report to %s and figures to %s", stats_dir, figures_dir)
    return report
