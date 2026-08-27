"""
representative_selector.py — Unsupervised representative selection.

DMSG/subsumption has been removed. Representatives are selected with UBIG-RS:
Unsupervised Behavior-aware Information-Gain Representative Selection.

For each cluster:
  1. compute feature-space centrality so outliers are avoided;
  2. shortlist the most central candidates;
  3. pick the candidate with the highest behavioral entropy/information score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.clustering.cluster_mutants import extract_features
from src.mutation.mutation_engine import Mutant
from src.utils.config import CENTRALITY_QUANTILE, REPRESENTATIVES_PER_CLUSTER
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RepresentativeStack:
    cluster_id: int
    representatives: List[Mutant]
    non_representatives: List[Mutant] = field(default_factory=list)
    mapping: Dict[str, str] = field(default_factory=dict)
    strategy: str = "UBIG-RS"


def behavioral_entropy(signature: List[int]) -> float:
    if not signature:
        return 0.0
    counts: Dict[int, int] = {}
    for item in signature:
        counts[item] = counts.get(item, 0) + 1
    total = float(len(signature))
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    # normalize by max entropy for three states: same, diff, timeout
    return entropy / math.log2(3) if entropy > 0 else 0.0


def behavioral_difference_rate(signature: List[int]) -> float:
    if not signature:
        return 0.0
    return sum(1 for x in signature if x != 0) / len(signature)


class RepresentativeSelector:
    def __init__(self, representatives_per_cluster: int = REPRESENTATIVES_PER_CLUSTER,
                 centrality_quantile: float = CENTRALITY_QUANTILE):
        self.representatives_per_cluster = max(1, representatives_per_cluster)
        self.centrality_quantile = min(max(centrality_quantile, 0.05), 1.0)

    def select(self, cluster_id: int, mutants: List[Mutant]) -> RepresentativeStack:
        if not mutants:
            return RepresentativeStack(cluster_id=cluster_id, representatives=[])
        if len(mutants) == 1:
            m = mutants[0]
            m.centrality = 1.0
            m.information_score = self._information_score(m)
            return RepresentativeStack(cluster_id=cluster_id, representatives=[m], non_representatives=[])

        X = extract_features(mutants)
        X = StandardScaler().fit_transform(X)
        D = pairwise_distances(X, metric="euclidean")
        mean_dist = D.mean(axis=1)
        max_dist = float(mean_dist.max()) if mean_dist.size else 1.0
        centrality = 1.0 - (mean_dist / max(max_dist, 1e-9))

        for m, c in zip(mutants, centrality):
            m.centrality = float(c)
            m.information_score = self._information_score(m)

        n_shortlist = max(self.representatives_per_cluster, int(math.ceil(len(mutants) * self.centrality_quantile)))
        central_indices = list(np.argsort(-centrality)[:n_shortlist])
        candidates = [mutants[i] for i in central_indices]

        ranked = sorted(
            candidates,
            key=lambda m: (
                float(m.information_score or 0.0),
                float(m.centrality or 0.0),
                behavioral_difference_rate(m.behavior_signature),
            ),
            reverse=True,
        )
        reps = ranked[: self.representatives_per_cluster]
        rep_ids = {m.mutant_id for m in reps}
        non_reps = [m for m in mutants if m.mutant_id not in rep_ids]
        mapping = self._map_to_nearest_reps(mutants, reps, D)

        log.info("Cluster %s: selected %d UBIG-RS representative(s) from %d mutants",
                 cluster_id, len(reps), len(mutants))
        return RepresentativeStack(
            cluster_id=cluster_id,
            representatives=reps,
            non_representatives=non_reps,
            mapping=mapping,
        )

    def select_all(self, clusters: Dict[int, List[Mutant]]) -> List[RepresentativeStack]:
        return [self.select(cid, members) for cid, members in sorted(clusters.items())]

    @staticmethod
    def _information_score(mutant: Mutant) -> float:
        # Entropy rewards structured behavioral variation; difference rate rewards
        # mutants that the cheap probes already partially separate from the original.
        entropy = behavioral_entropy(mutant.behavior_signature)
        diff_rate = behavioral_difference_rate(mutant.behavior_signature)
        return round(0.70 * entropy + 0.30 * diff_rate, 6)

    @staticmethod
    def _map_to_nearest_reps(mutants: List[Mutant], reps: List[Mutant], distance_matrix: np.ndarray) -> Dict[str, str]:
        if not reps:
            return {}
        index_by_id = {m.mutant_id: i for i, m in enumerate(mutants)}
        rep_indices = [index_by_id[r.mutant_id] for r in reps]
        mapping: Dict[str, str] = {}
        for m in mutants:
            i = index_by_id[m.mutant_id]
            nearest_rep_index = min(rep_indices, key=lambda r_idx: distance_matrix[i, r_idx])
            mapping[m.mutant_id] = mutants[nearest_rep_index].mutant_id
        return mapping
