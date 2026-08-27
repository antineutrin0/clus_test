"""
cluster_mutants.py — Behavior-aware unsupervised clustering for mutants.

KMeans has been replaced by agglomerative clustering because HumanEval creates
small, variable-size mutant sets where spherical KMeans assumptions are weak.
Features combine static AST/diff information with cheap behavioral signatures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.mutation.mutation_engine import Mutant
from src.utils.config import MAX_CLUSTERS, MIN_CLUSTER_SIZE, MUTATION_OPERATORS
from src.utils.logger import get_logger

log = get_logger(__name__)


def _stable_bucket(text: str, modulo: int = 17) -> float:
    if not text:
        return 0.0
    return (sum(ord(c) for c in text) % modulo) / max(modulo - 1, 1)


def _one_hot(value: str, vocab: List[str]) -> List[float]:
    return [1.0 if value == item else 0.0 for item in vocab]


def extract_features(mutants: List[Mutant]) -> np.ndarray:
    if not mutants:
        return np.zeros((0, 0), dtype=np.float32)

    max_line = max((m.line_number for m in mutants), default=1) or 1
    node_vocab = sorted({m.ast_node_type or "Unknown" for m in mutants})
    parent_vocab = sorted({m.parent_node_type or "Unknown" for m in mutants})
    max_sig_len = max((len(m.behavior_signature) for m in mutants), default=0)

    rows: List[List[float]] = []
    for m in mutants:
        behavior = list(m.behavior_signature)
        if max_sig_len:
            behavior = behavior + [0] * (max_sig_len - len(behavior))
            behavior = [float(x) / 2.0 for x in behavior]

        line_pos = float(m.line_number) / float(max_line)
        desc_bucket = _stable_bucket(m.description)
        row = (
            _one_hot(m.operator, MUTATION_OPERATORS)
            + _one_hot(m.ast_node_type or "Unknown", node_vocab)
            + _one_hot(m.parent_node_type or "Unknown", parent_vocab)
            + [line_pos, desc_bucket]
            + behavior
        )
        rows.append(row)
    return np.array(rows, dtype=np.float32)


class MutantClusterer:
    def __init__(self, max_clusters: int = MAX_CLUSTERS, random_state: int = 42):
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.best_k: int = 1
        self.labels_: np.ndarray = np.array([], dtype=int)
        self.silhouette_: float = 0.0
        self._scaler = StandardScaler()
        self.feature_matrix_: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def fit_predict(self, mutants: List[Mutant]) -> Dict[int, List[Mutant]]:
        if len(mutants) < MIN_CLUSTER_SIZE:
            for m in mutants:
                m.cluster_id = 0
            self.best_k = 1
            self.labels_ = np.zeros(len(mutants), dtype=int)
            return {0: mutants}

        X = extract_features(mutants)
        self.feature_matrix_ = self._scaler.fit_transform(X)
        n = len(mutants)
        max_k = min(self.max_clusters, n - 1)
        if max_k < 2:
            labels = np.zeros(n, dtype=int)
            self.best_k = 1
        else:
            best_labels = None
            best_score = -1.0
            best_k = 2
            for k in range(2, max_k + 1):
                labels = self._agglomerative(self.feature_matrix_, k)
                if len(set(labels)) < 2:
                    continue
                try:
                    score = float(silhouette_score(self.feature_matrix_, labels))
                except Exception:
                    score = -1.0
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_k = k
            labels = best_labels if best_labels is not None else np.zeros(n, dtype=int)
            self.best_k = best_k if best_labels is not None else 1
            self.silhouette_ = max(best_score, 0.0)

        self.labels_ = np.asarray(labels, dtype=int)
        clusters: Dict[int, List[Mutant]] = {}
        for mutant, label in zip(mutants, self.labels_):
            cid = int(label)
            mutant.cluster_id = cid
            clusters.setdefault(cid, []).append(mutant)

        log.info("Clustered %d mutants into %d clusters (silhouette=%.4f)", len(mutants), len(clusters), self.silhouette_)
        return clusters

    @staticmethod
    def _agglomerative(X: np.ndarray, k: int) -> np.ndarray:
        try:
            model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        except TypeError:
            model = AgglomerativeClustering(n_clusters=k, affinity="euclidean", linkage="ward")
        return model.fit_predict(X)


def save_clusters(clusters: Dict[int, List[Mutant]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(cid): [m.to_dict() for m in members] for cid, members in clusters.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("Clusters saved -> %s", path)


def load_clusters(path: Path) -> Dict[int, List[Mutant]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(cid): [Mutant.from_dict(d) for d in rows] for cid, rows in raw.items()}
