
"""
v0.8_sleep_compressor.py

Scalable episodic memory with:
- protected memories
- approximate concept formation via LSH buckets
- bounded episodic + concept stores
- local nearest-neighbour retrieval with scipy.spatial.cKDTree
- causal/self-relevance aware retention score

Design goal:
experience -> episode -> relations -> sleep compression -> concept
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:
    raise RuntimeError("scipy is required for cKDTree retrieval") from exc


@dataclass(slots=True)
class Episode:
    eid: int
    z: np.ndarray
    utility: float
    novelty: float
    causal: float
    self_relevance: float
    timestamp: int
    actor: int = -1
    kind: int = 0
    protected: bool = False


@dataclass(slots=True)
class Concept:
    cid: int
    z: np.ndarray
    support: int
    utility: float
    causal: float
    self_relevance: float
    last_timestamp: int
    actor: int
    kind: int
    lsh_key: int


class SleepCompressor:
    def __init__(
        self,
        dim: int,
        max_episodes: int = 6000,
        target_episodes: int = 3000,
        max_concepts: int = 12000,
        protected_threshold: float = 0.78,
        lsh_bits: int = 10,
        seed: int = 7,
    ):
        if target_episodes >= max_episodes:
            raise ValueError("target_episodes must be < max_episodes")
        self.dim = int(dim)
        self.max_episodes = int(max_episodes)
        self.target_episodes = int(target_episodes)
        self.max_concepts = int(max_concepts)
        self.protected_threshold = float(protected_threshold)
        self.lsh_bits = int(lsh_bits)

        self.rng = np.random.default_rng(seed)
        self._planes = self.rng.normal(size=(self.lsh_bits, self.dim)).astype(np.float32)
        self._planes /= np.linalg.norm(self._planes, axis=1, keepdims=True) + 1e-12

        self.episodes: List[Episode] = []
        self.concepts: Dict[Tuple[int, int, int], Concept] = {}
        self._next_eid = 0
        self._next_cid = 0
        self._tree: Optional[cKDTree] = None
        self._tree_vectors: Optional[np.ndarray] = None
        self._tree_refs: List[Tuple[str, int]] = []
        self._dirty = True
        self.sleep_count = 0
        self.total_ingested = 0

    @staticmethod
    def _clip01(x: float) -> float:
        return float(min(1.0, max(0.0, x)))

    def retention_score(
        self, utility: float, novelty: float, causal: float, self_relevance: float
    ) -> float:
        # Intentional bias: causal + self-changing events are hardest to discard.
        return (
            0.22 * self._clip01(utility)
            + 0.13 * self._clip01(novelty)
            + 0.35 * self._clip01(causal)
            + 0.30 * self._clip01(self_relevance)
        )

    def _lsh_key(self, z: np.ndarray) -> int:
        bits = (self._planes @ z) >= 0
        key = 0
        for i, bit in enumerate(bits):
            key |= int(bit) << i
        return key

    def add(
        self,
        z,
        *,
        utility: float,
        novelty: float,
        causal: float,
        self_relevance: float,
        timestamp: int,
        actor: int = -1,
        kind: int = 0,
    ) -> int:
        v = np.asarray(z, dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"expected vector shape {(self.dim,)}, got {v.shape}")
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n

        score = self.retention_score(utility, novelty, causal, self_relevance)
        ep = Episode(
            eid=self._next_eid,
            z=v,
            utility=self._clip01(utility),
            novelty=self._clip01(novelty),
            causal=self._clip01(causal),
            self_relevance=self._clip01(self_relevance),
            timestamp=int(timestamp),
            actor=int(actor),
            kind=int(kind),
            protected=score >= self.protected_threshold,
        )
        self._next_eid += 1
        self.total_ingested += 1
        self.episodes.append(ep)
        self._dirty = True

        if len(self.episodes) > self.max_episodes:
            self.sleep()

        return ep.eid

    def add_batch(
        self,
        vectors: np.ndarray,
        utility: np.ndarray,
        novelty: np.ndarray,
        causal: np.ndarray,
        self_relevance: np.ndarray,
        timestamps: np.ndarray,
        actors: Optional[np.ndarray] = None,
        kinds: Optional[np.ndarray] = None,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        n = len(vectors)
        if vectors.shape != (n, self.dim):
            raise ValueError("bad batch vector shape")
        actors = np.full(n, -1, dtype=np.int32) if actors is None else np.asarray(actors)
        kinds = np.zeros(n, dtype=np.int32) if kinds is None else np.asarray(kinds)

        for i in range(n):
            self.add(
                vectors[i],
                utility=float(utility[i]),
                novelty=float(novelty[i]),
                causal=float(causal[i]),
                self_relevance=float(self_relevance[i]),
                timestamp=int(timestamps[i]),
                actor=int(actors[i]),
                kind=int(kinds[i]),
            )

    def _merge_into_concept(self, eps: List[Episode], key: Tuple[int, int, int]) -> None:
        support = len(eps)
        weights = np.array(
            [
                0.35 + self.retention_score(
                    e.utility, e.novelty, e.causal, e.self_relevance
                )
                for e in eps
            ],
            dtype=np.float32,
        )
        mat = np.stack([e.z for e in eps])
        centroid = np.average(mat, axis=0, weights=weights)
        centroid /= np.linalg.norm(centroid) + 1e-12

        u = float(np.average([e.utility for e in eps], weights=weights))
        c = float(np.average([e.causal for e in eps], weights=weights))
        s = float(np.average([e.self_relevance for e in eps], weights=weights))
        last_ts = max(e.timestamp for e in eps)

        if key in self.concepts:
            old = self.concepts[key]
            total = old.support + support
            merged = (old.z * old.support + centroid * support) / total
            merged /= np.linalg.norm(merged) + 1e-12
            old.z = merged.astype(np.float32)
            old.utility = (old.utility * old.support + u * support) / total
            old.causal = (old.causal * old.support + c * support) / total
            old.self_relevance = (old.self_relevance * old.support + s * support) / total
            old.support = total
            old.last_timestamp = max(old.last_timestamp, last_ts)
        else:
            actor, kind, lsh_key = key
            self.concepts[key] = Concept(
                cid=self._next_cid,
                z=centroid.astype(np.float32),
                support=support,
                utility=u,
                causal=c,
                self_relevance=s,
                last_timestamp=last_ts,
                actor=actor,
                kind=kind,
                lsh_key=lsh_key,
            )
            self._next_cid += 1

    def _trim_concepts(self) -> None:
        if len(self.concepts) <= self.max_concepts:
            return

        ranked = sorted(
            self.concepts.items(),
            key=lambda kv: (
                0.35 * np.log1p(kv[1].support)
                + 0.25 * kv[1].utility
                + 0.25 * kv[1].causal
                + 0.15 * kv[1].self_relevance
            ),
            reverse=True,
        )
        self.concepts = dict(ranked[: self.max_concepts])

    def sleep(self) -> None:
        if len(self.episodes) <= self.target_episodes:
            return

        self.sleep_count += 1

        protected = [e for e in self.episodes if e.protected]
        ordinary = [e for e in self.episodes if not e.protected]

        # If protected memories alone exceed target, we keep them all by design.
        slots = max(0, self.target_episodes - len(protected))

        # Keep the highest-value ordinary episodes.
        ordinary.sort(
            key=lambda e: (
                self.retention_score(e.utility, e.novelty, e.causal, e.self_relevance),
                e.timestamp,
            ),
            reverse=True,
        )
        keep = ordinary[:slots]
        compress = ordinary[slots:]

        buckets: Dict[Tuple[int, int, int], List[Episode]] = {}
        for e in compress:
            key = (e.actor, e.kind, self._lsh_key(e.z))
            buckets.setdefault(key, []).append(e)

        # Singleton buckets are not promoted to "concepts": too little evidence.
        # They simply fade. Repeated evidence becomes a concept.
        for key, group in buckets.items():
            if len(group) >= 2:
                self._merge_into_concept(group, key)

        self.episodes = protected + keep
        self._trim_concepts()
        self._dirty = True

    def _rebuild_index(self) -> None:
        refs: List[Tuple[str, int]] = []
        vectors = []

        for e in self.episodes:
            refs.append(("episode", e.eid))
            vectors.append(e.z)

        for c in self.concepts.values():
            refs.append(("concept", c.cid))
            vectors.append(c.z)

        if vectors:
            mat = np.stack(vectors).astype(np.float32)
            self._tree = cKDTree(mat)
            self._tree_vectors = mat
            self._tree_refs = refs
        else:
            self._tree = None
            self._tree_vectors = None
            self._tree_refs = []

        self._dirty = False

    def query(self, z, k: int = 8):
        if self._dirty:
            self._rebuild_index()
        if self._tree is None:
            return []

        q = np.asarray(z, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        k = min(int(k), len(self._tree_refs))
        dist, idx = self._tree.query(q, k=k)

        dist = np.atleast_1d(dist)
        idx = np.atleast_1d(idx)
        return [
            {"ref": self._tree_refs[int(i)], "distance": float(d)}
            for d, i in zip(dist, idx)
        ]

    def stats(self):
        protected = sum(e.protected for e in self.episodes)
        concept_support = sum(c.support for c in self.concepts.values())
        return {
            "total_ingested": self.total_ingested,
            "episodes": len(self.episodes),
            "protected_episodes": protected,
            "concepts": len(self.concepts),
            "concept_support": concept_support,
            "sleep_count": self.sleep_count,
        }
