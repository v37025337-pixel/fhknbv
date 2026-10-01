
"""
v0.13_unified_runtime.py

Unified runtime for the current kernel line.

Integrates:
- v0.8  SleepCompressor       : derived/episodic memory only
- v0.9  OtherModel            : social hidden-state model
- v0.10 MultiTimescalePlanner : social planning
- v0.11 DomainRouter          : repository/code routing + CodeWorldModel
- v0.12 Multimodal evidence   : NETWORK_CONFIG / SECURITY_EVENT evidence worlds

Separation rule
---------------
RAW EVIDENCE (source files, screenshots, images) is immutable source-of-truth
and is NEVER sent to SleepCompressor.

Only derived observations, hypotheses, outcomes, and summaries may enter
episodic compression.

Perception boundary
-------------------
This runtime does not implement OCR/vision itself. Images enter as:
    raw image path + structured observations extracted by a perception layer.
The runtime then owns routing, evidence preservation, world-model construction,
memory, and planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import importlib.util
import sys
import numpy as np


def _load_sibling(module_name: str, filename: str):
    here = Path(__file__).resolve().parent
    path = here / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required sibling module {filename!r} was not found next to "
            f"{Path(__file__).name!r}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


v08 = _load_sibling("kernel_v08_memory", "v0.8_sleep_compressor.py")
v09 = _load_sibling("kernel_v09_other", "v0.9_other_model.py")
v10 = _load_sibling("kernel_v10_planner", "v0.10_social_planner.py")
v11 = _load_sibling("kernel_v11_domain", "v0.11_domain_router.py")
v12 = _load_sibling("kernel_v12_evidence", "v0.12_multimodal_evidence.py")


@dataclass(slots=True)
class AuditEvent:
    seq: int
    kind: str
    domain: str
    summary: str
    raw_evidence: Optional[str] = None
    sensitive: bool = False


@dataclass(slots=True)
class RuntimeSnapshot:
    sequence: int
    active_domains: List[str]
    code_world: Optional[dict]
    evidence_world: dict
    memory: dict
    social_actors: List[str]
    audit_events: int


class StableDerivedEncoder:
    """
    Deterministic fingerprint encoder used only when the caller does not supply
    a real semantic vector.

    This is NOT an embedding model and must not be presented as semantic
    understanding. It merely gives the memory module a stable numeric key.
    """

    def __init__(self, dim: int = 32):
        self.dim = int(dim)

    def encode(self, text: str) -> np.ndarray:
        data = b""
        counter = 0
        while len(data) < self.dim * 4:
            data += hashlib.sha256(
                text.encode("utf-8") + counter.to_bytes(4, "big")
            ).digest()
            counter += 1
        raw = np.frombuffer(data[: self.dim * 4], dtype=np.uint32).astype(np.float32)
        v = (raw / np.float32(2**32 - 1)) * 2.0 - 1.0
        v = v[: self.dim]
        v /= np.linalg.norm(v) + 1e-12
        return v.astype(np.float32)


class UnifiedRuntime:
    def __init__(
        self,
        *,
        memory_dim: int = 32,
        memory_max_episodes: int = 6000,
        memory_target_episodes: int = 3000,
        memory_max_concepts: int = 12000,
        seed: int = 7,
    ):
        self.memory_dim = int(memory_dim)
        self.encoder = StableDerivedEncoder(self.memory_dim)
        self.memory = v08.SleepCompressor(
            dim=self.memory_dim,
            max_episodes=memory_max_episodes,
            target_episodes=memory_target_episodes,
            max_concepts=memory_max_concepts,
            seed=seed,
        )

        self.domain_builder = v11.DomainModelBuilder()
        self.code_worlds: Dict[str, Any] = {}
        self.active_code_world_id: Optional[str] = None
        self.code_world = None

        self.evidence_kernel = v12.MultiDomainEvidenceKernel()

        self.social_models: Dict[str, Any] = {}
        self.social_planners: Dict[str, Any] = {}

        self.audit: List[AuditEvent] = []
        self.sequence = 0
        self.active_domains = set()

    # ------------------------------------------------------------------
    # Internal bookkeeping
    # ------------------------------------------------------------------

    def _audit(
        self,
        kind: str,
        domain: str,
        summary: str,
        *,
        raw_evidence: Optional[str] = None,
        sensitive: bool = False,
    ) -> None:
        self.sequence += 1
        self.audit.append(
            AuditEvent(
                seq=self.sequence,
                kind=kind,
                domain=domain,
                summary=summary,
                raw_evidence=raw_evidence,
                sensitive=sensitive,
            )
        )

    def remember_derived(
        self,
        label: str,
        *,
        vector: Optional[np.ndarray] = None,
        utility: float = 0.5,
        novelty: float = 0.5,
        causal: float = 0.5,
        self_relevance: float = 0.3,
        actor: int = -1,
        kind: int = 0,
    ) -> int:
        """
        Store DERIVED information only.

        Do not pass raw file contents, screenshots, API keys, IP addresses,
        account identifiers, or other sensitive raw evidence here.
        """
        z = self.encoder.encode(label) if vector is None else np.asarray(
            vector, dtype=np.float32
        )
        if z.shape != (self.memory_dim,):
            raise ValueError(
                f"derived vector must have shape {(self.memory_dim,)}, got {z.shape}"
            )

        return self.memory.add(
            z,
            utility=utility,
            novelty=novelty,
            causal=causal,
            self_relevance=self_relevance,
            timestamp=self.sequence,
            actor=actor,
            kind=kind,
        )

    # ------------------------------------------------------------------
    # CODE
    # ------------------------------------------------------------------

    def ingest_repository(
        self,
        paths: Iterable[str],
        *,
        manifest_signals: Optional[Dict[str, bool]] = None,
        extension_histogram: Optional[Dict[str, int]] = None,
        workspace_id: str = "default",
    ):
        paths = list(paths)
        inv = v11.RepositoryInventory.from_paths(
            paths,
            manifest_signals=manifest_signals,
        )
        if extension_histogram is not None:
            inv.extension_histogram = dict(extension_histogram)

        built = self.domain_builder.build_repository(inv)
        decision = built.decision

        self._audit(
            "route",
            decision.domain.value,
            f"repository routed to {decision.domain.value} "
            f"(confidence={decision.confidence:.3f})",
        )

        if decision.domain == v11.Domain.CODE:
            self.code_worlds[workspace_id] = built.model
            self.active_code_world_id = workspace_id
            self.code_world = built.model
            self.active_domains.add("code")
            self.remember_derived(
                "derived:repository:code-domain-detected",
                utility=0.7,
                novelty=0.6,
                causal=0.7,
                kind=10,
            )

        return decision

    def ingest_code_source(
        self,
        path: str,
        content: str,
        *,
        workspace_id: Optional[str] = None,
    ):
        wid = workspace_id or self.active_code_world_id
        if wid is None or wid not in self.code_worlds:
            raise RuntimeError("No matching CodeWorld. Ingest repository inventory first.")

        world = self.code_worlds[wid]
        self.active_code_world_id = wid
        self.code_world = world
        parsed = world.ingest_source(path, content)
        self._audit(
            "code_parse",
            "code",
            f"workspace={wid}; parsed {path}: {len(parsed.classes)} classes, "
            f"{len(parsed.functions)} functions, {len(parsed.imports)} imports",
            raw_evidence=path,
        )
        self.remember_derived(
            f"derived:code-structure:{path}:"
            f"c{len(parsed.classes)}:f{len(parsed.functions)}:i{len(parsed.imports)}",
            utility=0.65,
            novelty=0.75,
            causal=0.55,
            kind=11,
        )
        return parsed

    # ------------------------------------------------------------------
    # MULTIMODAL EVIDENCE
    # ------------------------------------------------------------------

    def ingest_dns_image(self, path: str, rows: List[Dict[str, str]]):
        domain, confidence = self.evidence_kernel.ingest_dns_image(path, rows)
        self.active_domains.add(domain.value)
        self._audit(
            "evidence",
            domain.value,
            f"DNS table: {len(rows)} provider rows",
            raw_evidence=path,
        )
        self.remember_derived(
            f"derived:dns-table:rows={len(rows)}",
            utility=0.5,
            novelty=0.6,
            causal=0.35,
            kind=20,
        )
        return domain, confidence

    def ingest_security_image(
        self,
        path: str,
        observations: List[Any],
    ):
        domain, confidence = self.evidence_kernel.ingest_security_image(
            path, observations
        )
        self.active_domains.add(domain.value)
        self._audit(
            "evidence",
            domain.value,
            "security event ingested; sensitive fields kept out of derived memory",
            raw_evidence=path,
            sensitive=True,
        )
        # Intentionally coarse and redacted.
        self.remember_derived(
            "derived:security-event:verification-required",
            utility=0.85,
            novelty=0.8,
            causal=0.85,
            self_relevance=0.5,
            kind=21,
        )
        return domain, confidence

    # ------------------------------------------------------------------
    # SOCIAL
    # ------------------------------------------------------------------

    def register_social_actor(
        self,
        actor_id: str,
        goal_vectors: np.ndarray,
        n_actions: int,
        *,
        rationality: float = 4.0,
        hazard: float = 0.02,
        unknown_prior: float = 0.10,
        empirical_strength: float = 0.20,
        gamma: float = 0.92,
        horizons=(1, 2, 3),
        value_tolerance: float = 0.055,
        unknown_risk_penalty: float = 0.30,
    ):
        model = v09.OtherModel(
            goal_vectors,
            n_actions,
            rationality=rationality,
            hazard=hazard,
            unknown_prior=unknown_prior,
            empirical_strength=empirical_strength,
        )
        planner = v10.MultiTimescalePlanner(
            model,
            gamma=gamma,
            horizons=horizons,
            value_tolerance=value_tolerance,
            unknown_risk_penalty=unknown_risk_penalty,
        )
        self.social_models[actor_id] = model
        self.social_planners[actor_id] = planner
        self.active_domains.add("social")
        self._audit("social_actor", "social", f"registered actor {actor_id}")
        return model

    def plan_social(self, actor_id: str, state, env):
        if actor_id not in self.social_planners:
            raise KeyError(f"unknown social actor {actor_id!r}")
        result = self.social_planners[actor_id].plan(state, env)
        self._audit(
            "plan",
            "social",
            f"actor={actor_id}; action={result.action}; horizon={result.horizon}; "
            f"unknown={result.unknown_probability:.3f}",
        )
        return result

    def observe_social(
        self,
        actor_id: str,
        action_features: np.ndarray,
        chosen_action: int,
        *,
        context_key=None,
        promised_action: Optional[int] = None,
        success: Optional[bool] = None,
    ):
        if actor_id not in self.social_models:
            raise KeyError(f"unknown social actor {actor_id!r}")
        model = self.social_models[actor_id]
        model.observe(
            action_features,
            chosen_action,
            context_key=context_key,
            promised_action=promised_action,
            success=success,
        )
        self._audit(
            "observe",
            "social",
            f"actor={actor_id}; observed action={chosen_action}",
        )
        self.remember_derived(
            f"derived:social-observation:{actor_id}:a{chosen_action}",
            utility=0.5,
            novelty=0.4,
            causal=0.6,
            actor=int.from_bytes(hashlib.sha256(actor_id.encode("utf-8")).digest()[:4], "big") % 1_000_000,
            kind=30,
        )

    # ------------------------------------------------------------------
    # Maintenance / inspection
    # ------------------------------------------------------------------

    def sleep(self):
        before = self.memory.stats()
        self.memory.sleep()
        after = self.memory.stats()
        self._audit(
            "sleep",
            "memory",
            f"episodes {before['episodes']}->{after['episodes']}; "
            f"concepts {before['concepts']}->{after['concepts']}",
        )
        return before, after

    def snapshot(self) -> RuntimeSnapshot:
        code = None
        if self.code_world is not None:
            code = dict(self.code_world.architecture_summary())
            code["active_workspace_id"] = self.active_code_world_id
            code["workspace_count"] = len(self.code_worlds)
            code["all_workspaces"] = {
                wid: world.architecture_summary()
                for wid, world in self.code_worlds.items()
            }

        return RuntimeSnapshot(
            sequence=self.sequence,
            active_domains=sorted(self.active_domains),
            code_world=code,
            evidence_world=self.evidence_kernel.safe_summary(),
            memory=self.memory.stats(),
            social_actors=sorted(self.social_models),
            audit_events=len(self.audit),
        )
