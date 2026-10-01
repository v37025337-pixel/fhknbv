
"""
v0.11_domain_router.py

L0-style domain routing and code-world construction.

Core rule:
    raw input -> entities -> relations -> constraints -> domain model

This module intentionally keeps raw source artifacts OUTSIDE episodic
compression. Unique source files are immutable evidence references.

Supported first-class domain in v0.11:
    CODE

Other domains are detected but remain adapter placeholders:
    SOCIAL, DATA, DOCUMENT, UNKNOWN

The code parser is explicitly lexical/static. It does not claim full AST
semantics for every language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple, Any
import hashlib
import math
import re


# ---------------------------------------------------------------------------
# L0 primitives
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Entity:
    eid: str
    kind: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Relation:
    src: str
    rel: str
    dst: str
    confidence: float = 1.0
    evidence: Optional[str] = None


@dataclass(slots=True)
class Constraint:
    kind: str
    expression: str
    evidence: Optional[str] = None


@dataclass(slots=True)
class L0World:
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.eid] = entity

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def stats(self) -> dict:
        rel_counts: Dict[str, int] = {}
        ent_counts: Dict[str, int] = {}
        for e in self.entities.values():
            ent_counts[e.kind] = ent_counts.get(e.kind, 0) + 1
        for r in self.relations:
            rel_counts[r.rel] = rel_counts.get(r.rel, 0) + 1
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "constraints": len(self.constraints),
            "entity_kinds": ent_counts,
            "relation_kinds": rel_counts,
        }


# ---------------------------------------------------------------------------
# Raw artifact store: NEVER compressed as episodic memory
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class ArtifactRef:
    artifact_id: str
    path: str
    sha256: str
    size: int
    media_kind: str


class RawArtifactStore:
    """
    Immutable evidence registry.

    Raw artifacts are source-of-truth references and do not participate in
    SleepCompressor pruning. Derived observations may be compressed elsewhere.
    """

    def __init__(self):
        self._items: Dict[str, ArtifactRef] = {}
        self._path_index: Dict[str, str] = {}

    @staticmethod
    def _media_kind(path: str) -> str:
        ext = PurePosixPath(path).suffix.lower()
        if ext in {
            ".dart", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt",
            ".swift", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".go",
            ".rb", ".php", ".sh", ".gradle", ".cmake"
        }:
            return "source"
        if ext in {".yaml", ".yml", ".toml", ".json", ".xml", ".plist"}:
            return "config"
        if ext in {".md", ".txt", ".rst"}:
            return "document"
        if ext in {".csv", ".tsv", ".parquet", ".xlsx"}:
            return "data"
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"}:
            return "image"
        return "other"

    def register(
        self,
        path: str,
        *,
        content: Optional[str] = None,
        size: Optional[int] = None,
        external_sha: Optional[str] = None,
    ) -> ArtifactRef:
        if content is not None:
            raw = content.encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            actual_size = len(raw)
        else:
            seed = f"{path}|{size or 0}|{external_sha or ''}".encode("utf-8")
            digest = external_sha or hashlib.sha256(seed).hexdigest()
            actual_size = int(size or 0)

        # Artifact identity is the source location; content hash is version evidence.
        # Re-reading/upgrading the same path must not create a second "memory".
        artifact_id = "artifact:" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:20]
        ref = ArtifactRef(
            artifact_id=artifact_id,
            path=path,
            sha256=digest,
            size=actual_size,
            media_kind=self._media_kind(path),
        )
        self._items[artifact_id] = ref
        self._path_index[path] = artifact_id
        return ref

    def __len__(self) -> int:
        return len(self._items)

    def values(self):
        return self._items.values()


# ---------------------------------------------------------------------------
# Domain routing
# ---------------------------------------------------------------------------

class Domain(str, Enum):
    CODE = "code"
    SOCIAL = "social"
    DATA = "data"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class RepositoryInventory:
    paths: List[str]
    manifest_signals: Dict[str, bool] = field(default_factory=dict)
    extension_histogram: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str],
        manifest_signals: Optional[Dict[str, bool]] = None,
    ) -> "RepositoryInventory":
        paths = list(paths)
        hist: Dict[str, int] = {}
        for p in paths:
            name = PurePosixPath(p).name
            suffix = PurePosixPath(name).suffix.lower()
            ext = suffix if suffix else "<none>"
            hist[ext] = hist.get(ext, 0) + 1
        return cls(
            paths=paths,
            manifest_signals=dict(manifest_signals or {}),
            extension_histogram=hist,
        )


@dataclass(slots=True)
class RouteDecision:
    domain: Domain
    scores: Dict[str, float]
    confidence: float
    margin: float
    reasons: List[str]


class DomainRouter:
    CODE_EXTS = {
        ".dart", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt",
        ".swift", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".go",
        ".rb", ".php", ".sh", ".gradle", ".cmake",
    }
    DATA_EXTS = {".csv", ".tsv", ".parquet", ".xlsx", ".arrow"}
    DOC_EXTS = {".md", ".txt", ".rst", ".pdf", ".docx"}

    CODE_MANIFEST_NAMES = {
        "pubspec.yaml", "pyproject.toml", "requirements.txt", "package.json",
        "cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts",
        "cmakelists.txt",
    }

    def __init__(self, min_confidence: float = 0.50, min_margin: float = 0.15):
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)

    @staticmethod
    def _normalize(scores: Dict[Domain, float]) -> Dict[Domain, float]:
        # Positive softmax with a moderate temperature.
        vals = {k: math.exp(min(12.0, max(-12.0, v))) for k, v in scores.items()}
        z = sum(vals.values()) or 1.0
        return {k: v / z for k, v in vals.items()}

    def route_repository(self, inv: RepositoryInventory) -> RouteDecision:
        total = max(1, len(inv.paths))
        hist = inv.extension_histogram

        code_count = sum(hist.get(ext, 0) for ext in self.CODE_EXTS)
        data_count = sum(hist.get(ext, 0) for ext in self.DATA_EXTS)
        doc_count = sum(hist.get(ext, 0) for ext in self.DOC_EXTS)

        lower_names = {PurePosixPath(p).name.lower() for p in inv.paths}
        manifests = sum(1 for n in self.CODE_MANIFEST_NAMES if n in lower_names)
        has_tests = any(
            p.startswith("test/") or p.startswith("tests/") or "/test/" in p
            for p in inv.paths
        )
        has_ci = any(p.startswith(".github/workflows/") for p in inv.paths)
        has_src_tree = any(
            p.startswith(("lib/", "src/", "app/", "packages/"))
            for p in inv.paths
        )

        code_ratio = code_count / total
        data_ratio = data_count / total
        doc_ratio = doc_count / total

        # Scores are evidence, not a learned universal classifier.
        raw = {
            Domain.CODE:
                0.4
                + 3.1 * code_ratio
                + 1.5 * min(1, manifests)
                + 0.7 * int(has_tests)
                + 0.4 * int(has_ci)
                + 0.8 * int(has_src_tree),
            Domain.DATA:
                0.3 + 4.0 * data_ratio,
            Domain.DOCUMENT:
                0.3 + 3.0 * doc_ratio,
            Domain.SOCIAL:
                0.15,   # repository inventory alone is not social evidence
            Domain.UNKNOWN:
                0.45,
        }

        probs = self._normalize(raw)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        (best_domain, best), (_, second) = ranked[:2]
        margin = best - second

        reasons: List[str] = []
        if code_count:
            reasons.append(f"{code_count}/{total} artifacts have source-code extensions")
        if manifests:
            reasons.append(f"{manifests} recognized code manifest(s)")
        if has_src_tree:
            reasons.append("source-tree structure detected")
        if has_tests:
            reasons.append("test tree detected")
        if has_ci:
            reasons.append("CI/workflow files detected")

        if best < self.min_confidence or margin < self.min_margin:
            chosen = Domain.UNKNOWN
        else:
            chosen = best_domain

        return RouteDecision(
            domain=chosen,
            scores={k.value: float(v) for k, v in probs.items()},
            confidence=float(best),
            margin=float(margin),
            reasons=reasons,
        )


# ---------------------------------------------------------------------------
# Code-world model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ParsedCode:
    path: str
    language: str
    imports: List[str]
    classes: List[str]
    functions: List[str]


class LexicalCodeParser:
    """
    Conservative lexical extractor.

    It recognizes declarations/imports. It is NOT a full compiler AST.
    """

    LANG_BY_EXT = {
        ".dart": "dart",
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".java": "java",
        ".kt": "kotlin",
        ".swift": "swift",
        ".rs": "rust",
        ".go": "go",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "c-header",
        ".hpp": "cpp-header",
        ".sh": "shell",
    }

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        seen = set()
        out = []
        for x in values:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def parse(self, path: str, content: str) -> ParsedCode:
        ext = PurePosixPath(path).suffix.lower()
        lang = self.LANG_BY_EXT.get(ext, "unknown")

        imports: List[str] = []
        classes: List[str] = []
        funcs: List[str] = []

        if lang == "dart":
            imports = re.findall(r"\bimport\s+['\"]([^'\"]+)['\"]", content)
            classes = re.findall(r"\bclass\s+([A-Za-z_]\w*)", content)
            # Avoid matching control-flow; require a type-ish token before function.
            func_pat = re.compile(
                r"(?:^|\n)\s*(?:static\s+)?"
                r"(?:Future(?:<[^>\n]+>)?|void|int|double|bool|String|Widget|Uri|"
                r"List(?:<[^>\n]+>)?|Map(?:<[^>\n]+>)?|[A-Z][A-Za-z0-9_<>, ?]*)"
                r"\s+([A-Za-z_]\w*)\s*\(",
                re.MULTILINE,
            )
            funcs = func_pat.findall(content)

        elif lang == "python":
            imports += re.findall(r"^\s*import\s+([A-Za-z0-9_\.]+)", content, re.M)
            imports += re.findall(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\b", content, re.M)
            classes = re.findall(r"^\s*class\s+([A-Za-z_]\w*)", content, re.M)
            funcs = re.findall(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", content, re.M)

        elif lang in {"javascript", "typescript"}:
            imports += re.findall(r"\bfrom\s+['\"]([^'\"]+)['\"]", content)
            imports += re.findall(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", content)
            classes = re.findall(r"\bclass\s+([A-Za-z_$]\w*)", content)
            funcs += re.findall(r"\bfunction\s+([A-Za-z_$]\w*)\s*\(", content)

        else:
            # Generic fallback: conservative class/function-like declarations.
            classes = re.findall(r"\bclass\s+([A-Za-z_]\w*)", content)
            funcs = re.findall(
                r"(?:^|\n)\s*(?:[A-Za-z_]\w*(?:<[^>]+>)?)\s+([A-Za-z_]\w*)\s*\(",
                content,
            )

        return ParsedCode(
            path=path,
            language=lang,
            imports=self._dedupe(imports),
            classes=self._dedupe(classes),
            functions=self._dedupe(funcs),
        )


class CodeWorldModel:
    def __init__(self):
        self.world = L0World()
        self.artifacts = RawArtifactStore()
        self.parser = LexicalCodeParser()
        self.parsed: Dict[str, ParsedCode] = {}

    @staticmethod
    def _file_eid(path: str) -> str:
        return f"file:{path}"

    @staticmethod
    def _symbol_eid(path: str, kind: str, name: str) -> str:
        return f"{kind}:{path}::{name}"

    @staticmethod
    def _package_import_candidate(import_name: str) -> Optional[str]:
        # A package import is only a *candidate* for a local path.
        # It becomes local only if that lib/... file already exists in inventory.
        # This prevents package:flutter/... or package:dio/... from being
        # falsely rewritten as project-local files.
        m = re.match(r"package:[^/]+/(.+)", import_name)
        if m:
            return "lib/" + m.group(1)
        return None

    def ingest_inventory(self, inv: RepositoryInventory) -> None:
        repo_id = "repo:root"
        self.world.add_entity(Entity(repo_id, "repository", {
            "file_count": len(inv.paths),
            "manifest_signals": dict(inv.manifest_signals),
        }))

        for p in inv.paths:
            ref = self.artifacts.register(p)
            feid = self._file_eid(p)
            self.world.add_entity(Entity(feid, "file", {
                "path": p,
                "artifact_id": ref.artifact_id,
                "media_kind": ref.media_kind,
            }))
            self.world.add_relation(Relation(repo_id, "CONTAINS", feid, 1.0, p))

        # Repository constraints are facts about structure, not "memories".
        if inv.manifest_signals.get("has_tests"):
            self.world.add_constraint(Constraint(
                "structural",
                "repository contains an explicit test tree",
                "inventory",
            ))
        if inv.manifest_signals.get("has_ci"):
            self.world.add_constraint(Constraint(
                "structural",
                "repository contains CI/workflow configuration",
                "inventory",
            ))

    def ingest_source(self, path: str, content: str) -> ParsedCode:
        ref = self.artifacts.register(path, content=content)
        feid = self._file_eid(path)

        # Upgrade/replace inventory-level file entity with content evidence.
        self.world.add_entity(Entity(feid, "file", {
            "path": path,
            "artifact_id": ref.artifact_id,
            "sha256": ref.sha256,
            "media_kind": "source",
        }))

        parsed = self.parser.parse(path, content)
        self.parsed[path] = parsed

        for name in parsed.classes:
            sid = self._symbol_eid(path, "class", name)
            self.world.add_entity(Entity(sid, "class", {"name": name, "file": path}))
            self.world.add_relation(Relation(feid, "DEFINES", sid, 0.98, path))

        for name in parsed.functions:
            sid = self._symbol_eid(path, "function", name)
            self.world.add_entity(Entity(sid, "function", {"name": name, "file": path}))
            self.world.add_relation(Relation(feid, "DEFINES", sid, 0.94, path))

        for imp in parsed.imports:
            candidate = self._package_import_candidate(imp)
            target = self._file_eid(candidate) if candidate else None

            if target is not None and target in self.world.entities:
                self.world.add_relation(Relation(feid, "IMPORTS", target, 0.99, imp))
            else:
                dep_id = f"dependency:{imp}"
                if dep_id not in self.world.entities:
                    self.world.add_entity(Entity(dep_id, "external_dependency", {
                        "name": imp
                    }))
                self.world.add_relation(Relation(feid, "IMPORTS", dep_id, 0.99, imp))

        return parsed

    def infer_test_relations(self) -> int:
        """
        Conservative filename-based test linking:
        test/foo_test.dart -> any unique lib/**/foo.dart
        """
        lib_by_base: Dict[str, List[str]] = {}
        for e in self.world.entities.values():
            if e.kind != "file":
                continue
            p = e.attrs.get("path") or ""
            if p.startswith("lib/"):
                lib_by_base.setdefault(PurePosixPath(p).name, []).append(p)

        added = 0
        for e in list(self.world.entities.values()):
            if e.kind != "file":
                continue
            p = e.attrs.get("path") or ""
            if not p.startswith(("test/", "tests/")):
                continue
            name = PurePosixPath(p).name
            if not name.endswith("_test.dart"):
                continue
            target_name = name[:-len("_test.dart")] + ".dart"
            candidates = lib_by_base.get(target_name, [])
            if len(candidates) == 1:
                self.world.add_relation(Relation(
                    self._file_eid(p),
                    "TESTS",
                    self._file_eid(candidates[0]),
                    0.85,
                    "filename convention",
                ))
                added += 1
        return added

    def architecture_summary(self) -> dict:
        stats = self.world.stats()
        imports = [r for r in self.world.relations if r.rel == "IMPORTS"]
        defs = [r for r in self.world.relations if r.rel == "DEFINES"]
        tests = [r for r in self.world.relations if r.rel == "TESTS"]
        return {
            **stats,
            "raw_artifacts_retained": len(self.artifacts),
            "parsed_source_files": len(self.parsed),
            "import_edges": len(imports),
            "definition_edges": len(defs),
            "test_edges": len(tests),
            "parser_semantics": "lexical_static",
        }


# ---------------------------------------------------------------------------
# Router facade
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BuiltDomainModel:
    decision: RouteDecision
    model: Optional[Any]


class DomainModelBuilder:
    def __init__(self):
        self.router = DomainRouter()

    def build_repository(self, inv: RepositoryInventory) -> BuiltDomainModel:
        decision = self.router.route_repository(inv)
        if decision.domain == Domain.CODE:
            model = CodeWorldModel()
            model.ingest_inventory(inv)
            return BuiltDomainModel(decision, model)
        # Refuse to pretend another domain adapter exists.
        return BuiltDomainModel(decision, None)
