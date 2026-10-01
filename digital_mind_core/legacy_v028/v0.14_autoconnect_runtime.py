
"""
v0.14_autoconnect_runtime.py

AutoConnect layer for v0.13 Unified Runtime.

Goal
----
Given an arbitrary source, choose and attach the appropriate read-only adapter
without the caller manually selecting a domain-specific method.

Supported source classes:
- local source/config/document file
- local directory / repository inventory
- image file (with pluggable perception callback)
- HTTPS URL (with pluggable/read-only fetcher)
- already structured observations

Safety boundary:
- automatic connections are READ-ONLY
- mutating external actions are not implemented here
- raw evidence stays outside episodic compression
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Optional
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import sys
from urllib.parse import urlparse


def _load_sibling(module_name: str, filename: str):
    here = Path(__file__).resolve().parent
    path = here / filename
    if not path.exists():
        raise FileNotFoundError(f"Required sibling module missing: {filename}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod



v13 = _load_sibling("kernel_v13_runtime", "v0.13_unified_runtime.py")

try:
    capability_registry_mod = _load_sibling(
        "kernel_v16_capability_registry",
        "v0.16_capability_registry.py",
    )
except FileNotFoundError:
    capability_registry_mod = None

try:
    data_world_mod = _load_sibling(
        "kernel_v17_data_world",
        "v0.17_data_world.py",
    )
except FileNotFoundError:
    data_world_mod = None


class SourceKind(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"
    IMAGE = "image"
    URL = "url"
    STRUCTURED = "structured"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ConnectionDecision:
    source_kind: SourceKind
    adapter: str
    confidence: float
    reasons: List[str]
    read_only: bool = True


@dataclass(slots=True)
class ConnectionResult:
    decision: ConnectionDecision
    domain: str
    attached: bool
    details: Dict[str, Any] = field(default_factory=dict)


class AutoConnectPolicy:
    """
    Read-only policy. URLs must be HTTPS by default.
    """

    def __init__(
        self,
        *,
        allow_https: bool = True,
        allow_http: bool = False,
        allow_local_files: bool = True,
        allow_local_directories: bool = True,
        max_text_bytes: int = 2_000_000,
    ):
        self.allow_https = bool(allow_https)
        self.allow_http = bool(allow_http)
        self.allow_local_files = bool(allow_local_files)
        self.allow_local_directories = bool(allow_local_directories)
        self.max_text_bytes = int(max_text_bytes)

    def permits_url(self, url: str) -> bool:
        scheme = urlparse(url).scheme.lower()
        if scheme == "https":
            return self.allow_https
        if scheme == "http":
            return self.allow_http
        return False


class SourceDetector:
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    CODE_EXTS = {
        ".dart", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt",
        ".swift", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".go", ".rb",
        ".php", ".sh", ".gradle", ".cmake",
    }
    TEXT_EXTS = CODE_EXTS | {
        ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".json", ".xml",
        ".html", ".css", ".csv", ".tsv",
    }

    @classmethod
    def detect(cls, source: Any) -> ConnectionDecision:
        if isinstance(source, dict) or isinstance(source, list):
            return ConnectionDecision(
                SourceKind.STRUCTURED,
                "structured",
                0.98,
                ["source is already structured data"],
            )

        if isinstance(source, (str, os.PathLike)):
            s = str(source)
            parsed = urlparse(s)
            if parsed.scheme in {"https", "http"} and parsed.netloc:
                return ConnectionDecision(
                    SourceKind.URL,
                    "https_readonly",
                    0.96 if parsed.scheme == "https" else 0.80,
                    [f"{parsed.scheme.upper()} URL detected"],
                )

            p = Path(s)
            if p.exists() and p.is_dir():
                return ConnectionDecision(
                    SourceKind.DIRECTORY,
                    "repository_or_directory",
                    0.98,
                    ["existing local directory detected"],
                )

            if p.exists() and p.is_file():
                ext = p.suffix.lower()
                if ext in cls.IMAGE_EXTS:
                    return ConnectionDecision(
                        SourceKind.IMAGE,
                        "image_perception",
                        0.99,
                        [f"image extension {ext}"],
                    )
                return ConnectionDecision(
                    SourceKind.FILE,
                    "local_file",
                    0.95,
                    [f"existing local file; extension={ext or '<none>'}"],
                )

        return ConnectionDecision(
            SourceKind.UNKNOWN,
            "none",
            0.0,
            ["no supported source signature"],
        )


class PerceptionAdapter:
    """
    Callback contract:
        perception(path: str) -> dict

    Expected examples:
      {"kind":"dns_table", "rows":[...]}
      {"kind":"security_event", "observations":[Observation,...]}
      {"kind":"text", "text":"...", "language":"..."}
    """

    def __init__(self, callback: Optional[Callable[[str], Dict[str, Any]]] = None):
        self.callback = callback

    def perceive(self, path: str) -> Dict[str, Any]:
        if self.callback is None:
            raise RuntimeError(
                "Image detected, but no perception callback is attached."
            )
        result = self.callback(path)
        if not isinstance(result, dict):
            raise TypeError("perception callback must return a dict")
        return result


class ReadOnlyURLAdapter:
    """
    Fetch callback contract:
        fetch(url: str) -> str | bytes | dict

    A default network implementation is intentionally not embedded in the
    kernel core. Runtime hosts can inject their own HTTP client.
    """

    def __init__(self, fetcher: Optional[Callable[[str], Any]] = None):
        self.fetcher = fetcher

    def fetch(self, url: str) -> Any:
        if self.fetcher is None:
            raise RuntimeError(
                "URL detected, but no read-only fetcher is attached."
            )
        return self.fetcher(url)


class AutoConnectRuntime:
    def __init__(
        self,
        *,
        policy: Optional[AutoConnectPolicy] = None,
        perception: Optional[Callable[[str], Dict[str, Any]]] = None,
        url_fetcher: Optional[Callable[[str], Any]] = None,
        runtime_kwargs: Optional[Dict[str, Any]] = None,
        provider_registry=None,
        provider_dirs: Optional[List[str]] = None,
    ):
        self.policy = policy or AutoConnectPolicy()
        self.detector = SourceDetector()

        # Auto-discover providers when explicit callbacks were not supplied.
        self.provider_registry = provider_registry
        if self.provider_registry is None and capability_registry_mod is not None:
            default_dir = Path(__file__).resolve().parent / "providers"
            dirs = list(provider_dirs or [str(default_dir)])
            self.provider_registry = capability_registry_mod.CapabilityRegistry(dirs)
            self.provider_registry.discover()

        if perception is None and self.provider_registry is not None:
            perception = self.provider_registry.resolve("vision.perceive")
        if url_fetcher is None and self.provider_registry is not None:
            url_fetcher = self.provider_registry.resolve("web.fetch.readonly")

        self.perception = PerceptionAdapter(perception)
        self.url_adapter = ReadOnlyURLAdapter(url_fetcher)
        self.runtime = v13.UnifiedRuntime(**(runtime_kwargs or {}))
        self.data_world = data_world_mod.DataWorldModel() if data_world_mod is not None else None
        self.connections: List[ConnectionResult] = []

    # ---------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _manifest_signals(paths: List[str]) -> Dict[str, bool]:
        ps = set(paths)
        return {
            "has_pubspec": "pubspec.yaml" in ps,
            "has_flutter_main": "lib/main.dart" in ps,
            "has_tests": any(p.startswith(("test/", "tests/")) for p in paths),
            "has_ci": any(p.startswith(".github/workflows/") for p in paths),
            "has_android": any(p.startswith("android/") for p in paths),
            "has_ios": any(p.startswith("ios/") for p in paths),
            "has_linux": any(p.startswith("linux/") for p in paths),
            "has_macos": any(p.startswith("macos/") for p in paths),
            "has_windows": any(p.startswith("windows/") for p in paths),
            "has_web": any(p.startswith("web/") for p in paths),
        }

    @staticmethod
    def _walk_directory(root: Path) -> List[str]:
        paths = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            # Skip VCS/cache/build internals.
            rel = p.relative_to(root).as_posix()
            parts = set(PurePosixPath(rel).parts)
            if parts & {".git", ".dart_tool", "build", "__pycache__", ".venv", "node_modules"}:
                continue
            paths.append(rel)
        return sorted(paths)

    def _read_text_file(self, path: Path) -> str:
        if path.stat().st_size > self.policy.max_text_bytes:
            raise ValueError(
                f"text file exceeds max_text_bytes={self.policy.max_text_bytes}"
            )
        return path.read_text(encoding="utf-8", errors="replace")

    # ---------------------------------------------------------------
    # connectors
    # ---------------------------------------------------------------

    def connect(self, source: Any) -> ConnectionResult:
        decision = self.detector.detect(source)

        if decision.source_kind == SourceKind.DIRECTORY:
            if not self.policy.allow_local_directories:
                raise PermissionError("local directory auto-connect disabled")
            result = self._connect_directory(Path(source), decision)

        elif decision.source_kind == SourceKind.FILE:
            if not self.policy.allow_local_files:
                raise PermissionError("local file auto-connect disabled")
            result = self._connect_file(Path(source), decision)

        elif decision.source_kind == SourceKind.IMAGE:
            if not self.policy.allow_local_files:
                raise PermissionError("local image auto-connect disabled")
            result = self._connect_image(str(source), decision)

        elif decision.source_kind == SourceKind.URL:
            result = self._connect_url(str(source), decision)

        elif decision.source_kind == SourceKind.STRUCTURED:
            result = self._connect_structured(source, decision)

        else:
            result = ConnectionResult(
                decision=decision,
                domain="unknown",
                attached=False,
                details={"reason": "unsupported source"},
            )

        self.connections.append(result)
        return result

    def _connect_directory(
        self, root: Path, decision: ConnectionDecision
    ) -> ConnectionResult:
        paths = self._walk_directory(root)
        signals = self._manifest_signals(paths)
        workspace_id = "dir:" + hashlib.sha256(
            str(root.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        route = self.runtime.ingest_repository(
            paths,
            manifest_signals=signals,
            workspace_id=workspace_id,
        )

        parsed = 0
        skipped = 0
        if route.domain.value == "code" and self.runtime.code_world is not None:
            for rel in paths:
                ext = PurePosixPath(rel).suffix.lower()
                if ext not in SourceDetector.CODE_EXTS:
                    continue
                full = root / rel
                try:
                    content = self._read_text_file(full)
                    self.runtime.ingest_code_source(
                        rel, content, workspace_id=workspace_id
                    )
                    parsed += 1
                except Exception:
                    skipped += 1
            self.runtime.code_world.infer_test_relations()

        return ConnectionResult(
            decision=decision,
            domain=route.domain.value,
            attached=True,
            details={
                "root": str(root),
                "files_discovered": len(paths),
                "source_files_parsed": parsed,
                "source_files_skipped": skipped,
                "route_confidence": route.confidence,
                "workspace_id": workspace_id,
            },
        )

    def _connect_file(
        self, path: Path, decision: ConnectionDecision
    ) -> ConnectionResult:
        ext = path.suffix.lower()

        if ext in SourceDetector.CODE_EXTS:
            # A lone code file is represented as a one-file repository.
            workspace_id = "file:" + hashlib.sha256(
                str(path.resolve()).encode("utf-8")
            ).hexdigest()[:12]
            route = self.runtime.ingest_repository(
                [path.name],
                manifest_signals={},
                workspace_id=workspace_id,
            )
            if route.domain.value == "code" and self.runtime.code_world is not None:
                content = self._read_text_file(path)
                parsed = self.runtime.ingest_code_source(
                    path.name, content, workspace_id=workspace_id
                )
                return ConnectionResult(
                    decision=decision,
                    domain="code",
                    attached=True,
                    details={
                        "path": str(path),
                        "language": parsed.language,
                        "imports": len(parsed.imports),
                        "classes": len(parsed.classes),
                        "functions": len(parsed.functions),
                    },
                )

        # Structured data gets a dedicated world model before generic document routing.
        if self.data_world is not None and ext in {".csv", ".tsv"}:
            ds = self.data_world.ingest_delimited(
                str(path), delimiter="," if ext == ".csv" else "\t"
            )
            schema = self.data_world.schema_dict(ds)
            self.runtime.active_domains.add("data")
            self.runtime._audit(
                "autoconnect_data",
                "data",
                f"structured table attached: {path.name}; rows={ds.row_count}",
                raw_evidence=str(path),
            )
            self.runtime.remember_derived(
                f"derived:data:{path.name}:rows={ds.row_count}:fields={len(ds.fields)}",
                utility=0.55, novelty=0.6, causal=0.35, kind=41,
            )
            return ConnectionResult(
                decision=decision,
                domain="data",
                attached=True,
                details={"path": str(path), "schema": schema, "read_only": True},
            )

        if self.data_world is not None and ext == ".json":
            ds = self.data_world.ingest_json_file(str(path))
            schema = self.data_world.schema_dict(ds)
            self.runtime.active_domains.add("structured_data")
            self.runtime._audit(
                "autoconnect_data",
                "structured_data",
                f"JSON data attached: {path.name}",
                raw_evidence=str(path),
            )
            self.runtime.remember_derived(
                f"derived:json:{path.name}:rows={ds.row_count}:fields={len(ds.fields)}",
                utility=0.55, novelty=0.6, causal=0.35, kind=42,
            )
            return ConnectionResult(
                decision=decision,
                domain="structured_data",
                attached=True,
                details={"path": str(path), "schema": schema, "read_only": True},
            )

        # Non-code text/config becomes a generic structured document event.
        if ext in SourceDetector.TEXT_EXTS:
            text = self._read_text_file(path)
            self.runtime._audit(
                "autoconnect_file",
                "document",
                f"read-only text artifact attached: {path.name}",
                raw_evidence=str(path),
            )
            self.runtime.active_domains.add("document")
            self.runtime.remember_derived(
                f"derived:document:{path.name}:bytes={len(text.encode('utf-8'))}",
                utility=0.4,
                novelty=0.5,
                causal=0.2,
                kind=40,
            )
            return ConnectionResult(
                decision=decision,
                domain="document",
                attached=True,
                details={"path": str(path), "chars": len(text)},
            )

        return ConnectionResult(
            decision=decision,
            domain="unknown",
            attached=False,
            details={"path": str(path), "reason": "unsupported file type"},
        )

    def _connect_image(
        self, path: str, decision: ConnectionDecision
    ) -> ConnectionResult:
        perceived = self.perception.perceive(path)
        kind = perceived.get("kind")

        if kind == "dns_table":
            domain, confidence = self.runtime.ingest_dns_image(
                path, perceived.get("rows", [])
            )
            return ConnectionResult(
                decision=decision,
                domain=domain.value,
                attached=True,
                details={"perception_kind": kind, "confidence": confidence},
            )

        if kind == "security_event":
            domain, confidence = self.runtime.ingest_security_image(
                path, perceived.get("observations", [])
            )
            return ConnectionResult(
                decision=decision,
                domain=domain.value,
                attached=True,
                details={"perception_kind": kind, "confidence": confidence},
            )

        if kind == "text":
            self.runtime._audit(
                "perception",
                "document",
                "image perceived as generic text document",
                raw_evidence=path,
            )
            self.runtime.active_domains.add("document")
            return ConnectionResult(
                decision=decision,
                domain="document",
                attached=True,
                details={"perception_kind": kind},
            )

        return ConnectionResult(
            decision=decision,
            domain="unknown",
            attached=False,
            details={"reason": f"unsupported perception kind: {kind!r}"},
        )

    def _connect_url(
        self, url: str, decision: ConnectionDecision
    ) -> ConnectionResult:
        if not self.policy.permits_url(url):
            raise PermissionError(f"URL policy rejected: {url}")
        payload = self.url_adapter.fetch(url)

        # Structured remote payload can route recursively.
        if isinstance(payload, (dict, list)):
            nested = self._connect_structured(payload, ConnectionDecision(
                SourceKind.STRUCTURED, "structured", 0.98,
                ["structured payload returned by read-only URL fetcher"],
            ))
            nested.details["url"] = url
            return nested

        if isinstance(payload, bytes):
            try:
                payload = payload.decode("utf-8")
            except UnicodeDecodeError:
                return ConnectionResult(
                    decision=decision,
                    domain="unknown",
                    attached=False,
                    details={"url": url, "reason": "binary remote payload unsupported"},
                )

        if isinstance(payload, str):
            suffix = PurePosixPath(urlparse(url).path).suffix.lower()
            if suffix in SourceDetector.CODE_EXTS:
                name = PurePosixPath(urlparse(url).path).name or "remote_source"
                workspace_id = "url:" + hashlib.sha256(
                    url.encode("utf-8")
                ).hexdigest()[:12]
                route = self.runtime.ingest_repository(
                    [name],
                    manifest_signals={},
                    workspace_id=workspace_id,
                )
                if route.domain.value == "code" and workspace_id in self.runtime.code_worlds:
                    parsed = self.runtime.ingest_code_source(
                        name, payload, workspace_id=workspace_id
                    )
                    return ConnectionResult(
                        decision=decision,
                        domain="code",
                        attached=True,
                        details={
                            "url": url,
                            "language": parsed.language,
                            "read_only": True,
                            "workspace_id": workspace_id,
                        },
                    )

            self.runtime.active_domains.add("document")
            self.runtime._audit(
                "autoconnect_url",
                "document",
                f"read-only URL attached: {urlparse(url).netloc}",
            )
            return ConnectionResult(
                decision=decision,
                domain="document",
                attached=True,
                details={"url": url, "chars": len(payload), "read_only": True},
            )

        return ConnectionResult(
            decision=decision,
            domain="unknown",
            attached=False,
            details={"url": url, "reason": "unsupported fetch payload"},
        )

    def _connect_structured(
        self, source: Any, decision: ConnectionDecision
    ) -> ConnectionResult:
        if isinstance(source, dict) and source.get("kind") == "dns_table":
            # Structured evidence without an image has no raw image path.
            rows = source.get("rows", [])
            self.runtime.active_domains.add("network_config")
            self.runtime.remember_derived(
                f"derived:structured-dns:rows={len(rows)}",
                utility=0.4, novelty=0.4, causal=0.3, kind=22,
            )
            return ConnectionResult(
                decision=decision,
                domain="network_config",
                attached=True,
                details={"rows": len(rows), "raw_evidence": False},
            )

        if isinstance(source, dict) and source.get("kind") == "security_event":
            self.runtime.active_domains.add("security_event")
            self.runtime.remember_derived(
                "derived:structured-security-event:verification-required",
                utility=0.8, novelty=0.7, causal=0.8, kind=23,
            )
            return ConnectionResult(
                decision=decision,
                domain="security_event",
                attached=True,
                details={"raw_evidence": False},
            )

        if self.data_world is not None and isinstance(source, (dict, list)):
            ds = self.data_world.ingest_payload(source)
            schema = self.data_world.schema_dict(ds)
            self.runtime.active_domains.add("structured_data")
            self.runtime.remember_derived(
                f"derived:structured-data:rows={ds.row_count}:fields={len(ds.fields)}",
                utility=0.5, novelty=0.55, causal=0.3, kind=43,
            )
            return ConnectionResult(
                decision=decision,
                domain="structured_data",
                attached=True,
                details={"schema": schema, "raw_evidence": False},
            )

        self.runtime.active_domains.add("document")
        return ConnectionResult(
            decision=decision,
            domain="document",
            attached=True,
            details={"structured_type": type(source).__name__},
        )

    def snapshot(self):
        return {
            "connections": [
                {
                    "source_kind": x.decision.source_kind.value,
                    "adapter": x.decision.adapter,
                    "domain": x.domain,
                    "attached": x.attached,
                    "details": x.details,
                }
                for x in self.connections
            ],
            "runtime": self.runtime.snapshot(),
            "data_world": None if self.data_world is None else self.data_world.stats(),
        }
