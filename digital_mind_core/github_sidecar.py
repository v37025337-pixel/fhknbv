"""GitHub repository sidecar for DIGITAL_MIND / UnifiedKernel.

The cognitive kernel stays independent from GitHub.  This module only binds a
kernel instance to a canonical repository/branch and verifies the remote HEAD.

Two transport modes are supported:
- host callback: preferred inside ChatGPT or another managed host;
- native HTTPS GET: useful in Codespaces or another Linux environment.

No GitHub write credentials are stored or requested by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable
from urllib.parse import quote
import json
import re
import time
import urllib.request


_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class RepositoryBinding:
    repository: str
    branch: str = "main"
    bound_head: str | None = None

    def __post_init__(self):
        if not _REPO_RE.fullmatch(self.repository):
            raise ValueError("repository must be owner/name")
        if not self.branch or not _BRANCH_RE.fullmatch(self.branch):
            raise ValueError("invalid branch name")
        if self.bound_head is not None and not _SHA_RE.fullmatch(self.bound_head):
            raise ValueError("bound_head must be a 40-character lowercase SHA")


class GitHubRepositorySidecar:
    """Read-only repository binding used by a host/model plugin."""

    def __init__(
        self,
        repository: str = "v37025337-pixel/fhknbv",
        branch: str = "main",
        bound_head: str | None = None,
        *,
        timeout_seconds: float = 10.0,
    ):
        self.binding = RepositoryBinding(repository, branch, bound_head)
        self.timeout_seconds = float(timeout_seconds)
        if not 0.1 <= self.timeout_seconds <= 60.0:
            raise ValueError("timeout_seconds must be in [0.1, 60]")
        self.last_check: dict[str, Any] | None = None

    @staticmethod
    def _normalize_payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("GitHub HEAD payload must be a JSON object")
        sha = str(payload.get("sha", "")).lower()
        if not _SHA_RE.fullmatch(sha):
            raise ValueError("GitHub HEAD payload has no valid commit SHA")
        commit = payload.get("commit")
        message = None
        if isinstance(commit, dict):
            message = commit.get("message")
        return {
            "sha": sha,
            "message": message if isinstance(message, str) else None,
            "html_url": payload.get("html_url"),
        }

    def native_fetch_head(self) -> dict[str, Any]:
        owner, name = self.binding.repository.split("/", 1)
        url = (
            "https://api.github.com/repos/"
            + quote(owner, safe="")
            + "/"
            + quote(name, safe="")
            + "/commits/"
            + quote(self.binding.branch, safe="")
        )
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "digital-mind-core-github-sidecar/1",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            if int(getattr(response, "status", 200)) != 200:
                raise RuntimeError(f"GitHub returned HTTP {response.status}")
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise RuntimeError("GitHub response exceeds size limit")
        return self._normalize_payload(json.loads(raw.decode("utf-8")))

    def verify(
        self,
        fetch_head: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if fetch_head is None:
            head = self.native_fetch_head()
            transport = "native_https"
        else:
            head = self._normalize_payload(
                fetch_head(self.binding.repository, self.binding.branch)
            )
            transport = "host_callback"

        bound = self.binding.bound_head
        status = "UNBOUND" if bound is None else (
            "SYNCED" if head["sha"] == bound else "DRIFTED"
        )
        result = {
            "repository": self.binding.repository,
            "branch": self.binding.branch,
            "bound_head": bound,
            "remote_head": head["sha"],
            "status": status,
            "message": head["message"],
            "html_url": head["html_url"],
            "transport": transport,
            "checked_at": time.time(),
        }
        self.last_check = result
        return dict(result)

    def snapshot(self) -> dict[str, Any]:
        return {
            "binding": asdict(self.binding),
            "last_check": None if self.last_check is None else dict(self.last_check),
            "write_credentials_held": False,
            "model_weights_modified": False,
        }
