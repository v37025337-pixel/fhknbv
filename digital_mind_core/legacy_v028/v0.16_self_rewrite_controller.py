
"""
v0.16_self_rewrite_controller.py

Permissioned self-rewrite controller.

The kernel may rewrite its OWN local source tree, but every rewrite is:
1. staged in an isolated candidate copy,
2. syntax-compiled,
3. tested with allow-listed Python test scripts,
4. promoted atomically only if gates pass,
5. otherwise rejected without changing the active tree.

No shell=True, no arbitrary external deployment, no writes outside workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import py_compile


@dataclass(slots=True)
class RewritePolicy:
    allow_self_modify: bool = True
    allowed_extensions: tuple[str, ...] = (".py", ".json", ".txt")
    max_changed_files: int = 12
    max_total_write_bytes: int = 1_500_000
    test_timeout_seconds: int = 60
    require_tests: bool = True


@dataclass(slots=True)
class FileChange:
    path: str
    content: str


@dataclass(slots=True)
class RewriteProposal:
    proposal_id: str
    goal: str
    rationale: str
    changes: List[FileChange]
    tests: List[str] = field(default_factory=list)


@dataclass(slots=True)
class RewriteResult:
    proposal_id: str
    accepted: bool
    stage: str
    reason: str
    changed_files: List[str]
    test_results: Dict[str, dict]
    candidate_path: Optional[str] = None
    promoted_revision: Optional[str] = None


class SelfRewriteController:
    def __init__(self, workspace_root: str, policy: Optional[RewritePolicy] = None):
        self.root = Path(workspace_root).resolve()
        self.policy = policy or RewritePolicy()

        self.current = self.root / "current"
        self.candidates = self.root / "candidates"
        self.history = self.root / "history"
        self.audit_file = self.root / "rewrite_audit.jsonl"

        self.current.mkdir(parents=True, exist_ok=True)
        self.candidates.mkdir(parents=True, exist_ok=True)
        self.history.mkdir(parents=True, exist_ok=True)

    # ----------------------------- validation -----------------------------

    def _safe_rel(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe target path: {rel!r}")
        if p.suffix not in self.policy.allowed_extensions:
            raise ValueError(f"extension not allowed: {p.suffix!r}")
        return p

    def _validate_proposal(self, proposal: RewriteProposal) -> None:
        if not self.policy.allow_self_modify:
            raise PermissionError("self modification is disabled")
        if not proposal.changes:
            raise ValueError("proposal contains no changes")
        if len(proposal.changes) > self.policy.max_changed_files:
            raise ValueError("too many changed files")

        total = 0
        seen = set()
        for change in proposal.changes:
            rel = self._safe_rel(change.path)
            key = rel.as_posix()
            if key in seen:
                raise ValueError(f"duplicate target: {key}")
            seen.add(key)
            total += len(change.content.encode("utf-8"))

        if total > self.policy.max_total_write_bytes:
            raise ValueError("proposal exceeds write-byte limit")

        for test in proposal.tests:
            t = self._safe_rel(test)
            if not t.name.startswith("test_") or t.suffix != ".py":
                raise ValueError(
                    f"only local test_*.py scripts may be executed: {test!r}"
                )

        if self.policy.require_tests and not proposal.tests:
            raise ValueError("at least one test is required")

    # ----------------------------- audit ----------------------------------

    def _audit(self, event: dict) -> None:
        event = dict(event)
        event["ts"] = time.time()
        with self.audit_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    # ----------------------------- hashing --------------------------------

    @staticmethod
    def _tree_hash(root: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            rel = p.relative_to(root).as_posix().encode("utf-8")
            h.update(rel)
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
        return h.hexdigest()

    # ----------------------------- gates ----------------------------------

    @staticmethod
    def _compile_tree(root: Path) -> Tuple[bool, str]:
        try:
            for p in sorted(root.rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                py_compile.compile(str(p), doraise=True)
            return True, "all Python files compiled"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _run_tests(self, candidate: Path, tests: List[str]) -> Dict[str, dict]:
        results = {}
        for rel in tests:
            target = candidate / rel
            if not target.exists():
                results[rel] = {
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "pass": False,
                    "reason": "test file missing",
                }
                continue

            proc = subprocess.run(
                [sys.executable, str(target)],
                cwd=candidate,
                text=True,
                capture_output=True,
                timeout=self.policy.test_timeout_seconds,
            )
            results[rel] = {
                "returncode": proc.returncode,
                "stdout": proc.stdout[-5000:],
                "stderr": proc.stderr[-5000:],
                "pass": proc.returncode == 0,
            }
        return results

    # ----------------------------- transaction ----------------------------

    def stage_and_evaluate(self, proposal: RewriteProposal) -> RewriteResult:
        try:
            self._validate_proposal(proposal)
        except Exception as exc:
            result = RewriteResult(
                proposal.proposal_id, False, "validation",
                f"{type(exc).__name__}: {exc}", [], {}
            )
            self._audit({"proposal": proposal.proposal_id, "accepted": False,
                         "stage": "validation", "reason": result.reason})
            return result

        candidate = self.candidates / proposal.proposal_id
        if candidate.exists():
            shutil.rmtree(candidate)
        shutil.copytree(self.current, candidate)

        changed = []
        try:
            for change in proposal.changes:
                rel = self._safe_rel(change.path)
                target = candidate / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(change.content, encoding="utf-8")
                changed.append(rel.as_posix())
        except Exception as exc:
            shutil.rmtree(candidate, ignore_errors=True)
            result = RewriteResult(
                proposal.proposal_id, False, "write",
                f"{type(exc).__name__}: {exc}", changed, {}
            )
            self._audit({"proposal": proposal.proposal_id, "accepted": False,
                         "stage": "write", "reason": result.reason})
            return result

        compile_ok, compile_msg = self._compile_tree(candidate)
        if not compile_ok:
            result = RewriteResult(
                proposal.proposal_id, False, "compile",
                compile_msg, changed, {}, str(candidate)
            )
            self._audit({"proposal": proposal.proposal_id, "accepted": False,
                         "stage": "compile", "reason": compile_msg,
                         "changed_files": changed})
            return result

        test_results = self._run_tests(candidate, proposal.tests)
        failed = [name for name, r in test_results.items() if not r["pass"]]
        if failed:
            result = RewriteResult(
                proposal.proposal_id, False, "tests",
                f"failed tests: {failed}", changed, test_results, str(candidate)
            )
            self._audit({"proposal": proposal.proposal_id, "accepted": False,
                         "stage": "tests", "reason": result.reason,
                         "changed_files": changed})
            return result

        revision = self._tree_hash(candidate)
        result = RewriteResult(
            proposal.proposal_id, True, "ready",
            "candidate passed compile and tests", changed, test_results,
            str(candidate), revision
        )
        self._audit({"proposal": proposal.proposal_id, "accepted": True,
                     "stage": "ready", "revision": revision,
                     "changed_files": changed})
        return result

    def promote(self, result: RewriteResult) -> RewriteResult:
        if not result.accepted or result.stage != "ready" or not result.candidate_path:
            raise ValueError("only a ready accepted candidate can be promoted")

        candidate = Path(result.candidate_path).resolve()
        if self.candidates not in candidate.parents:
            raise ValueError("candidate is outside candidate tree")

        old_revision = self._tree_hash(self.current)
        backup = self.history / old_revision[:16]
        if not backup.exists():
            shutil.copytree(self.current, backup)

        incoming = self.root / ".incoming"
        if incoming.exists():
            shutil.rmtree(incoming)
        shutil.copytree(candidate, incoming)

        previous = self.root / ".previous"
        if previous.exists():
            shutil.rmtree(previous)

        os.replace(self.current, previous)
        os.replace(incoming, self.current)
        shutil.rmtree(previous, ignore_errors=True)

        new_revision = self._tree_hash(self.current)
        promoted = RewriteResult(
            result.proposal_id, True, "promoted",
            "candidate promoted atomically", result.changed_files,
            result.test_results, result.candidate_path, new_revision
        )
        self._audit({"proposal": result.proposal_id, "accepted": True,
                     "stage": "promoted", "old_revision": old_revision,
                     "new_revision": new_revision})
        return promoted
