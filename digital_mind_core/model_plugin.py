"""Host/model plugin facade for the DIGITAL_MIND / UnifiedKernel codebase.

The model itself is not modified.  A host process can call before_model() and
after_model() around each model turn, while repository and durable-state access
remain explicit injected capabilities.

Raw chat text is not sent to persistence callbacks; only bounded derived turn
metadata is emitted.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from typing import Any, Callable

from .github_sidecar import GitHubRepositorySidecar
from .kernel import KernelConfig
from .internet import WebReader
from .tool_module import MAX_REQUEST_CHARS, MindModule, _json_result, _reject_nonfinite


def _features(text):
    text = str(text)
    words = [x for x in text.replace("\n", " ").split(" ") if x]
    low = text.lower()
    return {
        "chars": len(text),
        "words": len(words),
        "questions": text.count("?"),
        "code_hint": any(token in low for token in (
            "python", "код", "def ", "class ", "test", "ошиб",
            "kernel", "ядро", "algorithm", "алгоритм",
        )),
    }


class UnifiedKernelModelPlugin:
    """Stateful plugin wrapper with explicit host capabilities."""

    def __init__(
        self,
        config=None,
        *,
        repository="v37025337-pixel/fhknbv",
        branch="main",
        fetch_head: Callable[[str, str], dict[str, Any]] | None = None,
        state_reader: Callable[[], dict[str, Any]] | None = None,
        event_writer: Callable[[dict[str, Any]], Any] | None = None,
        web_reader=None,
        web_memory_path=None,
    ):
        self.module = MindModule(config if config is not None else KernelConfig(),
                                 web_reader=web_reader, web_memory_path=web_memory_path)
        self.repository = GitHubRepositorySidecar(repository=repository, branch=branch)
        self.fetch_head = fetch_head
        self.state_reader = state_reader
        self.event_writer = event_writer
        self.turn_counter = 0
        self.pending_turn = None

    def repository_status(self):
        if self.fetch_head is None:
            return {
                **self.repository.snapshot(),
                "status": "HOST_TRANSPORT_NOT_ATTACHED",
            }
        result = self.repository.verify(self.fetch_head)
        # Branch tracking means any valid current main HEAD is accepted.  The
        # exact SHA is retained as provenance rather than a permanent expected
        # value that would make the next commit look like drift.
        return {**result, "tracking_mode": "branch_head"}

    def external_state(self):
        if self.state_reader is None:
            return {"attached": False, "state": None}
        state = self.state_reader()
        if not isinstance(state, dict):
            raise TypeError("state_reader must return a dict")
        return {"attached": True, "state": copy.deepcopy(state)}

    def before_model(self, text, task=None):
        self.turn_counter += 1
        turn_id = self.turn_counter
        features = _features(text)
        work = copy.deepcopy(task) if task is not None else {}
        if not isinstance(work, dict):
            raise ValueError("task must be a JSON object")

        observations = list(work.get("observations", []))
        observations.append({
            "source": "model_host",
            "turn_id": turn_id,
            "features": features,
        })
        work["observations"] = observations
        work.setdefault("context", "model_plugin")
        work.setdefault("preserve_context", True)
        work.setdefault("candidate_actions", [
            "answer", "inspect", "test", "synthesize",
        ])

        result = self.module.mind.process(work)
        self.pending_turn = turn_id
        return _json_result({
            "turn_id": turn_id,
            "features": features,
            "kernel": {
                "action": result.get("action"),
                "reason": result.get("reason"),
                "cycle": result.get("cycle"),
                "inferences": result.get("inferences"),
                "plan": result.get("plan"),
            },
            "repository": self.repository_status(),
            "external_state": self.external_state(),
        })

    def after_model(self, text, *, success=None, metadata=None):
        if self.pending_turn is None:
            raise RuntimeError("before_model must be called first")
        turn_id = self.pending_turn
        features = _features(text)

        task = {
            "context": "model_plugin",
            "preserve_context": True,
            "observations": [{
                "source": "model_response",
                "turn_id": turn_id,
                "features": features,
            }],
            "candidate_actions": ["continue"],
        }
        if success is not None:
            if type(success) is not bool:
                raise ValueError("success must be boolean or null")
            task["outcome"] = {
                "capability": "host_response",
                "success": success,
            }

        result = self.module.mind.process(task)
        event = {
            "schema": "unified-kernel.model-turn.v1",
            "turn_id": turn_id,
            "response_features": features,
            "success": success,
            "metadata": copy.deepcopy(metadata or {}),
            "kernel_cycle": result.get("cycle"),
        }
        if self.event_writer is not None:
            self.event_writer(copy.deepcopy(event))

        self.pending_turn = None
        estimate = self.module.mind.cognition.self_model.estimate("host_response")
        return _json_result({
            "turn_id": turn_id,
            "recorded": True,
            "host_response_estimate": estimate,
            "event_persisted": self.event_writer is not None,
        })

    def status(self):
        return _json_result({
            "plugin": "UnifiedKernelModelPlugin",
            "turn_counter": self.turn_counter,
            "pending_turn": self.pending_turn,
            "repository": self.repository_status(),
            "external_state": self.external_state(),
            "kernel": self.module.mind.report(),
            "internet": self.module.internet_status(),
            "boundaries": {
                "model_weights_modified": False,
                "host_capabilities_explicit": True,
                "raw_chat_persisted_by_plugin": False,
            },
        })

    def execute(self, request):
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        op = request.get("op")
        if op == "status":
            if set(request) - {"op", "id"}:
                raise ValueError("unknown request fields")
            return self.status()
        if op == "before_model":
            if set(request) - {"op", "id", "text", "task"}:
                raise ValueError("unknown request fields")
            if not isinstance(request.get("text"), str):
                raise ValueError("text must be a string")
            return self.before_model(request["text"], request.get("task"))
        if op == "after_model":
            if set(request) - {"op", "id", "text", "success", "metadata"}:
                raise ValueError("unknown request fields")
            if not isinstance(request.get("text"), str):
                raise ValueError("text must be a string")
            metadata = request.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object or null")
            return self.after_model(
                request["text"],
                success=request.get("success"),
                metadata=metadata,
            )
        if op == "verify_repository":
            if set(request) - {"op", "id"}:
                raise ValueError("unknown request fields")
            return self.repository_status()
        if op == "external_state":
            if set(request) - {"op", "id"}:
                raise ValueError("unknown request fields")
            return self.external_state()
        # Preserve the original MindModule request surface.
        return self.module.execute(request)


def serve(plugin, input_stream, output_stream):
    while True:
        raw = input_stream.readline(MAX_REQUEST_CHARS + 1)
        if not raw:
            return
        identifier = None
        try:
            if len(raw) > MAX_REQUEST_CHARS:
                while raw and not raw.endswith("\n"):
                    raw = input_stream.readline(MAX_REQUEST_CHARS + 1)
                raise ValueError("request exceeds the character limit")
            if not raw.strip():
                continue
            request = json.loads(raw, parse_constant=_reject_nonfinite)
            if isinstance(request, dict) and type(request.get("id")) in (str, int):
                identifier = request["id"]
            result = plugin.execute(request)
            response = {"id": identifier, "ok": True, "result": result}
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            response = {
                "id": identifier,
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        output_stream.write(encoded + "\n")
        output_stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run UnifiedKernel as a model plugin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--internet', action='store_true', help='attach real HTTP(S) GET transport')
    parser.add_argument('--web-memory', help='persist the last 16 web sources as JSON; requires --internet')
    args = parser.parse_args(argv)
    if args.web_memory and not args.internet:
        parser.error('--web-memory requires --internet')
    try:
        config = KernelConfig(seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    serve(UnifiedKernelModelPlugin(config, web_reader=WebReader() if args.internet else None,
          web_memory_path=args.web_memory), sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
