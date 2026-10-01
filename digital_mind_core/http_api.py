"""Minimal stateful HTTP adapter for the DIGITAL_MIND kernel.

The adapter intentionally reuses MindModule instead of duplicating kernel logic.
It is meant for black-box integration tests and Codespaces port forwarding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import __version__
from .google_search import GoogleSearchUnavailable
from .kernel import KernelConfig
from .self_diagnostic import diagnose
from .tool_module import MAX_REQUEST_CHARS, MindModule, _json_result, _reject_nonfinite


class DiagnosticBusy(RuntimeError):
    """Raised when a repository self-diagnostic is already running."""


class KernelHTTPService:
    def __init__(self, module=None, *, root=None, diagnostic_fn=diagnose):
        self.module = module if module is not None else MindModule()
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        self.diagnostic_fn = diagnostic_fn
        self._module_lock = threading.RLock()
        self._diagnostic_lock = threading.Lock()

    def execute(self, request):
        with self._module_lock:
            return self.module.execute(request)

    def status(self):
        with self._module_lock:
            kernel = self.module.execute({"op": "status"})
            research = self.module.execute({"op": "google_status"})
        return {
            "schema": "digital-mind.http-status.v1",
            "version": __version__,
            "transport": "http",
            "stateful": True,
            "kernel": kernel,
            "research": research,
        }

    def research(self, request):
        if not isinstance(request, dict):
            raise ValueError("research request must be a JSON object")
        allowed = {"query", "num"}
        if set(request) - allowed:
            raise ValueError("unknown research request fields")
        query = request.get("query")
        num = request.get("num", 5)
        with self._module_lock:
            result = self.module.execute({
                "op": "google_search",
                "query": query,
                "num": num,
            })
        return {
            "schema": "digital-mind.research.v1",
            "result": result,
        }

    def self_diagnostic(self):
        if not self._diagnostic_lock.acquire(blocking=False):
            raise DiagnosticBusy("self-diagnostic is already running")
        try:
            return self.diagnostic_fn(self.root.resolve())
        finally:
            self._diagnostic_lock.release()


class KernelHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, service):
        self.service = service
        super().__init__(server_address, KernelRequestHandler)


class KernelRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DigitalMindHTTP/1"

    def _write_json(self, status, payload):
        encoded = json.dumps(
            _json_result(payload),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status, exc):
        self._write_json(status, {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        })

    def _read_json(self):
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if not 0 < length <= MAX_REQUEST_CHARS:
            raise ValueError(
                f"request body must be from 1 to {MAX_REQUEST_CHARS} bytes"
            )
        raw = self.rfile.read(length)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("request body must be UTF-8 JSON") from exc
        return json.loads(text, parse_constant=_reject_nonfinite)

    @property
    def service(self):
        return self.server.service

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/kernel/status":
                result = self.service.status()
            elif path == "/kernel/self-diagnostic":
                result = self.service.self_diagnostic()
            else:
                self._write_json(404, {
                    "ok": False,
                    "error": {"type": "NotFound", "message": "unknown endpoint"},
                })
                return
            self._write_json(200, {"ok": True, "result": result})
        except DiagnosticBusy as exc:
            self._error(409, exc)
        except Exception as exc:
            self._error(500, exc)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/kernel/execute":
                if not isinstance(body, dict):
                    raise ValueError("execute request must be a JSON object")
                identifier = (
                    body.get("id")
                    if type(body.get("id")) in (str, int)
                    else None
                )
                result = self.service.execute(body)
                self._write_json(200, {
                    "id": identifier,
                    "ok": True,
                    "result": result,
                })
                return
            if path == "/kernel/research":
                result = self.service.research(body)
                self._write_json(200, {"ok": True, "result": result})
                return
            self._write_json(404, {
                "ok": False,
                "error": {"type": "NotFound", "message": "unknown endpoint"},
            })
        except ValueError as exc:
            self._error(400, exc)
        except GoogleSearchUnavailable as exc:
            self._error(503, exc)
        except Exception as exc:
            self._error(500, exc)

    def log_message(self, fmt, *args):
        # Keep operational logs on stderr while avoiding request-body logging.
        super().log_message(fmt, *args)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Expose DIGITAL_MIND as a small stateful HTTP service"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be from 1 to 65535")
    try:
        config = KernelConfig(seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    service = KernelHTTPService(
        MindModule(config),
        root=args.root,
    )
    server = KernelHTTPServer((args.host, args.port), service)
    print(json.dumps({
        "service": "digital-mind-http",
        "host": args.host,
        "port": args.port,
        "version": __version__,
        "endpoints": [
            "POST /kernel/execute",
            "GET /kernel/status",
            "POST /kernel/research",
            "GET /kernel/self-diagnostic",
        ],
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
