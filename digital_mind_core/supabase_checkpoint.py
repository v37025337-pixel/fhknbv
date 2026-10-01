"""Service-only Supabase materialization for the atomic kernel checkpoint.

The checkpoint stays a single JSON document in one singleton Postgres row.
This client uses only the standard library and supports both legacy JWT
service_role keys and current sb_secret_* backend keys.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode, urlparse
import urllib.error
import urllib.request

from .checkpoint import (
    read_checkpoint_document,
    restore_checkpoint_document,
    validate_checkpoint_document,
)


TABLE = "digital_mind_checkpoint_current"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class SupabaseCheckpointUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SupabaseCheckpointConfig:
    url: str | None = None
    secret_key: str | None = None
    timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls):
        key = (
            os.environ.get("SUPABASE_SECRET_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        )
        return cls(
            url=os.environ.get("SUPABASE_URL"),
            secret_key=key,
        )

    def ready(self):
        return bool(self.url and self.secret_key)

    def endpoint(self):
        if not self.url:
            raise SupabaseCheckpointUnavailable("SUPABASE_URL is not configured")
        parsed = urlparse(self.url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("SUPABASE_URL must be an https URL")
        return self.url.rstrip("/")


def _default_request(method, url, headers, body, timeout):
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise SupabaseCheckpointUnavailable("Supabase response exceeds size limit")
            return response.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read(MAX_RESPONSE_BYTES + 1)
        message = raw.decode("utf-8", errors="replace")[:4000]
        raise SupabaseCheckpointUnavailable(
            f"Supabase HTTP {exc.code}: {message}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SupabaseCheckpointUnavailable(
            f"Supabase request failed: {exc.reason}"
        ) from exc


class SupabaseCheckpointStore:
    def __init__(self, config=None, *, request_fn=None):
        self.config = config or SupabaseCheckpointConfig.from_environment()
        self.request_fn = request_fn or _default_request

    def status(self):
        host = None
        if self.config.url:
            host = urlparse(self.config.url).netloc or None
        return {
            "configured": self.config.ready(),
            "host": host,
            "table": TABLE,
            "secret_exposed": False,
        }

    def _headers(self, *, prefer=None):
        if not self.config.ready():
            raise SupabaseCheckpointUnavailable(
                "Supabase checkpoint store requires SUPABASE_URL and "
                "SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY"
            )
        key = self.config.secret_key
        headers = {
            "apikey": key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "digital-mind-checkpoint/1",
        }
        # Current sb_secret_* keys are opaque and must not be sent as Bearer JWTs.
        if not key.startswith("sb_secret_"):
            headers["Authorization"] = "Bearer " + key
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _json_request(self, method, path, *, body=None, prefer=None):
        encoded = None
        if body is not None:
            encoded = json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        status, raw = self.request_fn(
            method,
            self.config.endpoint() + path,
            self._headers(prefer=prefer),
            encoded,
            float(self.config.timeout_seconds),
        )
        if not 200 <= status < 300:
            raise SupabaseCheckpointUnavailable(f"unexpected Supabase HTTP status: {status}")
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SupabaseCheckpointUnavailable("Supabase returned invalid JSON") from exc

    @staticmethod
    def row_for(document):
        document = validate_checkpoint_document(document)
        identity = document["identity"]
        integrity = document["integrity"]
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        return {
            "id": 1,
            "checkpoint_id": integrity["checkpoint_id"],
            "package_version": identity["package_version"],
            "git_commit": identity.get("git_commit"),
            "source_sha256": identity["source_sha256"],
            "state_sha256": identity["state_sha256"],
            "payload_sha256": integrity["payload_sha256"],
            "payload": document,
            "byte_size": len(encoded),
            "verification_error": None,
        }

    def push_document(self, document):
        row = self.row_for(document)
        query = urlencode({
            "on_conflict": "id",
            "select": (
                "id,generation,checkpoint_id,package_version,git_commit,"
                "source_sha256,state_sha256,payload_sha256,byte_size,"
                "materialized_at,verified_at,verification_error"
            ),
        })
        result = self._json_request(
            "POST",
            f"/rest/v1/{TABLE}?{query}",
            body=row,
            prefer="resolution=merge-duplicates,return=representation",
        )
        if not isinstance(result, list) or len(result) != 1:
            raise SupabaseCheckpointUnavailable("Supabase upsert returned no singleton row")
        stored = result[0]
        if stored.get("checkpoint_id") != row["checkpoint_id"]:
            raise SupabaseCheckpointUnavailable("Supabase returned a different checkpoint id")
        return stored

    def push_file(self, path):
        return self.push_document(read_checkpoint_document(path))

    def pull_row(self):
        query = urlencode({
            "id": "eq.1",
            "select": (
                "id,generation,checkpoint_id,package_version,git_commit,"
                "source_sha256,state_sha256,payload_sha256,payload,byte_size,"
                "materialized_at,verified_at,verification_error"
            ),
            "limit": "1",
        })
        result = self._json_request("GET", f"/rest/v1/{TABLE}?{query}")
        if not isinstance(result, list) or len(result) != 1:
            raise SupabaseCheckpointUnavailable("no materialized kernel checkpoint exists")
        return result[0]

    def pull_document(self):
        row = self.pull_row()
        document = validate_checkpoint_document(row.get("payload"))
        if row.get("checkpoint_id") != document["integrity"]["checkpoint_id"]:
            raise SupabaseCheckpointUnavailable("checkpoint id differs from stored payload")
        if row.get("payload_sha256") != document["integrity"]["payload_sha256"]:
            raise SupabaseCheckpointUnavailable("checkpoint digest differs from stored payload")
        return document

    def restore_mind(self):
        return restore_checkpoint_document(self.pull_document())

    def pull_to_file(self, path):
        document = self.pull_document()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return destination


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Push/pull the current DIGITAL_MIND atomic checkpoint through Supabase"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    push = commands.add_parser("push")
    push.add_argument("path", type=Path)
    pull = commands.add_parser("pull")
    pull.add_argument("path", type=Path)
    commands.add_parser("status")
    args = parser.parse_args(argv)

    store = SupabaseCheckpointStore()
    if args.command == "status":
        print(json.dumps(store.status(), ensure_ascii=False, sort_keys=True))
        return
    if args.command == "push":
        row = store.push_file(args.path)
        print(json.dumps({
            "checkpoint_id": row["checkpoint_id"],
            "generation": row["generation"],
            "byte_size": row["byte_size"],
            "materialized_at": row["materialized_at"],
            "verified_at": row["verified_at"],
        }, ensure_ascii=False, sort_keys=True))
        return
    path = store.pull_to_file(args.path)
    mind = restore_checkpoint_document(read_checkpoint_document(path))
    print(json.dumps({
        "path": str(path),
        "steps": mind.fast_steps,
        "checkpoint_id": mind._atomic_checkpoint_metadata["integrity"]["checkpoint_id"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
