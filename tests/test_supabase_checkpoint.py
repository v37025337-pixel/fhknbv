import json
from pathlib import Path
import tempfile
import unittest

from digital_mind_core.checkpoint import build_checkpoint_document
from digital_mind_core.kernel import KernelConfig, UnifiedMind
from digital_mind_core.supabase_checkpoint import (
    SupabaseCheckpointConfig,
    SupabaseCheckpointStore,
)


class FakePostgREST:
    def __init__(self):
        self.row = None
        self.generation = 0
        self.last_headers = None

    def __call__(self, method, url, headers, body, timeout):
        self.last_headers = dict(headers)
        if method == "POST":
            incoming = json.loads(body.decode("utf-8"))
            self.generation += 1
            self.row = {
                **incoming,
                "generation": self.generation,
                "materialized_at": "2026-10-01T21:00:00Z",
                "verified_at": "2026-10-01T21:00:00Z",
            }
            selected = {
                key: self.row.get(key)
                for key in (
                    "id", "generation", "checkpoint_id", "package_version",
                    "git_commit", "source_sha256", "state_sha256",
                    "payload_sha256", "byte_size", "materialized_at",
                    "verified_at", "verification_error",
                )
            }
            return 201, json.dumps([selected]).encode("utf-8")
        if method == "GET":
            return 200, json.dumps([self.row]).encode("utf-8")
        raise AssertionError(method)


class SupabaseCheckpointTests(unittest.TestCase):
    def make_store(self, transport):
        return SupabaseCheckpointStore(
            SupabaseCheckpointConfig(
                url="https://example.supabase.co",
                secret_key="sb_secret_test_only",
            ),
            request_fn=transport,
        )

    def test_push_pull_roundtrip_restores_kernel(self):
        mind = UnifiedMind(KernelConfig(seed=4, l0_learning=False))
        mind.run(16)
        document = build_checkpoint_document(
            mind,
            autodev_state={
                "schema": "digital-mind.autodev-state.v1",
                "development_cycle": 1,
                "deficits": [],
                "attempt_history": [],
                "capability_state": {},
            },
        )
        transport = FakePostgREST()
        store = self.make_store(transport)

        stored = store.push_document(document)
        self.assertEqual(stored["generation"], 1)
        self.assertEqual(stored["checkpoint_id"], document["integrity"]["checkpoint_id"])
        self.assertIn("apikey", transport.last_headers)
        self.assertNotIn("Authorization", transport.last_headers)

        pulled = store.pull_document()
        self.assertEqual(pulled["integrity"], document["integrity"])
        restored = store.restore_mind()
        self.assertEqual(restored.fast_steps, 16)

    def test_legacy_service_role_uses_bearer_too(self):
        transport = FakePostgREST()
        store = SupabaseCheckpointStore(
            SupabaseCheckpointConfig(
                url="https://example.supabase.co",
                secret_key="eyJlegacy-service-role-test",
            ),
            request_fn=transport,
        )
        mind = UnifiedMind(KernelConfig(l0_learning=False))
        doc = build_checkpoint_document(mind, autodev_state=None)
        store.push_document(doc)
        self.assertEqual(
            transport.last_headers["Authorization"],
            "Bearer eyJlegacy-service-role-test",
        )

    def test_status_never_exposes_secret(self):
        store = self.make_store(FakePostgREST())
        status = store.status()
        self.assertTrue(status["configured"])
        self.assertFalse(status["secret_exposed"])
        self.assertNotIn("key", json.dumps(status).lower())

    def test_pull_to_file_revalidates_payload(self):
        mind = UnifiedMind(KernelConfig(seed=2, l0_learning=False))
        doc = build_checkpoint_document(mind, autodev_state=None)
        transport = FakePostgREST()
        store = self.make_store(transport)
        store.push_document(doc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            store.pull_to_file(path)
            self.assertEqual(
                json.loads(path.read_text())["integrity"]["checkpoint_id"],
                doc["integrity"]["checkpoint_id"],
            )


if __name__ == "__main__":
    unittest.main()
