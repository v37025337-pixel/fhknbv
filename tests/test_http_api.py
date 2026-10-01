import json
from pathlib import Path
import threading
import unittest
import urllib.error
import urllib.request

from digital_mind_core.http_api import KernelHTTPServer, KernelHTTPService


class FakeModule:
    def __init__(self):
        self.steps = 0
        self.searches = 0

    def execute(self, request):
        op = request.get("op")
        if op == "status":
            return {"steps": self.steps}
        if op == "simulate":
            self.steps += request.get("steps", 1)
            return {"steps": self.steps}
        if op == "google_status":
            return {
                "status": "READY_TEST_GOOGLE",
                "search_count": self.searches,
            }
        if op == "google_search":
            self.searches += 1
            return {
                "provider": "google_test",
                "query": request.get("query"),
                "results": [{"title": "result", "link": "https://example.org"}],
            }
        raise ValueError("unsupported fake operation")


class HTTPAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = FakeModule()
        cls.service = KernelHTTPService(
            cls.module,
            root=Path("."),
            diagnostic_fn=lambda root: {
                "schema": "diagnostic.test.v1",
                "root_exists": root.exists(),
            },
        )
        cls.server = KernelHTTPServer(("127.0.0.1", 0), cls.service)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, payload=None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_execute_preserves_state(self):
        before = self.module.steps
        status, first = self.request(
            "POST",
            "/kernel/execute",
            {"id": "run-1", "op": "simulate", "steps": 3},
        )
        self.assertEqual(status, 200)
        self.assertEqual(first["id"], "run-1")
        self.assertTrue(first["ok"])
        self.assertEqual(first["result"]["steps"], before + 3)

        _, second = self.request("GET", "/kernel/status")
        self.assertEqual(second["result"]["kernel"]["steps"], before + 3)
        self.assertTrue(second["result"]["stateful"])

    def test_research_uses_google_adapter(self):
        status, body = self.request(
            "POST",
            "/kernel/research",
            {"query": "bounded program synthesis", "num": 2},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(
            body["result"]["result"]["query"],
            "bounded program synthesis",
        )

    def test_self_diagnostic_endpoint(self):
        status, body = self.request("GET", "/kernel/self-diagnostic")
        self.assertEqual(status, 200)
        self.assertEqual(
            body["result"]["schema"],
            "diagnostic.test.v1",
        )
        self.assertTrue(body["result"]["root_exists"])

    def test_invalid_execute_request_returns_400(self):
        req = urllib.request.Request(
            self.base + "/kernel/execute",
            data=json.dumps({"op": "unknown"}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=3)
        self.assertEqual(caught.exception.code, 400)
        body = json.loads(caught.exception.read().decode("utf-8"))
        self.assertFalse(body["ok"])


if __name__ == "__main__":
    unittest.main()
