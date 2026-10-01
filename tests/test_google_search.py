import unittest

from digital_mind_core.google_search import (
    GoogleSearchClient,
    GoogleSearchConfig,
    GoogleSearchUnavailable,
)
from digital_mind_core.tool_module import MindModule


class GoogleSearchTests(unittest.TestCase):
    def fake_google(self, query, num):
        return {
            "provider": "google",
            "results": [
                {
                    "title": "Example result",
                    "link": "https://example.org/result",
                    "snippet": f"result for {query}",
                    "displayLink": "example.org",
                }
            ][:num],
        }

    def test_unconfigured_client_fails_closed(self):
        client = GoogleSearchClient(
            GoogleSearchConfig(api_key=None, search_engine_id=None)
        )
        self.assertEqual(
            client.status()["status"],
            "NEEDS_GOOGLE_CREDENTIALS_OR_VERIFIED_HOST",
        )
        with self.assertRaisesRegex(GoogleSearchUnavailable, "not connected"):
            client.search("test")

    def test_verified_host_google_callback(self):
        client = GoogleSearchClient(
            GoogleSearchConfig(api_key=None, search_engine_id=None),
            host_search=self.fake_google,
            host_declares_google=True,
        )
        result = client.search("digital mind", 1)
        self.assertEqual(result["provider"], "google_host")
        self.assertEqual(result["results"][0]["title"], "Example result")
        self.assertEqual(client.status()["last_transport"], "host_google_callback")

    def test_unverified_host_is_rejected(self):
        client = GoogleSearchClient(
            GoogleSearchConfig(api_key=None, search_engine_id=None),
            host_search=self.fake_google,
            host_declares_google=False,
        )
        self.assertEqual(
            client.status()["status"],
            "HOST_SEARCH_PRESENT_BUT_NOT_GOOGLE_VERIFIED",
        )
        with self.assertRaises(GoogleSearchUnavailable):
            client.search("test")

    def test_module_exposes_google_status_and_search(self):
        client = GoogleSearchClient(
            GoogleSearchConfig(api_key=None, search_engine_id=None),
            host_search=self.fake_google,
            host_declares_google=True,
        )
        module = MindModule(google_client=client)
        status = module.execute({"op": "google_status"})
        self.assertEqual(status["status"], "READY_HOST_GOOGLE")
        result = module.execute({
            "op": "google_search",
            "query": "kernel research",
            "num": 1,
        })
        self.assertEqual(result["provider"], "google_host")

    def test_credentials_never_appear_in_status(self):
        client = GoogleSearchClient(
            GoogleSearchConfig(
                api_key="secret-api-key",
                search_engine_id="secret-cx",
            )
        )
        status = repr(client.status())
        self.assertNotIn("secret-api-key", status)
        self.assertNotIn("secret-cx", status)
        self.assertFalse(client.status()["credentials_exposed"])


if __name__ == "__main__":
    unittest.main()
