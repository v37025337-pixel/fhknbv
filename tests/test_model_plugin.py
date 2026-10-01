import unittest

from digital_mind_core.kernel import KernelConfig
from digital_mind_core.model_plugin import UnifiedKernelModelPlugin


HEAD = "e291ab9787a4fabc12cf1ab286ad88417a0893ff"


class ModelPluginTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.remote = {
            "kernel_version": "1.5-selfdeploy-runtime",
            "lineage_id": "d4908980-fca5-4b35-941c-5bb5c9c066fd",
            "health": {"ok": True},
        }

        def fetch_head(repo, branch):
            return {
                "sha": HEAD,
                "commit": {"message": "Fix UnifiedKernel sidecar verification"},
                "html_url": f"https://github.com/{repo}/commit/{HEAD}",
            }

        self.plugin = UnifiedKernelModelPlugin(
            KernelConfig(seed=123),
            fetch_head=fetch_head,
            state_reader=lambda: self.remote,
            event_writer=self.events.append,
        )

    def test_two_way_turn(self):
        before = self.plugin.before_model("Проверь ядро")
        self.assertEqual(before["turn_id"], 1)
        self.assertEqual(before["repository"]["status"], "UNBOUND")
        self.assertTrue(before["external_state"]["attached"])
        after = self.plugin.after_model("Проверено", success=True)
        self.assertTrue(after["recorded"])
        self.assertTrue(after["event_persisted"])
        self.assertGreater(after["host_response_estimate"], 0.5)
        self.assertEqual(len(self.events), 1)
        self.assertNotIn("Проверено", repr(self.events[0]))

    def test_status_is_explicit_about_boundaries(self):
        status = self.plugin.status()
        self.assertFalse(status["boundaries"]["model_weights_modified"])
        self.assertFalse(status["boundaries"]["raw_chat_persisted_by_plugin"])
        self.assertTrue(status["external_state"]["attached"])

    def test_original_module_surface_still_works(self):
        result = self.plugin.execute({"op": "status"})
        self.assertIn("kernel", result)
        simulated = self.plugin.execute({"op": "simulate", "steps": 1})
        self.assertIsInstance(simulated, dict)

    def test_after_requires_before(self):
        plugin = UnifiedKernelModelPlugin(KernelConfig(seed=2))
        with self.assertRaises(RuntimeError):
            plugin.after_model("orphan response")


if __name__ == "__main__":
    unittest.main()
