import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from digital_mind_core.checkpoint import (
    AtomicCheckpointError,
    CHECKPOINT_FORMAT,
    component_documents,
    load_checkpoint,
    restore_autodev_state,
    save_checkpoint,
)
from digital_mind_core.kernel import KernelConfig, UnifiedMind
from digital_mind_core.persistence import state_fingerprint


AUTODEV = {
    "schema": "digital-mind.autodev-state.v1",
    "development_cycle": 7,
    "deficits": [{"deficit_id": "test", "severity": 0.5}],
    "attempt_history": [{"attempt_id": "a1", "result": "REJECTED"}],
    "capability_state": {"github": "CONNECTED"},
}


class AtomicCheckpointTests(unittest.TestCase):
    def test_single_checkpoint_roundtrip_includes_all_named_components(self):
        mind = UnifiedMind(KernelConfig(seed=2, l0_learning=False))
        mind.run(96)
        self.assertIn("choose_plan_cost", mind.cognition.mechanism_info())
        mind.process({
            "facts": [("external_fact", "alpha")],
            "persist_facts": True,
            "queries": [("external_fact", "alpha")],
        })
        mind.run_mechanism("stability_switch", history=10.0, current=12.0)
        mind.predict_self_counterfactual(replan=1.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.checkpoint.json"
            checkpoint_id = save_checkpoint(mind, path, autodev_state=AUTODEV)
            raw = json.loads(path.read_text())
            self.assertEqual(raw["format"], CHECKPOINT_FORMAT)
            self.assertEqual(raw["integrity"]["checkpoint_id"], checkpoint_id)
            self.assertEqual(
                set(raw["components"]),
                {"cognition", "memory", "self_model", "mechanisms", "runtime"},
            )
            self.assertTrue(raw["autodev"]["present"])
            self.assertEqual(raw["autodev"]["state"]["development_cycle"], 7)
            self.assertEqual(len(raw["identity"]["source_sha256"]), 64)

            restored = load_checkpoint(path)
            self.assertEqual(state_fingerprint(restored), state_fingerprint(mind))
            self.assertEqual(component_documents(restored), component_documents(mind))
            self.assertEqual(restored._autodev_state["state"], AUTODEV)

    def test_external_process_is_replayed_at_its_original_step(self):
        mind = UnifiedMind(KernelConfig(seed=3, l0_learning=False))
        mind.run(24)
        mind.process({
            "facts": [("persistent_external", "yes")],
            "persist_facts": True,
            "queries": [("persistent_external", "yes")],
        })
        mind.run(16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_checkpoint(mind, path, autodev_state=AUTODEV)
            restored = load_checkpoint(path)
            answer = restored.process({"queries": [("persistent_external", "yes")], "persist_facts": True})
            self.assertTrue(answer["inferences"][0]["value"])
            self.assertEqual(restored.fast_steps, 40)

    def test_connection_restore_does_not_need_original_file(self):
        mind = UnifiedMind(KernelConfig(seed=4, l0_learning=False))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "states.csv"
            data.write_text(
                "Rank,State,Population\n1,Alabama,4849377\n2,Alaska,736732\n",
                encoding="utf-8",
            )
            result = mind.connect(data)
            self.assertTrue(result.attached)
            checkpoint = root / "checkpoint.json"
            save_checkpoint(mind, checkpoint, autodev_state=AUTODEV)
            data.unlink()
            restored = load_checkpoint(checkpoint)
            self.assertEqual(restored._connections, 1)
            query = restored.process({
                "persist_facts": True,
                "queries": [
                    ("dataset_rows", "connection_1", 2),
                    ("dataset_field", "connection_1", "Population", "int"),
                ],
            })
            self.assertTrue(all(x["value"] for x in query["inferences"]))

    def test_tamper_is_detected_before_restore(self):
        mind = UnifiedMind(KernelConfig(l0_learning=False))
        mind.run(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_checkpoint(mind, path, autodev_state=AUTODEV)
            data = json.loads(path.read_text())
            data["components"]["runtime"]["steps"] += 1
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(AtomicCheckpointError, "integrity mismatch"):
                load_checkpoint(path)

    def test_atomic_replace_failure_preserves_previous_checkpoint(self):
        first = UnifiedMind(KernelConfig(seed=1, l0_learning=False))
        first.run(4)
        second = UnifiedMind(KernelConfig(seed=1, l0_learning=False))
        second.run(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_checkpoint(first, path, autodev_state=AUTODEV)
            before = path.read_bytes()
            with mock.patch(
                "digital_mind_core.checkpoint.os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    save_checkpoint(
                        second,
                        path,
                        autodev_state=AUTODEV,
                        verify=False,
                    )
            self.assertEqual(path.read_bytes(), before)
            leftovers = list(path.parent.glob(path.name + ".*.tmp"))
            self.assertEqual(leftovers, [])

    def test_autodev_can_be_materialized_from_same_checkpoint(self):
        mind = UnifiedMind(KernelConfig(l0_learning=False))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.json"
            destination = root / "restored" / "autodev_state.json"
            save_checkpoint(mind, checkpoint, autodev_state=AUTODEV)
            restore_autodev_state(checkpoint, destination)
            self.assertEqual(json.loads(destination.read_text()), AUTODEV)


if __name__ == "__main__":
    unittest.main()
