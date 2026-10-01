import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy

import digital_mind_core.kernel as kernel_module
from digital_mind_core.kernel import KernelConfig, UnifiedMind
import digital_mind_core.persistence as persistence
from digital_mind_core.persistence import (
    REPLAY_FORMAT,
    ReplayMigrationRequired,
    load_replay,
    save_replay,
    state_fingerprint,
    upgrade_replay,
)


class PersistenceTests(unittest.TestCase):
    def test_replay_restores_and_continues_real_learning(self):
        mind = UnifiedMind(KernelConfig(seed=5))
        mind.run(32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_replay(mind, path)
            restored = load_replay(path)
            np.testing.assert_array_equal(restored.model.W, mind.model.W)
            self.assertEqual(restored.fast_steps, mind.fast_steps)
            self.assertEqual(
                restored.model.runtime.models["world_innovation"].version,
                32,
            )
            mind.run(8)
            restored.run(8)
            np.testing.assert_array_equal(restored.obs, mind.obs)
            np.testing.assert_array_equal(restored.model.W, mind.model.W)

    def test_modified_replay_is_detected(self):
        mind = UnifiedMind(KernelConfig(seed=0, l0_learning=False))
        mind.run(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_replay(mind, path)
            data = json.loads(path.read_text())
            data["steps"] = 9
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ReplayMigrationRequired, "does not match"):
                load_replay(path)

    def test_v3_provenance_drift_is_allowed_only_when_state_matches(self):
        mind = UnifiedMind(KernelConfig(seed=3, l0_learning=False))
        mind.run(12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_replay(mind, path)
            data = json.loads(path.read_text())
            data["provenance"]["source_sha256"] = "0" * 64
            data["provenance"]["numpy"] = "future-numpy"
            data["provenance"]["scipy"] = "future-scipy"
            path.write_text(json.dumps(data))
            restored = load_replay(path)
            self.assertEqual(state_fingerprint(restored), state_fingerprint(mind))
            self.assertEqual(
                restored._replay_metadata["provenance"]["source_sha256"],
                "0" * 64,
            )

    def test_v3_fingerprint_ignores_package_display_version(self):
        mind = UnifiedMind(KernelConfig(seed=4, l0_learning=False))
        mind.run(10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_replay(mind, path)
            previous = kernel_module.__version__
            kernel_module.__version__ = "99.0-runtime-label-only"
            try:
                restored = load_replay(path)
            finally:
                kernel_module.__version__ = previous
            self.assertEqual(restored.fast_steps, mind.fast_steps)

    def test_legacy_v2_migrates_across_source_and_library_changes(self):
        mind = UnifiedMind(KernelConfig(seed=6, l0_learning=False))
        mind.run(14)
        legacy = {
            "format": "digital-mind-replay-v2",
            "config": asdict(mind.config),
            "steps": mind.fast_steps,
            "source_sha256": "1" * 64,
            "numpy": "old-numpy",
            "scipy": "old-scipy",
            "state_sha256": persistence._legacy_state_fingerprint(
                mind,
                report_version="0.2.1",
            ),
            "initial_mechanisms": copy.deepcopy(mind._initial_mechanisms),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(legacy))
            restored = load_replay(path)
            self.assertEqual(restored.fast_steps, mind.fast_steps)
            self.assertEqual(
                restored._replay_metadata["migration_history"][0]["from"],
                "digital-mind-replay-v2",
            )
            self.assertEqual(
                restored._replay_metadata["compatibility"]["legacy_report_version"],
                "0.2.1",
            )

    def test_upgrade_replay_materializes_current_schema(self):
        mind = UnifiedMind(KernelConfig(seed=7, l0_learning=False))
        mind.run(9)
        legacy = {
            "format": "digital-mind-replay-v2",
            "config": asdict(mind.config),
            "steps": mind.fast_steps,
            "source_sha256": "2" * 64,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "state_sha256": persistence._legacy_state_fingerprint(
                mind,
                report_version="0.2.1",
            ),
            "initial_mechanisms": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.json"
            new = Path(directory) / "new.json"
            old.write_text(json.dumps(legacy))
            upgrade_replay(old, new)
            upgraded = json.loads(new.read_text())
            self.assertEqual(upgraded["format"], REPLAY_FORMAT)
            self.assertEqual(upgraded["schema_version"], 3)
            self.assertEqual(
                upgraded["state_fingerprint"]["algorithm"],
                persistence.STATE_FINGERPRINT_ALGORITHM,
            )


    def test_external_reasoning_events_are_part_of_replay(self):
        mind = UnifiedMind(KernelConfig(seed=9, l0_learning=False))
        mind.run(12)
        mind.process({
            "facts": [("persisted_external", "fact")],
            "persist_facts": True,
            "queries": [("persisted_external", "fact")],
        })
        mind.run(6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_replay(mind, path)
            restored = load_replay(path)
            self.assertEqual(state_fingerprint(restored), state_fingerprint(mind))
            self.assertEqual(restored._external_tasks, 1)
            self.assertEqual(len(restored._external_journal), 1)


if __name__ == "__main__":
    unittest.main()
