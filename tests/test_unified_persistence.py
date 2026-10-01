from pathlib import Path
import tempfile
import unittest

import numpy as np

from digital_mind_core.kernel import KernelConfig, UnifiedMind
from digital_mind_core.persistence import load_replay, save_replay, state_fingerprint


class UnifiedPersistenceTests(unittest.TestCase):
    def test_replay_recreates_generated_rule_and_subsequent_choices(self):
        mind = UnifiedMind(KernelConfig(seed=1, l0_learning=False))
        mind.run(96)
        self.assertTrue(mind.evolution.active)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.json'
            save_replay(mind, path)
            restored = load_replay(path)
            self.assertEqual(state_fingerprint(restored), state_fingerprint(mind))
            mind.run(8)
            restored.run(8)
            self.assertEqual(state_fingerprint(restored), state_fingerprint(mind))

    def test_imported_mechanisms_are_part_of_replay_initial_conditions(self):
        trained = UnifiedMind(KernelConfig(seed=2, l0_learning=False))
        trained.run(96)
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory) / 'mechanisms.json'
            replay = Path(directory) / 'replay.json'
            trained.save_mechanisms(store)
            fresh = UnifiedMind(KernelConfig(seed=5, l0_learning=False))
            fresh.load_mechanisms(store)
            fresh.run(16)
            save_replay(fresh, replay)
            restored = load_replay(replay)
            self.assertEqual(state_fingerprint(fresh), state_fingerprint(restored))

    def test_external_tasks_are_replayed_in_order(self):
        mind = UnifiedMind(KernelConfig(l0_learning=False))
        mind.run(6)
        mind.process({
            'facts': [('p', 'a')],
            'persist_facts': True,
            'queries': [('p', 'a')],
        })
        mind.run(4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.json'
            save_replay(mind, path)
            restored = load_replay(path)
            self.assertEqual(state_fingerprint(mind), state_fingerprint(restored))
            self.assertEqual(restored._external_tasks, 1)


if __name__ == '__main__':
    unittest.main()
