import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from digital_mind_core.kernel import KernelConfig, UnifiedMind
from digital_mind_core.persistence import load_replay, save_replay


class PersistenceTests(unittest.TestCase):
    def test_replay_restores_and_continues_real_learning(self):
        mind = UnifiedMind(KernelConfig(seed=5))
        mind.run(32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            save_replay(mind, path)
            restored = load_replay(path)
            np.testing.assert_array_equal(restored.model.W, mind.model.W)
            self.assertEqual(restored.fast_steps, mind.fast_steps)
            self.assertEqual(restored.model.runtime.models['world_innovation'].version, 32)
            mind.run(8)
            restored.run(8)
            np.testing.assert_array_equal(restored.obs, mind.obs)
            np.testing.assert_array_equal(restored.model.W, mind.model.W)

    def test_modified_replay_is_detected(self):
        mind = UnifiedMind(KernelConfig(seed=0, l0_learning=False))
        mind.run(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            save_replay(mind, path)
            data = json.loads(path.read_text())
            data['steps'] = 9
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                load_replay(path)


if __name__ == '__main__':
    unittest.main()
