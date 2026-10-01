import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from digital_mind_core.learning import ProspectiveSelection
from digital_mind_core.kernel import AdaptivePredictiveModel, KernelConfig, UnifiedMind
from digital_mind_core.persistence import load_replay, save_replay, state_fingerprint


class LearningTests(unittest.TestCase):
    def test_candidate_needs_future_predictions_before_admission(self):
        selector = ProspectiveSelection(('recent',))
        predictions = {'baseline': np.array([0.]), 'recent': np.array([1.])}
        for _ in range(64):
            self.assertEqual(selector.active, 'baseline')
            selector.observe(np.array([1.]), predictions, 'baseline')
        self.assertEqual(selector.active, 'recent')
        self.assertEqual(selector.report()['used_steps']['baseline'], 64)

    def test_recent_failures_revoke_candidate_even_after_long_success(self):
        selector = ProspectiveSelection(('recent',))
        predictions = {'baseline': np.array([0.]), 'recent': np.array([1.])}
        for _ in range(80):
            selector.observe(np.array([1.]), predictions, selector.active)
        self.assertEqual(selector.active, 'recent')
        for _ in range(8):
            selector.observe(np.array([0.]), predictions, selector.active)
        self.assertEqual(selector.active, 'baseline')
        self.assertEqual(selector.report()['events'][-1]['event'], 'returned_to_baseline')

    def test_no_gain_does_not_grant_authority(self):
        selector = ProspectiveSelection(('recent',))
        for _ in range(80):
            selector.observe(np.array([1.]), {'baseline': np.array([0.]),
                                            'recent': np.array([0.])}, selector.active)
        self.assertEqual(selector.active, 'baseline')

    def test_invalid_observation_does_not_change_evidence(self):
        selector = ProspectiveSelection(('recent',))
        before = selector.state_document()
        for predictions in ({'baseline': [0.], 'recent': [float('nan')]},
                            {'baseline': [0.], 'recent': [0., 1.]},
                            {'baseline': [0.]}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                selector.observe(np.array([1.]), predictions, 'baseline')
            self.assertEqual(selector.state_document(), before)

    def test_target_cannot_change_its_own_frozen_prediction(self):
        for enabled in (False, True):
            trained = AdaptivePredictiveModel(18, 6, 3, enabled=enabled)
            rng = np.random.default_rng(7)
            for _ in range(80):
                trained.update(rng.normal(size=18), rng.normal(size=6), rng.normal(size=18))
            obs, action = rng.normal(size=18), rng.normal(size=6)
            for selected in ('baseline', 'recent', 'stable'):
                with self.subTest(l0=enabled, selected=selected):
                    first = copy.deepcopy(trained)
                    # Exercise every predictor branch; admission itself is tested above.
                    first.selection.active = selected
                    second = copy.deepcopy(first)
                    expected = first.predict(obs, action)
                    p1, _, _ = first.update(obs, action, np.zeros(18))
                    p2, _, _ = second.update(obs, action, np.ones(18) * 9.)
                    np.testing.assert_array_equal(p1, p2)
                    np.testing.assert_array_equal(p1, expected)

    def test_planning_prediction_does_not_train_or_change_selection(self):
        model = AdaptivePredictiveModel(18, 6, 0, enabled=False)
        rng = np.random.default_rng(4)
        for _ in range(80):
            model.update(rng.normal(size=18), rng.normal(size=6), rng.normal(size=18))
        before = model.learning_state_fingerprint()
        for _ in range(5):
            result = model.predict(np.zeros(18), np.zeros(6))
            self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(before, model.learning_state_fingerprint())

    def test_full_kernel_uses_experienced_forecasts_and_can_disable_extension(self):
        mind = UnifiedMind(KernelConfig(seed=2, l0_learning=False))
        mind.run(96)
        report = mind.report()['adaptive_learning']
        self.assertTrue(report['enabled'])
        self.assertEqual(report['observations'], 96)
        self.assertEqual(sum(report['used_steps'].values()), 96)
        legacy = UnifiedMind(KernelConfig(adaptive_learning=False, l0_learning=False))
        legacy.run(16)
        self.assertFalse(legacy.report()['adaptive_learning']['enabled'])

    def test_replay_preserves_candidates_and_subsequent_predictions(self):
        first = UnifiedMind(KernelConfig(seed=1, l0_learning=False))
        first.run(96)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.json'
            save_replay(first, path)
            second = load_replay(path)
            self.assertEqual(first.model.learning_state_fingerprint(),
                             second.model.learning_state_fingerprint())
            first.run(8)
            second.run(8)
            self.assertEqual(state_fingerprint(first), state_fingerprint(second))
            second.model.experts['recent'].P[0, 0] += 1.
            self.assertNotEqual(state_fingerprint(first), state_fingerprint(second))


if __name__ == '__main__':
    unittest.main()
