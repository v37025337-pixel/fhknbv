import itertools
import unittest

import numpy as np

from digital_mind_core.components import (
    DelayedPlanningBenchmark, GoalHierarchy, MultiTimescalePlanner,
    PredictiveModel,
)
from digital_mind_core.l0 import spectral_runtime, spectral_scalar


class ComponentTests(unittest.TestCase):
    def test_goal_deadline_and_completion(self):
        goals = GoalHierarchy(18)
        obs = np.zeros(18)
        obs[0] = 1
        leaf = goals.spawn_state_goal(obs, 0)
        goals.refresh(leaf.horizon + 1, obs, set())
        self.assertEqual(leaf.status, 'expired')
        self.assertNotEqual(goals.active_leaf().gid, leaf.gid)
        leaf = goals.spawn_state_goal(obs, 64)
        goals.refresh(65, leaf.target, set())
        self.assertEqual(leaf.status, 'completed')

    def test_model_remembers_actual_delayed_action(self):
        model = PredictiveModel(18, 6)
        actions = [np.full(6, i) for i in (1., 2., 3.)]
        for a in actions:
            model.prev_actions.append(a)
        phi = model.features(np.zeros(18), np.zeros(6))
        i = model.feature_names.index('a3_0')
        self.assertEqual(phi[i], 1.)

    def test_exact_reference_matches_exhaustive_search(self):
        score, _ = DelayedPlanningBenchmark.best_sequence(8)
        expected = max(DelayedPlanningBenchmark.rollout(s)[0]
                       for s in itertools.product((0., 1.), repeat=8))
        self.assertAlmostEqual(score, expected)

    def test_short_spectral_data_rejected(self):
        spec = spectral_runtime.ProgramSpec('small', ['x'], 'y',
            spectral_runtime.RelationSpec('r', ['x'], 'y', {}), [])
        with self.assertRaisesRegex(ValueError, 'at least 9'):
            spectral_runtime.execute(spec, spectral_scalar, np.zeros((5, 2)), .7)

    def test_invalid_spectral_fraction_rejected(self):
        spec = spectral_runtime.ProgramSpec('small', ['x'], 'y',
            spectral_runtime.RelationSpec('r', ['x'], 'y', {}), [])
        with self.assertRaisesRegex(ValueError, 'train_fraction'):
            spectral_runtime.execute(spec, spectral_scalar, np.zeros((12, 2)), 1.5)


if __name__ == '__main__':
    unittest.main()
