import copy
import unittest

import numpy as np

from digital_mind_core.components import FreeQuestionGenerator, MultiTimescalePlanner, OpenQuestion
from digital_mind_core.kernel import Intervention, KernelConfig, L0PredictiveModel, UnifiedMind, planning_benchmark


class KernelTests(unittest.TestCase):
    def test_latest_question_can_be_explained(self):
        gen = FreeQuestionGenerator(2, ['bias', 'a0'])
        q = OpenQuestion(1, 0, np.array([1., 0.]), (1,), np.array([.7]),
                         ('a0',), 1., 'test')
        gen.questions.append(q)
        for t in range(1, 41):
            gen.observe(np.zeros(2), np.ones(2), t)
            if t % 8 == 0:
                gen.update_resolution(t)
        self.assertTrue(q.resolved)

    def test_shared_planner_matches_independent_exact_answer(self):
        transition = lambda s, a: (.84 * s[0] + .72 * s[1], .72 * s[1] + a)
        utility = lambda s, a, n: s[0] - .3 * a
        score, seq = MultiTimescalePlanner.search(
            (0., 0.), (0., 1.), transition, utility, horizon=8, beam=None,
            discount=1., terminal=lambda s: 1.6 * s[0])
        # Independently derived linear reward coefficients: only the last action
        # costs more than its terminal contribution on this horizon.
        self.assertEqual(seq, [1.] * 7 + [0.])
        self.assertAlmostEqual(score, 34.51374103820697)

    def test_planning_benchmark_uses_equal_executed_horizons(self):
        result = planning_benchmark(16)
        self.assertEqual(result['1']['steps'], result['8']['steps'])
        self.assertGreater(result['8']['score'], result['1']['score'])
        self.assertEqual(result['1']['actions'], [0.] * 16)

    def test_l0_current_target_cannot_change_current_prediction(self):
        a = L0PredictiveModel(18, 6, 3)
        rng = np.random.default_rng(4)
        for _ in range(80):
            a.update(rng.normal(size=18), rng.normal(size=6), rng.normal(size=18))
        b = copy.deepcopy(a)
        obs, action = rng.normal(size=18), rng.normal(size=6)
        state = a.state_for(obs, action)
        pa = a.runtime.step(state, targets={'residual': np.zeros(18)}).predictions['world_innovation']
        pb = b.runtime.step(state, targets={'residual': np.ones(18) * 9}).predictions['world_innovation']
        np.testing.assert_array_equal(pa, pb)
        self.assertEqual(a.runtime.models['world_innovation'].version, 81)

    def test_new_goal_is_replanned_even_when_episode_already_selected_it(self):
        mind = UnifiedMind(KernelConfig(l0_learning=False, questions=False))
        mind.run(1)
        before = mind.plan_ticks
        leaf = mind.goals._new(mind.goals.root.gid, 3, 1, 40, 2.,
                               np.ones(18), np.ones(3), 'new goal')
        mind.current_goal = leaf
        mind.step()
        self.assertEqual(mind.plan_ticks, before + 1)
        self.assertEqual(mind.planned_goal_id, leaf.gid)

    def test_prediction_is_read_only(self):
        model = L0PredictiveModel(18, 6, 0)
        rng = np.random.default_rng(1)
        for _ in range(64):
            model.update(rng.normal(size=18), rng.normal(size=6), rng.normal(size=18))
        version = model.runtime.models['world_innovation'].version
        history = len(model.runtime.models['world_innovation'].raw_history)
        pred = model.runtime.predict_model('world_innovation',
                                          model.state_for(np.zeros(18), np.zeros(6)))
        self.assertTrue(np.all(np.isfinite(pred)))
        self.assertEqual(model.runtime.models['world_innovation'].version, version)
        self.assertEqual(len(model.runtime.models['world_innovation'].raw_history), history)

    def test_unvalidated_l0_correction_stays_in_shadow(self):
        model = L0PredictiveModel(18, 6, 0)
        model.baseline_losses.extend([1.] * 80)
        model.cross_ema = model.norm_ema = 1.
        model.shadow_benefits.extend([-.01] * 80)
        self.assertEqual(model.l0_weight, 0.)

    def test_one_randomized_experiment_has_one_fixed_analysis(self):
        exp = Intervention(1, 0, np.ones(2), 0)
        for i in range(48):
            sign = -1 if i % 2 else 1
            exp.observe(sign, 2 * sign + .01 * (i % 3), 48)
            if i < 47:
                self.assertEqual(exp.status, 'collecting')
        self.assertEqual(exp.status, 'supported')
        self.assertGreater(exp.confidence_interval[0], 0)
        exp.observe(1, 10000, 48)
        self.assertEqual(len(exp.samples), 48)

    def test_no_effect_experiment_is_inconclusive(self):
        exp = Intervention(1, 0, np.ones(2), 0)
        for i in range(48):
            exp.observe(-1 if i % 2 else 1, 0., 48)
        self.assertEqual(exp.status, 'inconclusive')

    def test_continuation_keeps_world_and_cognitive_clocks_aligned(self):
        config = KernelConfig(seed=3, l0_learning=False, questions=False)
        continued = UnifiedMind(config)
        continued.run(16)
        continued.run(16)
        continuous = UnifiedMind(config)
        continuous.run(32)
        np.testing.assert_array_equal(continued.obs, continuous.obs)
        self.assertEqual(continued.world.t, continued.fast_steps)
        self.assertEqual(continued.executive_ticks, 32)
        self.assertEqual(continued.meso_ticks, 4)
        self.assertEqual(len(continued.questions.questions), 0)
        self.assertEqual(continued.model.runtime.models['world_innovation'].version, 0)

    def test_l0_decisions_change_with_surprise_and_clocks(self):
        mind = UnifiedMind(KernelConfig(l0_learning=False))
        first = mind._executive_decision(0)
        self.assertTrue(first['replan'])
        self.assertFalse(first['reflect'])
        mind.model.ensemble_losses.extend([.001] * 31 + [.1])
        last = mind._executive_decision(63)
        self.assertTrue(last['reflect'])
        self.assertTrue(last['consolidate'])
        self.assertGreater(last['exploration'], first['exploration'])

    def test_integration_reaches_real_l0_learning_and_memory(self):
        mind = UnifiedMind(KernelConfig(seed=2))
        mind.run(80)
        self.assertEqual(mind.executive_ticks, 80)
        self.assertEqual(mind.model.runtime.models['world_innovation'].version, 80)
        self.assertEqual(len(mind.narrative.nodes), 10)
        self.assertEqual(mind.consolidated_self['t'], 64)
        self.assertTrue(np.all(np.isfinite(mind.obs)))
        self.assertTrue(mind.experiments)
        self.assertLessEqual(mind.probe_steps, 20)


if __name__ == '__main__':
    unittest.main()
