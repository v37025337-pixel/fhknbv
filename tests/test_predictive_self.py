import unittest

import numpy as np

from digital_mind_core.kernel import KernelConfig, UnifiedMind
from digital_mind_core.predictive_self import PredictiveSelfModel


class PredictiveSelfTests(unittest.TestCase):
    def test_synthetic_self_transition_beats_persistence(self):
        model = PredictiveSelfModel(ridge=8.0, forgetting=0.999)
        rng = np.random.default_rng(123)
        state = np.ones(model.state_dim) * 0.45
        for t in range(700):
            decision = rng.uniform(0.0, 1.0, size=model.decision_dim)
            drift = np.zeros(model.state_dim)
            drift[0] = 0.025 * (decision[0] - 0.5)
            drift[1] = 0.035 * (decision[5] - 0.5)
            drift[2] = 0.020 * (decision[4] - 0.5)
            drift[3] = 0.030 * (decision[1] - 0.5)
            drift[4] = 0.040 * (decision[3] - 0.5)
            drift[5] = 0.030 * (decision[2] - 0.5)
            drift[6] = 0.020 * (decision[0] - decision[4])
            drift[7] = 0.025 * (decision[6] - 0.5)
            drift[8] = 0.030 * (decision[7] - 0.5)
            next_state = np.clip(
                state + drift + rng.normal(scale=0.001, size=model.state_dim),
                0.0, 1.0,
            )
            model.observe(state, decision, next_state)
            state = next_state
        report = model.report(warmup=128, window=256)
        self.assertGreater(report["evaluation_count"], 200)
        self.assertLess(
            report["prediction_mse"],
            0.55 * report["persistence_baseline_mse"],
        )

    def test_counterfactual_is_read_only(self):
        model = PredictiveSelfModel()
        state = np.ones(model.state_dim) * 0.5
        decision = np.zeros(model.decision_dim)
        before_W = model.W.copy()
        before_P = model.P.copy()
        before_n = model.observations
        result = model.counterfactual(state, decision, replan=1.0)
        self.assertEqual(set(result), set(model.STATE_NAMES))
        np.testing.assert_array_equal(model.W, before_W)
        np.testing.assert_array_equal(model.P, before_P)
        self.assertEqual(model.observations, before_n)
        self.assertEqual(model.counterfactual_calls, 1)

    def test_integrated_kernel_updates_predictive_self(self):
        mind = UnifiedMind(KernelConfig(
            seed=3, l0_learning=False, questions=False
        ))
        mind.run(256)
        report = mind.report()["predictive_self"]
        self.assertEqual(report["mode"], "SHADOW")
        self.assertEqual(report["observations"], 256)
        self.assertGreater(report["evaluation_count"], 100)
        self.assertTrue(np.isfinite(report["prediction_mse"]))
        self.assertTrue(np.isfinite(report["persistence_baseline_mse"]))
        before = mind.predictive_self.observations
        result = mind.predict_self_counterfactual(replan=1.0, reflect=1.0)
        self.assertEqual(set(result), set(mind.predictive_self.STATE_NAMES))
        self.assertEqual(mind.predictive_self.observations, before)
        self.assertFalse(mind.predictive_self.report()["causal_claim"])


if __name__ == "__main__":
    unittest.main()
