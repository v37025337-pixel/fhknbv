import unittest
import numpy as np

from digital_mind_core.l0_research_candidate import DriftAwareCorrectionRouter


class DriftAwareCorrectionRouterTests(unittest.TestCase):
    def test_fail_closed_during_warmup(self):
        router = DriftAwareCorrectionRouter(warmup=3)
        for _ in range(2):
            self.assertEqual(
                router.select(drift_score=0.0, shadow_weight=0.5), 0.0
            )
            router.observe(
                baseline_error=np.ones(4),
                correction=np.ones(4),
                shadow_weight=0.5,
                drift_score=0.0,
            )

    def test_helpful_correction_gets_authority(self):
        router = DriftAwareCorrectionRouter(
            warmup=3, decay=0.9, margin_ratio=0.0
        )
        for _ in range(4):
            router.select(drift_score=0.0, shadow_weight=1.0)
            router.observe(
                baseline_error=np.ones(4),
                correction=np.ones(4),
                shadow_weight=1.0,
                drift_score=0.0,
            )
        self.assertGreater(
            router.select(drift_score=0.0, shadow_weight=1.0), 0.0
        )

    def test_harmful_correction_is_retired(self):
        router = DriftAwareCorrectionRouter(warmup=3, decay=0.9)
        for _ in range(8):
            router.select(drift_score=0.0, shadow_weight=1.0)
            router.observe(
                baseline_error=np.ones(4),
                correction=-np.ones(4),
                shadow_weight=1.0,
                drift_score=0.0,
            )
        self.assertEqual(
            router.select(drift_score=0.0, shadow_weight=1.0), 0.0
        )

    def test_drift_contexts_learn_separately(self):
        router = DriftAwareCorrectionRouter(
            drift_threshold=0.5, warmup=2, decay=0.9
        )
        for _ in range(4):
            router.select(drift_score=0.1, shadow_weight=1.0)
            router.observe(
                baseline_error=np.ones(3),
                correction=np.ones(3),
                shadow_weight=1.0,
                drift_score=0.1,
            )
            router.select(drift_score=0.9, shadow_weight=1.0)
            router.observe(
                baseline_error=np.ones(3),
                correction=-np.ones(3),
                shadow_weight=1.0,
                drift_score=0.9,
            )
        self.assertGreater(router.select(drift_score=0.1, shadow_weight=1.0), 0.0)
        self.assertEqual(router.select(drift_score=0.9, shadow_weight=1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
