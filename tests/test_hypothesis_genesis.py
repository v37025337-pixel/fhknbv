import unittest

from digital_mind_core.hypothesis_genesis import (
    choose_primary_deficit, synthesize_hypothesis,
)


class HypothesisGenesisTests(unittest.TestCase):
    def sample(self):
        return {
            "deficits": [
                {
                    "deficit_id": "b",
                    "severity": 0.8,
                    "question": "How improve b?",
                    "success_test": "b must improve blind loss",
                },
                {
                    "deficit_id": "a",
                    "severity": 1.0,
                    "question": "How stabilize residual correction under drift?",
                    "success_test": "reduce prequential future loss",
                },
            ],
            "research_plan": [{
                "deficit_id": "a",
                "objective": "prequential drift residual stability",
            }],
            "external_evidence": {"sources": [
                {
                    "source_id": "s1",
                    "deficit_id": "a",
                    "title": "one",
                    "url": "https://example.org/1",
                    "candidate_mechanisms": [
                        "evaluate residual prediction prequentially before learning",
                        "maintain bounded residual experts",
                    ],
                },
                {
                    "source_id": "s2",
                    "deficit_id": "a",
                    "title": "two",
                    "url": "https://example.org/2",
                    "candidate_mechanisms": [
                        "estimate drift before granting residual authority",
                    ],
                },
            ]},
        }

    def test_primary_deficit_by_measured_severity(self):
        self.assertEqual(
            choose_primary_deficit(self.sample())["deficit_id"], "a"
        )

    def test_hypothesis_is_shadow_and_source_grounded(self):
        result = synthesize_hypothesis(self.sample())
        self.assertEqual(result["deficit"]["deficit_id"], "a")
        self.assertEqual(result["candidate_mode"], "SHADOW_ONLY")
        self.assertFalse(result["internet_evidence_is_executable"])
        self.assertGreaterEqual(len(result["selected_mechanisms"]), 2)
        self.assertTrue(all(x["source_id"] for x in result["selected_mechanisms"]))

    def test_missing_evidence_fails_closed(self):
        data = self.sample()
        data["external_evidence"]["sources"] = []
        with self.assertRaisesRegex(ValueError, "no external evidence"):
            synthesize_hypothesis(data)


if __name__ == "__main__":
    unittest.main()
