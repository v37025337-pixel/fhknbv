import unittest

from digital_mind_core.self_research import (
    Deficit, assess, build_research_plan, validate_evidence,
)


class SelfResearchTests(unittest.TestCase):
    def test_measured_deficits_generate_research_plan(self):
        training = {
            "training": {"runs": [
                {"l0_final_weight": 0.0},
                {"l0_final_weight": 0.0},
            ]},
            "negative_capability_probes": [
                {"capability": "string_program_synthesis", "supported": False}
            ],
        }
        l0 = {"selected_on_design_only": None}
        predictive = {"blind_verdict": {"pass": True}}
        counterfactual = {
            "design_verdict": {"pass": False},
            "blind_verdict": {"pass": True},
        }
        deficits = assess(
            training, l0, predictive, counterfactual,
            {"general_algorithm_synthesis": False},
        )
        ids = [d.deficit_id for d in deficits]
        self.assertEqual(ids[0], "l0_correction_stability")
        self.assertIn("self_counterfactual_identifiability", ids)
        self.assertIn("algorithm_genesis", ids)
        plan = build_research_plan(deficits, top_k=3)
        self.assertEqual(len(plan), 3)
        self.assertTrue(all(item["query"] for item in plan))
        self.assertTrue(all(item["success_test"] for item in plan))

    def test_no_false_l0_deficit_when_authority_is_used(self):
        training = {
            "training": {"runs": [{"l0_final_weight": 0.2}]},
            "negative_capability_probes": [],
        }
        deficits = assess(
            training, {}, {},
            {"design_verdict": {"pass": True}, "blind_verdict": {"pass": True}},
            {"general_algorithm_synthesis": True},
        )
        self.assertNotIn(
            "l0_correction_stability",
            [d.deficit_id for d in deficits],
        )

    def test_external_evidence_is_data_not_code(self):
        doc = {
            "sources": [{
                "source_id": "s1",
                "deficit_id": "algorithm_genesis",
                "url": "https://example.org/paper",
                "title": "Paper",
                "summary": "Evidence summary.",
                "candidate_mechanisms": ["typed DSL"],
            }]
        }
        clean = validate_evidence(doc, ["algorithm_genesis"])
        self.assertEqual(clean["source_count"], 1)
        self.assertEqual(clean["sources"][0]["candidate_mechanisms"], ["typed DSL"])

    def test_non_https_evidence_rejected(self):
        doc = {
            "sources": [{
                "source_id": "s1",
                "deficit_id": "algorithm_genesis",
                "url": "http://example.org/paper",
                "title": "Paper",
                "summary": "Evidence summary.",
                "candidate_mechanisms": [],
            }]
        }
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            validate_evidence(doc, ["algorithm_genesis"])


if __name__ == "__main__":
    unittest.main()
