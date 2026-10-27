"""Regression tests for prospect updates and scoring."""

import copy
import json
import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


class ProspectUpdateTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.object(data_service, "PROSPECTS", copy.deepcopy(data_service.PROSPECTS))
        self.records.start()
        self.addCleanup(self.records.stop)
        self.profiles = patch.object(data_service, "_PROFILES", {})
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_update_is_visible_in_source_and_cached_profile(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})["prospect_profile"]
        data_service.save_profile_to_db("LEAD-39002", copy.deepcopy(profile))
        result = agent.update_prospect_info.invoke({"prospect_id": "LEAD-39002", "technology": "Terraform"})
        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.fetch_tech_stack("LEAD-39002"))
        self.assertIn("Terraform", data_service.get_profile_from_db("LEAD-39002")["prospect_profile"]["tech_stack"])
        rebuilt = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

    def test_update_before_build_and_repeated_update(self):
        for _ in range(2):
            data_service.update_prospect_info("LEAD-39002", "Terraform")
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})["prospect_profile"]
        self.assertEqual(profile["tech_stack"].count("Terraform"), 1)

    def score_from_input(self, messages):
        offering_text, profile_text = messages[1]["content"].split("\n\nProspect profile:\n")
        offering = json.loads(offering_text.removeprefix("Offering:\n"))
        profile = json.loads(profile_text)
        present = set(profile["tech_stack"])
        required = set(offering["required_tech_stack"])
        self.assertIn("Terraform", present)
        self.assertFalse(required - present)
        return agent.ProspectScore(
            score=100,
            justification="All required technologies, including Terraform, are present.",
            rubric_breakdown=agent.RubricBreakdown(revenue_fit=100, tech_stack_match=100, segment_fit=100),
        )

    def test_update_then_score_uses_updated_stack(self):
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})["prospect_profile"]
        with patch.object(agent, "_scoring_llm") as scorer:
            scorer.invoke.side_effect = self.score_from_input
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": data_service.get_offering("OFFER-10004")})
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)
        self.assertNotIn("warning", result)

    def test_supplied_stack_is_scored_and_storage_mismatch_is_disclosed(self):
        profile = copy.deepcopy(agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})["prospect_profile"])
        profile["tech_stack"].append("Terraform")
        with patch.object(agent, "_scoring_llm") as scorer:
            scorer.invoke.side_effect = self.score_from_input
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": data_service.get_offering("OFFER-10004")})
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)
        self.assertIn("differs from the stored record", result["warning"])
        self.assertNotIn("Terraform", data_service.fetch_tech_stack("LEAD-39002"))

    def test_unknown_prospect_is_not_reported_updated(self):
        self.assertEqual(data_service.update_prospect_info("unknown", "Terraform"), {"updated": False, "found": False})


if __name__ == "__main__":
    unittest.main()
