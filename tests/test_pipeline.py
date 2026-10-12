import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.deidentify import TextDeidentifier
from src.features.psycholinguistic import PsycholinguisticExtractor
from src.models.baseline_classifier import RiskAssessmentModel

class TestRiskScreeningPipeline(unittest.TestCase):

    def setUp(self):
        self.deidentifier = TextDeidentifier()
        self.extractor = PsycholinguisticExtractor()

    def test_deidentification(self):
        raw = "Contact me at user@example.com or visit https://reddit.com/r/test by u/testuser"
        cleaned = self.deidentifier.clean(raw)
        self.assertNotIn("user@example.com", cleaned)
        self.assertNotIn("https://reddit.com", cleaned)
        self.assertNotIn("u/testuser", cleaned)
        self.assertIn("[EMAIL]", cleaned)
        self.assertIn("[URL]", cleaned)
        self.assertIn("[USER]", cleaned)

    def test_psycholinguistic_features(self):
        at_risk_text = "I feel completely hopeless and tired. Nobody ever listens to me."
        features = self.extractor.extract_features(at_risk_text)

        self.assertGreater(features['first_person_singular_ratio'], 0)
        self.assertGreater(features['absolutist_ratio'], 0)
        self.assertGreater(features['distress_affect_ratio'], 0)

    def test_risk_classifier_end_to_end(self):
        texts = [
            "I feel so empty, hopeless and completely alone every day.",
            "I am broken, tired, and in endless pain with nobody to talk to.",
            "We enjoyed our road trip across the national parks this summer.",
            "The community garden planted fresh tomatoes and sunflowers together."
        ]
        labels = [1, 1, 0, 0]

        model = RiskAssessmentModel()
        model.fit(texts, labels)

        # Test at-risk inference
        result_at_risk = model.assess_risk("I am exhausted, empty, and constantly crying in my room.")
        self.assertIn("risk_score", result_at_risk)
        self.assertGreater(result_at_risk["risk_score"], 0.5)

        # Test stable inference
        result_stable = model.assess_risk("Our group made a delicious dinner and watched a movie.")
        self.assertLess(result_stable["risk_score"], 0.5)

if __name__ == '__main__':
    unittest.main()
