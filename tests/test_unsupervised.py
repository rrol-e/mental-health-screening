import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.unsupervised_discovery import UnsupervisedTopicDiscovery

class TestUnsupervisedDiscovery(unittest.TestCase):

    def setUp(self):
        self.texts = [
            "I feel so empty, hopeless and completely alone every day. Can't sleep.",
            "Panic attacks and chest pain keep me awake at night with insomnia.",
            "My partner has been shouting and emotionally abusive towards me.",
            "We enjoyed our camping trip near the lake and cooked on the campfire.",
            "The community garden planted fresh tomatoes and sunflowers together.",
            "Looking for book recommendations on history and science fiction."
        ]
        self.labels = [1, 1, 1, 0, 0, 0]

    def test_fit_and_transform(self):
        engine = UnsupervisedTopicDiscovery(n_topics=3, method="lda", min_df=1, random_state=42)
        doc_topics = engine.fit_transform(self.texts)

        self.assertEqual(doc_topics.shape, (len(self.texts), 3))
        # Check rows sum approximately to 1
        row_sums = np.sum(doc_topics, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(self.texts)), atol=1e-5)

        # Descriptors
        descriptors = engine.get_topic_descriptors()
        self.assertEqual(len(descriptors), 3)

        # Risk correlation
        corrs = engine.correlate_topics_with_risk(doc_topics, self.labels)
        self.assertEqual(len(corrs), 3)

    def test_nmf_method(self):
        engine = UnsupervisedTopicDiscovery(n_topics=2, method="nmf", min_df=1, random_state=42)
        doc_topics = engine.fit_transform(self.texts)
        self.assertEqual(doc_topics.shape, (len(self.texts), 2))

if __name__ == '__main__':
    unittest.main()
