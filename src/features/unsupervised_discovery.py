import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import joblib

class UnsupervisedTopicDiscovery:
    """
    Unsupervised Pattern & Topic Discovery Engine.
    Discovers latent linguistic themes across digital footprints without
    supervision or human labeling bias.
    """

    def __init__(
        self,
        n_topics: int = 6,
        method: str = "lda",
        max_features: int = 2500,
        min_df: int = 2,
        random_state: int = 42
    ):
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        self.min_df = min_df
        self.random_state = random_state

        if method == "lda":
            self.vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=self.min_df,
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                learning_method='batch',
                max_iter=25
            )
        elif method == "nmf":
            self.vectorizer = TfidfVectorizer(
                max_df=0.95,
                min_df=self.min_df,
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=300
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'lda' or 'nmf'.")

        self.is_fitted = False
        self.topic_keywords_: List[List[str]] = []

    def fit(self, texts: List[str]):
        """
        Fits the unsupervised topic model strictly on unlabeled texts.
        """
        dtm = self.vectorizer.fit_transform(texts)
        self.model.fit(dtm)
        self.is_fitted = True

        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_keywords_ = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_indices]
            self.topic_keywords_.append(top_words)

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforms texts into document-topic distribution vectors (shape: [N, n_topics]).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming.")
        dtm = self.vectorizer.transform(texts)
        doc_topics = self.model.transform(dtm)
        # Normalize each row to sum to 1
        row_sums = doc_topics.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return doc_topics / row_sums

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def get_topic_descriptors(self) -> List[Dict[str, Any]]:
        """
        Returns top descriptive terms for each discovered latent cluster.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted.")
        descriptors = []
        for i, words in enumerate(self.topic_keywords_):
            descriptors.append({
                "topic_id": i,
                "name": f"Latent_Theme_{i+1}",
                "top_keywords": words
            })
        return descriptors

    def correlate_topics_with_risk(
        self,
        doc_topics: np.ndarray,
        labels: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Measures the empirical correlation between discovered latent themes and
        the downstream at-risk status. Identifies which clusters signal distress.
        """
        y = np.array(labels)
        correlations = []

        for i in range(self.n_topics):
            topic_weights = doc_topics[:, i]
            corr = np.corrcoef(topic_weights, y)[0, 1]
            mean_at_risk = float(np.mean(topic_weights[y == 1]))
            mean_control = float(np.mean(topic_weights[y == 0]))
            risk_lift = (mean_at_risk - mean_control) / (mean_control + 1e-6)

            correlations.append({
                "topic_id": i,
                "name": f"Theme_{i+1}",
                "top_keywords": self.topic_keywords_[i][:6],
                "correlation": round(float(corr), 4),
                "mean_weight_at_risk": round(mean_at_risk, 4),
                "mean_weight_control": round(mean_control, 4),
                "risk_association": "Elevated Distress" if corr > 0.05 else ("Protective / Stable" if corr < -0.05 else "Neutral")
            })

        # Sort by strongest risk association
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        return correlations

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "UnsupervisedTopicDiscovery":
        return joblib.load(filepath)
