import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.utils.deidentify import TextDeidentifier
from src.features.psycholinguistic import PsycholinguisticExtractor
from src.features.unsupervised_discovery import UnsupervisedTopicDiscovery

class HybridRiskModel:
    """
    Hybrid Mental Health Screening Model.
    Combines:
    1. Research-backed Psycholinguistic vectors (12 markers)
    2. Unsupervised LDA Latent Topics (K discovered themes)
    """

    PSYCHO_FEATURES = [
        "word_count",
        "first_person_singular_ratio",
        "first_person_plural_ratio",
        "self_focus_index",
        "absolutist_ratio",
        "distress_affect_ratio",
        "somatic_ratio",
        "lexical_diversity",
        "avg_word_length",
        "ellipsis_frequency",
        "question_density",
        "exclamation_density"
    ]

    def __init__(
        self,
        n_topics: int = 6,
        model_type: str = "logistic_regression",
        feature_mode: str = "hybrid",  # "psycholinguistic", "unsupervised", "hybrid"
        risk_threshold: float = 0.5
    ):
        self.n_topics = n_topics
        self.model_type = model_type
        self.feature_mode = feature_mode
        self.risk_threshold = risk_threshold

        self.deidentifier = TextDeidentifier()
        self.extractor = PsycholinguisticExtractor()
        self.topic_engine = UnsupervisedTopicDiscovery(n_topics=n_topics, method="lda", random_state=42)
        self.scaler = StandardScaler()

        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        else:
            self.model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

        self.is_fitted = False
        self.feature_names_: List[str] = []

    def _extract_psycholinguistic(self, texts: List[str]) -> np.ndarray:
        rows = []
        for text in texts:
            clean = self.deidentifier.clean(text)
            f = self.extractor.extract_features(clean)
            rows.append([f[k] for k in self.PSYCHO_FEATURES])
        return np.array(rows)

    def fit(self, texts: List[str], labels: List[int]):
        clean_texts = [self.deidentifier.clean(t) for t in texts]

        # 1. Fit unsupervised topic engine on unlabeled clean text
        if self.feature_mode in ["unsupervised", "hybrid"]:
            self.topic_engine.fit(clean_texts)

        # 2. Build feature matrix
        feature_blocks = []
        feature_names = []

        if self.feature_mode in ["psycholinguistic", "hybrid"]:
            X_psycho = self._extract_psycholinguistic(clean_texts)
            feature_blocks.append(X_psycho)
            feature_names.extend(self.PSYCHO_FEATURES)

        if self.feature_mode in ["unsupervised", "hybrid"]:
            X_topics = self.topic_engine.transform(clean_texts)
            feature_blocks.append(X_topics)
            topic_names = [f"latent_topic_{i+1}" for i in range(self.n_topics)]
            feature_names.extend(topic_names)

        X = np.hstack(feature_blocks)
        self.feature_names_ = feature_names

        X_scaled = self.scaler.fit_transform(X)
        y = np.array(labels)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def _build_features(self, texts: List[str]) -> np.ndarray:
        clean_texts = [self.deidentifier.clean(t) for t in texts]
        feature_blocks = []

        if self.feature_mode in ["psycholinguistic", "hybrid"]:
            X_psycho = self._extract_psycholinguistic(clean_texts)
            feature_blocks.append(X_psycho)

        if self.feature_mode in ["unsupervised", "hybrid"]:
            X_topics = self.topic_engine.transform(clean_texts)
            feature_blocks.append(X_topics)

        return np.hstack(feature_blocks)

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")

        X = self._build_features(texts)
        X_scaled = self.scaler.transform(X)
        y = np.array(labels)

        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= self.risk_threshold).astype(int)

        report = classification_report(y, preds, target_names=["Stable", "At-Risk"], output_dict=True)
        cm = confusion_matrix(y, preds).tolist()
        auc = roc_auc_score(y, probs)

        return {
            "metrics": report,
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
            "feature_mode": self.feature_mode,
            "feature_count": len(self.feature_names_)
        }

    def explain_features(self) -> List[Tuple[str, float]]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted.")
        if self.model_type == "logistic_regression":
            weights = self.model.coef_[0]
            ranking = sorted(zip(self.feature_names_, weights), key=lambda x: abs(x[1]), reverse=True)
        else:
            weights = self.model.feature_importances_
            ranking = sorted(zip(self.feature_names_, weights), key=lambda x: x[1], reverse=True)
        return [(name, round(float(w), 4)) for name, w in ranking]
