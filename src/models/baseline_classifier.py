import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

from src.utils.deidentify import TextDeidentifier
from src.features.psycholinguistic import PsycholinguisticExtractor

class RiskAssessmentModel:
    """
    Supervised Risk Screening Model using psycholinguistic feature vectors.
    Prioritizes sensitivity (recall) and model interpretability.
    """

    FEATURE_NAMES = [
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

    def __init__(self, model_type: str = "logistic_regression", risk_threshold: float = 0.5):
        self.model_type = model_type
        self.risk_threshold = risk_threshold
        self.deidentifier = TextDeidentifier()
        self.extractor = PsycholinguisticExtractor()
        self.scaler = StandardScaler()

        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        else:
            self.model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

        self.is_fitted = False

    def extract_dataset_features(self, texts: List[str]) -> np.ndarray:
        feature_rows = []
        for text in texts:
            clean_text = self.deidentifier.clean(text)
            feat_dict = self.extractor.extract_features(clean_text)
            feature_rows.append([feat_dict[k] for k in self.FEATURE_NAMES])
        return np.array(feature_rows)

    def fit(self, texts: List[str], labels: List[int]):
        X = self.extract_dataset_features(texts)
        y = np.array(labels)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")

        X = self.extract_dataset_features(texts)
        y = np.array(labels)
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probabilities >= self.risk_threshold).astype(int)

        report = classification_report(y, preds, target_names=["Stable", "At-Risk"], output_dict=True)
        cm = confusion_matrix(y, preds).tolist()

        try:
            auc = roc_auc_score(y, probabilities)
        except Exception:
            auc = 0.5

        return {
            "metrics": report,
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
            "threshold_used": self.risk_threshold
        }

    def explain_global_features(self) -> List[Tuple[str, float]]:
        """
        Returns feature importance or model coefficients to explain
        global indicators of risk.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature explanations.")

        if self.model_type == "logistic_regression":
            coeffs = self.model.coef_[0]
            ranking = sorted(zip(self.FEATURE_NAMES, coeffs), key=lambda x: abs(x[1]), reverse=True)
        else:
            importances = self.model.feature_importances_
            ranking = sorted(zip(self.FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)

        return [(feat, round(float(val), 4)) for feat, val in ranking]

    def assess_risk(self, raw_text: str) -> Dict[str, Any]:
        """
        Inference endpoint: takes raw text, de-identifies, extracts psycholinguistic markers,
        and outputs an interpretable risk assessment.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before running inference.")

        clean_text = self.deidentifier.clean(raw_text)
        features = self.extractor.extract_features(clean_text)
        
        feature_vector = np.array([[features[k] for k in self.FEATURE_NAMES]])
        scaled_vec = self.scaler.transform(feature_vector)

        risk_prob = float(self.model.predict_proba(scaled_vec)[0, 1])

        # Stratified triage categories
        if risk_prob < 0.35:
            category = "Low / Stable"
            recommendation = "No elevated risk markers detected in current sample."
        elif risk_prob < 0.65:
            category = "Moderate / Observational"
            recommendation = "Some distress or cognitive markers present. Continued monitoring or supportive resources advised."
        else:
            category = "Elevated / At-Risk"
            recommendation = "High concentration of clinical risk markers (self-focus, absolutist cognition, somatic distress). Human triage recommended."

        # Compute local feature drivers
        key_drivers = []
        if self.model_type == "logistic_regression":
            coeffs = self.model.coef_[0]
            contributions = scaled_vec[0] * coeffs
            for feat, cont in zip(self.FEATURE_NAMES, contributions):
                if cont > 0.2:  # pushes towards at-risk
                    key_drivers.append({
                        "feature": feat,
                        "value": features[feat],
                        "impact": "Increases Risk Score"
                    })

        return {
            "risk_score": round(risk_prob, 4),
            "triage_category": category,
            "threshold": self.risk_threshold,
            "recommendation": recommendation,
            "linguistic_features": features,
            "top_risk_drivers": key_drivers
        }
