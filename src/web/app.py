import os
import sys
import re
from typing import List, Dict, Any
from flask import Flask, request, jsonify, send_from_directory

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.dataset_loader import DatasetLoader
from src.features.psycholinguistic import PsycholinguisticExtractor
from src.features.unsupervised_discovery import UnsupervisedTopicDiscovery
from src.models.hybrid_classifier import HybridRiskModel
from src.utils.deidentify import TextDeidentifier

app = Flask(__name__, static_folder="static")

# Global model state
hybrid_model: HybridRiskModel = None
extractor = PsycholinguisticExtractor()
deidentifier = TextDeidentifier()

def initialize_model():
    global hybrid_model
    if hybrid_model is not None and hybrid_model.is_fitted:
        return

    print("[*] Initializing Hybrid Risk Assessment Model for Web Dashboard...")
    loader = DatasetLoader(deidentify=True)
    df_train, df_test = loader.load_dreaddit(use_cache=True)
    
    train_texts = df_train['clean_text'].tolist()
    train_labels = df_train['label'].tolist()

    hybrid_model = HybridRiskModel(
        n_topics=6,
        model_type="logistic_regression",
        feature_mode="hybrid",
        risk_threshold=0.5
    )
    hybrid_model.fit(train_texts, train_labels)
    print("[+] Hybrid Model initialized and ready for real-time inference.")

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    raw_text = data.get('text', '').strip()

    if not raw_text:
        return jsonify({"error": "Text cannot be empty."}), 400

    clean_text = deidentifier.clean(raw_text)

    # 1. Psycholinguistic extraction
    psycho_features = extractor.extract_features(clean_text)

    # 2. Topic transformation
    topic_weights = hybrid_model.topic_engine.transform([clean_text])[0]
    topic_descriptors = hybrid_model.topic_engine.get_topic_descriptors()
    
    topics_breakdown = []
    for i, w in enumerate(topic_weights):
        topics_breakdown.append({
            "topic_id": i + 1,
            "name": f"Theme #{i+1}",
            "keywords": topic_descriptors[i]['top_keywords'][:5],
            "weight": round(float(w), 4)
        })

    # 3. Model inference
    feature_vec = hybrid_model._build_features([clean_text])
    scaled_vec = hybrid_model.scaler.transform(feature_vec)
    prob_risk = float(hybrid_model.model.predict_proba(scaled_vec)[0, 1])

    # 4. Risk Triage categorization
    if prob_risk < 0.35:
        category = "Low / Stable"
        badge_color = "emerald"
        recommendation = "No elevated psychiatric risk markers detected. Language pattern is grounded and socially connected."
    elif prob_risk < 0.65:
        category = "Moderate / Observational"
        badge_color = "amber"
        recommendation = "Sub-clinical distress markers or situational stress detected. Supportive resources and wellness check-ins recommended."
    else:
        category = "Elevated / At-Risk"
        badge_color = "rose"
        recommendation = "High concentration of clinical risk markers (intense self-focus, absolutist cognition, somatic exhaustion). Human triage and support intervention recommended."

    # 5. Local Feature Contributions (Explainability)
    coeffs = hybrid_model.model.coef_[0]
    contributions = scaled_vec[0] * coeffs
    drivers = []
    for name, cont in zip(hybrid_model.feature_names_, contributions):
        if abs(cont) > 0.15:
            drivers.append({
                "feature": name.replace('_', ' ').title(),
                "impact": round(float(cont), 3),
                "direction": "Elevates Risk" if cont > 0 else "Protective/Stabilizing"
            })
    drivers.sort(key=lambda x: abs(x['impact']), reverse=True)

    # 6. Highlight tokens in original text
    tokens = extractor.word_token_pattern.findall(raw_text)
    annotated_words = []
    for token in tokens:
        low = token.lower()
        tag = None
        if low in extractor.FIRST_PERSON_SINGULAR:
            tag = "self_focus"
        elif low in extractor.ABSOLUTIST_WORDS:
            tag = "absolutist"
        elif low in extractor.DISTRESS_AFFECT_WORDS:
            tag = "distress"
        elif low in extractor.SOMATIC_WORDS:
            tag = "somatic"
        
        annotated_words.append({
            "word": token,
            "tag": tag
        })

    return jsonify({
        "clean_text": clean_text,
        "risk_score": round(prob_risk, 4),
        "risk_percentage": round(prob_risk * 100, 1),
        "triage_category": category,
        "badge_color": badge_color,
        "recommendation": recommendation,
        "psycholinguistic_features": psycho_features,
        "topics": topics_breakdown,
        "top_drivers": drivers[:6],
        "annotated_tokens": annotated_words
    })

@app.route('/api/sample_cases', methods=['GET'])
def sample_cases():
    return jsonify([
        {
            "title": "Severe Rumination & Despair",
            "category": "High Risk",
            "text": "I feel so completely empty and exhausted. Every single day is unbearable and I have nobody left to talk to. I just want this endless pain to stop."
        },
        {
            "title": "Somatic Panic & Insomnia",
            "category": "High Risk",
            "text": "I haven't slept in three days. My chest is tight from constant anxiety and panic, and I'm scared that I am losing my mind."
        },
        {
            "title": "Social & Recreation (Control)",
            "category": "Stable",
            "text": "We went camping near the lake over the weekend with our family. The trail was beautiful and we had great conversations around the campfire."
        },
        {
            "title": "Everyday Work Stress",
            "category": "Moderate/Stable",
            "text": "Work was pretty hectic today with tight deadlines and client meetings, but looking forward to unwinding with a book tonight."
        }
    ])

@app.route('/api/benchmark_stats', methods=['GET'])
def benchmark_stats():
    return jsonify({
        "dataset_name": "Dreaddit (EMNLP 2019)",
        "total_posts": 3553,
        "train_posts": 2838,
        "test_posts": 715,
        "subreddits": ["ptsd", "anxiety", "relationships", "domesticviolence", "survivorsofabuse", "homeless", "almosthomeless", "stress", "assistance", "food_pantry"],
        "metrics": {
            "recall": 0.751,
            "precision": 0.705,
            "f1": 0.727,
            "roc_auc": 0.7813
        }
    })

if __name__ == '__main__':
    initialize_model()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[+] Starting MindSignals Web Dashboard on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
