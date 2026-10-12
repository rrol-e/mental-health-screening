import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_loader import DatasetLoader
from src.models.baseline_classifier import RiskAssessmentModel

def run_pipeline(dataset_choice: str = "dreaddit", model_type: str = "logistic_regression"):
    print("=" * 75)
    print(f"  MENTAL HEALTH AT-RISK SCREENING PIPELINE [{dataset_choice.upper()}]")
    print("=" * 75)

    loader = DatasetLoader(deidentify=True)

    if dataset_choice == "dreaddit":
        print("[*] Loading cached Dreaddit benchmark (de-identified)...")
        df_train, df_test = loader.load_dreaddit(use_cache=True)
        train_texts = df_train['clean_text'].tolist()
        train_labels = df_train['label'].tolist()
        test_texts = df_test['clean_text'].tolist()
        test_labels = df_test['label'].tolist()
    else:
        print("[*] Loading local seed dataset...")
        df_seed = loader.load_seed()
        train_texts = df_seed['clean_text'].tolist()
        train_labels = df_seed['risk_label'].tolist()
        test_texts = train_texts
        test_labels = train_labels

    print(f"\n[+] Dataset Partition:")
    print(f"    - Training Set: {len(train_texts):,} samples (At-Risk: {sum(train_labels):,}, Stable: {len(train_labels)-sum(train_labels):,})")
    print(f"    - Test Set:     {len(test_texts):,} samples (At-Risk: {sum(test_labels):,}, Stable: {len(test_labels)-sum(test_labels):,})")

    # Train model
    print(f"\n[+] Fitting {model_type.replace('_', ' ').title()} on Psycholinguistic Features...")
    model = RiskAssessmentModel(model_type=model_type, risk_threshold=0.5)
    model.fit(train_texts, train_labels)

    # Evaluate on out-of-sample Test Set
    eval_results = model.evaluate(test_texts, test_labels)
    metrics = eval_results['metrics']

    print("\n[+] Out-of-Sample Test Evaluation Metrics:")
    print(f"    - Overall Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    - At-Risk Precision: {metrics['At-Risk']['precision']:.4f}")
    print(f"    - At-Risk Recall:    {metrics['At-Risk']['recall']:.4f} (Sensitivity / False Negative Guard)")
    print(f"    - At-Risk F1-Score:  {metrics['At-Risk']['f1-score']:.4f}")
    print(f"    - Macro F1:          {metrics['macro avg']['f1-score']:.4f}")
    print(f"    - ROC-AUC:           {eval_results['roc_auc']:.4f}")

    cm = eval_results['confusion_matrix']
    print(f"\n[+] Confusion Matrix (Test Set):")
    print(f"    TN (True Stable):   {cm[0][0]:<5} | FP (False Alert):    {cm[0][1]:<5}")
    print(f"    FN (Missed Risk):   {cm[1][0]:<5} | TP (True At-Risk):   {cm[1][1]:<5}")

    # Global explanations
    print("\n[+] Top Global Psycholinguistic Indicators:")
    explanations = model.explain_global_features()
    for feat, weight in explanations:
        direction = "Elevates Risk" if weight > 0 else "Protective/Stable"
        print(f"    * {feat:<28} | Weight: {weight:>6.4f} ({direction})")

    # Sample test inferences
    test_cases = [
        (
            "High Distress / At-Risk Signal",
            "I feel completely empty and exhausted. Every single day is unbearable and I have nobody left to talk to. I just want this endless pain to stop."
        ),
        (
            "Everyday Conversational / Control Signal",
            "We spent the afternoon hiking up the mountain trail with our friends. The views from the summit were fantastic and we had a picnic lunch."
        ),
        (
            "Mild Daily Stress / Non-Pathological",
            "My laptop battery died during the presentation today which was slightly frustrating, but my manager was understanding."
        )
    ]

    print("\n" + "=" * 75)
    print("  SAMPLE INFERENCE & RISK TRIAGE")
    print("=" * 75)

    for title, text in test_cases:
        assessment = model.assess_risk(text)
        print(f"\n>>> Input: [{title}]")
        print(f"    \"{text}\"")
        print(f"    - Predicted Risk Score: {assessment['risk_score'] * 100:.1f}%")
        print(f"    - Triage Category:      {assessment['triage_category']}")
        print(f"    - Clinical Note:        {assessment['recommendation']}")
        if assessment['top_risk_drivers']:
            print("    - Primary Linguistic Drivers:")
            for d in assessment['top_risk_drivers']:
                print(f"       * {d['feature']} ({d['impact']})")

    print("\n" + "=" * 75)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 75)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mental Health Risk Screening Pipeline")
    parser.add_argument("--dataset", choices=["dreaddit", "seed"], default="dreaddit", help="Dataset to train and evaluate on")
    parser.add_argument("--model", choices=["logistic_regression", "random_forest"], default="logistic_regression", help="Model type")
    args = parser.parse_args()

    run_pipeline(dataset_choice=args.dataset, model_type=args.model)
