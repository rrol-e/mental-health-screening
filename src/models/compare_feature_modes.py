import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.dataset_loader import DatasetLoader
from src.models.hybrid_classifier import HybridRiskModel

def run_feature_comparison():
    print("=" * 85)
    print("  FEATURE ABLATION EXPERIMENT: PSYCHOLINGUISTIC vs. UNSUPERVISED vs. HYBRID")
    print("=" * 85)

    loader = DatasetLoader(deidentify=True)
    df_train, df_test = loader.load_dreaddit(use_cache=True)

    train_texts = df_train['clean_text'].tolist()
    train_labels = df_train['label'].tolist()
    test_texts = df_test['clean_text'].tolist()
    test_labels = df_test['label'].tolist()

    print(f"\n[+] Dataset Partition:")
    print(f"    - Training Set: {len(train_texts):,} samples")
    print(f"    - Test Set:     {len(test_texts):,} samples (unseen out-of-sample)")

    modes = [
        ("Psycholinguistic Only", "psycholinguistic"),
        ("Unsupervised Topics Only (LDA)", "unsupervised"),
        ("Hybrid (Psycholinguistic + Discovered Topics)", "hybrid")
    ]

    results = []

    for label, mode in modes:
        print(f"\n[*] Training [{label}]...")
        model = HybridRiskModel(n_topics=6, model_type="logistic_regression", feature_mode=mode)
        model.fit(train_texts, train_labels)

        eval_res = model.evaluate(test_texts, test_labels)
        m = eval_res['metrics']
        
        results.append({
            "Mode": label,
            "Features": eval_res['feature_count'],
            "Accuracy": m['accuracy'],
            "Recall": m['At-Risk']['recall'],
            "Precision": m['At-Risk']['precision'],
            "F1": m['At-Risk']['f1-score'],
            "ROC_AUC": eval_res['roc_auc']
        })

    print("\n" + "=" * 85)
    print("  OUT-OF-SAMPLE TEST SET PERFORMANCE COMPARISON (DREADDIT)")
    print("=" * 85)
    print(f"  {'Feature Architecture':<42} | {'Feats':>5} | {'Recall':>7} | {'Prec':>6} | {'F1':>6} | {'ROC-AUC':>7}")
    print("  " + "-" * 81)

    for r in results:
        print(f"  {r['Mode']:<42} | {r['Features']:>5} | {r['Recall']*100:>6.1f}% | {r['Precision']*100:>5.1f}% | {r['F1']:>6.3f} | {r['ROC_AUC']:>7.4f}")

    print("=" * 85)

if __name__ == '__main__':
    run_feature_comparison()
