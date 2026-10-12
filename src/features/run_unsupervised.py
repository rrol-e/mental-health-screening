import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.dataset_loader import DatasetLoader
from src.features.unsupervised_discovery import UnsupervisedTopicDiscovery

def run_unsupervised_analysis():
    print("=" * 80)
    print("  UNSUPERVISED LATENT PATTERN & THEME DISCOVERY (BLIND EXTRACTION)")
    print("=" * 80)

    loader = DatasetLoader(deidentify=True)
    df_train, df_test = loader.load_dreaddit(use_cache=True)
    
    # Train unsupervised model strictly on text (no labels provided)
    train_texts = df_train['clean_text'].tolist()
    train_labels = df_train['label'].tolist()
    test_texts = df_test['clean_text'].tolist()
    test_labels = df_test['label'].tolist()

    all_texts = train_texts + test_texts
    all_labels = train_labels + test_labels

    print(f"\n[1] Fitting Unsupervised Topic Model (LDA, K=6) on {len(all_texts):,} unlabeled posts...")
    topic_engine = UnsupervisedTopicDiscovery(n_topics=6, method="lda", random_state=42)
    doc_topics = topic_engine.fit_transform(all_texts)

    print("\n[2] DISCOVERED LATENT THEMATIC CLUSTERS (No Supervision):")
    descriptors = topic_engine.get_topic_descriptors()
    for d in descriptors:
        keywords = ", ".join(d['top_keywords'][:8])
        print(f"    * Topic #{d['topic_id'] + 1:<2}: {keywords}")

    print("\n[3] CLUSTER-TO-RISK CORRELATION ANALYSIS (Post-Hoc Verification):")
    print("    Measuring how each discovered cluster correlates with at-risk profiles:")
    print("    " + "-" * 76)
    print(f"    {'Topic':<10} | {'Correlation':>11} | {'At-Risk Wt':>11} | {'Control Wt':>11} | {'Association':<18}")
    print("    " + "-" * 76)

    correlations = topic_engine.correlate_topics_with_risk(doc_topics, all_labels)
    for c in correlations:
        print(f"    {c['name']:<10} | {c['correlation']:>11.4f} | {c['mean_weight_at_risk']:>11.4f} | {c['mean_weight_control']:>11.4f} | {c['risk_association']:<18}")
        keywords = ", ".join(c['top_keywords'])
        print(f"      -> Core Vocabulary: [{keywords}]")

    # Save artifact
    output_path = os.path.join(loader.PROCESSED_DIR, "unsupervised_lda_model.joblib")
    topic_engine.save(output_path)
    print(f"\n[+] Saved fitted topic model to: {output_path}")
    print("=" * 80)

if __name__ == '__main__':
    run_unsupervised_analysis()
