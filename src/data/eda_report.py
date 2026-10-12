import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.dataset_loader import DatasetLoader
from src.features.psycholinguistic import PsycholinguisticExtractor

def generate_eda_report():
    print("=" * 75)
    print("     DREADDIT BENCHMARK DATASET: CLINICAL NLP EXPLORATORY ANALYSIS")
    print("=" * 75)

    loader = DatasetLoader()
    df_train, df_test = loader.load_dreaddit(use_cache=True)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    total_samples = len(df_all)
    at_risk_count = int((df_all['label'] == 1).sum())
    control_count = int((df_all['label'] == 0).sum())

    print(f"\n[1] DATASET OVERVIEW:")
    print(f"    - Total Labeled Posts:  {total_samples:,}")
    print(f"    - At-Risk / High Stress: {at_risk_count:,} ({at_risk_count / total_samples * 100:.1f}%)")
    print(f"    - Stable / Non-Stressed: {control_count:,} ({control_count / total_samples * 100:.1f}%)")

    print(f"\n[2] SUBREDDIT COMPOSITION & CLASS DISTRIBUTION:")
    sub_summary = df_all.groupby('subreddit')['label'].agg(
        Total='count',
        At_Risk=lambda x: (x == 1).sum(),
        Control=lambda x: (x == 0).sum(),
        Risk_Pct=lambda x: (x == 1).mean() * 100
    ).sort_values(by='Total', ascending=False)

    print(f"    {'Subreddit':<20} | {'Total':>7} | {'At-Risk':>8} | {'Control':>8} | {'% At-Risk':>10}")
    print("    " + "-" * 62)
    for sub, row in sub_summary.iterrows():
        print(f"    {sub:<20} | {int(row['Total']):>7} | {int(row['At_Risk']):>8} | {int(row['Control']):>8} | {row['Risk_Pct']:>9.1f}%")

    print(f"\n[3] EXTRACTING PSYCHOLINGUISTIC MARKERS ACROSS ALL POSTS...")
    extractor = PsycholinguisticExtractor()
    feature_list = []
    for text in df_all['clean_text']:
        feature_list.append(extractor.extract_features(text))
    
    df_features = pd.DataFrame(feature_list)
    df_combined = pd.concat([df_all[['label']], df_features], axis=1)

    print(f"\n[4] PSYCHOLINGUISTIC FEATURE COMPARISON (AT-RISK vs. CONTROL):")
    print(f"    {'Feature Name':<28} | {'Control (0)':>11} | {'At-Risk (1)':>11} | {'Welch t-stat':>13} | {'p-value':>10}")
    print("    " + "-" * 82)

    for col in df_features.columns:
        group_0 = df_combined[df_combined['label'] == 0][col].values
        group_1 = df_combined[df_combined['label'] == 1][col].values
        
        mean_0 = np.mean(group_0)
        mean_1 = np.mean(group_1)
        
        t_stat, p_val = stats.ttest_ind(group_1, group_0, equal_var=False)
        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"

        print(f"    {col:<28} | {mean_0:>11.4f} | {mean_1:>11.4f} | {t_stat:>13.2f} | {p_str:>10}")

    print("\n" + "=" * 75)
    print("  EXPLORATORY DATA ANALYSIS COMPLETE")
    print("=" * 75)

if __name__ == '__main__':
    generate_eda_report()
