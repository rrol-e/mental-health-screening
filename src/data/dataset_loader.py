import os
import pandas as pd
from typing import Optional, Tuple
from datasets import load_dataset
from src.utils.deidentify import TextDeidentifier

class DatasetLoader:
    """
    Unified Dataset Loader for Mental Health Risk Screening.
    Supports academic benchmarks (Dreaddit) and local custom/seed datasets.
    Automatically de-identifies texts to maintain ethical standards.
    """

    PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed'))
    SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'samples'))

    def __init__(self, deidentify: bool = True):
        self.deidentifier = TextDeidentifier() if deidentify else None
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)

    def load_dreaddit(self, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the Dreaddit academic benchmark dataset (3,553 posts across 10 subreddits).
        Caches locally to parquet for instant offline reloading.
        """
        train_cache = os.path.join(self.PROCESSED_DIR, "dreaddit_train.parquet")
        test_cache = os.path.join(self.PROCESSED_DIR, "dreaddit_test.parquet")

        if use_cache and os.path.exists(train_cache) and os.path.exists(test_cache):
            df_train = pd.read_parquet(train_cache)
            df_test = pd.read_parquet(test_cache)
            return df_train, df_test

        print("[*] Fetching Dreaddit benchmark from Hugging Face...")
        hf_dataset = load_dataset('andreagasparini/dreaddit')

        df_train = hf_dataset['train'].to_pandas()
        df_test = hf_dataset['test'].to_pandas()

        # Standardize core columns
        keep_cols = ['text', 'label', 'subreddit', 'confidence', 'social_timestamp', 'social_karma']
        df_train = df_train[[c for c in keep_cols if c in df_train.columns]]
        df_test = df_test[[c for c in keep_cols if c in df_test.columns]]

        # Apply ethical de-identification
        if self.deidentifier:
            print("[*] De-identifying training set (scrubbing PII/URLs/handles)...")
            df_train['clean_text'] = df_train['text'].apply(self.deidentifier.clean)
            print("[*] De-identifying test set...")
            df_test['clean_text'] = df_test['text'].apply(self.deidentifier.clean)
        else:
            df_train['clean_text'] = df_train['text']
            df_test['clean_text'] = df_test['text']

        # Cache locally
        df_train.to_parquet(train_cache, index=False)
        df_test.to_parquet(test_cache, index=False)
        print(f"[+] Cached processed Dreaddit to: {self.PROCESSED_DIR}")

        return df_train, df_test

    def load_seed(self) -> pd.DataFrame:
        """
        Loads local seed dataset for unit testing and quick smoke tests.
        """
        seed_path = os.path.join(self.SAMPLES_DIR, "seed_dataset.json")
        df = pd.read_json(seed_path)
        if self.deidentifier:
            df['clean_text'] = df['text'].apply(self.deidentifier.clean)
        else:
            df['clean_text'] = df['text']
        return df
