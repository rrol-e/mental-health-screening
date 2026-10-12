# Identifying Mentally At-Risk Individuals via Computational Psychiatry & NLP

An ethical machine learning framework designed to screen and triage digital footprints for mental health risk markers **without diagnostic overreach**.

---

## 1. Ethical Guardrails & Project Philosophy
1. **Screening, Not Diagnosis**: This system is designed as an early-detection triage and risk scoring mechanism for support routing—it **does not** output psychiatric diagnoses (e.g., DSM-5 / ICD-11 classifications).
2. **PII Stripping & De-identification**: All inputs pass through an automated anonymization pipeline that strips URLs, usernames, mentions, emails, and phone numbers before feature extraction.
3. **Bias Mitigation**: Ground-truth baselines rely on clinically grounded linguistic markers and verified community contexts rather than ad-hoc subjective labeling.

---

## 2. Feature Vector Architecture
The model relies on mathematically modeled psycholinguistic features established in computational psychiatry literature:
* **First-Person Pronoun Ratio**: High self-referential focus ($I$, $me$, $my$) vs. collective engagement ($we$, $our$).
* **Absolutist Cognitive Distortion Index**: Frequency of dichotomous/black-and-white terms (*always, never, completely, impossible*).
* **Distress & Negative Affect**: Empirical lexicon of anhedonia, exhaustion, and psychological distress.
* **Somatic Disruption Indicators**: Sleep deprivation and physical exhaustion mentions.
* **Syntactic Hesitation & Punctuation**: Ellipses frequency and question density.

---

## 3. Benchmark Dataset: Dreaddit
The system integrates and caches the **Dreaddit** academic benchmark dataset (*Turcan & McKeown, EMNLP 2019*):
* **3,553 labeled Reddit posts** across 10 specialized subreddits (`r/ptsd`, `r/anxiety`, `r/relationships`, `r/domesticviolence`, `r/survivorsofabuse`, `r/homeless`, `r/almosthomeless`, `r/stress`, `r/assistance`, `r/food_pantry`).
* **Balanced Split**: 1,857 At-Risk / High Distress (52.3%) vs. 1,696 Stable / Control (47.7%).
* **Pre-processed & Anonymized**: Stored locally in `data/processed/dreaddit_train.parquet` and `data/processed/dreaddit_test.parquet`.

---

## 4. Directory Layout
```
Health-Project/
├── .venv/                         # Python 3.14 virtual environment
├── data/
│   ├── processed/
│   │   ├── dreaddit_train.parquet # 2,838 de-identified training samples
│   │   └── dreaddit_test.parquet  # 715 out-of-sample test samples
│   └── samples/
│       └── seed_dataset.json      # Unit test seed dataset
├── src/
│   ├── data/
│   │   ├── dataset_loader.py      # Unified benchmark loader & caching
│   │   └── eda_report.py          # Empirical statistical validation
│   ├── utils/
│   │   └── deidentify.py          # Automated PII stripping
│   ├── features/
│   │   └── psycholinguistic.py    # Computational psychiatry feature extractor
│   ├── models/
│   │   └── baseline_classifier.py # Interpretable risk classifier & explainability
│   └── pipeline.py                # End-to-end training & evaluation
├── tests/
│   └── test_pipeline.py           # Unit test suite
├── requirements.txt
└── README.md
```

---

## 5. Quickstart

### Run the Dreaddit Pipeline
```powershell
.\.venv\Scripts\python.exe src/pipeline.py --dataset dreaddit --model logistic_regression
```

### Run the Exploratory Data Analysis (EDA)
```powershell
.\.venv\Scripts\python.exe src/data/eda_report.py
```

### Run Unit Tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```
