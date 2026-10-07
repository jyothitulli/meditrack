# MediTrack: Existing Implementation Audit

## 1. Audit Overview
This audit inspects the current 30% working prototype built on the Kaggle Longitudinal Daily Pharmaceutical Dispensing Dataset prior to investigating the S2 Dryad dataset.

---

## 2. Component-by-Component Assessment Matrix

| Component | Current Implementation | Dataset-Dependent? | Reusable? | Required Changes for S2 Migration |
|---|---|:---:|:---:|---|
| **Raw Dataset Ingestion** (`data/raw/salesdaily.csv`) | Ingests daily sales across 8 WHO ATC medicine categories (2,106 daily records). | **YES** | **NO** | Replace with real S2 Dryad dataset ingestion (`hf_pk`, `productID`, `date`, `stockout`, `openBalance`, `closeBalance`, `received`, `consumption`). |
| **Data Preprocessing** (`src/data/preprocess.py`) | Wide-to-long panel reshaping; simulated periodic review $(s, S)$ inventory ledger. | **YES** | **PARTIALLY** | Remove simulated inventory ledger since S2 provides empirical physical balances (`openBalance`, `closeBalance`, `received`, `stockout`). Adapt to multi-facility panel structure. |
| **Feature Engineering** (`src/data/feature_engineering.py`) | 14 rolling features (7d/14d/30d), consumption velocity, days since replenishment, arithmetic days of stock. | **YES** | **YES** | Rebuild feature calculations on S2 temporal resolution (e.g. monthly/weekly/daily per facility-medicine time series). Maintain zero-leakage guarantee. |
| **Target Generation** (`src/data/feature_engineering.py`) | Forward 14-day rolling minimum stock $\le 0$. | **YES** | **PARTIALLY** | Formulate target based on real empirical `stockout` records in S2 over the true forward horizon without data leakage. |
| **Configuration & Metadata** (`src/utils/config.py`) | Paths, 8 ATC codes, 14-day horizon, prototype risk thresholds (0.35 / 0.65). | **YES** | **YES** | Update medicine metadata dictionary, facility mappings, and prediction horizon constants based on S2. |
| **Chronological Train/Test Split** (`src/ml/train.py`) | 80% earliest calendar dates for training, 20% latest dates for testing. | **NO** | **YES** | Retain strict chronological date boundary splitting logic across all facility-medicine pairs. |
| **Baseline ML Model** (`src/ml/train.py`) | `RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced')`. | **NO** | **YES** | Fully reusable baseline architecture and evaluation pipeline. |
| **Predictor & Inference Engine** (`src/ml/predict.py`) | Batch and single-instance inference, prototype risk categorization (`get_risk_level`). | **NO** | **YES** | Fully reusable; adapt input schema for facility and product identifiers. |
| **Explainability Engine** (`src/ml/explain.py`) | SHAP TreeExplainer, beeswarm/bar plots, natural language attribution narratives. | **NO** | **YES** | Fully reusable; adapt feature labels and narrative templates to S2 features. |
| **Demonstration Dashboard** (`dashboard/app.py`) | Streamlit application with KPI cards, surveillance table, 180d trends, local SHAP breakdowns. | **PARTIALLY** | **YES** | Enhance UI to support facility-level selection and real empirical stockout histories. |
| **Test Suite** (`tests/`) | 10 unit tests covering preprocessing, feature engineering, zero-leakage, split, model, and SHAP. | **PARTIALLY** | **YES** | Update schema assertions to validate S2 fields and multi-facility integrity. |
| **Master Pipeline Runner** (`run_pipeline.py`) | Single orchestrator script executing ingestion to explainability. | **NO** | **YES** | Fully reusable orchestration flow. |

---

## 3. Preservation & Rollback Plan
The Kaggle baseline implementation has been safely preserved in `archive/kaggle_baseline/` to guarantee non-destructive evaluation and enable complete rollback or side-by-side benchmarking.
