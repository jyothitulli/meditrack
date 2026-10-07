# MediTrack: Explainable Machine Learning Early Warning System for Essential Medicine Stock-Out Risk

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passing-brightgreen.svg)]()
[![Data](https://img.shields.io/badge/Data-Dryad%20(S2%20DHIS2)-0B7285.svg)](https://doi.org/10.5061/dryad.h9w0vt4tw)

> **Milestone Status**: S2 Dataset Migrated 30% Working Prototype (Academic Decision-Support Early Warning System)

---

## 1. Project Overview
MediTrack is an explainable machine learning decision-support tool designed for healthcare supply chain managers. It forecasts the risk that public health facilities will experience a stock-out of essential medicines (e.g., Oxytocin, Salbutamol, Ampicillin, Paracetamol, ORS) during the **next monthly reporting cycle (1-month horizon)**, enabling proactive replenishment interventions.

---

## 2. Directory Structure

```text
meditrack/
│
├── data/
│   ├── s2_raw/
│   │   ├── S2_Dhis2Data.csv            # S2 National DHIS2 Dataset (457,225 records)
│   │   ├── S1.csv                      # Health facility master list
│   │   └── README.md                   # Dryad source documentation
│   ├── processed/
│   │   ├── processed_inventory.csv     # Cleaned longitudinal inventory panel
│   │   └── features_dataset.csv        # Engineered feature matrix with forward target
│   └── DATASET_DOCUMENTATION.md        # Provenance and schema documentation
│
├── archive/
│   └── kaggle_baseline/                # Preserved Kaggle baseline implementation (Rollback safe)
│
├── src/
│   ├── data/
│   │   ├── preprocess.py               # Ingestion, validation, and categorical encoding
│   │   └── feature_engineering.py      # Leakage-free temporal features & forward target
│   │
│   ├── ml/
│   │   ├── train.py                    # Chronological split, Random Forest training & evaluation
│   │   ├── predict.py                  # Predictor class, risk categorization, snapshot scoring
│   │   └── explain.py                  # SHAP TreeExplainer global & local attributions
│   │
│   └── utils/
│       └── config.py                   # Paths, thresholds, and hyperparameters
│
├── models/
│   ├── baseline_rf_model.joblib        # Persisted trained Random Forest model
│   └── model_metadata.json             # Feature names, timelines, and configuration
│
├── outputs/
│   ├── metrics.json                    # Full quantitative performance metrics
│   ├── evaluation_report.txt           # Detailed evaluation summary
│   ├── confusion_matrix.png            # Confusion matrix plot on chronological test set
│   └── explainability/
│       ├── shap_summary.png            # Global SHAP beeswarm plot
│       └── shap_bar.png                # Global SHAP mean absolute importance bar plot
│
├── dashboard/
│   └── app.py                          # Interactive Streamlit surveillance dashboard
│
├── docs/
│   ├── EXISTING_IMPLEMENTATION_AUDIT.md # Pre-migration baseline audit
│   ├── S2_DATASET_FEASIBILITY.md       # Feasibility assessment of S2 Dryad dataset
│   ├── DATASET_COMPARISON.md           # Comparative analysis: Kaggle vs. S2
│   ├── STOCKOUT_DEFINITION.md          # Stockout semantics and target formulation
│   ├── ML_PROBLEM_FORMULATION.md       # Machine learning task specification
│   ├── FEATURE_LEAKAGE_AUDIT.md        # Feature timing and anti-leakage audit
│   ├── TEMPORAL_SPLIT.md               # Chronological train/test split strategy
│   └── REVIEW_30_PERCENT.md            # Comprehensive academic defense document
│
├── tests/
│   ├── test_preprocessing.py           # Preprocessing & schema tests
│   ├── test_feature_engineering.py     # Feature calculation & zero future leakage tests
│   ├── test_model.py                   # Chronological split & inference tests
│   └── test_explainability.py          # SHAP attribution tests
│
├── run_pipeline.py                     # Master orchestrator script (End-to-end execution)
├── requirements.txt                    # Project dependencies
├── FINAL_30_PERCENT_CHECKLIST.md       # Verification checklist
└── README.md                           # Quickstart guide
```

---

## 3. Quickstart & Execution Guide

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Run Automated Unit Tests
```powershell
pytest tests/ -v
```

### Step 3: Run the Complete Machine Learning Pipeline
```powershell
python run_pipeline.py
```

### Step 4: Launch the Multi-Facility Interactive Dashboard
```powershell
streamlit run dashboard/app.py
```

---

## 4. Key Performance Highlights (Chronological Holdout Test Set)

*Evaluated on 92,406 unseen future instances across 2023 (1,091 health facilities):*

| Metric | Score | Interpretation |
|---|---|---|
| **Recall (Stockout Class)** | **81.95%** | Primary clinical priority: Catches 12,380 out of 15,106 future stockout events |
| **ROC-AUC** | **0.8014** | Strong class separability on real-world noisy DHIS2 data |
| **PR-AUC** | **0.4486** | 2.7x lift over random baseline prevalence (16.35%) |
| **Precision** | **29.75%** | Provides actionable decision-support early warning alerts |
| **Accuracy** | **65.42%** | Reflects balanced decision threshold prioritizing stockout recall |

---

## 5. Academic Honesty & Disclaimer
- **Authentic Dataset**: Uses the real national DHIS2 dataset from Sierra Leone (Data Dryad DOI: 10.5061/dryad.h9w0vt4tw).
- **Zero Temporal Leakage**: Enforces strict chronological train/test partitioning ($\max(T_{\text{train}}) < \min(T_{\text{test}})$).
- **Decision-Support Prototype**: This tool is an early warning prototype and does not claim clinical validation.
