# MediTrack: Explainable Machine Learning Early Warning System for Essential Medicine Stock-Out Risk

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-10%2F10%20Passing-brightgreen.svg)]()

> **Milestone Status**: First 30% Working Prototype (Academic Decision-Support Prototype)

---

## 1. Project Overview
MediTrack is an explainable machine learning decision-support tool designed for healthcare and hospital inventory managers. It forecasts the risk that essential medicines will experience a stock-out within a defined tactical horizon (**14 days**), allowing procurement teams to intervene proactively before supplies are exhausted.

---

## 2. Directory Structure

```text
meditrack/
│
├── data/
│   ├── raw/
│   │   └── salesdaily.csv              # Raw longitudinal dispensing records (2,106 days)
│   ├── processed/
│   │   ├── processed_inventory.csv     # Daily inventory tracking ledger
│   │   └── features_dataset.csv        # Engineered feature matrix with forward target
│   └── DATASET_DOCUMENTATION.md        # Dataset provenance, statistics, and audit
│
├── src/
│   ├── data/
│   │   ├── preprocess.py               # Ingestion, validation, and ledger simulation
│   │   └── feature_engineering.py      # Leakage-free temporal features & 14-day target
│   │
│   ├── ml/
│   │   ├── train.py                    # Chronological split, Random Forest training & evaluation
│   │   ├── predict.py                  # Predictor class, risk categorization, snapshot scoring
│   │   └── explain.py                  # SHAP TreeExplainer global & local attributions
│   │
│   └── utils/
│       └── config.py                   # Paths, thresholds, medicine metadata, random seeds
│
├── models/
│   ├── baseline_rf_model.joblib        # Persisted trained Random Forest model
│   └── model_metadata.json             # Feature names, timelines, and hyperparameters
│
├── outputs/
│   ├── metrics.json                    # Full quantitative performance metrics
│   ├── evaluation_report.txt           # Detailed evaluation summary
│   ├── confusion_matrix.png            # Confusion matrix plot on test set
│   └── explainability/
│       ├── shap_summary.png            # Global SHAP beeswarm plot
│       └── shap_bar.png                # Global SHAP mean absolute importance bar plot
│
├── dashboard/
│   └── app.py                          # Interactive Streamlit surveillance dashboard
│
├── docs/
│   ├── IMPLEMENTATION_PLAN.md          # First 30% milestone technical specification
│   ├── DATASET_FEASIBILITY.md          # Candidate dataset evaluation matrix
│   ├── TARGET_DEFINITION.md            # Mathematical target formulation & horizon rationale
│   ├── FEATURE_ENGINEERING.md          # Feature dictionary & timing guarantees
│   ├── DATA_LEAKAGE_PREVENTION.md      # Chronological partitioning methodology
│   └── REVIEW_30_PERCENT.md            # Comprehensive academic defense document
│
├── tests/
│   ├── test_preprocessing.py           # Preprocessing & conservation tests
│   ├── test_feature_engineering.py     # Feature calculation & zero future leakage tests
│   ├── test_model.py                   # Chronological split & inference tests
│   └── test_explainability.py          # SHAP attribution tests
│
├── run_pipeline.py                     # Master orchestrator script (End-to-end execution)
├── requirements.txt                    # Project dependencies
├── FINAL_30_PERCENT_CHECKLIST.md       # Milestone delivery verification checklist
└── README.md                           # Quickstart and setup guide
```

---

## 3. Quickstart & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Step 1: Clone Repository & Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Run Automated Tests
Execute the test suite to verify zero-leakage, ledger conservation, and pipeline integrity:
```powershell
pytest tests/ -v
```

### Step 3: Run the Complete Machine Learning Pipeline
Execute all stages (data loading, preprocessing, feature extraction, chronological training, test evaluation, SHAP explainability, and live predictions):
```powershell
python run_pipeline.py
```

### Step 4: Launch the Interactive Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
```

---

## 4. Key Performance Highlights (Chronological Holdout Test Set)

| Metric | Baseline Score | Interpretation |
|---|---|---|
| **Recall (Stockout Class)** | **96.41%** | Prioritized metric: Successfully catches 96.4% of imminent stockouts |
| **Precision** | **83.56%** | Low false alarm rate for hospital inventory staff |
| **F1-Score** | **0.8953** | Strong balanced performance |
| **ROC-AUC** | **0.9719** | Excellent class discrimination |
| **Accuracy** | **90.63%** | Overall correct classification |

---

## 5. Academic Honesty & Disclaimer
- **Authentic Dataset**: Uses real 6-year daily dispensing records across 8 essential medicine classes.
- **Zero Temporal Leakage**: Enforces strict chronological train/test split ($\max(T_{\text{train}}) < \min(T_{\text{test}})$).
- **Decision-Support Prototype**: This tool is designed to assist inventory managers and does not claim medical or clinical validation.
