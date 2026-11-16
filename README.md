# MediTrack: Explainable Machine Learning Early Warning System for Essential Medicine Stock-Out Risk

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-15%2F15%20Passing-brightgreen.svg)]()
[![Data](https://img.shields.io/badge/Data-Dryad%20(S2%20DHIS2)-0B7285.svg)](https://doi.org/10.5061/dryad.h9w0vt4tw)

> **Milestone Status**: S2 Dataset Migrated 50% Milestone Working Prototype (Benchmarked, Calibrated & Prescriptive Early Warning System)

---

## 1. Project Overview
MediTrack is an explainable machine learning decision-support tool designed for healthcare supply chain managers. It forecasts the risk that public health facilities will experience a stock-out of essential medicines (e.g., Oxytocin, Salbutamol, Ampicillin, Paracetamol, ORS) during the **next monthly reporting cycle (1-month horizon)**, enabling proactive replenishment interventions and intra-district lateral stock rebalancing.

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
│   │   ├── features_dataset.csv        # Engineered feature matrix v2 (Seasonality, Surge, Acceleration)
│   │   └── features_dataset.parquet    # Compressed high-speed parquet format
│   └── DATASET_DOCUMENTATION.md        # Provenance and schema documentation
│
├── archive/
│   └── kaggle_baseline/                # Preserved Kaggle baseline implementation (Rollback safe)
│
├── src/
│   ├── data/
│   │   ├── preprocess.py               # Ingestion, validation, and categorical encoding
│   │   └── feature_engineering.py      # Leakage-free temporal features & forward target (v2 features)
│   │
│   ├── ml/
│   │   ├── train.py                    # Chronological split & Random Forest training
│   │   ├── benchmark.py                # Multi-model tournament & Walk-Forward temporal CV
│   │   ├── calibrate.py                # Platt Scaling & Isotonic probability calibration
│   │   ├── prescriptive.py             # Reorder quantities (ROQ) & District lateral transfer rebalancing
│   │   ├── predict.py                  # Predictor class (supports calibrated, RF, and GBM)
│   │   └── explain.py                  # SHAP TreeExplainer global & local attributions
│   │
│   └── utils/
│       └── config.py                   # Paths, thresholds, hyperparameters & feature definitions
│
├── models/
│   ├── baseline_rf_model.joblib        # Persisted trained Random Forest model
│   ├── calibrated_model.joblib         # Calibrated Isotonic probability model
│   ├── gradient_boosting_model.joblib  # Trained HistGradientBoosting model
│   └── model_metadata.json             # Feature names, timelines, and configuration
│
├── outputs/
│   ├── metrics.json                    # Baseline quantitative performance metrics
│   ├── model_benchmark.json            # Multi-model tournament & Walk-forward CV results
│   ├── model_benchmark_report.txt      # Comprehensive benchmark text summary
│   ├── roc_comparison.png              # Multi-model ROC curves on chronological holdout
│   ├── pr_comparison.png               # Multi-model Precision-Recall curves on holdout
│   ├── calibration_curve.png           # Reliability diagram (Uncalibrated vs Platt vs Isotonic)
│   ├── confusion_matrix.png            # Confusion matrix plot on chronological test set
│   └── explainability/
│       ├── shap_summary.png            # Global SHAP beeswarm plot
│       └── shap_bar.png                # Global SHAP mean absolute importance bar plot
│
├── dashboard/
│   └── app.py                          # 5-view Streamlit interactive surveillance & prescriptive dashboard
│
├── docs/
│   ├── MILESTONE_50_PERCENT.md         # Comprehensive 50% Milestone academic defense document
│   ├── REVIEW_30_PERCENT.md            # Preserved 30% Milestone review document
│   ├── EXISTING_IMPLEMENTATION_AUDIT.md # Pre-migration baseline audit
│   ├── S2_DATASET_FEASIBILITY.md       # Feasibility assessment of S2 Dryad dataset
│   ├── DATASET_COMPARISON.md           # Comparative analysis: Kaggle vs. S2
│   ├── STOCKOUT_DEFINITION.md          # Stockout semantics and target formulation
│   ├── ML_PROBLEM_FORMULATION.md       # Machine learning task specification
│   ├── FEATURE_LEAKAGE_AUDIT.md        # Feature timing and anti-leakage audit
│   └── TEMPORAL_SPLIT.md               # Chronological train/test split strategy
│
├── tests/
│   ├── test_preprocessing.py           # Preprocessing & schema tests
│   ├── test_feature_engineering.py     # Feature calculation & zero future leakage tests
│   ├── test_model.py                   # Chronological split & multi-model inference tests
│   ├── test_benchmark.py               # Model tournament initialization & metrics tests
│   ├── test_calibration.py             # Probability calibration & ECE tests
│   ├── test_prescriptive.py            # Reorder quantities & lateral transfer tests
│   └── test_explainability.py          # SHAP attribution tests
│
├── run_pipeline.py                     # Master orchestrator script (7-step end-to-end execution)
├── requirements.txt                    # Project dependencies
├── FINAL_50_PERCENT_CHECKLIST.md       # 50% Milestone Verification checklist
├── FINAL_30_PERCENT_CHECKLIST.md       # Preserved 30% Verification checklist
└── README.md                           # Quickstart guide
```

---

## 3. Quickstart & Execution Guide

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Run Automated Unit Tests (15 Tests)
```powershell
pytest tests/ -v
```

### Step 3: Run the Complete Machine Learning Pipeline (End-to-End)
```powershell
python run_pipeline.py
```

### Step 4: Launch the Multi-Facility Interactive Surveillance Dashboard
```powershell
streamlit run dashboard/app.py
```

---

## 4. Key Performance Highlights (Model Tournament on 92,406 Test Instances across 2023)

| Algorithm | Recall (Stockout Class) | Precision | F1-Score | ROC-AUC | PR-AUC | Brier Score | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | 81.64% (12,333 / 15,106) | 29.85% | **0.4371** | **0.8069** | **0.4570** | **0.2033** | **0.0006 ms** |
| **Random Forest (Balanced)** | **82.23%** (12,422 / 15,106) | 29.80% | 0.4365 | 0.8028 | 0.4543 | 0.2034 | 0.0039 ms |
| **Logistic Regression (L2)** | 73.62% (11,121 / 15,106) | 23.47% | 0.3559 | 0.7398 | 0.3356 | 0.2314 | 0.0001 ms |

---

## 5. Walk-Forward Temporal Cross-Validation (3 Expanding Windows)

| Algorithm | Mean Recall ± Std | Mean F1 ± Std | Mean ROC-AUC ± Std |
|---|:---:|:---:|:---:|
| **Random Forest (Balanced)** | **80.95% ± 1.13%** | 0.4418 ± 0.0041 | **0.8052 ± 0.0028** |
| **HistGradientBoosting** | 80.12% ± 1.28% | **0.4431 ± 0.0049** | 0.8041 ± 0.0031 |
| **Logistic Regression (L2)** | 72.88% ± 0.94% | 0.3592 ± 0.0038 | 0.7411 ± 0.0016 |

---

## 6. Academic Honesty & Disclaimer
- **Authentic Dataset**: Uses the real national DHIS2 dataset from Sierra Leone (Data Dryad DOI: 10.5061/dryad.h9w0vt4tw, *Nature* 2026).
- **Zero Temporal Leakage**: Enforces strict chronological train/test partitioning ($\max(T_{\text{train}}) < \min(T_{\text{test}})$) and walk-forward CV.
- **Decision-Support Prototype**: This tool is an early warning and prescriptive decision-support prototype to assist authorized inventory managers.

