# MediTrack: S2 Dataset Migration & 30% Milestone Review & Academic Defense

## 1. Problem Statement
Essential medicine stock-outs across healthcare networks directly undermine public health, disrupt maternal and emergency care, and lead to preventable mortality. In developing healthcare systems, clinics frequently report inventory stock-outs post-hoc. MediTrack formulates an explainable machine learning early warning system that predicts whether a specific healthcare facility will experience a stock-out of an essential medicine during the subsequent monthly reporting cycle.

---

## 2. Project Objective
To establish a fully verified, leakage-free, reproducible, and academically defensible 30% working prototype of MediTrack using the empirical **S2 National DHIS2 Dataset** from Sierra Leone (Data Dryad DOI: [10.5061/dryad.h9w0vt4tw](https://doi.org/10.5061/dryad.h9w0vt4tw), *Nature*, 2026).

---

## 3. Dataset Provenance & Characteristics
- **Dataset**: S2 Longitudinal Medicine Consumption and Inventory Dataset
- **Repository**: Data Dryad ([https://datadryad.org/dataset/doi:10.5061/dryad.h9w0vt4tw](https://datadryad.org/dataset/doi:10.5061/dryad.h9w0vt4tw))
- **Study Reference**: Chung et al., "Improving access to essential medicines via decision-aware machine learning," *Nature*, 2026.
- **Scale**: 457,225 total records across 1,091 public health facilities and 36 essential medicines/supplies over 50 continuous monthly reporting periods (October 2019 to November 2023).
- **Quality**: 0 missing values across core reporting fields; continuous monthly temporal sequences.

### Essential Medicine & Clinical Supplies Coverage (36 Products):
- **Reproductive Health**: Oxytocin (10IU amp), Misoprostol (200mcg tab), Pregnancy Rapid Test Kits.
- **Respiratory & Emergency**: Salbutamol (100mcg inhaler), Prednisolone (5mg tab), Dextrose 50% & 5% IV.
- **Antimicrobials & Analgesics**: Ampicillin (500mg inj), Ceftriaxone (250mg inj), Ciprofloxacin (500mg tab), Erythromycin, Paracetamol (500mg & 250mg).
- **Child Health & Nutrition**: Oral Rehydration Salts (ORS), Zinc Sulphate (20mg tab), Albendazole (400mg), Folic Acid (5mg), Ferrous Sulphate (200mg).
- **Clinical Consumables**: Disposable gloves, surgical masks, IV cannulas (18G & 24G), syringes (2ml, 10ml, 20ml), IV giving sets.

---

## 4. Stock-Out Definition & Target Formulation
- **Source Field**: `stockout` ($1 = \text{Stockout occurred during reporting month}, 0 = \text{Maintained positive inventory}$).
- **Target Formulation**: Binary indicator of whether the health facility experiences a stock-out of that product in the **next monthly cycle ($T+1$)**:
$$\text{is\_stockout\_next\_month}_{i, j, T} = \text{stockout}_{i, j, T+1}$$
- **Prediction Horizon**: 1 Month (aligned with national DHIS2 procurement and delivery review cycles).
- **Target Distribution**: 59,845 positive instances (13.91%), 370,288 negative instances (86.09%) across 430,133 labeled records.

---

## 5. Strict Data Leakage Prevention
- **Chronological Cutoff**:
  - **Training Window (80%)**: 2019-10-01 to 2022-12-01 (337,727 samples).
  - **Testing Window (20%)**: 2023-01-01 to 2023-10-01 (92,406 unseen holdout samples).
  - Guarantee: $\max(T_{\text{train}}) = \text{2022-12-01} < \min(T_{\text{test}}) = \text{2023-01-01}$.
- **Backward-Looking Features ($t \le T$)**:
  - `current_stock` (`closeBalance_T`)
  - `open_balance` (`openBalance_T`)
  - `current_consumption` (`consumption_T`)
  - `current_received` (`received_T`)
  - `current_stockout` (`stockout_T`)
  - `consumption_roll3_mean`, `consumption_roll6_mean`, `consumption_roll3_std`
  - `stock_change_1m`, `stock_to_consumption_ratio`, `estimated_days_of_stock`
  - `historical_stockout_rate` (expanding historical frequency)
  - `facility_type_encoded`, `district_encoded`
- Verified by automated unit test `test_zero_future_leakage_s2`.

---

## 6. Baseline ML Model & Actual Chronological Performance
- **Algorithm**: `RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=20, min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1)`
- **Evaluation on 92,406 Unseen Future Test Instances (2023)**:

| Metric | Score | Operational Significance |
|---|---|---|
| **Recall (Stockout Class)** | **81.95%** (12,380 / 15,106) | Prioritized metric: Successfully catches 82% of imminent stockouts across 1,091 facilities. |
| **Precision** | **29.75%** (12,380 / 41,608) | Balanced for low false negative rate on imbalanced (16.35%) test stream. |
| **ROC-AUC** | **0.8014** | Strong class separability across probability thresholds. |
| **PR-AUC** | **0.4486** | 2.7x lift over random baseline (16.35%). |
| **Accuracy** | **65.42%** | Reflects balanced decision boundary favoring high recall. |
| **F1-Score** | **0.4366** | Harmonic mean reflecting decision-support prioritization. |

### Confusion Matrix Breakdown:
- **True Negatives ($TN$)**: 48,072
- **False Positives ($FP$)**: 29,228 (Early warning alert triggered)
- **False Negatives ($FN$)**: 2,726 (Missed stockouts)
- **True Positives ($TP$)**: 12,380 (Stockouts successfully anticipated)

---

## 7. Explainability & Decision Attributions (SHAP)
- **Global Feature Importance**:
  1. `estimated_days_of_stock` (33.75%)
  2. `stock_to_consumption_ratio` (25.75%)
  3. `historical_stockout_rate` (11.60%)
  4. `current_stock` (9.84%)
  5. `open_balance` (6.21%)
- **Local Explanations**: Generates natural language attribution statements for each facility-medicine prediction.

---

## 8. Prototype Risk Categorization
- 🟢 **LOW Risk**: Probability $< 0.20$
- 🟡 **MEDIUM Risk**: $0.20 \le \text{Probability} < 0.50$
- 🔴 **HIGH Risk**: $\text{Probability} \ge 0.50$

---

## 9. Completed 30% vs. Remaining 70%
- **Completed (30%)**:
  - Full ingestion, cleaning, and feature engineering of S2 DHIS2 dataset.
  - Chronological holdout evaluation on 92,406 unseen test instances.
  - SHAP global and local explainability layer.
  - Multi-facility surveillance dashboard (`dashboard/app.py`).
  - Automated unit test suite (8/8 passing).
  - Kaggle baseline safely preserved in `archive/kaggle_baseline/`.
- **Remaining Future Work (70%)**:
  - Gradient boosted trees (LightGBM / XGBoost) and Temporal Fusion Transformers.
  - Probability calibration (Platt scaling / Isotonic regression).
  - District-level pooled inventory transfer optimization.
  - Automated SMS/email alert dispatchers and HIS integration.
