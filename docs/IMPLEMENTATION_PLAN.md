# MediTrack Implementation Plan (First 30% Milestone)

## 1. Project Objective
MediTrack is an explainable machine learning-based early warning system designed for hospital and healthcare inventory managers to identify essential medicines at risk of experiencing a stock-out within a defined future prediction horizon (N=14 days).

## 2. Proposed ML Problem Formulation
- **Paradigm**: Supervised Binary Classification.
- **Prediction Target**: is_stockout_next_14d (1 if closing stock <= 0 or critical threshold reached at any day in [T+1, T+14], 0 otherwise).
- **Prediction Horizon**: 14 Days (2 weeks), aligned with standard hospital procurement lead times.

## 3. Preprocessing & Temporal Structure
- Transform raw wide-format daily dispensing into long-format longitudinal panel data (Medicine x Date).
- Integrate inventory balance ledger using standard periodic review policy.
- Chronological Train/Test Partition: Earliest 80% dates for Training, latest 20% dates for Testing.

## 4. Feature Engineering
Strictly pre-prediction date (t <= T) features:
- **Inventory metrics**: current_stock, stock_change_7d, stock_min_14d, stock_max_14d, stock_std_14d
- **Consumption metrics**: consumption_7d_mean, consumption_14d_mean, consumption_30d_mean, consumption_7d_std, consumption_velocity
- **Replenishment metrics**: days_since_last_replenishment, last_replenishment_qty
- **Operational indicators**: estimated_days_of_stock, consumption_to_stock_ratio

## 5. Baseline ML Model
- **Algorithm**: Random Forest Classifier (n_estimators=100, max_depth=6, class_weight='balanced', random_state=42).
- **Rationale**: Robust to tabular nonlinearities, resilient to moderate collinearity, outputs well-calibrated class probabilities, supports TreeExplainer SHAP.

## 6. Evaluation & Explainability
- **Metrics**: Confusion Matrix, Precision, Recall (Stock-out class), F1-Score, ROC-AUC, PR-AUC.
- **Explainability**: TreeExplainer SHAP for global feature importance and local instance-level decision factor breakdowns.
- **Risk Categorization**: Prototype thresholds (Low < 0.35, Medium 0.35-0.65, High >= 0.65).
