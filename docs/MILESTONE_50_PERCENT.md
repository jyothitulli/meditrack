# MediTrack: 50% Milestone Review & Academic Defense Document

## 1. Executive Summary & Milestone Transition
MediTrack has successfully progressed from the **30% Milestone** (Data Ingestion, Baseline Random Forest, Zero-Leakage Split, Basic Prototype Dashboard) to the **50% Milestone** (Benchmarked, Calibrated, and Prescriptive Healthcare Decision-Support System).

The 50% milestone introduces five major clinical and algorithmic advances:
1. **Multi-Model Tournament**: Quantitative benchmarking of Balanced Random Forest, Modern Tree Boosting (`HistGradientBoostingClassifier`), and Regularized Linear Models (`LogisticRegression`).
2. **Walk-Forward Chronological Cross-Validation**: Expanding-window evaluation over three distinct multi-year periods (2021 H2, 2022 H2, 2023 Holdout) establishing temporal stability without look-ahead bias.
3. **Probability Calibration**: Implementation of Platt Scaling (Sigmoid) and Isotonic Regression, reducing Expected Calibration Error (ECE) and Brier Score for clinically trustworthy risk estimates.
4. **Advanced Feature Engineering (v2)**: Integration of monsoon/malaria seasonality ($\sin/\cos$ cyclical month encodings, rainy season indicators), consumption volatility surge ratios, and stockout acceleration metrics.
5. **Prescriptive Action Engine**: Moving beyond risk prediction to automated decision-making through Recommended Order Quantities (ROQ) with Safety Stock (95% service level) and Intra-District Lateral Stock Transfer Rebalancing between nearby health facilities.

---

## 2. Multi-Model Tournament & Comparative Holdout Evaluation
All models were evaluated on the **92,406 unseen chronological test instances across 2023** (1,091 health facilities, 36 essential medicines):

| Algorithm | Recall (Stockout Class) | Precision | F1-Score | ROC-AUC | PR-AUC | Brier Score | Inference Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Balanced)** | **82.23%** (12,422 / 15,106) | 29.80% | 0.4365 | 0.8028 | 0.4543 | 0.2034 | 0.0039 ms/sample |
| **HistGradientBoosting** | **81.64%** (12,333 / 15,106) | 29.85% | 0.4371 | **0.8069** | **0.4570** | **0.2033** | **0.0006 ms/sample** |
| **Logistic Regression (L2)** | 73.62% (11,121 / 15,106) | 23.47% | 0.3559 | 0.7398 | 0.3356 | 0.2314 | 0.0001 ms/sample |

### Key Academic Findings:
- **Non-Linear Dominance**: Non-linear tree ensembles achieve over **+8.6% higher recall** and **+0.067 higher ROC-AUC** than the linear baseline, proving that non-linear feature interactions (such as the interaction between rolling demand surges and remaining days of stock) are critical.
- **HistGradientBoosting Efficiency**: Gradient Boosting achieves the highest ROC-AUC (0.8069) and PR-AUC (0.4570) while offering **6.5x faster inference latency** (0.0006 ms vs 0.0039 ms per sample).
- **PR-AUC Lift**: Both tree algorithms provide a **2.8x lift** over the random baseline stockout prevalence (16.35%).

Artifacts Generated:
- `outputs/roc_comparison.png` (Comparative ROC Curves)
- `outputs/pr_comparison.png` (Comparative PR Curves)
- `outputs/model_benchmark.json` (Machine-readable benchmark matrix)
- `outputs/model_benchmark_report.txt` (Comprehensive benchmark report)

---

## 3. Walk-Forward Chronological Cross-Validation
To guarantee that model performance does not depend on a single temporal partition, a 3-fold expanding-window cross-validation was conducted:

| Algorithm | Mean Recall ± Std | Mean F1 ± Std | Mean ROC-AUC ± Std |
|---|:---:|:---:|:---:|
| **Random Forest (Balanced)** | **80.95% ± 1.13%** | 0.4418 ± 0.0041 | **0.8052 ± 0.0028** |
| **HistGradientBoosting** | 80.12% ± 1.28% | **0.4431 ± 0.0049** | 0.8041 ± 0.0031 |
| **Logistic Regression (L2)** | 72.88% ± 0.94% | 0.3592 ± 0.0038 | 0.7411 ± 0.0016 |

- **Zero Temporal Leakage**: Folds preserve the arrow of time ($\max(T_{\text{train}}) < \min(T_{\text{val}})$).
- **Low Standard Deviation ($\sigma_{\text{AUC}} < 0.003$)**: Confirms consistent model behavior across pre- and post-COVID reporting years.

---

## 4. Probability Calibration & Reliability Analysis
Raw ensemble classifiers often output uncalibrated voting margins rather than true Bayesian posterior probabilities. In public health decision-support, miscalibrated probabilities distort risk prioritization and clinical resource allocation.

### Calibration Results on Chronological Validation Partition:
- **Calibration Split**: Train (2019-10 to 2021-12, 249,555 samples) $\rightarrow$ Val (2022-01 to 2022-12, 88,172 samples) $\rightarrow$ Test (2023-01 to 2023-10, 92,406 samples).
- **Methods Evaluated**:
  1. *Uncalibrated Baseline*
  2. *Platt Scaling (Sigmoid)*
  3. *Isotonic Regression (Non-parametric piecewise constant)*
- **Optimal Model**: **Isotonic Regression** achieved the lowest Brier Score loss and lowest Expected Calibration Error (ECE).
- Persisted to: `models/calibrated_model.joblib`
- Reliability Diagram: `outputs/calibration_curve.png`

---

## 5. Advanced Feature Engineering v2
The feature matrix was extended from 14 to **19 strictly backward-looking features ($t \le T$)**:
- **Epidemiological Seasonality**:
  - `is_rainy_season`: Binary indicator for May–October monsoon months (correlating with peak malaria and respiratory illness).
  - `sin_month`, `cos_month`: Cyclical harmonic calendar features preserving annual periodicity.
- **Consumption Volatility & Surges**:
  - `consumption_surge_ratio`: $\text{Consumption}_T / (\bar{C}_{6m} + \epsilon)$ (detects sudden spikes above the 6-month historical baseline).
- **Stockout Acceleration**:
  - `stockout_acceleration`: Frequency of stockouts in the last 3 months minus the expanding historical average ($\text{Stockout}_{3m} - \text{Stockout}_{\text{hist}}$).

Zero future leakage was verified by automated unit test `tests/test_feature_engineering.py::test_zero_future_leakage_s2`.

---

## 6. Prescriptive Healthcare Replenishment & Lateral Rebalancing
MediTrack transitions from *descriptive/predictive* to *prescriptive* decision support:

### 1. Recommended Order Quantity (ROQ) with Safety Stock:
$$\text{Daily Consumption } (D) = \frac{\bar{C}_{\text{roll3}}}{30}$$
$$\text{Safety Stock } (SS) = \max\left(7 \times D, \; Z_{0.95} \times \frac{\sigma_C}{\sqrt{30}} \times \sqrt{L}\right)$$
$$\text{Order-Up-To Level } (OUL) = (L + R) \times D + SS$$
$$\text{Recommended Order Qty } (ROQ) = \max(0, \; OUL - S_{\text{current}})$$
Where $L = 14 \text{ days}$ (lead time), $R = 30 \text{ days}$ (review period), and $Z = 1.65$ (95% service level).

### 2. Intra-District Lateral Stock Rebalancing:
When a healthcare facility faces imminent stockout risk ($\text{Risk} \ge 0.50$, Days of Stock $< 10$), the engine queries health facilities within the same geographical district:
- Filters for candidate clinics with surplus inventory ($\ge 45$ days of stock runway).
- Enforces a clinical safety buffer: Donor clinic must retain $\ge 30$ days of stock for its own population.
- Suggests transfer volume and displays post-transfer runway for both clinics.

---

## 7. Interactive Surveillance Dashboard Enhancements
The Streamlit application (`dashboard/app.py`) was restructured into five purpose-built views:
1. **🏥 Hospital Inventory**: Facility-specific surveillance table with live risk dots, days of stock, prescriptive order quantities, and SHAP diagnosis cards.
2. **🔄 Prescriptive Rebalancing**: Interactive district emergency transfer finder pairing deficit clinics with nearby surplus donor facilities.
3. **🏆 Model Tournament & Benchmarking**: Academic evaluation console displaying ROC and PR curves, algorithm comparison tables, and walk-forward cross-validation results.
4. **🎯 Probability Calibration**: Reliability diagrams and clinical explanation of calibrated probabilities.
5. **📚 Methodology & About**: Detailed technical documentation and provenance references.

---

## 8. Verification & Test Suite
The automated test suite in `tests/` now comprises **15 comprehensive unit tests (100% passing)**:
- `test_preprocessing.py`: Schema validation, categorical encodings, zero missing values.
- `test_feature_engineering.py`: Temporal feature computation, zero future data leakage.
- `test_model.py`: Chronological partition non-overlap, multi-model inference, risk thresholds.
- `test_explainability.py`: Instance-level SHAP attributions and natural-language narrative generation.
- `test_prescriptive.py`: ROQ calculations, deficit handling, surplus constraints, intra-district transfer logic.
- `test_calibration.py`: Expected Calibration Error (ECE) for perfect and uncalibrated scenarios.
- `test_benchmark.py`: Model suite initialization, quantitative classification metrics.

---

## 9. Comparison: 30% Milestone vs. 50% Milestone

| Dimension | 30% Milestone Status | 50% Milestone Status |
|---|---|---|
| **Models Evaluated** | Single baseline (Random Forest) | 3-Model Tournament (RF, HistGBM, Logistic Regression) |
| **Cross-Validation** | Single train/test chronological split | 3-Fold Walk-Forward Expanding Temporal Cross-Validation |
| **Probability Quality** | Raw uncalibrated ensemble fractions | Calibrated via Platt Scaling & Isotonic Regression (Lowest Brier Score) |
| **Feature Set** | 14 basic inventory & rolling features | 19 features including seasonality ($\sin/\cos$, monsoon), surge ratio, acceleration |
| **Prescriptive Capability** | None (Risk prediction only) | Recommended Order Quantities (ROQ) & District Lateral Stock Transfer Rebalancing |
| **Dashboard Views** | 2 views (Inventory & Baseline Evaluation) | 5 views (Inventory, Prescriptive Rebalancing, Tournament, Calibration, Methodology) |
| **Test Suite Coverage** | 8 unit tests | 15 unit tests covering ML, calibration, and prescriptive logic |
