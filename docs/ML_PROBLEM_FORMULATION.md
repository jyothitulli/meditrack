# MediTrack: Machine Learning Problem Formulation (S2 Dataset)

## 1. Evaluated ML Paradigms

### Option A: Binary Stock-Out Classification (SELECTED)
- **Task**: Predict whether facility $i$ will experience a stock-out of essential medicine $j$ during the next monthly reporting cycle ($T+1$).
- **Output**: Calibrated probability $P(\\text{Stockout}_{T+1} = 1 \\mid X_{\\le T}) \\in [0, 1]$.
- **Justification**: 
  1. Matches the core decision problem of hospital pharmacy supply chains (Early Warning: Alert procurement before inventory runs out).
  2. Supported natively by the empirical binary `stockout` reporting variable in DHIS2.
  3. Direct operational translation into prototype risk categories (Low, Medium, High).
  4. Natively compatible with SHAP TreeExplainer for decision factor attributions.

### Option B: Multi-Class Risk Classification (REJECTED for Baseline)
- Multi-class categorization requires imposing arbitrary intermediate severity classes that are not recorded in DHIS2.

### Option C: Inventory Balance Regression (REJECTED for Baseline)
- Predicting exact unit stock counts introduces high variance across heterogeneous facility sizes and does not directly output a normalized stock-out risk probability.

## 2. Selected Mathematical Formulation
- **Input Feature Vector**: $X_{i, j, T} \\in \\mathbb{R}^D$ (Constructed strictly from historical observations up to month $T$).
- **Target Label**: $Y_{i, j, T} = \\text{stockout}_{i, j, T+1} \\in \\{0, 1\\}$.
- **Baseline Model**: Random Forest Classifier with balanced class weights to account for the 13.55% empirical positive rate.
- **Evaluation Strategy**: Out-of-time chronological evaluation on the final 12 calendar months.
