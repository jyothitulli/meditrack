"""
MediTrack Configuration Module (S2 Dataset)
Contains project paths, feature lists, target definitions, risk thresholds, and metadata.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_S2_PATH = DATA_DIR / "s2_raw" / "S2_Dhis2Data.csv"
FACILITY_METADATA_PATH = DATA_DIR / "s2_raw" / "S1.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_inventory.csv"
FEATURES_DATA_PATH = DATA_DIR / "processed" / "features_dataset.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODELS_DIR / "baseline_rf_model.joblib"
METADATA_SAVE_PATH = MODELS_DIR / "model_metadata.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPLAINABILITY_DIR = OUTPUTS_DIR / "explainability"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
EVALUATION_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"

# Reproducibility
RANDOM_SEED = 42

# Prediction Horizon & Chronological Cutoff
PREDICTION_HORIZON_LABEL = "Next Monthly Reporting Cycle (1 Month)"
TRAIN_SPLIT_RATIO = 0.80

# Feature Names (Strictly Historical t <= T)
FEATURE_COLUMNS = [
    "current_stock",
    "open_balance",
    "current_consumption",
    "current_received",
    "current_stockout",
    "consumption_roll3_mean",
    "consumption_roll6_mean",
    "consumption_roll3_std",
    "stock_change_1m",
    "stock_to_consumption_ratio",
    "estimated_days_of_stock",
    "historical_stockout_rate",
    "facility_type_encoded",
    "district_encoded"
]

TARGET_COLUMN = "is_stockout_next_month"

# Prototype Risk Thresholds (Aligned to S2 empirical stockout base rate ~13.55%)
# Low: < 0.20, Medium: 0.20 - 0.50, High: >= 0.50
RISK_THRESHOLD_LOW = 0.20
RISK_THRESHOLD_HIGH = 0.50

def get_risk_level(probability: float) -> str:
    """
    Categorizes predicted stock-out probability into Prototype Risk Levels.
    
    Args:
        probability: Float in [0.0, 1.0] representing predicted stockout probability.
        
    Returns:
        Risk label: 'LOW', 'MEDIUM', or 'HIGH'.
    """
    if probability < RISK_THRESHOLD_LOW:
        return "LOW"
    elif probability < RISK_THRESHOLD_HIGH:
        return "MEDIUM"
    else:
        return "HIGH"
