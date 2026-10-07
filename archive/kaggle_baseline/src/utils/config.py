"""
MediTrack Configuration Module
Contains project constants, paths, thresholds, and metadata.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "salesdaily.csv"
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

# Medicine Metadata & ATC Classifications
MEDICINE_METADATA = {
    "M01AB": {
        "name": "Diclofenac / Acetic Acid Derivatives",
        "category": "Anti-inflammatory & Antirheumatic (NSAID)",
        "essential_use": "Pain management, musculoskeletal inflammation"
    },
    "M01AE": {
        "name": "Ibuprofen / Propionic Acid Derivatives",
        "category": "Anti-inflammatory & Antirheumatic (NSAID)",
        "essential_use": "Fever, mild-to-moderate pain, arthritis"
    },
    "N02BA": {
        "name": "Acetylsalicylic Acid (Aspirin)",
        "category": "Analgesic & Antipyretic",
        "essential_use": "Cardiovascular prophylaxis, acute pain"
    },
    "N02BE": {
        "name": "Paracetamol / Acetaminophen",
        "category": "Analgesic & Antipyretic (Anilide)",
        "essential_use": "First-line fever & pain management"
    },
    "N05B": {
        "name": "Diazepam / Anxiolytics",
        "category": "Psycholeptic (Anxiolytic)",
        "essential_use": "Anxiety disorders, status epilepticus, sedation"
    },
    "N05C": {
        "name": "Hypnotics & Sedatives",
        "category": "Psycholeptic (Sedative)",
        "essential_use": "Severe sleep disturbance, acute crisis"
    },
    "R03": {
        "name": "Salbutamol / Anti-asthmatics",
        "category": "Respiratory System (Bronchodilator)",
        "essential_use": "Acute asthma attacks, COPD bronchospasm"
    },
    "R06": {
        "name": "Cetirizine / Antihistamines",
        "category": "Systemic Antihistamine",
        "essential_use": "Anaphylaxis support, acute allergic reactions"
    }
}

# Machine Learning Target & Parameters
PREDICTION_HORIZON_DAYS = 14
TRAIN_SPLIT_RATIO = 0.80

# Feature Names
FEATURE_COLUMNS = [
    "current_stock",
    "stock_change_7d",
    "stock_min_14d",
    "stock_max_14d",
    "stock_std_14d",
    "consumption_7d_mean",
    "consumption_14d_mean",
    "consumption_30d_mean",
    "consumption_7d_std",
    "consumption_velocity",
    "days_since_last_replenishment",
    "last_replenishment_qty",
    "estimated_days_of_stock",
    "consumption_to_stock_ratio"
]

TARGET_COLUMN = "is_stockout_next_14d"

# Prototype Risk Thresholds (Heuristic Decision Boundaries for decision support)
# NOTE: These are prototype boundaries and NOT clinically calibrated.
RISK_THRESHOLD_LOW = 0.35
RISK_THRESHOLD_HIGH = 0.65

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
