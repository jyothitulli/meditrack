"""
MediTrack Probability Calibration Module (50% Milestone)
Calibrates model output probabilities using Platt Scaling (Sigmoid) and Isotonic Regression
over a chronological validation partition to eliminate overconfidence and provide trustworthy risk scores.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    FEATURES_DATA_PATH,
    FEATURES_PARQUET_PATH,
    MODEL_SAVE_PATH,
    CALIBRATED_MODEL_PATH,
    CALIBRATION_CURVE_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_SEED
)


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) weighted by bin size.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1]) if i < n_bins - 1 else (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def run_probability_calibration(
    features_df: pd.DataFrame = None,
    base_model = None
):
    """
    Executes chronological validation calibration and plots reliability curves.
    """
    if features_df is None:
        if FEATURES_PARQUET_PATH.exists():
            features_df = pd.read_parquet(FEATURES_PARQUET_PATH)
        else:
            features_df = pd.read_csv(FEATURES_DATA_PATH)
            
    df_clean = features_df.dropna(subset=[TARGET_COLUMN]).copy()
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    
    from sklearn.model_selection import PredefinedSplit
    
    # Chronological partition:
    # Train: 2019-10 to 2021-12 (earliest)
    # Val (Calibration): 2022-01 to 2022-12
    # Test (Holdout): 2023-01 to 2023-10
    train_mask = df_clean["date"] < "2022-01-01"
    val_mask = (df_clean["date"] >= "2022-01-01") & (df_clean["date"] < "2023-01-01")
    test_mask = df_clean["date"] >= "2023-01-01"
    
    X_train = df_clean.loc[train_mask, FEATURE_COLUMNS]
    y_train = df_clean.loc[train_mask, TARGET_COLUMN].astype(int)
    
    X_val = df_clean.loc[val_mask, FEATURE_COLUMNS]
    y_val = df_clean.loc[val_mask, TARGET_COLUMN].astype(int)
    
    X_test = df_clean.loc[test_mask, FEATURE_COLUMNS]
    y_test = df_clean.loc[test_mask, TARGET_COLUMN].astype(int)
    
    # Combined Train + Val for PredefinedSplit
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    
    # test_fold: -1 for training instances, 0 for calibration/validation instances
    test_fold = np.array([-1] * len(X_train) + [0] * len(X_val))
    cv_split = PredefinedSplit(test_fold)
    
    print(f"[CALIBRATE] Calibration Split: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    
    # Baseline model
    from sklearn.ensemble import RandomForestClassifier
    base_model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    base_model.fit(X_train, y_train)
            
    # Uncalibrated baseline on test set
    prob_uncal = base_model.predict_proba(X_test)[:, 1]
    brier_uncal = float(brier_score_loss(y_test, prob_uncal))
    ece_uncal = compute_expected_calibration_error(y_test.values, prob_uncal)
    
    # 1. Sigmoid Calibration (Platt Scaling) via PredefinedSplit
    cal_sigmoid = CalibratedClassifierCV(
        estimator=RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
        method="sigmoid",
        cv=cv_split
    )
    cal_sigmoid.fit(X_train_val, y_train_val)
    prob_sig = cal_sigmoid.predict_proba(X_test)[:, 1]
    brier_sig = float(brier_score_loss(y_test, prob_sig))
    ece_sig = compute_expected_calibration_error(y_test.values, prob_sig)
    
    # 2. Isotonic Calibration via PredefinedSplit
    cal_iso = CalibratedClassifierCV(
        estimator=RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
        method="isotonic",
        cv=cv_split
    )
    cal_iso.fit(X_train_val, y_train_val)
    prob_iso = cal_iso.predict_proba(X_test)[:, 1]
    brier_iso = float(brier_score_loss(y_test, prob_iso))
    ece_iso = compute_expected_calibration_error(y_test.values, prob_iso)
    
    # Pick best model (lowest Brier Score)
    if brier_sig <= brier_iso:
        best_cal_model = cal_sigmoid
        best_method = "Sigmoid (Platt Scaling)"
        best_brier = brier_sig
        best_ece = ece_sig
    else:
        best_cal_model = cal_iso
        best_method = "Isotonic Regression"
        best_brier = brier_iso
        best_ece = ece_iso
        
    CALIBRATED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_cal_model, CALIBRATED_MODEL_PATH)
    print(f"[CALIBRATE] Best calibrated model ({best_method}) saved to: {CALIBRATED_MODEL_PATH}")
    
    # Generate Reliability Curve Plot
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    
    frac_uncal, mean_uncal = calibration_curve(y_test, prob_uncal, n_bins=10)
    frac_sig, mean_sig = calibration_curve(y_test, prob_sig, n_bins=10)
    frac_iso, mean_iso = calibration_curve(y_test, prob_iso, n_bins=10)
    
    plt.plot(mean_uncal, frac_uncal, "s-", color="#ef4444", label=f"Uncalibrated RF (Brier={brier_uncal:.4f}, ECE={ece_uncal:.4f})")
    plt.plot(mean_sig, frac_sig, "o-", color="#3b82f6", label=f"Platt Sigmoid (Brier={brier_sig:.4f}, ECE={ece_sig:.4f})")
    plt.plot(mean_iso, frac_iso, "^-", color="#10b981", label=f"Isotonic (Brier={brier_iso:.4f}, ECE={ece_iso:.4f})")
    
    plt.xlabel("Mean Predicted Stock-out Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Stock-outs", fontsize=11)
    plt.title("MediTrack Reliability Diagram (Probability Calibration)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    CALIBRATION_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CALIBRATION_CURVE_PATH, dpi=200)
    plt.close()
    print(f"[CALIBRATE] Reliability diagram saved to: {CALIBRATION_CURVE_PATH}")
    
    results = {
        "uncalibrated": {"brier_score": brier_uncal, "ece": ece_uncal},
        "sigmoid": {"brier_score": brier_sig, "ece": ece_sig},
        "isotonic": {"brier_score": brier_iso, "ece": ece_iso},
        "best_method": best_method,
        "best_brier": best_brier,
        "best_ece": best_ece
    }
    return results


if __name__ == "__main__":
    run_probability_calibration()
