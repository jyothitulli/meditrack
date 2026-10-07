"""
MediTrack Baseline Model Training & Evaluation Module (S2 Dataset)
Enforces strict chronological train/test split, trains Random Forest baseline,
computes evaluation metrics, and persists artifacts.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    FEATURES_DATA_PATH,
    MODEL_SAVE_PATH,
    METADATA_SAVE_PATH,
    METRICS_PATH,
    EVALUATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_SPLIT_RATIO,
    RANDOM_SEED,
    PREDICTION_HORIZON_LABEL
)


def perform_chronological_split(df: pd.DataFrame, split_ratio: float = TRAIN_SPLIT_RATIO):
    """
    Splits multi-facility panel dataset chronologically into earliest train set and latest test set.
    """
    df_clean = df.dropna(subset=[TARGET_COLUMN]).copy()
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    
    unique_dates = np.sort(df_clean["date"].unique())
    split_idx = int(len(unique_dates) * split_ratio)
    split_date = unique_dates[split_idx]
    
    train_mask = df_clean["date"] < split_date
    test_mask = df_clean["date"] >= split_date
    
    train_df = df_clean[train_mask].copy()
    test_df = df_clean[test_mask].copy()
    
    # Verify temporal non-overlap
    max_train_date = train_df["date"].max()
    min_test_date = test_df["date"].min()
    assert max_train_date < min_test_date, (
        f"Temporal leakage detected: max train ({max_train_date}) >= min test ({min_test_date})"
    )
    
    return train_df, test_df, max_train_date, min_test_date


def train_baseline_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list = FEATURE_COLUMNS,
    target: str = TARGET_COLUMN,
    seed: int = RANDOM_SEED
):
    """
    Trains Random Forest Classifier baseline and evaluates on chronological test set.
    """
    X_train = train_df[features]
    y_train = train_df[target].astype(int)
    
    X_test = test_df[features]
    y_test = test_df[target].astype(int)
    
    print(f"[TRAIN] Training set: {len(X_train):,} samples (Positive class: {y_train.sum():,} [{100.0*y_train.mean():.2f}%])")
    print(f"[TRAIN] Test set: {len(X_test):,} samples (Positive class: {y_test.sum():,} [{100.0*y_test.mean():.2f}%])")
    
    # Initialize Random Forest baseline with balanced class weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )
    
    print("[TRAIN] Fitting Random Forest model...")
    model.fit(X_train, y_train)
    
    # Predict on unseen chronological holdout test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall_stockout_class": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        },
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "prediction_horizon": PREDICTION_HORIZON_LABEL,
        "model_type": "RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced')"
    }
    
    # Feature importances
    feature_importances = {
        feat: float(imp) for feat, imp in zip(features, model.feature_importances_)
    }
    # Sort descending
    feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
    metrics["feature_importances"] = feature_importances
    
    return model, metrics, cm, X_test, y_test, y_pred, y_prob


def save_artifacts(
    model,
    metrics: dict,
    cm: np.ndarray,
    train_date_range: tuple,
    test_date_range: tuple
):
    """
    Saves trained model, metadata, metrics JSON, text report, and confusion matrix plot.
    """
    # 1. Save model
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"[SAVE] Model persisted to: {MODEL_SAVE_PATH}")
    
    # 2. Save metadata
    metadata = {
        "model_file": str(MODEL_SAVE_PATH.name),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "prediction_horizon": PREDICTION_HORIZON_LABEL,
        "train_date_start": str(train_date_range[0].date()),
        "train_date_end": str(train_date_range[1].date()),
        "test_date_start": str(test_date_range[0].date()),
        "test_date_end": str(test_date_range[1].date()),
        "random_seed": RANDOM_SEED
    }
    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SAVE] Metadata persisted to: {METADATA_SAVE_PATH}")
    
    # 3. Save metrics JSON
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SAVE] Metrics persisted to: {METRICS_PATH}")
    
    # 4. Save evaluation report text
    report_text = f"""================================================================================
MEDITRACK - S2 DATASET BASELINE EVALUATION REPORT
================================================================================
Prediction Target: {TARGET_COLUMN} ({PREDICTION_HORIZON_LABEL})
Model Algorithm:   {metrics['model_type']}
Evaluation Set:    Chronological Hold-Out Test Set (Zero Temporal Leakage)
Date Range:        {metadata['test_date_start']} to {metadata['test_date_end']}
Total Test Count:  {metrics['test_samples']:,} instances across 1,091 health facilities

--------------------------------------------------------------------------------
CLASSIFICATION PERFORMANCE METRICS:
--------------------------------------------------------------------------------
Accuracy:                 {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
Precision (Stockout):     {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
Recall (Stockout Class):  {metrics['recall_stockout_class']:.4f} ({metrics['recall_stockout_class']*100:.2f}%)
F1-Score (Stockout):      {metrics['f1_score']:.4f}
ROC-AUC:                  {metrics['roc_auc']:.4f}
PR-AUC:                   {metrics['pr_auc']:.4f}

--------------------------------------------------------------------------------
CONFUSION MATRIX:
--------------------------------------------------------------------------------
True Negatives  (Adequate Stock correctly predicted): {metrics['confusion_matrix']['true_negatives']:,}
False Positives (False Alarm early warnings):         {metrics['confusion_matrix']['false_positives']:,}
False Negatives (Missed Stock-out emergencies):       {metrics['confusion_matrix']['false_negatives']:,}
True Positives  (Stock-outs correctly alerted):       {metrics['confusion_matrix']['true_positives']:,}

--------------------------------------------------------------------------------
GLOBAL FEATURE IMPORTANCES (GINI IMPURITY DECREASE):
--------------------------------------------------------------------------------
"""
    for feat, imp in metrics["feature_importances"].items():
        report_text += f"  - {feat:<30} : {imp:.4f} ({imp*100:.2f}%)\n"
        
    report_text += """================================================================================
ACADEMIC INTERPRETATION:
- Recall on empirical stockout class is prioritized for healthcare decision-support.
- Evaluated on real-world DHIS2 reporting data from 1,091 public health facilities.
================================================================================
"""
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[SAVE] Evaluation report written to: {EVALUATION_REPORT_PATH}")
    
    # 5. Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_arr = np.array([
        [metrics["confusion_matrix"]["true_negatives"], metrics["confusion_matrix"]["false_positives"]],
        [metrics["confusion_matrix"]["false_negatives"], metrics["confusion_matrix"]["true_positives"]]
    ])
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Stockout (0)", "Stockout (1)"],
        yticklabels=["No Stockout (0)", "Stockout (1)"]
    )
    plt.title(f"MediTrack S2 Confusion Matrix (Holdout Test Set)")
    plt.ylabel("Actual Ground Truth Class")
    plt.xlabel("Model Predicted Class")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()
    print(f"[SAVE] Confusion matrix plot saved to: {CONFUSION_MATRIX_PATH}")


def run_training():
    """
    End-to-end execution of train/test split, model training, evaluation, and persistence.
    """
    print(f"[TRAIN] Loading features from: {FEATURES_DATA_PATH}")
    if not FEATURES_DATA_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found at: {FEATURES_DATA_PATH}")
        
    df = pd.read_csv(FEATURES_DATA_PATH)
    
    print("[TRAIN] Performing chronological train/test split...")
    train_df, test_df, max_train_dt, min_test_dt = perform_chronological_split(df)
    
    train_date_range = (train_df["date"].min(), max_train_dt)
    test_date_range = (min_test_dt, test_df["date"].max())
    
    print(f"[TRAIN] Train timeline: {train_date_range[0].date()} to {train_date_range[1].date()}")
    print(f"[TRAIN] Test timeline:  {test_date_range[0].date()} to {test_date_range[1].date()}")
    
    model, metrics, cm, X_test, y_test, y_pred, y_prob = train_baseline_model(train_df, test_df)
    
    save_artifacts(model, metrics, cm, train_date_range, test_date_range)
    
    print("\n[TRAIN] Training and Evaluation Complete!")
    print(f"        Accuracy: {metrics['accuracy']:.4f} | Recall: {metrics['recall_stockout_class']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    return model, metrics


if __name__ == "__main__":
    run_training()
