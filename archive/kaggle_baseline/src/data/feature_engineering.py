"""
MediTrack Feature Engineering Module
Builds leakage-free historical features and future stock-out target labels.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    PROCESSED_DATA_PATH,
    FEATURES_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PREDICTION_HORIZON_DAYS
)


def compute_temporal_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly backward-looking features (t <= T) for a single medicine group.
    
    Args:
        group: DataFrame of a single medicine sorted chronologically by date.
        
    Returns:
        DataFrame with feature columns attached.
    """
    df = group.copy().sort_values("date").reset_index(drop=True)
    
    # 1. Inventory features (strictly backward looking)
    df["current_stock"] = df["stock"]
    df["stock_change_7d"] = df["stock"] - df["stock"].shift(7)
    df["stock_min_14d"] = df["stock"].rolling(window=14, min_periods=1).min()
    df["stock_max_14d"] = df["stock"].rolling(window=14, min_periods=1).max()
    df["stock_std_14d"] = df["stock"].rolling(window=14, min_periods=2).std().fillna(0.0)
    
    # 2. Consumption features (strictly backward looking)
    df["consumption_7d_mean"] = df["consumption"].rolling(window=7, min_periods=1).mean()
    df["consumption_14d_mean"] = df["consumption"].rolling(window=14, min_periods=1).mean()
    df["consumption_30d_mean"] = df["consumption"].rolling(window=30, min_periods=1).mean()
    df["consumption_7d_std"] = df["consumption"].rolling(window=7, min_periods=2).std().fillna(0.0)
    df["consumption_velocity"] = df["consumption_7d_mean"] / (df["consumption_30d_mean"] + 1e-5)
    
    # 3. Replenishment tracking (strictly backward looking)
    days_since_replenishment = []
    last_replenishment_qty = []
    
    last_rep_day = -999
    last_rep_val = 0.0
    
    for t, rep_val in enumerate(df["replenishment"]):
        if rep_val > 0.0:
            last_rep_day = t
            last_rep_val = rep_val
            
        if last_rep_day == -999:
            days_since = 30.0  # Default initial prior
            qty = 0.0
        else:
            days_since = float(t - last_rep_day)
            qty = float(last_rep_val)
            
        days_since_replenishment.append(days_since)
        last_replenishment_qty.append(qty)
        
    df["days_since_last_replenishment"] = days_since_replenishment
    df["last_replenishment_qty"] = last_replenishment_qty
    
    # 4. Operational derived indicators (strictly non-ML deterministic arithmetic)
    df["estimated_days_of_stock"] = df["current_stock"] / (df["consumption_7d_mean"] + 1e-5)
    df["consumption_to_stock_ratio"] = df["consumption_7d_mean"] / (df["current_stock"] + 1e-5)
    
    # 5. Impute initial warm-up shifts
    df["stock_change_7d"] = df["stock_change_7d"].fillna(0.0)
    
    return df


def create_future_target(group: pd.DataFrame, horizon: int = PREDICTION_HORIZON_DAYS) -> pd.DataFrame:
    """
    Computes the binary stock-out target label using strictly future observations [T+1, T+horizon].
    
    Target:
        1 if min(Stock_{T+1 : T+horizon}) <= 0 else 0
        
    Args:
        group: Single medicine DataFrame.
        horizon: Prediction window in days (default: 14).
        
    Returns:
        DataFrame with TARGET_COLUMN added. Records lacking complete future window are marked NaN.
    """
    df = group.copy().sort_values("date").reset_index(drop=True)
    n = len(df)
    
    target_values = []
    for t in range(n):
        if t + horizon < n:
            future_window_stock = df.loc[t + 1 : t + horizon, "stock"]
            is_stockout = 1 if (future_window_stock <= 0.0).any() else 0
            target_values.append(is_stockout)
        else:
            # Future horizon extends beyond dataset boundary
            target_values.append(np.nan)
            
    df[TARGET_COLUMN] = target_values
    return df


def build_feature_dataset(
    processed_path: Path = PROCESSED_DATA_PATH,
    output_path: Path = FEATURES_DATA_PATH,
    horizon: int = PREDICTION_HORIZON_DAYS
) -> pd.DataFrame:
    """
    Executes full feature engineering and target label generation across all medicines.
    """
    print(f"[FEATURES] Loading processed inventory data from: {processed_path}")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed inventory file not found at: {processed_path}")
        
    df = pd.read_csv(processed_path)
    df["date"] = pd.to_datetime(df["date"])
    
    engineered_dfs = []
    for med_id, group in df.groupby("medicine_id"):
        group_feats = compute_temporal_features(group)
        group_target = create_future_target(group_feats, horizon=horizon)
        engineered_dfs.append(group_target)
        
    final_df = pd.concat(engineered_dfs, ignore_index=True)
    final_df = final_df.sort_values(["date", "medicine_id"]).reset_index(drop=True)
    
    # Save full dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"[FEATURES] Saved complete feature matrix to: {output_path}")
    
    # Summary stats
    labeled_df = final_df.dropna(subset=[TARGET_COLUMN])
    pos_count = int((labeled_df[TARGET_COLUMN] == 1).sum())
    neg_count = int((labeled_df[TARGET_COLUMN] == 0).sum())
    pos_rate = 100.0 * pos_count / len(labeled_df)
    
    print(f"[FEATURES] Labeled records: {len(labeled_df)}")
    print(f"[FEATURES] Stock-out positive class: {pos_count} ({pos_rate:.2f}%)")
    print(f"[FEATURES] Stock-out negative class: {neg_count} ({100.0 - pos_rate:.2f}%)")
    
    return final_df


if __name__ == "__main__":
    build_feature_dataset()
