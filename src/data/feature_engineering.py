"""
MediTrack Feature Engineering Module (S2 Dataset)
Constructs leakage-free temporal features and future 1-month stockout target labels.
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
    FEATURES_PARQUET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


def compute_group_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly backward-looking temporal features for a single facility-product series.
    """
    df = group.copy().sort_values("date").reset_index(drop=True)
    
    # 1. Base historical metrics at cutoff month T
    df["current_stock"] = df["closeBalance"].astype(float)
    df["open_balance"] = df["openBalance"].astype(float)
    df["current_consumption"] = df["consumption"].astype(float)
    df["current_received"] = df["received"].astype(float)
    df["current_stockout"] = df["stockout"].astype(int)
    
    # 2. Rolling consumption features (strictly backward looking t <= T)
    df["consumption_roll3_mean"] = df["consumption"].rolling(window=3, min_periods=1).mean()
    df["consumption_roll6_mean"] = df["consumption"].rolling(window=6, min_periods=1).mean()
    df["consumption_roll3_std"] = df["consumption"].rolling(window=3, min_periods=2).std().fillna(0.0)
    
    # 3. Inventory delta and operational coverage
    df["stock_change_1m"] = (df["closeBalance"] - df["closeBalance"].shift(1)).fillna(0.0)
    df["stock_to_consumption_ratio"] = df["current_stock"] / (df["current_consumption"] + 1e-5)
    
    # Operational Days of Stock estimate (Monthly demand converted to daily runout)
    daily_demand = (df["consumption_roll3_mean"] / 30.0) + 1e-5
    df["estimated_days_of_stock"] = df["current_stock"] / daily_demand
    
    # 4. Expanding historical stockout frequency
    df["historical_stockout_rate"] = df["stockout"].expanding(min_periods=1).mean()
    
    # 5. Future Target Label (Next Month T+1)
    df[TARGET_COLUMN] = df["stockout"].shift(-1)
    
    return df


def build_feature_dataset(
    processed_path: Path = PROCESSED_DATA_PATH,
    output_path: Path = FEATURES_DATA_PATH
) -> pd.DataFrame:
    """
    Builds the complete feature matrix across all facility-product time series.
    """
    print(f"[FEATURES] Loading processed data from: {processed_path}")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed inventory file not found at: {processed_path}")
        
    df = pd.read_csv(processed_path)
    df["date"] = pd.to_datetime(df["date"])
    
    print("[FEATURES] Computing temporal features and forward targets across facility-product groups...")
    # Group by facility and product
    engineered_dfs = []
    grouped = df.groupby(["hf_pk", "productID"], as_index=False)
    
    # Vectorized / group iteration
    for _, group in grouped:
        if len(group) >= 1:
            group_feats = compute_group_features(group)
            engineered_dfs.append(group_feats)
            
    final_df = pd.concat(engineered_dfs, ignore_index=True)
    final_df = final_df.sort_values(["date", "hf_pk", "productID"]).reset_index(drop=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"[FEATURES] Saved engineered feature dataset to: {output_path}")
    
    try:
        final_df.to_parquet(FEATURES_PARQUET_PATH, index=False)
        print(f"[FEATURES] Saved compressed parquet feature dataset to: {FEATURES_PARQUET_PATH}")
    except Exception as e:
        print(f"[FEATURES] Warning: Could not save parquet: {e}")
    
    # Summary of labeled records
    labeled_df = final_df.dropna(subset=[TARGET_COLUMN])
    pos_count = int((labeled_df[TARGET_COLUMN] == 1).sum())
    neg_count = int((labeled_df[TARGET_COLUMN] == 0).sum())
    pos_rate = 100.0 * pos_count / len(labeled_df)
    
    print(f"[FEATURES] Total records: {len(final_df):,} | Labeled records: {len(labeled_df):,}")
    print(f"[FEATURES] Target Class 1 (Next-Month Stockout): {pos_count:,} ({pos_rate:.2f}%)")
    print(f"[FEATURES] Target Class 0 (Next-Month In-Stock): {neg_count:,} ({100.0 - pos_rate:.2f}%)")
    
    return final_df


if __name__ == "__main__":
    build_feature_dataset()
