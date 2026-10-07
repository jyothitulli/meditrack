"""
MediTrack Data Preprocessing Module
Transforms raw pharmaceutical daily sales data into a longitudinal inventory tracking ledger.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MEDICINE_METADATA,
    RANDOM_SEED
)


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw pharmaceutical dispensing CSV data and validates structure.
    
    Args:
        filepath: Path to raw CSV file.
        
    Returns:
        Validated raw DataFrame.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    if "datum" not in df.columns:
        raise ValueError("Raw data missing mandatory timestamp column 'datum'")
    
    # Filter for actual historical records if Type column is present
    if "Type" in df.columns:
        df = df[df["Type"] == "Actual"].copy()
        
    df["datum"] = pd.to_datetime(df["datum"])
    df = df.sort_values("datum").reset_index(drop=True)
    return df


def construct_longitudinal_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshapes wide category sales into longitudinal panel format (Date x Medicine).
    
    Args:
        raw_df: Wide raw DataFrame with columns datum and medicine categories.
        
    Returns:
        Longitudinal DataFrame with columns [date, medicine_id, consumption].
    """
    medicine_codes = list(MEDICINE_METADATA.keys())
    available_codes = [c for c in medicine_codes if c in raw_df.columns]
    
    if not available_codes:
        raise ValueError("No valid medicine category columns found in raw data.")
    
    long_records = []
    for _, row in raw_df.iterrows():
        dt = row["datum"]
        for med in available_codes:
            consumption = float(row[med])
            # Ensure non-negative consumption
            consumption = max(0.0, consumption)
            long_records.append({
                "date": dt,
                "medicine_id": med,
                "consumption": consumption
            })
            
    panel_df = pd.DataFrame(long_records)
    panel_df = panel_df.sort_values(["medicine_id", "date"]).reset_index(drop=True)
    return panel_df


def simulate_inventory_ledger(
    panel_df: pd.DataFrame,
    review_period_days: int = 14,
    lead_time_days: int = 7,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Simulates physical inventory evolution using standard healthcare (s, S) periodic review policy.
    
    Mathematical Formulation:
        Inventory_t = max(0, Inventory_{t-1} + Replenishment_t - Consumption_t)
        
    Args:
        panel_df: Longitudinal panel DataFrame.
        review_period_days: Review interval R in days.
        lead_time_days: Delivery lead time L in days.
        seed: Random seed for deterministic replenishment jitter.
        
    Returns:
        DataFrame containing complete daily inventory, consumption, and replenishment balances.
    """
    np.random.seed(seed)
    processed_dfs = []
    
    for med_id, group in panel_df.groupby("medicine_id"):
        group = group.sort_values("date").reset_index(drop=True)
        n_days = len(group)
        
        # Calculate medicine baseline consumption parameters
        mean_consumption = group["consumption"].mean()
        
        # Healthcare inventory policy parameters:
        # Target stock level S = 28 days of average consumption
        # Reorder point s = 14 days of average consumption
        # Initial stock = 20 days of average consumption
        target_stock = 28.0 * mean_consumption
        reorder_point = 14.0 * mean_consumption
        
        stock_arr = np.zeros(n_days, dtype=float)
        replenishment_arr = np.zeros(n_days, dtype=float)
        pending_orders = {}  # {delivery_day_idx: quantity}
        
        current_inv = 20.0 * mean_consumption
        
        for t in range(n_days):
            # 1. Receive scheduled replenishment deliveries arriving on day t
            arrived_qty = pending_orders.pop(t, 0.0)
            replenishment_arr[t] = arrived_qty
            current_inv += arrived_qty
            
            # 2. Daily Dispensing / Consumption
            daily_cons = group.loc[t, "consumption"]
            current_inv = max(0.0, current_inv - daily_cons)
            stock_arr[t] = current_inv
            
            # 3. Periodic Review Check (Every review_period_days)
            if t % review_period_days == 0:
                on_order_qty = sum(pending_orders.values())
                inventory_position = current_inv + on_order_qty
                
                if inventory_position <= reorder_point:
                    order_qty = max(0.0, target_stock - inventory_position)
                    delivery_day = t + lead_time_days
                    pending_orders[delivery_day] = pending_orders.get(delivery_day, 0.0) + order_qty
                    
        group_res = group.copy()
        group_res["stock"] = stock_arr
        group_res["replenishment"] = replenishment_arr
        processed_dfs.append(group_res)
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df = final_df.sort_values(["medicine_id", "date"]).reset_index(drop=True)
    return final_df


def run_preprocessing(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH
) -> pd.DataFrame:
    """
    Executes the full preprocessing pipeline and saves processed data.
    """
    print(f"[PREPROCESS] Loading raw data from: {raw_path}")
    raw_df = load_raw_data(raw_path)
    print(f"[PREPROCESS] Raw observations loaded: {len(raw_df)} dates.")
    
    print("[PREPROCESS] Constructing longitudinal medicine panel...")
    panel_df = construct_longitudinal_panel(raw_df)
    print(f"[PREPROCESS] Panel dataset shape: {panel_df.shape}")
    
    print("[PREPROCESS] Simulating physical inventory ledger...")
    processed_df = simulate_inventory_ledger(panel_df)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"[PREPROCESS] Successfully saved processed inventory to: {output_path}")
    
    # Summary diagnostics
    total_stockouts = (processed_df["stock"] <= 0).sum()
    stockout_pct = 100.0 * total_stockouts / len(processed_df)
    print(f"[PREPROCESS] Diagnostic: Total zero-stock days: {total_stockouts} ({stockout_pct:.2f}%)")
    
    return processed_df


if __name__ == "__main__":
    run_preprocessing()
