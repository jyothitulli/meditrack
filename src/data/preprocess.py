"""
MediTrack Preprocessing Module (S2 Dataset)
Ingests, validates, cleans, and structures the S2 DHIS2 multi-facility longitudinal dataset.
"""
import sys
import json
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    RAW_S2_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_SEED
)


def load_raw_s2_data(filepath: Path = RAW_S2_PATH) -> pd.DataFrame:
    """
    Loads raw S2 DHIS2 dataset and validates mandatory schema fields.
    Automatically unzips S2.csv.zip if CSV is missing.
    """
    if not filepath.exists():
        zip_path = filepath.parent / "S2.csv.zip"
        if zip_path.exists():
            print(f"[PREPROCESS] Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(filepath.parent)
        else:
            raise FileNotFoundError(f"Neither {filepath} nor {zip_path} was found.")
            
    df = pd.read_csv(filepath)
    required_cols = [
        "hf_pk", "name1", "date", "stockout", "received", "consumption",
        "closeBalance", "openBalance", "facility_type", "district", "productID"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"S2 dataset missing required columns: {missing}")
        
    # Parse date
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    return df


def clean_and_encode_s2_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans anomalies, sorts chronologically, and encodes categorical attributes.
    """
    df_clean = df.copy()
    
    # 1. Ensure non-negative numerical quantities
    num_cols = ["openBalance", "closeBalance", "received", "consumption"]
    for col in num_cols:
        df_clean[col] = df_clean[col].clip(lower=0)
        
    # 2. Binary stockout indicator validation
    df_clean["stockout"] = df_clean["stockout"].clip(lower=0, upper=1).astype(int)
    
    # 3. Categorical encoding with deterministic sorting
    facility_types = sorted(df_clean["facility_type"].dropna().unique().tolist())
    districts = sorted(df_clean["district"].dropna().unique().tolist())
    
    ft_map = {ft: idx for idx, ft in enumerate(facility_types)}
    dist_map = {d: idx for idx, d in enumerate(districts)}
    
    df_clean["facility_type_encoded"] = df_clean["facility_type"].map(ft_map).fillna(0).astype(int)
    df_clean["district_encoded"] = df_clean["district"].map(dist_map).fillna(0).astype(int)
    
    # 4. Sort chronologically by facility, product, date
    df_clean = df_clean.sort_values(["hf_pk", "productID", "date"]).reset_index(drop=True)
    
    return df_clean


def run_preprocessing(
    raw_path: Path = RAW_S2_PATH,
    output_path: Path = PROCESSED_DATA_PATH
) -> pd.DataFrame:
    """
    Executes full preprocessing and persists processed inventory.
    """
    print(f"[PREPROCESS] Loading raw S2 dataset from: {raw_path}")
    raw_df = load_raw_s2_data(raw_path)
    print(f"[PREPROCESS] Ingested {len(raw_df):,} raw records across {raw_df['hf_pk'].nunique():,} facilities.")
    
    print("[PREPROCESS] Cleaning data and encoding categorical fields...")
    processed_df = clean_and_encode_s2_data(raw_df)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"[PREPROCESS] Saved processed inventory dataset to: {output_path}")
    
    # Diagnostic summary
    n_stockouts = (processed_df["stockout"] == 1).sum()
    pct_stockout = 100.0 * n_stockouts / len(processed_df)
    print(f"[PREPROCESS] Total records: {len(processed_df):,} | Stockout occurrences: {n_stockouts:,} ({pct_stockout:.2f}%)")
    
    return processed_df


if __name__ == "__main__":
    run_preprocessing()
