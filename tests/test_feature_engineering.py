"""
Unit tests for S2 feature engineering and zero-leakage guarantee.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.feature_engineering import compute_group_features
from src.utils.config import FEATURE_COLUMNS, TARGET_COLUMN


def test_compute_group_features():
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    sample_group = pd.DataFrame({
        "hf_pk": np.full(10, 1),
        "productID": np.full(10, 101),
        "name1": ["Paracetamol"] * 10,
        "date": dates,
        "stockout": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "received": [100] * 10,
        "consumption": [50] * 10,
        "closeBalance": [100, 50, 0, 100, 50, 100, 0, 100, 50, 100],
        "openBalance": [50] * 10,
        "facility_type_encoded": np.zeros(10, dtype=int),
        "district_encoded": np.zeros(10, dtype=int)
    })
    
    feats = compute_group_features(sample_group)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"Missing feature: {col}"
        
    assert TARGET_COLUMN in feats.columns
    # Target on month 1 should equal stockout on month 2 (which is 1)
    assert feats.loc[1, TARGET_COLUMN] == 1
    # Tail record has no future month -> NaN
    assert pd.isna(feats.loc[9, TARGET_COLUMN])


def test_zero_future_leakage_s2():
    """
    Asserts that features at month T are strictly invariant to future rows (t > T).
    """
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    base_data = pd.DataFrame({
        "hf_pk": np.full(12, 1),
        "productID": np.full(12, 101),
        "name1": ["Paracetamol"] * 12,
        "date": dates,
        "stockout": np.random.choice([0, 1], size=12),
        "received": np.random.uniform(0, 200, size=12),
        "consumption": np.random.uniform(10, 100, size=12),
        "closeBalance": np.random.uniform(0, 300, size=12),
        "openBalance": np.random.uniform(0, 300, size=12),
        "facility_type_encoded": np.zeros(12, dtype=int),
        "district_encoded": np.zeros(12, dtype=int)
    })
    
    cutoff_idx = 6
    full_feats = compute_group_features(base_data)
    trunc_feats = compute_group_features(base_data.iloc[:cutoff_idx + 1])
    
    for col in FEATURE_COLUMNS:
        val_full = full_feats.loc[cutoff_idx, col]
        val_trunc = trunc_feats.loc[cutoff_idx, col]
        assert np.isclose(val_full, val_trunc, atol=1e-5), f"Feature {col} leaked future data!"
