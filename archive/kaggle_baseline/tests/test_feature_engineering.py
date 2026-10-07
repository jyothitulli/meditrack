"""
Unit tests for feature engineering and target creation.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.feature_engineering import compute_temporal_features, create_future_target
from src.utils.config import FEATURE_COLUMNS, TARGET_COLUMN, PREDICTION_HORIZON_DAYS


def test_compute_temporal_features():
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    sample_data = pd.DataFrame({
        "date": dates,
        "medicine_id": "M01AB",
        "stock": np.linspace(100.0, 10.0, 50),
        "consumption": np.full(50, 5.0),
        "replenishment": np.zeros(50)
    })
    sample_data.loc[20, "replenishment"] = 50.0
    
    feats = compute_temporal_features(sample_data)
    
    # Check all feature columns exist
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"Missing feature: {col}"
        
    # Check estimated days of stock
    # For day 0, stock=100, consumption_7d_mean=5.0 -> est days ~ 20
    assert np.isclose(feats.loc[0, "estimated_days_of_stock"], 20.0, atol=0.1)


def test_zero_future_leakage():
    """
    Verifies that feature values computed at date T are strictly independent of future rows (t > T).
    """
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    base_data = pd.DataFrame({
        "date": dates,
        "medicine_id": "M01AB",
        "stock": np.random.uniform(20.0, 100.0, size=60),
        "consumption": np.random.uniform(2.0, 8.0, size=60),
        "replenishment": np.zeros(60)
    })
    
    cutoff_idx = 30
    full_feats = compute_temporal_features(base_data)
    truncated_feats = compute_temporal_features(base_data.iloc[:cutoff_idx + 1])
    
    # Values at cutoff_idx must be identical
    for col in FEATURE_COLUMNS:
        val_full = full_feats.loc[cutoff_idx, col]
        val_trunc = truncated_feats.loc[cutoff_idx, col]
        assert np.isclose(val_full, val_trunc, atol=1e-5), f"Feature {col} leaked future information!"


def test_create_future_target():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    # Stock drops to 0 on day 10
    stock_vals = np.full(30, 50.0)
    stock_vals[10] = 0.0
    
    sample_data = pd.DataFrame({
        "date": dates,
        "medicine_id": "M01AB",
        "stock": stock_vals,
        "consumption": np.full(30, 5.0),
        "replenishment": np.zeros(30)
    })
    
    df_with_target = create_future_target(sample_data, horizon=14)
    assert TARGET_COLUMN in df_with_target.columns
    
    # Days 0 to 9 should have target = 1 because day 10 is within their 14-day future window
    assert df_with_target.loc[0, TARGET_COLUMN] == 1
    assert df_with_target.loc[9, TARGET_COLUMN] == 1
    
    # Tail records (last 14 days) should have NaN target due to boundary
    assert pd.isna(df_with_target.loc[20, TARGET_COLUMN])
