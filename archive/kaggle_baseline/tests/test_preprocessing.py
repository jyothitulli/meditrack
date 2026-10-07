"""
Unit tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocess import load_raw_data, construct_longitudinal_panel, simulate_inventory_ledger
from src.utils.config import RAW_DATA_PATH


def test_load_raw_data():
    raw_df = load_raw_data(RAW_DATA_PATH)
    assert not raw_df.empty
    assert "datum" in raw_df.columns
    assert pd.api.types.is_datetime64_any_dtype(raw_df["datum"])
    # Verify chronological sorting
    assert raw_df["datum"].is_monotonic_increasing


def test_construct_longitudinal_panel():
    raw_df = load_raw_data(RAW_DATA_PATH)
    panel_df = construct_longitudinal_panel(raw_df)
    
    assert set(panel_df.columns) == {"date", "medicine_id", "consumption"}
    assert len(panel_df) > 0
    # No negative consumption
    assert (panel_df["consumption"] >= 0).all()


def test_simulate_inventory_ledger():
    # Construct small test panel
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    sample_panel = pd.DataFrame({
        "date": dates,
        "medicine_id": "TEST_MED",
        "consumption": np.random.uniform(5.0, 15.0, size=60)
    })
    
    ledger = simulate_inventory_ledger(sample_panel, review_period_days=14, lead_time_days=7)
    assert "stock" in ledger.columns
    assert "replenishment" in ledger.columns
    # Inventory stock must never be negative
    assert (ledger["stock"] >= 0.0).all()
    # Replenishments must be non-negative
    assert (ledger["replenishment"] >= 0.0).all()
