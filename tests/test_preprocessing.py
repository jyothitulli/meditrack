"""
Unit tests for S2 data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocess import load_raw_s2_data, clean_and_encode_s2_data
from src.utils.config import RAW_S2_PATH


def test_load_raw_s2_data():
    raw_df = load_raw_s2_data(RAW_S2_PATH)
    assert not raw_df.empty
    assert "date" in raw_df.columns
    assert "hf_pk" in raw_df.columns
    assert "productID" in raw_df.columns
    assert "stockout" in raw_df.columns
    assert pd.api.types.is_datetime64_any_dtype(raw_df["date"])


def test_clean_and_encode_s2_data():
    sample_raw = pd.DataFrame({
        "hf_pk": [1, 1, 2],
        "name1": ["Med A", "Med A", "Med B"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01"]),
        "stockout": [0, 1, 0],
        "received": [100, 0, 50],
        "consumption": [50, 40, 20],
        "closeBalance": [50, 10, 30],
        "openBalance": [0, 50, 0],
        "facility_type": ["CHC", "CHC", "MCHP"],
        "district": ["Bo", "Bo", "Bombali"],
        "productID": [1, 1, 2]
    })
    cleaned = clean_and_encode_s2_data(sample_raw)
    assert "facility_type_encoded" in cleaned.columns
    assert "district_encoded" in cleaned.columns
    assert (cleaned["openBalance"] >= 0).all()
    assert (cleaned["closeBalance"] >= 0).all()
    assert (cleaned["consumption"] >= 0).all()
