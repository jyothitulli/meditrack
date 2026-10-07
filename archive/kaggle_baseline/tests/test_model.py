"""
Unit tests for ML model training, chronological split, and inference.
"""
import pytest
import pandas as pd
import numpy as np
from src.ml.train import perform_chronological_split, train_baseline_model
from src.ml.predict import MediTrackPredictor
from src.utils.config import FEATURES_DATA_PATH, get_risk_level, FEATURE_COLUMNS


def test_chronological_split_non_overlapping():
    df = pd.read_csv(FEATURES_DATA_PATH)
    train_df, test_df, max_train_dt, min_test_dt = perform_chronological_split(df)
    
    assert not train_df.empty
    assert not test_df.empty
    # Strict temporal ordering
    assert max_train_dt < min_test_dt
    assert train_df["date"].max() < test_df["date"].min()


def test_risk_categorization_function():
    assert get_risk_level(0.10) == "LOW"
    assert get_risk_level(0.349) == "LOW"
    assert get_risk_level(0.35) == "MEDIUM"
    assert get_risk_level(0.50) == "MEDIUM"
    assert get_risk_level(0.65) == "HIGH"
    assert get_risk_level(0.95) == "HIGH"


def test_meditrack_predictor():
    predictor = MediTrackPredictor()
    df = pd.read_csv(FEATURES_DATA_PATH)
    sample_row = df.iloc[-1].to_dict()
    
    res = predictor.predict_single_medicine(sample_row)
    assert "stockout_probability" in res
    assert "risk_level" in res
    assert 0.0 <= res["stockout_probability"] <= 1.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
