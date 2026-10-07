"""
Unit tests for S2 ML model training, chronological splitting, and inference.
"""
import pytest
import pandas as pd
from src.ml.train import perform_chronological_split
from src.ml.predict import MediTrackPredictor
from src.utils.config import FEATURES_DATA_PATH, get_risk_level


def test_chronological_split_s2():
    df = pd.read_csv(FEATURES_DATA_PATH)
    train_df, test_df, max_train_dt, min_test_dt = perform_chronological_split(df)
    
    assert not train_df.empty
    assert not test_df.empty
    assert max_train_dt < min_test_dt
    assert train_df["date"].max() < test_df["date"].min()


def test_risk_categorization_s2():
    assert get_risk_level(0.05) == "LOW"
    assert get_risk_level(0.199) == "LOW"
    assert get_risk_level(0.20) == "MEDIUM"
    assert get_risk_level(0.499) == "MEDIUM"
    assert get_risk_level(0.50) == "HIGH"
    assert get_risk_level(0.85) == "HIGH"


def test_predictor_single_instance_s2():
    predictor = MediTrackPredictor()
    df = pd.read_csv(FEATURES_DATA_PATH)
    sample_row = df.iloc[-1].to_dict()
    
    res = predictor.predict_single_instance(sample_row)
    assert "stockout_probability" in res
    assert "risk_level" in res
    assert 0.0 <= res["stockout_probability"] <= 1.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
