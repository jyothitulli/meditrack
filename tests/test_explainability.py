"""
Unit tests for S2 SHAP explainability module.
"""
import pytest
import pandas as pd
from src.ml.explain import MediTrackExplainer
from src.utils.config import FEATURES_DATA_PATH


def test_explain_instance_s2():
    df = pd.read_csv(FEATURES_DATA_PATH)
    explainer = MediTrackExplainer()
    
    sample_row = df.iloc[-1]
    res = explainer.explain_instance(sample_row)
    
    assert "predicted_probability" in res
    assert "top_contributions" in res
    assert "narratives" in res
    assert len(res["narratives"]) > 0
    assert len(res["top_contributions"]) == len(explainer.features)
