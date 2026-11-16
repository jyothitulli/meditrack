"""
Unit tests for Model Tournament and Benchmarking (50% Milestone).
"""
import pytest
import numpy as np
from src.ml.benchmark import build_candidate_models, evaluate_predictions


def test_build_candidate_models():
    models = build_candidate_models(seed=42)
    assert "Random Forest (Balanced)" in models
    assert "HistGradientBoosting" in models
    assert "Logistic Regression (L2)" in models


def test_evaluate_predictions_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.8, 0.2, 0.7, 0.4, 0.3])
    
    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall_stockout_class" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert "confusion_matrix" in metrics
    
    # 2 true positives (1, 1), 1 false negative (true=1, pred=0) -> recall = 2/3 = 0.6667
    assert np.isclose(metrics["recall_stockout_class"], 2.0 / 3.0, atol=1e-3)
