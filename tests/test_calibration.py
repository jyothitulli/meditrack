"""
Unit tests for probability calibration and reliability metrics (50% Milestone).
"""
import pytest
import numpy as np
from src.ml.calibrate import compute_expected_calibration_error


def test_compute_expected_calibration_error_perfect():
    # Perfectly calibrated case
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert np.isclose(ece, 0.0, atol=1e-5)


def test_compute_expected_calibration_error_imperfect():
    # Overconfident wrong predictions
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.9, 0.9, 0.9, 0.9])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece > 0.5
