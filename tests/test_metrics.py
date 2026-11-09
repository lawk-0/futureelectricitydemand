"""
Tests for Evaluation Metrics and Peak Demand Analytics modules.
Member 5: Performance Evaluation and Energy Analytics Lead
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation.metrics import (
    mae,
    rmse,
    smape,
    r_squared,
    pinball_loss,
    picp,
    mpiw,
    evaluate_forecast,
)
from src.analytics.peak_detection import PeakDemandAnalyzer


def test_standard_metrics_exact_values():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    assert np.isclose(mae(y_true, y_pred), (10 + 10 + 0) / 3.0)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt((100 + 100 + 0) / 3.0))
    assert 0.0 <= smape(y_true, y_pred) <= 100.0
    assert r_squared(y_true, y_pred) > 0.90


def test_uncertainty_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    lower = np.array([90.0, 180.0, 290.0, 410.0])  # 3 of 4 covered (last misses: 400 < 410)
    upper = np.array([110.0, 220.0, 310.0, 450.0])

    cov = picp(y_true, lower, upper)
    assert cov == 75.0  # 3 out of 4 = 75%

    width = mpiw(lower, upper)
    expected_width = (20.0 + 40.0 + 20.0 + 40.0) / 4.0
    assert np.isclose(width, expected_width)

    # Pinball loss must be >= 0
    p10 = pinball_loss(y_true, lower, 0.10)
    assert p10 >= 0.0


def test_evaluate_forecast_integration():
    y_true = pd.Series([100.0, 150.0, 200.0, np.nan])
    y_pred = pd.Series([102.0, 148.0, 195.0, 250.0])
    lower = pd.Series([90.0, 130.0, 180.0, 230.0])
    upper = pd.Series([115.0, 165.0, 220.0, 270.0])

    metrics = evaluate_forecast(y_true, y_pred, lower, upper)
    assert metrics["n_samples"] == 3
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "sMAPE" in metrics
    assert "PICP" in metrics
    assert "MPIW" in metrics


def test_peak_demand_analyzer():
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    # Base load 20,000 with a peak of 35,000 at index 50
    loads = np.full(100, 20000.0)
    loads[50] = 35000.0

    df_hist = pd.DataFrame({"timestamp": dates, "consumption": loads})
    analyzer = PeakDemandAnalyzer()

    hist_stats = analyzer.analyze_historical_peaks(df_hist)
    assert hist_stats["all_time_peak_mw"] == 35000.0

    # Test detection on forecast
    future_dates = pd.date_range("2023-01-06", periods=24, freq="1h")
    preds = np.full(24, 22000.0)
    preds[15] = 32000.0  # Exceeds 90th percentile threshold

    df_forecast = pd.DataFrame({
        "timestamp": future_dates,
        "predicted": preds,
        "lower_bound": preds - 1000,
        "upper_bound": preds + 1000,
    })

    peak_alert = analyzer.detect_forecast_peaks(
        df_forecast, threshold_mw=25000.0, critical_threshold_mw=30000.0
    )

    assert peak_alert["status"] == "CRITICAL PEAK RISK"
    assert peak_alert["expected_peak_mw"] == 32000.0
    assert peak_alert["high_demand_hours_count"] == 1
    assert "disclaimer" in peak_alert

