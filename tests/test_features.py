"""
Tests for Feature Engineering module: calendar, cyclical, lag, and rolling statistics.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_timeseries():
    """Generates 200 hours of regular synthetic time series."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=200, freq="1h")
    values = 30000 + 4000 * np.sin(np.linspace(0, 8 * np.pi, 200))
    return pd.DataFrame({"timestamp": dates, "consumption": values})


def test_calendar_and_cyclical_features(sample_timeseries):
    fe = FeatureEngineer()
    df_cal = fe.add_calendar_features(sample_timeseries)

    assert "hour" in df_cal.columns
    assert "dayofweek" in df_cal.columns
    assert "month" in df_cal.columns
    assert "is_weekend" in df_cal.columns
    assert "is_holiday" in df_cal.columns

    df_cyc = fe.add_cyclical_features(df_cal)
    assert "hour_sin" in df_cyc.columns
    assert "hour_cos" in df_cyc.columns
    # Sine and cosine bounds [-1.0, 1.0]
    assert df_cyc["hour_sin"].between(-1.0001, 1.0001).all()
    assert df_cyc["hour_cos"].between(-1.0001, 1.0001).all()


def test_lag_features_alignment(sample_timeseries):
    fe = FeatureEngineer()
    df_lag = fe.add_lag_features(sample_timeseries, min_lag=1)

    # lag_1 at index 1 must equal consumption at index 0
    assert df_lag["lag_1"].iloc[1] == sample_timeseries["consumption"].iloc[0]
    # lag_24 at index 24 must equal consumption at index 0
    assert df_lag["lag_24"].iloc[24] == sample_timeseries["consumption"].iloc[0]


def test_rolling_features_shift_leakage_protection(sample_timeseries):
    fe = FeatureEngineer()
    df_roll = fe.add_rolling_features(sample_timeseries, shift=1)

    # Rolling mean for window 24 at index 24 is calculated from indices 0 to 23 (shifted by 1)
    # The consumption at index 24 itself must NOT be in the mean at index 24
    expected_mean_24 = sample_timeseries["consumption"].iloc[0:24].mean()
    assert np.isclose(df_roll["rolling_mean_24"].iloc[24], expected_mean_24, atol=1e-5)


def test_full_feature_pipeline(sample_timeseries):
    fe = FeatureEngineer()
    feat_df = fe.create_features(sample_timeseries, min_lag=1, drop_na=True)

    feature_cols = fe.get_feature_names(feat_df)
    assert len(feature_cols) >= 10
    assert "timestamp" not in feature_cols
    assert "consumption" not in feature_cols
    assert feat_df.isna().sum().sum() == 0

