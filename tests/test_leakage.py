"""
Temporal data leakage prevention and split integrity tests.
Member 2: Temporal Validation Lead
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation.backtesting import chronological_split, verify_temporal_boundaries, RollingOriginBacktester
from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def long_timeseries():
    """Generates 2,000 hours of continuous energy data."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=2000, freq="1h")
    values = 28000 + 3500 * np.sin(np.linspace(0, 20 * np.pi, 2000)) + np.random.normal(0, 100, 2000)
    return pd.DataFrame({"timestamp": dates, "consumption": values})


def test_chronological_split_strict_ordering(long_timeseries):
    """Proves that no training timestamp occurs after validation or test boundaries."""
    train_df, val_df, test_df = chronological_split(
        long_timeseries,
        test_hours=168,
        val_hours=168,
        time_col="timestamp",
    )

    # 1. Train strictly precedes Validation
    assert train_df["timestamp"].max() < val_df["timestamp"].min()

    # 2. Validation strictly precedes Test
    assert val_df["timestamp"].max() < test_df["timestamp"].min()

    # 3. No timestamp overlap between splits
    train_set = set(train_df["timestamp"])
    val_set = set(val_df["timestamp"])
    test_set = set(test_df["timestamp"])
    assert len(train_set.intersection(val_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set.intersection(test_set)) == 0


def test_leakage_verifier_raises_on_reversed_timestamps():
    """Confirms verify_temporal_boundaries detects future leakage and raises error."""
    dates_train = pd.date_range("2023-01-02", periods=50, freq="1h")
    dates_val = pd.date_range("2023-01-01", periods=50, freq="1h")  # In the past relative to train!

    df_train = pd.DataFrame({"timestamp": dates_train, "consumption": 100})
    df_val = pd.DataFrame({"timestamp": dates_val, "consumption": 100})

    with pytest.raises(AssertionError, match="TEMPORAL LEAKAGE DETECTED"):
        verify_temporal_boundaries(df_train, df_val, time_col="timestamp")


def test_rolling_origin_folds_integrity(long_timeseries):
    """Verifies that every cross-validation fold maintains chronological integrity."""
    backtester = RollingOriginBacktester(
        n_splits=3,
        horizon_hours=48,
        min_train_hours=500,
        window_type="expanding",
    )

    for fold_train, fold_test, meta in backtester.split(long_timeseries):
        assert meta["train_end"] < meta["test_start"]
        assert fold_train["timestamp"].max() < fold_test["timestamp"].min()
        assert len(set(fold_train["timestamp"]).intersection(set(fold_test["timestamp"]))) == 0


def test_feature_engineering_no_current_target_in_predictors(long_timeseries):
    """
    Verifies that predictive feature matrix at row i has NO access to consumption at row i
    for lag and rolling variables.
    """
    fe = FeatureEngineer()
    # Add an extreme artificial spike at index 100
    df_spike = long_timeseries.copy()
    spike_index = 100
    df_spike.loc[spike_index, "consumption"] = 999999.0

    feat_df = fe.create_features(df_spike, min_lag=1, drop_na=False)

    # At row 100, the lag_1 must NOT be 999999.0 (it must reflect row 99)
    assert feat_df.loc[spike_index, "lag_1"] != 999999.0
    # At row 100, rolling_mean_24 must NOT contain the spike
    assert feat_df.loc[spike_index, "rolling_mean_24"] < 900000.0
    # At row 101, lag_1 should now reflect the spike from row 100
    assert feat_df.loc[spike_index + 1, "lag_1"] == 999999.0

