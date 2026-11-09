"""
Tests for Forecasting Models: Seasonal Naive, Prophet, LightGBM, Quantiles, and Inference.
Members 3 & 4: Modeling Leads
"""

import pytest
import pandas as pd
import numpy as np
import os
from src.features.feature_engineering import FeatureEngineer
from src.models.baseline import SeasonalNaiveModel
from src.models.prophet_model import ProphetDemandModel
from src.models.lightgbm_model import LightGBMDemandModel
from src.models.quantile_model import LightGBMQuantileRegressor
from src.forecasting.multi_horizon import RecursiveForecaster
from src.forecasting.inference import EnergyForecastingEngine


@pytest.fixture
def clean_train_data():
    """Generates 500 hours of realistic sinusoidal electricity demand."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=500, freq="1h")
    # Base load 25000 + daily cycle (period 24) + weekly variation + noise
    daily = 4000 * np.sin(2 * np.pi * np.arange(500) / 24.0)
    weekly = 2000 * np.cos(2 * np.pi * np.arange(500) / 168.0)
    noise = np.random.normal(0, 150, 500)
    values = 25000 + daily + weekly + noise
    return pd.DataFrame({"timestamp": dates, "consumption": values})


def test_seasonal_naive_forecaster(clean_train_data):
    model = SeasonalNaiveModel(season_lag=24)
    model.fit(clean_train_data)
    forecast = model.predict(horizon=24)

    assert len(forecast) == 24
    assert "predicted" in forecast.columns
    assert "lower_bound" in forecast.columns
    assert "upper_bound" in forecast.columns
    assert (forecast["lower_bound"] <= forecast["predicted"]).all()
    assert (forecast["predicted"] <= forecast["upper_bound"]).all()


def test_prophet_model_forecaster(clean_train_data):
    # Train Prophet on subset for speed in unit test
    model = ProphetDemandModel()
    model.fit(clean_train_data.iloc[-250:])
    forecast = model.predict(horizon=24)

    assert len(forecast) == 24
    assert "predicted" in forecast.columns
    assert "lower_bound" in forecast.columns
    assert "upper_bound" in forecast.columns
    assert (forecast["predicted"] > 0).all()


def test_lightgbm_point_and_importance(clean_train_data):
    fe = FeatureEngineer()
    feat_df = fe.create_features(clean_train_data, min_lag=1, drop_na=True)
    feature_cols = fe.get_feature_names(feat_df)

    X = feat_df[feature_cols]
    y = feat_df["consumption"]

    lgb_model = LightGBMDemandModel(params={"n_estimators": 50})
    lgb_model.fit(X.iloc[:-50], y.iloc[:-50])
    preds = lgb_model.predict(X.iloc[-50:])

    assert len(preds) == 50
    assert not np.isnan(preds).any()

    importance_df = lgb_model.get_feature_importance(importance_type="gain")
    assert len(importance_df) == len(feature_cols)
    assert "importance_pct" in importance_df.columns


def test_quantile_lightgbm_monotonicity_and_conformal(clean_train_data):
    fe = FeatureEngineer()
    feat_df = fe.create_features(clean_train_data, min_lag=1, drop_na=True)
    feature_cols = fe.get_feature_names(feat_df)

    X_train = feat_df.iloc[:-100][feature_cols]
    y_train = feat_df.iloc[:-100]["consumption"]
    X_val = feat_df.iloc[-100:-50][feature_cols]
    y_val = feat_df.iloc[-100:-50]["consumption"]
    X_test = feat_df.iloc[-50:][feature_cols]

    q_model = LightGBMQuantileRegressor(
        quantiles=[0.10, 0.50, 0.90],
        params={"n_estimators": 40},
    )
    q_model.fit(X_train, y_train)

    # 1. Monotonicity assertion: lower <= median <= upper for all test predictions
    lower, median, upper = q_model.predict_interval(X_test)
    assert len(lower) == 50
    assert (lower <= median + 1e-5).all(), "Quantile crossing: lower exceeds median!"
    assert (median <= upper + 1e-5).all(), "Quantile crossing: median exceeds upper!"

    # 2. Conformal calibration test
    offset = q_model.calibrate_conformal(X_val, y_val, target_coverage=0.80)
    assert isinstance(offset, float)

    lower_c, median_c, upper_c = q_model.predict_interval(X_test, apply_conformal=True)
    assert (lower_c <= median_c + 1e-5).all()
    assert (median_c <= upper_c + 1e-5).all()


def test_inference_engine_and_csv_export(clean_train_data, tmp_path):
    engine = EnergyForecastingEngine()
    forecast = engine.forecast_seasonal_naive(clean_train_data, horizon=24)

    assert len(forecast) == 24
    assert "predicted" in forecast.columns

    export_path = tmp_path / "forecast_export.csv"
    engine.export_to_csv(forecast, str(export_path))
    assert os.path.exists(export_path)

    reloaded = pd.read_csv(export_path)
    assert len(reloaded) == 24
    assert list(reloaded.columns) == list(forecast.columns)

