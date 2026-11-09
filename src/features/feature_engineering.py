"""
Feature engineering module for time-series energy demand forecasting.
Member 2: Feature Engineering Lead

Constructs calendar, cyclical, lag, and rolling statistics features
with strict protection against future temporal target leakage.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import holidays
from src.utils.helpers import get_logger, load_config

logger = get_logger("FeatureEngineer")


class FeatureEngineer:
    """
    Transforms clean energy demand time-series into model-ready feature matrices.
    Applies:
    - Calendar date/time breakdown
    - Holiday indicators using regional holiday calendar
    - Cyclical trigonometry encodings (sine & cosine)
    - Shifted lag features (e.g., t-1, t-2, t-24, t-48, t-168)
    - Shifted rolling statistics (e.g., 24h & 168h rolling mean and std)
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)
        feat_cfg = self.config.get("features", {})
        data_cfg = self.config.get("data", {})

        self.time_col = data_cfg.get("timestamp_col", "timestamp")
        self.target_col = data_cfg.get("consumption_col", "consumption")
        self.holiday_country = data_cfg.get("holiday_country", "US")
        self.holiday_subdiv = data_cfg.get("holiday_subdiv", "PA")

        self.lags: List[int] = feat_cfg.get("lags", [1, 2, 24, 48, 168])
        self.rolling_windows = feat_cfg.get(
            "rolling_windows",
            [{"window": 24, "stats": ["mean", "std"]}, {"window": 168, "stats": ["mean", "std"]}],
        )
        self.calendar_cfg = feat_cfg.get("calendar", {})

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates calendar-based temporal indicators."""
        data = df.copy()
        ts = data[self.time_col].dt

        if self.calendar_cfg.get("include_hour", True):
            data["hour"] = ts.hour
        if self.calendar_cfg.get("include_dayofweek", True):
            data["dayofweek"] = ts.dayofweek
        if self.calendar_cfg.get("include_dayofmonth", True):
            data["dayofmonth"] = ts.day
        if self.calendar_cfg.get("include_month", True):
            data["month"] = ts.month
        if self.calendar_cfg.get("include_quarter", True):
            data["quarter"] = ts.quarter
        if self.calendar_cfg.get("include_year", True):
            data["year"] = ts.year
        if self.calendar_cfg.get("include_weekend", True):
            data["is_weekend"] = ts.dayofweek.isin([5, 6]).astype(int)

        if self.calendar_cfg.get("include_holiday", True):
            try:
                # Query holiday calendar for range of years
                years = data[self.time_col].dt.year.unique()
                cal_holidays = holidays.country_holidays(
                    self.holiday_country,
                    subdiv=self.holiday_subdiv,
                    years=list(years)
                )
                data["is_holiday"] = data[self.time_col].dt.date.isin(cal_holidays).astype(int)
            except Exception as e:
                logger.warning(f"Could not load holidays for {self.holiday_country}: {e}. Defaulting to 0.")
                data["is_holiday"] = 0

        return data

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes cyclical variables using sine and cosine transformations:
        x_sin = sin(2 * pi * x / period)
        x_cos = cos(2 * pi * x / period)
        """
        data = df.copy()

        # Hour of day (0-23, period=24)
        if "hour" in data.columns:
            data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
            data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)

        # Day of week (0-6, period=7)
        if "dayofweek" in data.columns:
            data["dayofweek_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0)
            data["dayofweek_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0)

        # Month of year (1-12, period=12)
        if "month" in data.columns:
            data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
            data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

        return data

    def add_lag_features(self, df: pd.DataFrame, min_lag: int = 1) -> pd.DataFrame:
        """
        Generates historical lag features with shift >= min_lag.
        min_lag ensures that for multi-horizon forecasting (e.g. h hours ahead),
        we do not use any lag < h if direct forecasting is applied.
        For standard 1-step or recursive forecasting, min_lag = 1.
        """
        data = df.copy()
        for lag in self.lags:
            if lag >= min_lag:
                data[f"lag_{lag}"] = data[self.target_col].shift(lag)
        return data

    def add_rolling_features(self, df: pd.DataFrame, shift: int = 1) -> pd.DataFrame:
        """
        Calculates rolling window statistics (mean, std).
        CRITICAL LEAKAGE PROTECTION:
        Target series is shifted by `shift` steps (default 1) BEFORE computing
        rolling statistics, ensuring observation at timestamp t is never included
        in rolling stats for predicting t.
        """
        data = df.copy()
        shifted_target = data[self.target_col].shift(shift)

        for item in self.rolling_windows:
            w = item.get("window", 24)
            stats = item.get("stats", ["mean", "std"])
            rolling_obj = shifted_target.rolling(window=w, min_periods=w // 2)

            if "mean" in stats:
                data[f"rolling_mean_{w}"] = rolling_obj.mean()
            if "std" in stats:
                data[f"rolling_std_{w}"] = rolling_obj.std()

        return data

    def create_features(
        self,
        df: pd.DataFrame,
        min_lag: int = 1,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        Full feature engineering pipeline:
        Calendar -> Cyclical -> Lags -> Rolling Stats -> Drop NaN from lag initialization.
        """
        logger.info(f"Generating features (min_lag={min_lag})...")
        data = self.add_calendar_features(df)
        data = self.add_cyclical_features(data)
        data = self.add_lag_features(data, min_lag=min_lag)
        data = self.add_rolling_features(data, shift=min_lag)

        if drop_na:
            initial_len = len(data)
            data = data.dropna().reset_index(drop=True)
            dropped = initial_len - len(data)
            logger.info(f"Feature matrix generated: {len(data):,} records ({dropped} burn-in rows dropped).")
        return data

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Returns list of predictive feature column names, excluding timestamp and target."""
        excluded = [self.time_col, self.target_col]
        return [c for c in df.columns if c not in excluded]

