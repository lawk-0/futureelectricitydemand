"""
Unified inference pipeline and multi-model forecast generator.
Member 4 & Member 6: ML & Integration Leads
"""

from typing import Optional, Dict, Any, Union, List
import pandas as pd
import numpy as np
import os
from src.features.feature_engineering import FeatureEngineer
from src.models.baseline import SeasonalNaiveModel
from src.models.prophet_model import ProphetDemandModel
from src.models.lightgbm_model import LightGBMDemandModel
from src.models.quantile_model import LightGBMQuantileRegressor
from src.forecasting.multi_horizon import RecursiveForecaster
from src.utils.helpers import get_logger, load_config

logger = get_logger("InferenceEngine")


class EnergyForecastingEngine:
    """
    High-level facade for generating multi-horizon electricity demand forecasts
    across statistical, machine learning, and quantile models.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)
        self.fe = FeatureEngineer(config_path=config_path)
        self.time_col = self.fe.time_col
        self.target_col = self.fe.target_col

    def forecast_seasonal_naive(
        self,
        train_df: pd.DataFrame,
        horizon: int = 24,
        season_lag: int = 168,
        actual_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Runs Seasonal Naive forecasting."""
        model = SeasonalNaiveModel(season_lag=season_lag, time_col=self.time_col, target_col=self.target_col)
        model.fit(train_df)
        forecast_df = model.predict(horizon=horizon)
        return self._attach_actuals(forecast_df, actual_df)

    def forecast_prophet(
        self,
        train_df: pd.DataFrame,
        horizon: int = 24,
        actual_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Runs Prophet forecasting."""
        model = ProphetDemandModel(time_col=self.time_col, target_col=self.target_col)
        model.fit(train_df)
        forecast_df = model.predict(horizon=horizon)
        return self._attach_actuals(forecast_df, actual_df)

    def forecast_quantile_lightgbm(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        horizon: int = 24,
        actual_df: Optional[pd.DataFrame] = None,
        apply_conformal: bool = False,
    ) -> pd.DataFrame:
        """
        Runs LightGBM Quantile Regression:
        Generates 10th percentile (lower), 50th percentile (median), and 90th percentile (upper).
        """
        # 1. Feature generation
        train_feats = self.fe.create_features(train_df, min_lag=1, drop_na=True)
        feature_cols = self.fe.get_feature_names(train_feats)

        X_train = train_feats[feature_cols]
        y_train = train_feats[self.target_col]

        X_val, y_val = None, None
        if val_df is not None:
            # We must concatenate enough history so lags in val_df can be computed
            combined_val = pd.concat([train_df.tail(200), val_df], ignore_index=True)
            val_feats = self.fe.create_features(combined_val, min_lag=1, drop_na=True)
            # Filter to original val timestamps
            val_feats = val_feats[val_feats[self.time_col].isin(val_df[self.time_col])]
            if not val_feats.empty:
                X_val = val_feats[feature_cols]
                y_val = val_feats[self.target_col]

        q_model = LightGBMQuantileRegressor(quantiles=[0.10, 0.50, 0.90])
        q_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        if apply_conformal and X_val is not None and not X_val.empty:
            q_model.calibrate_conformal(X_val, y_val, target_coverage=0.80)

        # Multi-step generation: recursive strategy for median and interval expansion
        last_time = train_df[self.time_col].iloc[-1]
        history = train_df.copy().sort_values(by=self.time_col).reset_index(drop=True)

        preds_lower = []
        preds_median = []
        preds_upper = []
        timestamps = []

        for step in range(1, horizon + 1):
            next_time = last_time + pd.Timedelta(hours=step)
            timestamps.append(next_time)

            feat_df = self.fe.create_features(history, min_lag=1, drop_na=False)
            latest_row = feat_df.iloc[[-1]][feature_cols]

            lower, median, upper = q_model.predict_interval(latest_row, apply_conformal=apply_conformal)
            lower_val = max(0.0, float(lower[0]))
            median_val = max(0.0, float(median[0]))
            upper_val = max(median_val, float(upper[0]))

            # Horizon uncertainty expansion factor (sqrt of step ratio)
            step_factor = 1.0 + 0.15 * np.log(step) if step > 1 else 1.0
            lower_adjusted = max(0.0, median_val - (median_val - lower_val) * step_factor)
            upper_adjusted = median_val + (upper_val - median_val) * step_factor

            preds_lower.append(lower_adjusted)
            preds_median.append(median_val)
            preds_upper.append(upper_adjusted)

            # Append median forecast to history for next recursive lag calculation
            new_row = pd.DataFrame({
                self.time_col: [next_time],
                self.target_col: [median_val],
            })
            history = pd.concat([history, new_row], ignore_index=True)

        forecast_df = pd.DataFrame({
            "timestamp": timestamps,
            "predicted": np.round(preds_median, 2),
            "lower_bound": np.round(preds_lower, 2),
            "upper_bound": np.round(preds_upper, 2),
            "model": "LightGBM_Quantile",
        })

        return self._attach_actuals(forecast_df, actual_df)

    def _attach_actuals(
        self, forecast_df: pd.DataFrame, actual_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merges ground-truth actual demand if provided."""
        if actual_df is not None and not actual_df.empty:
            merged = pd.merge(
                forecast_df,
                actual_df[[self.time_col, self.target_col]],
                on=self.time_col,
                how="left",
            )
            merged = merged.rename(columns={self.target_col: "actual"})
            return merged
        else:
            forecast_df["actual"] = np.nan
            return forecast_df

    def export_to_csv(self, forecast_df: pd.DataFrame, file_path: str) -> str:
        """Exports forecasted records to CSV with standard schema."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        forecast_df.to_csv(file_path, index=False)
        logger.info(f"Exported {len(forecast_df)} forecast records to {file_path}")
        return file_path

