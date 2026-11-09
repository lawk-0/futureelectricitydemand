"""
Multi-step time-series forecasting strategies (Direct and Recursive).
Member 4: Core ML and Uncertainty Modelling Lead
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer
from src.models.lightgbm_model import LightGBMDemandModel
from src.models.quantile_model import LightGBMQuantileRegressor
from src.utils.helpers import get_logger

logger = get_logger("MultiHorizon")


class RecursiveForecaster:
    """
    Recursive (Iterated) Multi-Step Forecaster.
    Uses a single 1-step-ahead model iteratively:
    y_hat(t+1) is predicted, fed back as the new lag_1 to update lags and rolling statistics,
    and then used to predict y_hat(t+2), until the target horizon H is achieved.

    Tradeoff: Computationally lightweight (single model to train), but subject to error compounding.
    """

    def __init__(self, model: LightGBMDemandModel, feature_engineer: FeatureEngineer):
        self.model = model
        self.fe = feature_engineer
        self.time_col = feature_engineer.time_col
        self.target_col = feature_engineer.target_col

    def forecast(
        self,
        historical_df: pd.DataFrame,
        horizon: int = 24,
    ) -> pd.DataFrame:
        """
        Generates recursive point forecasts for `horizon` hourly steps into the future.
        """
        history = historical_df.copy().sort_values(by=self.time_col).reset_index(drop=True)
        predictions = []
        timestamps = []

        last_time = history[self.time_col].iloc[-1]

        # Iteratively predict each future step
        for step in range(1, horizon + 1):
            next_time = last_time + pd.Timedelta(hours=step)
            timestamps.append(next_time)

            # Feature generation using current expanded history
            feat_df = self.fe.create_features(history, min_lag=1, drop_na=False)
            latest_row = feat_df.iloc[[-1]]

            pred_val = float(self.model.predict(latest_row)[0])
            pred_val = max(0.0, pred_val)  # Energy demand is strictly non-negative
            predictions.append(pred_val)

            # Append synthetic step into history for subsequent lag/rolling updates
            new_row = pd.DataFrame({
                self.time_col: [next_time],
                self.target_col: [pred_val],
            })
            history = pd.concat([history, new_row], ignore_index=True)

        return pd.DataFrame({
            "timestamp": timestamps,
            "predicted": np.round(predictions, 2),
            "strategy": "recursive",
        })


class DirectHorizonForecaster:
    """
    Direct Multi-Horizon Forecaster.
    Trains separate, dedicated forecasting models for key horizons (e.g. h=24, h=48, h=168)
    or fits a model with min_lag = h, ensuring features strictly use data available at t=0.

    Tradeoff: Immune to recursive error propagation, but requires separate models or lag constraints.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        config_path: str = "config/settings.yaml",
    ):
        self.quantiles = quantiles or [0.10, 0.50, 0.90]
        self.config_path = config_path
        self.fe = FeatureEngineer(config_path=config_path)
        self.models: Dict[int, LightGBMQuantileRegressor] = {}

    def fit_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "DirectHorizonForecaster":
        """
        Fits a direct quantile forecasting model targeting horizon `horizon`
        by setting min_lag = horizon, guaranteeing no lag < horizon is referenced.
        """
        logger.info(f"Training Direct Multi-Horizon model for horizon h={horizon} hours...")

        train_features = self.fe.create_features(df, min_lag=1, drop_na=True)
        feature_cols = self.fe.get_feature_names(train_features)

        # Target shifted backwards by horizon to predict h steps ahead
        # y_target = y(t + horizon)
        train_features["target_h"] = train_features[self.fe.target_col].shift(-horizon)
        train_clean = train_features.dropna(subset=["target_h"] + feature_cols).reset_index(drop=True)

        X_train = train_clean[feature_cols]
        y_train = train_clean["target_h"]

        X_val, y_val = None, None
        if val_df is not None:
            val_features = self.fe.create_features(val_df, min_lag=1, drop_na=True)
            val_features["target_h"] = val_features[self.fe.target_col].shift(-horizon)
            val_clean = val_features.dropna(subset=["target_h"] + feature_cols).reset_index(drop=True)
            if not val_clean.empty:
                X_val = val_clean[feature_cols]
                y_val = val_clean["target_h"]

        q_model = LightGBMQuantileRegressor(
            quantiles=self.quantiles,
            config_path=self.config_path,
        )
        q_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        self.models[horizon] = q_model
        return self

