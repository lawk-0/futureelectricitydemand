"""
Seasonal Naive baseline forecasting models.
Member 3: Baseline and Statistical Modelling Lead
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from src.utils.helpers import get_logger

logger = get_logger("BaselineModel")


class SeasonalNaiveModel:
    """
    Seasonal Naive benchmark model.
    Forecasts future demand using historical observation from the same seasonal cycle:
    - Daily seasonality: y_{t+h} = y_{t+h-24} (same hour yesterday)
    - Weekly seasonality: y_{t+h} = y_{t+h-168} (same hour last week)
    """

    def __init__(self, season_lag: int = 168, time_col: str = "timestamp", target_col: str = "consumption"):
        self.season_lag = season_lag
        self.time_col = time_col
        self.target_col = target_col
        self.history_series: Optional[pd.Series] = None
        self.last_timestamp: Optional[pd.Timestamp] = None

    def fit(self, train_df: pd.DataFrame) -> "SeasonalNaiveModel":
        """Stores historical series required for seasonal repetition."""
        sorted_df = train_df.sort_values(by=self.time_col).reset_index(drop=True)
        self.history_series = sorted_df[self.target_col].copy()
        self.last_timestamp = sorted_df[self.time_col].iloc[-1]
        logger.info(
            f"SeasonalNaiveModel fitted with {len(self.history_series):,} records. "
            f"Seasonal lag: {self.season_lag} hours."
        )
        return self

    def predict(self, horizon: int = 24, start_timestamp: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Generates forecasts for the next `horizon` hours.
        Returns DataFrame with ['timestamp', 'predicted', 'lower_bound', 'upper_bound', 'model']
        """
        if self.history_series is None:
            raise RuntimeError("Model must be fitted before predict() is called.")

        start_time = start_timestamp if start_timestamp else self.last_timestamp + pd.Timedelta(hours=1)
        future_times = pd.date_range(start=start_time, periods=horizon, freq="1h")

        preds = []
        n_hist = len(self.history_series)

        for step in range(horizon):
            # Lookup historical value at (n_hist - season_lag + (step % season_lag))
            hist_offset = (n_hist - self.season_lag + (step % self.season_lag))
            if 0 <= hist_offset < n_hist:
                preds.append(float(self.history_series.iloc[hist_offset]))
            else:
                # Fallback to last available observation
                preds.append(float(self.history_series.iloc[-1]))

        pred_array = np.array(preds)
        
        # Empirical residual std based on training differences to provide benchmark interval
        if len(self.history_series) > self.season_lag:
            naive_diffs = self.history_series.diff(self.season_lag).dropna()
            std_err = float(naive_diffs.std())
        else:
            std_err = float(np.std(preds)) * 0.1

        # 80% interval (~1.28 standard deviations under normal assumption)
        lower_bound = np.maximum(0, pred_array - 1.28 * std_err)
        upper_bound = pred_array + 1.28 * std_err

        forecast_df = pd.DataFrame({
            "timestamp": future_times,
            "predicted": np.round(pred_array, 2),
            "lower_bound": np.round(lower_bound, 2),
            "upper_bound": np.round(upper_bound, 2),
            "model": f"SeasonalNaive_{self.season_lag}h",
        })
        return forecast_df

