"""
Peak electricity demand identification, threshold profiling, and load risk analytics.
Member 5: Performance Evaluation and Energy Analytics Lead
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from src.utils.helpers import get_logger, load_config, format_mw

logger = get_logger("PeakAnalytics")


class PeakDemandAnalyzer:
    """
    Analyzes historical and forecasted energy demand spikes:
    - Calculates all-time, monthly, and hourly peak benchmarks
    - Evaluates configurable percentile-based threshold alerts (e.g. 90th / 95th percentile)
    - Detects predicted peak occurrences with lower/upper uncertainty bounds
    - Categorizes load severity while strictly adhering to statistical alert terminology
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        time_col: str = "timestamp",
        target_col: str = "consumption",
    ):
        self.config = load_config(config_path)
        peak_cfg = self.config.get("peak_detection", {})
        self.time_col = time_col
        self.target_col = target_col
        self.default_percentile = peak_cfg.get("default_percentile", 90.0)
        self.critical_percentile = peak_cfg.get("critical_percentile", 95.0)

    def analyze_historical_peaks(self, history_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates historical peak demand benchmarks and statistical distribution."""
        consumption = history_df[self.target_col].dropna()
        if consumption.empty:
            return {"error": "Empty history dataframe."}

        max_idx = consumption.idxmax()
        all_time_peak_val = float(consumption.loc[max_idx])
        all_time_peak_time = str(history_df.loc[max_idx, self.time_col])

        p90 = float(consumption.quantile(self.default_percentile / 100.0))
        p95 = float(consumption.quantile(self.critical_percentile / 100.0))
        p99 = float(consumption.quantile(0.99))
        mean_val = float(consumption.mean())

        # Peak hours distribution (which hour of the day most frequently experiences peak)
        history_copy = history_df.copy()
        history_copy["hour"] = history_copy[self.time_col].dt.hour
        history_copy["month"] = history_copy[self.time_col].dt.month

        # Group by hour to see mean load profile
        hourly_profile = history_copy.groupby("hour")[self.target_col].mean().to_dict()
        top_peak_hour = max(hourly_profile, key=hourly_profile.get)

        return {
            "all_time_peak_mw": round(all_time_peak_val, 2),
            "all_time_peak_timestamp": all_time_peak_time,
            "threshold_p90_mw": round(p90, 2),
            "threshold_p95_mw": round(p95, 2),
            "threshold_p99_mw": round(p99, 2),
            "mean_demand_mw": round(mean_val, 2),
            "peak_to_average_ratio": round(all_time_peak_val / (mean_val + 1e-6), 2),
            "most_common_peak_hour": int(top_peak_hour),
            "hourly_mean_profile": {int(k): round(float(v), 1) for k, v in hourly_profile.items()},
        }

    def detect_forecast_peaks(
        self,
        forecast_df: pd.DataFrame,
        threshold_mw: float,
        critical_threshold_mw: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Scans forecast dataframe to identify upcoming high-demand events.
        
        Important distinction:
        This is a statistical demand alert based on historical distribution percentiles,
        not a hardware electrical capacity warning.
        """
        if forecast_df.empty or "predicted" not in forecast_df.columns:
            return {"error": "Invalid forecast dataframe."}

        crit_threshold = critical_threshold_mw or (threshold_mw * 1.06)

        df = forecast_df.copy()
        max_idx = df["predicted"].idxmax()
        peak_forecast = df.loc[max_idx]

        expected_peak_mw = float(peak_forecast["predicted"])
        expected_peak_time = str(peak_forecast["timestamp"])
        lower_at_peak = float(peak_forecast.get("lower_bound", expected_peak_mw))
        upper_at_peak = float(peak_forecast.get("upper_bound", expected_peak_mw))

        # Flag high-demand hours
        df["is_high_demand"] = df["predicted"] >= threshold_mw
        df["is_critical_risk"] = df["predicted"] >= crit_threshold

        high_demand_events = df[df["is_high_demand"]].copy()

        # Severity categorization
        if expected_peak_mw >= crit_threshold:
            status = "CRITICAL PEAK RISK"
            alert_level = 2
            description = (
                f"Predicted demand reaches {format_mw(expected_peak_mw)}, exceeding the critical "
                f"threshold ({format_mw(crit_threshold)}). Active demand response dispatch recommended."
            )
        elif expected_peak_mw >= threshold_mw:
            status = "HIGH DEMAND ALERT"
            alert_level = 1
            description = (
                f"Predicted demand reaches {format_mw(expected_peak_mw)}, exceeding the high-demand "
                f"threshold ({format_mw(threshold_mw)}). System reserves should be monitored."
            )
        else:
            status = "NORMAL DEMAND"
            alert_level = 0
            description = (
                f"Peak forecasted demand ({format_mw(expected_peak_mw)}) remains safely below "
                f"the alert threshold ({format_mw(threshold_mw)})."
            )

        events_summary = []
        for _, row in high_demand_events.iterrows():
            events_summary.append({
                "timestamp": str(row["timestamp"]),
                "predicted_mw": round(float(row["predicted"]), 1),
                "lower_bound_mw": round(float(row.get("lower_bound", row["predicted"])), 1),
                "upper_bound_mw": round(float(row.get("upper_bound", row["predicted"])), 1),
                "exceedance_mw": round(float(row["predicted"] - threshold_mw), 1),
            })

        return {
            "status": status,
            "alert_level": alert_level,
            "description": description,
            "expected_peak_timestamp": expected_peak_time,
            "expected_peak_mw": round(expected_peak_mw, 1),
            "lower_bound_at_peak": round(lower_at_peak, 1),
            "upper_bound_at_peak": round(upper_at_peak, 1),
            "threshold_used_mw": round(threshold_mw, 1),
            "critical_threshold_used_mw": round(crit_threshold, 1),
            "high_demand_hours_count": int(len(high_demand_events)),
            "high_demand_events": events_summary,
            "disclaimer": (
                "Notice: Statistical demand alerts reflect historical percentile exceedances. "
                "They represent consumption spikes and do not quantify hardware thermal limits or transformer capacity."
            ),
        }

