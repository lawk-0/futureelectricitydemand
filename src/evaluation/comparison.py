"""
Quantitative model benchmarking, residual analysis, and multi-horizon comparisons.
Member 5: Performance Evaluation and Energy Analytics Lead
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_forecast
from src.utils.helpers import get_logger

logger = get_logger("ModelComparison")


class ModelComparator:
    """
    Facilitates comparative performance benchmarking across diverse forecasting models
    evaluated on identical chronological test periods.
    """

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def evaluate_model(
        self,
        model_name: str,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        horizon_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluates a single model forecast and logs metrics."""
        metrics = evaluate_forecast(actual, predicted, lower=lower, upper=upper)
        entry = {"model": model_name, "horizon": horizon_hours or len(actual), **metrics}
        self.results.append(entry)
        return entry

    def get_comparison_table(self) -> pd.DataFrame:
        """Returns consolidated model benchmark comparison table."""
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame(self.results)
        # Order columns cleanly
        primary_cols = ["model", "horizon", "MAE", "RMSE", "sMAPE", "R2", "PICP", "MPIW"]
        existing = [c for c in primary_cols if c in df.columns]
        extra = [c for c in df.columns if c not in primary_cols]
        return df[existing + extra].sort_values(by="MAE").reset_index(drop=True)

    @staticmethod
    def compute_residuals(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
        """
        Computes residual error diagnostics: e_t = y_t - y_hat_t.
        Checks mean bias, spread, and normality parameters.
        """
        res = actual - predicted
        valid = res[~np.isnan(res)]

        if len(valid) == 0:
            return {}

        return {
            "residuals": valid,
            "mean_error": round(float(np.mean(valid)), 2),
            "std_error": round(float(np.std(valid)), 2),
            "min_error": round(float(np.min(valid)), 2),
            "max_error": round(float(np.max(valid)), 2),
            "median_error": round(float(np.median(valid)), 2),
            "skewness": round(
                float(((valid - np.mean(valid)) ** 3).mean() / (np.std(valid) ** 3 + 1e-6)), 3
            ),
        }

