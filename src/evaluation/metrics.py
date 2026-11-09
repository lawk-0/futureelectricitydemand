"""
Comprehensive forecasting and probabilistic uncertainty evaluation metrics.
Member 5: Performance Evaluation and Energy Analytics Lead
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.helpers import get_logger

logger = get_logger("Metrics")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in Megawatts."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in Megawatts."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) in percentage:
    sMAPE = (100% / n) * sum( 2 * |y_true - y_pred| / (|y_true| + |y_pred| + eps) )
    Gracefully bounded between 0% and 200%, robust to zero or near-zero values.
    """
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    diff = np.abs(y_true - y_pred)
    return float(np.mean(200.0 * diff / denominator))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R^2), supplementary metric."""
    return float(r2_score(y_true, y_pred))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Quantile / Pinball loss:
    L_alpha(y, q) = max(alpha * (y - q), (1 - alpha) * (q - y))
    """
    residuals = y_true - y_pred
    return float(np.mean(np.maximum(alpha * residuals, (alpha - 1.0) * residuals)))


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (PICP) in percentage (0-100%)."""
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered) * 100.0)


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width (MPIW) in Megawatts."""
    return float(np.mean(upper - lower))


def evaluate_forecast(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    lower: Optional[Union[np.ndarray, pd.Series]] = None,
    upper: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, Any]:
    """
    Calculates full suite of point-forecast and uncertainty-calibration metrics.
    Filters out any NaN pairs prior to computation.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # Filter NaNs
    valid_mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    if lower is not None and upper is not None:
        low = np.asarray(lower, dtype=float)
        upp = np.asarray(upper, dtype=float)
        valid_mask = valid_mask & ~np.isnan(low) & ~np.isnan(upp)
    else:
        low, upp = None, None

    y_t_val = y_t[valid_mask]
    y_p_val = y_p[valid_mask]

    if len(y_t_val) == 0:
        return {"error": "No overlapping actual and predicted values available for evaluation."}

    metrics: Dict[str, Any] = {
        "n_samples": int(len(y_t_val)),
        "MAE": round(mae(y_t_val, y_p_val), 2),
        "RMSE": round(rmse(y_t_val, y_p_val), 2),
        "sMAPE": round(smape(y_t_val, y_p_val), 2),
        "R2": round(r_squared(y_t_val, y_p_val), 4),
    }

    if low is not None and upp is not None:
        low_val = low[valid_mask]
        upp_val = upp[valid_mask]
        metrics["PICP"] = round(picp(y_t_val, low_val, upp_val), 2)
        metrics["MPIW"] = round(mpiw(low_val, upp_val), 2)
        metrics["Pinball_10"] = round(pinball_loss(y_t_val, low_val, 0.10), 2)
        metrics["Pinball_50"] = round(pinball_loss(y_t_val, y_p_val, 0.50), 2)
        metrics["Pinball_90"] = round(pinball_loss(y_t_val, upp_val, 0.90), 2)

    return metrics

