"""
Probabilistic energy load forecasting using LightGBM Quantile Regression and Conformal Calibration.
Member 4: Core ML and Uncertainty Modelling Lead
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.utils.helpers import get_logger, load_config

logger = get_logger("QuantileModel")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Computes asymmetric quantile loss (pinball loss) for quantile alpha:
    L_alpha(y, q) = max(alpha * (y - q), (1 - alpha) * (q - y))
    """
    residual = y_true - y_pred
    return float(np.mean(np.maximum(alpha * residual, (alpha - 1.0) * residual)))


def calculate_picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Prediction Interval Coverage Probability (PICP):
    Percentage of actual values falling within [lower, upper].
    PICP = (1/N) * sum(I(lower_i <= y_i <= upper_i)) * 100%
    """
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered) * 100.0)


def calculate_mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Mean Prediction Interval Width (MPIW):
    Average sharpness/width of the prediction interval in Megawatts.
    MPIW = (1/N) * sum(upper_i - lower_i)
    """
    return float(np.mean(upper - lower))


class LightGBMQuantileRegressor:
    """
    Multi-quantile gradient boosting forecaster.
    Trains dedicated quantile regressors for specified percentiles (default 10%, 50%, 90%),
    providing a non-parametric estimate of the conditional demand distribution.
    Enforces quantile monotonicity (lower <= median <= upper) to eliminate quantile crossing.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        config_path: str = "config/settings.yaml",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.config = load_config(config_path)
        base_params = self.config.get("models", {}).get("lightgbm", {})
        self.quantiles = quantiles or self.config.get("models", {}).get("quantiles", [0.10, 0.50, 0.90])

        self.base_params: Dict[str, Any] = {
            "objective": "quantile",
            "boosting_type": "gbdt",
            "learning_rate": base_params.get("learning_rate", 0.05),
            "n_estimators": base_params.get("n_estimators", 300),
            "max_depth": base_params.get("max_depth", 8),
            "num_leaves": base_params.get("num_leaves", 63),
            "subsample": base_params.get("subsample", 0.8),
            "colsample_bytree": base_params.get("colsample_bytree", 0.8),
            "random_state": base_params.get("random_state", 42),
            "verbose": -1,
            "n_jobs": base_params.get("n_jobs", -1),
        }

        if params:
            self.base_params.update(params)

        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False
        self.conformal_offset: float = 0.0  # Optional conformal calibration adjustment

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMQuantileRegressor":
        """Fits a distinct quantile model for each configured percentile."""
        self.feature_names_ = list(X_train.columns)
        logger.info(
            f"Training quantile models for percentiles: {[f'{int(q*100)}%' for q in self.quantiles]}..."
        )

        for q in self.quantiles:
            q_params = self.base_params.copy()
            q_params["alpha"] = q
            model_q = lgb.LGBMRegressor(**q_params)

            eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
            model_q.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None,
            )
            self.models[q] = model_q
            logger.info(f"Fitted quantile {q:.2f} model successfully.")

        self.is_fitted = True
        return self

    def calibrate_conformal(
        self, X_cal: pd.DataFrame, y_cal: pd.Series, target_coverage: float = 0.80
    ) -> float:
        """
        Split Conformal Prediction calibration layer (Shafer & Vovk 2008; Lei et al. 2018).
        Calculates empirical non-conformity residuals on holdout calibration data
        to adjust interval boundaries to guarantee target empirical coverage.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before conformal calibration.")

        raw_preds = self.predict_quantiles(X_cal, enforce_monotonicity=True)
        lower_raw = raw_preds[min(self.quantiles)]
        upper_raw = raw_preds[max(self.quantiles)]
        y_actual = y_cal.values

        # Non-conformity score: distance by which truth exceeds the raw interval
        # E_i = max(lower - y, y - upper)
        residuals = np.maximum(lower_raw - y_actual, y_actual - upper_raw)
        
        # Conformal quantile: (1 + 1/n) * target_coverage
        n = len(residuals)
        p_level = min(1.0, np.ceil((n + 1) * target_coverage) / n)
        self.conformal_offset = float(np.quantile(residuals, p_level))
        
        logger.info(
            f"Conformal calibration completed: offset={self.conformal_offset:.2f} MW "
            f"for nominal coverage {target_coverage*100:.1f}%."
        )
        return self.conformal_offset

    def predict_quantiles(
        self, X: pd.DataFrame, enforce_monotonicity: bool = True, apply_conformal: bool = False
    ) -> Dict[float, np.ndarray]:
        """
        Outputs predicted percentiles for input feature matrix.
        Post-processes raw predictions through row-wise sorting to enforce monotonicity:
        q_10 <= q_50 <= q_90, eliminating quantile crossing.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_quantiles() is called.")

        raw_predictions = {
            q: self.models[q].predict(X[self.feature_names_]) for q in self.quantiles
        }

        if enforce_monotonicity and len(self.quantiles) > 1:
            # Stack into matrix of shape (n_samples, n_quantiles)
            sorted_q_keys = sorted(self.quantiles)
            matrix = np.column_stack([raw_predictions[q] for q in sorted_q_keys])
            # Row-wise sort guarantees monotonicity without shifting overall distribution
            matrix_sorted = np.sort(matrix, axis=1)

            predictions = {
                q: matrix_sorted[:, idx] for idx, q in enumerate(sorted_q_keys)
            }
        else:
            predictions = raw_predictions

        if apply_conformal and self.conformal_offset != 0.0:
            min_q = min(self.quantiles)
            max_q = max(self.quantiles)
            predictions[min_q] = np.maximum(0, predictions[min_q] - self.conformal_offset)
            predictions[max_q] = predictions[max_q] + self.conformal_offset

        return predictions

    def predict_interval(
        self, X: pd.DataFrame, apply_conformal: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method returning:
        (lower_bound, median_forecast, upper_bound)
        """
        preds = self.predict_quantiles(
            X, enforce_monotonicity=True, apply_conformal=apply_conformal
        )
        lower = preds[min(self.quantiles)]
        median = preds[0.50] if 0.50 in preds else preds[sorted(self.quantiles)[len(self.quantiles) // 2]]
        upper = preds[max(self.quantiles)]
        return lower, median, upper

