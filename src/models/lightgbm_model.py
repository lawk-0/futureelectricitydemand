"""
Primary gradient boosting demand forecasting model using LightGBM.
Member 4: Core ML and Uncertainty Modelling Lead
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.utils.helpers import get_logger, load_config

logger = get_logger("LightGBMModel")


class LightGBMDemandModel:
    """
    LightGBM Gradient Boosted Decision Tree Regressor for energy load forecasting.
    Optimizes Mean Squared Error (L2) or Huber objective using histogram-based splitting.
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.config = load_config(config_path)
        base_params = self.config.get("models", {}).get("lightgbm", {})

        self.params: Dict[str, Any] = {
            "objective": "regression",
            "metric": "rmse",
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
            self.params.update(params)

        self.model = lgb.LGBMRegressor(**self.params)
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMDemandModel":
        """
        Trains LightGBM regressor with optional early stopping on validation fold.
        """
        self.feature_names_ = list(X_train.columns)
        logger.info(
            f"Fitting LightGBM on {len(X_train):,} samples with {len(self.feature_names_)} features..."
        )

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None,
        )

        self.is_fitted = True
        logger.info(
            f"LightGBM training complete. Best iteration: {getattr(self.model, 'best_iteration_', self.params['n_estimators'])}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates point forecasts for input feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("LightGBMDemandModel must be fitted before predict() is called.")
        return self.model.predict(X[self.feature_names_])

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Returns feature importance table sorted in descending order of contribution.
        importance_type: 'gain' (total informational gain) or 'split' (frequency of splits).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before extracting feature importance.")

        importances = self.model.booster_.feature_importance(importance_type=importance_type)
        df_importance = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        total_gain = df_importance["importance"].sum()
        if total_gain > 0:
            df_importance["importance_pct"] = (df_importance["importance"] / total_gain) * 100.0
        else:
            df_importance["importance_pct"] = 0.0

        return df_importance

