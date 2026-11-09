"""
Data preprocessing, regular time-grid alignment, and transparent imputation module.
Member 1: Data Engineering Lead
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.utils.helpers import get_logger, load_config

logger = get_logger("DataPreprocessor")


class DataPreprocessor:
    """
    Cleans raw energy time-series data:
    1. Resolves duplicate timestamps deterministically (aggregating via mean).
    2. Re-indexes along a strict contiguous hourly grid ('1h').
    3. Handles missing values using documented multi-tier imputation:
       - Linear interpolation for short gaps (<= 4 consecutive hours)
       - Hour-of-week seasonal profile median for larger gaps
    4. Audits and records all transformations to prevent silent alterations.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)
        data_cfg = self.config.get("data", {})
        self.time_col = data_cfg.get("timestamp_col", "timestamp")
        self.target_col = data_cfg.get("consumption_col", "consumption")
        self.freq = data_cfg.get("sampling_frequency", "1h")
        self.processed_dir = data_cfg.get("processed_dir", "data/processed")

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes complete preprocessing pipeline and returns:
        (clean_df, preprocessing_audit_log)
        """
        audit: Dict[str, Any] = {
            "initial_rows": len(df),
            "duplicate_timestamps_resolved": 0,
            "missing_timestamps_inserted": 0,
            "short_gap_interpolations": 0,
            "seasonal_median_imputations": 0,
            "imputation_details": [],
        }

        data = df.copy()

        # Step 1: Ensure sorted by timestamp
        data = data.sort_values(by=self.time_col).reset_index(drop=True)

        # Step 2: Resolve duplicate timestamps
        dup_count = int(data.duplicated(subset=[self.time_col]).sum())
        if dup_count > 0:
            logger.info(f"Resolving {dup_count} duplicate timestamps by averaging consumption.")
            audit["duplicate_timestamps_resolved"] = dup_count
            data = data.groupby(self.time_col, as_index=False)[self.target_col].mean()

        # Step 3: Reindex to contiguous hourly time grid
        min_ts = data[self.time_col].min()
        max_ts = data[self.time_col].max()
        ideal_index = pd.date_range(start=min_ts, end=max_ts, freq=self.freq, name=self.time_col)
        
        data = data.set_index(self.time_col)
        reindexed_data = data.reindex(ideal_index)
        
        inserted_gaps = int(reindexed_data[self.target_col].isna().sum())
        audit["missing_timestamps_inserted"] = inserted_gaps
        
        # Step 4: Transparent Imputation
        if inserted_gaps > 0:
            logger.info(f"Imputing {inserted_gaps} missing hourly consumption records.")
            
            # Identify short gaps (<= 4 hours) vs longer gaps
            is_na = reindexed_data[self.target_col].isna()
            # Calculate gap lengths using run-length encoding
            blocks = (~is_na).cumsum()
            gap_lengths = is_na.groupby(blocks).transform("sum")
            
            # 4a. Short gaps (<= 4 hours): Linear interpolation
            short_gap_mask = is_na & (gap_lengths <= 4)
            audit["short_gap_interpolations"] = int(short_gap_mask.sum())
            
            # Temporary linear interpolation
            interpolated = reindexed_data[self.target_col].interpolate(
                method="linear", limit=4, limit_direction="both"
            )
            
            # 4b. Longer gaps (> 4 hours): Hour-of-week seasonal median
            remaining_na = interpolated.isna()
            audit["seasonal_median_imputations"] = int(remaining_na.sum())
            
            if remaining_na.sum() > 0:
                # Compute hour-of-week median profile (7 days * 24 hours = 168 profile bins)
                reindexed_data["dayofweek"] = reindexed_data.index.dayofweek
                reindexed_data["hour"] = reindexed_data.index.hour
                seasonal_medians = reindexed_data.groupby(["dayofweek", "hour"])[self.target_col].transform("median")
                
                # If still any NaN (e.g. edge case), fallback to overall median
                overall_median = reindexed_data[self.target_col].median()
                seasonal_medians = seasonal_medians.fillna(overall_median)
                
                interpolated = interpolated.fillna(seasonal_medians)
                reindexed_data.drop(columns=["dayofweek", "hour"], inplace=True)
            
            reindexed_data[self.target_col] = interpolated

        # Reset index to return standard column structure
        clean_df = reindexed_data.reset_index().rename(columns={"index": self.time_col})
        clean_df = clean_df[[self.time_col, self.target_col]]
        clean_df = clean_df.sort_values(by=self.time_col).reset_index(drop=True)

        audit["final_rows"] = len(clean_df)
        audit["start_timestamp"] = str(clean_df[self.time_col].min())
        audit["end_timestamp"] = str(clean_df[self.time_col].max())
        
        logger.info(
            f"Preprocessing complete. Clean records: {len(clean_df):,}. "
            f"Gaps filled: {inserted_gaps} (Short: {audit['short_gap_interpolations']}, Seasonal: {audit['seasonal_median_imputations']})"
        )
        return clean_df, audit

    def save_processed(self, df: pd.DataFrame, filename: str = "clean_consumption.csv") -> str:
        """Saves preprocessed dataframe to processed directory."""
        os.makedirs(self.processed_dir, exist_ok=True)
        out_path = os.path.join(self.processed_dir, filename)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved processed dataset to {out_path}")
        return out_path

