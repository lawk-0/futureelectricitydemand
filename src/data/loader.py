"""
Dataset loading and column schema detection module.
Member 1: Data Engineering Lead
"""

import os
from typing import Union, Tuple, Optional, List, BinaryIO
import pandas as pd
from src.utils.helpers import get_logger, load_config

logger = get_logger("DataLoader")


class DataLoader:
    """
    Robust time-series energy dataset loader with automatic schema detection.
    Supports local CSV file paths and file-like objects (e.g., Streamlit upload).
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)
        data_cfg = self.config.get("data", {})
        self.timestamp_candidates = data_cfg.get(
            "raw_timestamp_candidates",
            ["Datetime", "datetime", "timestamp", "Timestamp", "Date", "DATE", "time", "Time"]
        )
        self.consumption_candidates = data_cfg.get(
            "raw_consumption_candidates",
            ["PJME_MW", "consumption", "MW", "Load", "load", "demand", "Demand", "Power", "power"]
        )
        self.target_col = data_cfg.get("consumption_col", "consumption")
        self.time_col = data_cfg.get("timestamp_col", "timestamp")

    def detect_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        Automatically detects timestamp and consumption columns from candidate names
        or by inspecting data types and heuristic names.
        """
        cols = list(df.columns)
        detected_time = None
        detected_cons = None

        # 1. Match against configured candidates
        for c in cols:
            if c in self.timestamp_candidates:
                detected_time = c
                break

        for c in cols:
            if c in self.consumption_candidates:
                detected_cons = c
                break

        # 2. Heuristic fallback for timestamp
        if not detected_time:
            for c in cols:
                c_lower = str(c).lower()
                if "date" in c_lower or "time" in c_lower:
                    detected_time = c
                    break

        # 3. Heuristic fallback for consumption (first numeric column that is not the timestamp)
        if not detected_cons:
            for c in cols:
                if c != detected_time:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        detected_cons = c
                        break
                    else:
                        # Try parsing as numeric
                        try:
                            pd.to_numeric(df[c].dropna().iloc[:20])
                            detected_cons = c
                            break
                        except Exception:
                            continue

        if not detected_time:
            raise ValueError(
                f"Could not automatically identify timestamp column in {cols}. "
                f"Please specify timestamp column manually."
            )
        if not detected_cons:
            raise ValueError(
                f"Could not automatically identify consumption column in {cols}. "
                f"Please specify consumption column manually."
            )

        logger.info(f"Detected columns -> Timestamp: '{detected_time}', Consumption: '{detected_cons}'")
        return detected_time, detected_cons

    def load_dataset(
        self,
        source: Union[str, BinaryIO, pd.DataFrame],
        timestamp_col: Optional[str] = None,
        consumption_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Loads CSV dataset from file path or file-like buffer, detects columns,
        standardizes schema, parses timestamps, and sorts chronologically.

        Returns DataFrame with standardized columns: ['timestamp', 'consumption']
        """
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif isinstance(source, (str, os.PathLike)):
            if not os.path.exists(str(source)):
                raise FileNotFoundError(f"Dataset file not found at: {source}")
            logger.info(f"Loading dataset from file: {source}")
            df = pd.read_csv(source)
        else:
            # File-like object (e.g. io.BytesIO or Streamlit UploadedFile)
            logger.info("Loading dataset from uploaded buffer/stream")
            df = pd.read_csv(source)

        if df.empty:
            raise ValueError("The loaded dataset is empty.")

        # Detect columns if not explicitly provided
        if timestamp_col is None or consumption_col is None:
            detected_time, detected_cons = self.detect_columns(df)
            timestamp_col = timestamp_col or detected_time
            consumption_col = consumption_col or detected_cons

        if timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{timestamp_col}' not found in dataset columns: {list(df.columns)}")
        if consumption_col not in df.columns:
            raise KeyError(f"Consumption column '{consumption_col}' not found in dataset columns: {list(df.columns)}")

        # Create standardized dataframe
        clean_df = pd.DataFrame()
        clean_df[self.time_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        clean_df[self.target_col] = pd.to_numeric(df[consumption_col], errors="coerce")

        # Check for unparseable timestamps
        invalid_ts_count = clean_df[self.time_col].isna().sum()
        if invalid_ts_count > 0:
            logger.warning(f"Found {invalid_ts_count} unparseable timestamps. Dropping them.")
            clean_df = clean_df.dropna(subset=[self.time_col])

        # Sort chronologically
        clean_df = clean_df.sort_values(by=self.time_col).reset_index(drop=True)

        logger.info(
            f"Successfully loaded {len(clean_df):,} records ranging from "
            f"{clean_df[self.time_col].min()} to {clean_df[self.time_col].max()}"
        )
        return clean_df

