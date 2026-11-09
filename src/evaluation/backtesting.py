"""
Temporal validation, chronological train/val/test splitting, and rolling-origin backtesting.
Member 2: Temporal Validation Lead
"""

from typing import Tuple, List, Dict, Generator, Any, Optional
import pandas as pd
import numpy as np
from src.utils.helpers import get_logger, load_config

logger = get_logger("TemporalValidation")


def chronological_split(
    df: pd.DataFrame,
    test_hours: int = 720,
    val_hours: int = 720,
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs strict chronological splitting into Train, Validation, and Test sets.
    No future data is ever included in train or validation sets.

    Parameters:
    - df: Chronologically sorted DataFrame
    - test_hours: Number of hourly records reserved for final holdout test set
    - val_hours: Number of hourly records reserved for validation/tuning set
    - time_col: Name of datetime column

    Returns:
    (train_df, val_df, test_df)
    """
    if not df[time_col].is_monotonic_increasing:
        df = df.sort_values(by=time_col).reset_index(drop=True)

    total_rows = len(df)
    min_required = test_hours + val_hours + 168  # At least 1 week for training
    if total_rows < min_required:
        raise ValueError(
            f"Dataset length ({total_rows}) is insufficient for split: "
            f"test_hours={test_hours}, val_hours={val_hours}, min_train=168"
        )

    test_start_idx = total_rows - test_hours
    val_start_idx = test_start_idx - val_hours

    train_df = df.iloc[:val_start_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_start_idx:test_start_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_start_idx:].copy().reset_index(drop=True)

    # Verify temporal ordering and no leakage
    verify_temporal_boundaries(train_df, val_df, test_df, time_col=time_col)

    logger.info(
        f"Chronological split successful: "
        f"Train={len(train_df):,} [{train_df[time_col].min()} to {train_df[time_col].max()}], "
        f"Val={len(val_df):,} [{val_df[time_col].min()} to {val_df[time_col].max()}], "
        f"Test={len(test_df):,} [{test_df[time_col].min()} to {test_df[time_col].max()}]"
    )
    return train_df, val_df, test_df


def verify_temporal_boundaries(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    time_col: str = "timestamp",
) -> bool:
    """
    Formally verifies that:
    max(train_timestamp) < min(val_timestamp)
    and (if test provided) max(val_timestamp) < min(test_timestamp).
    Raises AssertionError if temporal data leakage boundary is violated.
    """
    train_max = train_df[time_col].max()
    val_min = val_df[time_col].min()

    assert train_max < val_min, (
        f"TEMPORAL LEAKAGE DETECTED! "
        f"Train max timestamp ({train_max}) >= Validation min timestamp ({val_min})"
    )

    if test_df is not None and not test_df.empty:
        val_max = val_df[time_col].max()
        test_min = test_df[time_col].min()
        assert val_max < test_min, (
            f"TEMPORAL LEAKAGE DETECTED! "
            f"Validation max timestamp ({val_max}) >= Test min timestamp ({test_min})"
        )

    return True


class RollingOriginBacktester:
    """
    Time-series rolling-origin (expanding or sliding window) cross-validator.
    Ensures every test fold occurs strictly in the future relative to its corresponding training fold.
    """

    def __init__(
        self,
        n_splits: int = 4,
        horizon_hours: int = 168,
        min_train_hours: int = 24 * 60,  # 60 days
        window_type: str = "expanding",  # 'expanding' or 'sliding'
        time_col: str = "timestamp",
    ):
        self.n_splits = n_splits
        self.horizon_hours = horizon_hours
        self.min_train_hours = min_train_hours
        self.window_type = window_type
        self.time_col = time_col

    def split(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], None, None]:
        """
        Yields (fold_train_df, fold_test_df, fold_meta) for each chronological fold.
        """
        if not df[self.time_col].is_monotonic_increasing:
            df = df.sort_values(by=self.time_col).reset_index(drop=True)

        n_samples = len(df)
        total_test_horizon = self.n_splits * self.horizon_hours

        if n_samples < (self.min_train_hours + total_test_horizon):
            raise ValueError(
                f"Dataset size ({n_samples}) too small for {self.n_splits} splits of "
                f"horizon {self.horizon_hours} and min_train {self.min_train_hours}."
            )

        test_starts = [
            n_samples - (self.n_splits - i) * self.horizon_hours for i in range(self.n_splits)
        ]

        for fold_idx, test_start in enumerate(test_starts):
            test_end = test_start + self.horizon_hours

            if self.window_type == "expanding":
                train_start = 0
            else:  # sliding
                train_start = max(0, test_start - self.min_train_hours)

            fold_train = df.iloc[train_start:test_start].copy().reset_index(drop=True)
            fold_test = df.iloc[test_start:test_end].copy().reset_index(drop=True)

            # Strict assertion against leakage
            verify_temporal_boundaries(fold_train, fold_test, time_col=self.time_col)

            meta = {
                "fold": fold_idx + 1,
                "train_records": len(fold_train),
                "test_records": len(fold_test),
                "train_start": fold_train[self.time_col].min(),
                "train_end": fold_train[self.time_col].max(),
                "test_start": fold_test[self.time_col].min(),
                "test_end": fold_test[self.time_col].max(),
            }
            yield fold_train, fold_test, meta
