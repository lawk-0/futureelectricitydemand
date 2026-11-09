"""
Tests for Data Engineering modules: loader, validator, and preprocessor.
"""

import pytest
import pandas as pd
import numpy as np
import os
from src.data.loader import DataLoader
from src.data.validation import DataValidator
from src.data.preprocessing import DataPreprocessor


@pytest.fixture
def sample_raw_df():
    """Generates synthetic hourly data with intentional duplicate and gap."""
    # 50 hours sequence
    dates = pd.date_range("2023-01-01 00:00:00", periods=50, freq="1h")
    values = 25000 + 5000 * np.sin(np.linspace(0, 4 * np.pi, 50)) + np.random.normal(0, 200, 50)
    df = pd.DataFrame({"Datetime": dates, "PJME_MW": values})

    # Add a duplicate timestamp
    dup_row = pd.DataFrame({"Datetime": [dates[10]], "PJME_MW": [26000.0]})
    # Drop row 20 to create a 1-hour gap
    df = pd.concat([df.iloc[:20], df.iloc[21:], dup_row], ignore_index=True)
    return df


def test_loader_detects_columns_and_sorts(sample_raw_df, tmp_path):
    csv_file = tmp_path / "test_energy.csv"
    sample_raw_df.to_csv(csv_file, index=False)

    loader = DataLoader()
    loaded_df = loader.load_dataset(str(csv_file))

    assert "timestamp" in loaded_df.columns
    assert "consumption" in loaded_df.columns
    assert pd.api.types.is_datetime64_any_dtype(loaded_df["timestamp"])
    # Check sorted
    assert loaded_df["timestamp"].is_monotonic_increasing


def test_validation_reports_duplicates_and_gaps(sample_raw_df):
    loader = DataLoader()
    df = loader.load_dataset(sample_raw_df)

    validator = DataValidator(freq="1h")
    report = validator.validate(df)

    assert report["total_records"] > 0
    assert report["duplicate_timestamps_count"] >= 1
    assert report["missing_timestamps_count"] >= 1
    assert "summary_stats" in report
    assert report["integrity_score"] > 0


def test_preprocessing_resolves_duplicates_and_imputes(sample_raw_df):
    loader = DataLoader()
    raw_df = loader.load_dataset(sample_raw_df)

    preprocessor = DataPreprocessor()
    clean_df, audit = preprocessor.preprocess(raw_df)

    # Must have 0 duplicates after preprocessing
    assert clean_df["timestamp"].duplicated().sum() == 0
    # Must have no NaNs
    assert clean_df["consumption"].isna().sum() == 0
    # Must be strictly contiguous 1h intervals
    time_diffs = clean_df["timestamp"].diff().dropna()
    assert (time_diffs == pd.Timedelta(hours=1)).all()
    # Audit log must reflect resolved duplicates and gap imputations
    assert audit["duplicate_timestamps_resolved"] >= 1
    assert audit["missing_timestamps_inserted"] >= 1


def test_preprocessing_saves_output(sample_raw_df, tmp_path):
    loader = DataLoader()
    raw_df = loader.load_dataset(sample_raw_df)
    preprocessor = DataPreprocessor()
    preprocessor.processed_dir = str(tmp_path)

    clean_df, _ = preprocessor.preprocess(raw_df)
    out_file = preprocessor.save_processed(clean_df, "test_clean.csv")

    assert os.path.exists(out_file)
    reloaded = pd.read_csv(out_file)
    assert len(reloaded) == len(clean_df)

