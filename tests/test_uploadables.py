"""
Unit tests for custom uploadables, session state dataset management,
and model artifact uploads.
"""

import pytest
import io
import os
import pandas as pd
import numpy as np
import joblib
from src.utils.session_state import (
    get_active_dataset,
    set_active_dataset,
    reset_to_default_dataset,
    process_custom_csv_upload,
    load_custom_model_upload,
)
from src.models.quantile_model import LightGBMQuantileRegressor
from src.models.lightgbm_model import LightGBMDemandModel


@pytest.fixture
def custom_csv_buffer():
    """Simulates an uploaded custom energy CSV in memory."""
    dates = pd.date_range("2024-01-01 00:00:00", periods=100, freq="1h")
    # Custom column names: 'Date_Time' and 'Power_Load_MW'
    df = pd.DataFrame({
        "Date_Time": dates,
        "Power_Load_MW": 15000 + 2000 * np.sin(np.linspace(0, 4 * np.pi, 100)),
    })
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    buffer.name = "campus_microgrid_2024.csv"
    return buffer


def test_process_custom_csv_upload(custom_csv_buffer):
    clean_df, audit, info = process_custom_csv_upload(
        custom_csv_buffer, custom_name="Campus Microgrid"
    )

    assert "timestamp" in clean_df.columns
    assert "consumption" in clean_df.columns
    assert len(clean_df) == 100
    assert info["name"] == "Campus Microgrid"
    assert info["is_custom"] is True

    # Verify get_active_dataset reflects the newly uploaded dataset
    active_df, active_info = get_active_dataset()
    assert len(active_df) == 100
    assert active_info["name"] == "Campus Microgrid"


def test_reset_to_default_dataset():
    # After a custom upload, reset to default
    default_df, default_info = reset_to_default_dataset()
    assert default_info["is_custom"] is False
    assert "PJM" in default_info["name"]
    assert len(default_df) > 10000


def test_load_custom_model_upload():
    # Create and serialize a small dummy quantile model
    q_model = LightGBMQuantileRegressor(
        quantiles=[0.10, 0.50, 0.90], params={"n_estimators": 5}
    )
    X_dummy = pd.DataFrame({"feat_1": np.random.randn(50), "feat_2": np.random.randn(50)})
    y_dummy = pd.Series(np.random.randn(50) * 100 + 25000)
    q_model.fit(X_dummy, y_dummy)

    buffer = io.BytesIO()
    joblib.dump(q_model, buffer)
    buffer.seek(0)

    loaded_model, model_type = load_custom_model_upload(buffer)
    assert "Quantile" in model_type
    assert loaded_model.is_fitted is True

    # Point model test
    pt_model = LightGBMDemandModel(params={"n_estimators": 5})
    pt_model.fit(X_dummy, y_dummy)

    pt_buffer = io.BytesIO()
    joblib.dump(pt_model, pt_buffer)
    pt_buffer.seek(0)

    loaded_pt, pt_type = load_custom_model_upload(pt_buffer)
    assert "Point" in pt_type
    assert loaded_pt.is_fitted is True

