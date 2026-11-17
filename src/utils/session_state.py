"""
Session state manager for dynamic dataset selection, custom uploadables,
and model persistence across Streamlit pages.
"""

import os
import io
import pandas as pd
import joblib
import streamlit as st
from typing import Tuple, Dict, Any, Optional
from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator
from src.utils.helpers import get_logger

logger = get_logger("SessionManager")

DEFAULT_DATASET_PATH = "data/processed/clean_consumption.csv"


def get_active_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Retrieves the currently active dataset from st.session_state.
    Falls back to the default PJM East benchmark dataset if none is set.
    """
    if "active_dataset" in st.session_state and st.session_state["active_dataset"] is not None:
        df = st.session_state["active_dataset"]
        info = st.session_state.get("dataset_info", {
            "name": "Custom Dataset",
            "is_custom": True,
            "records": len(df),
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        })
        return df, info

    # Load default PJM East dataset
    if os.path.exists(DEFAULT_DATASET_PATH):
        df = pd.read_csv(DEFAULT_DATASET_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # Fallback to raw if processed does not exist yet
        raw_path = "data/raw/PJME_hourly.csv"
        loader = DataLoader()
        raw_df = loader.load_dataset(raw_path)
        preprocessor = DataPreprocessor()
        df, _ = preprocessor.preprocess(raw_df)

    info = {
        "name": "PJM East Benchmark (PJME)",
        "is_custom": False,
        "records": len(df),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }
    st.session_state["active_dataset"] = df
    st.session_state["dataset_info"] = info
    return df, info


def set_active_dataset(df: pd.DataFrame, name: str = "Custom Dataset", is_custom: bool = True) -> Dict[str, Any]:
    """Sets a validated dataframe as the global active dataset."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    info = {
        "name": name,
        "is_custom": is_custom,
        "records": len(df),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }
    st.session_state["active_dataset"] = df
    st.session_state["dataset_info"] = info
    # Clear cached forecasts that belonged to the previous dataset
    if "last_forecast" in st.session_state:
        del st.session_state["last_forecast"]
    return info


def reset_to_default_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resets the application to the default PJM East benchmark dataset."""
    if os.path.exists(DEFAULT_DATASET_PATH):
        df = pd.read_csv(DEFAULT_DATASET_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        loader = DataLoader()
        raw_df = loader.load_dataset("data/raw/PJME_hourly.csv")
        preprocessor = DataPreprocessor()
        df, _ = preprocessor.preprocess(raw_df)

    info = {
        "name": "PJM East Benchmark (PJME)",
        "is_custom": False,
        "records": len(df),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }
    st.session_state["active_dataset"] = df
    st.session_state["dataset_info"] = info
    if "last_forecast" in st.session_state:
        del st.session_state["last_forecast"]
    return df, info


def process_custom_csv_upload(file_buffer, custom_name: str = "Uploaded Dataset") -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Processes an uploaded CSV through schema detection, validation, and regular grid preprocessing.
    Returns: (clean_df, audit_report, dataset_info)
    """
    loader = DataLoader()
    raw_df = loader.load_dataset(file_buffer)
    
    validator = DataValidator()
    audit_report = validator.validate(raw_df)
    
    preprocessor = DataPreprocessor()
    clean_df, prep_audit = preprocessor.preprocess(raw_df)
    
    info = set_active_dataset(clean_df, name=custom_name, is_custom=True)
    return clean_df, audit_report, info


def load_custom_model_upload(file_buffer) -> Tuple[Any, str]:
    """
    Loads a user-uploaded trained model file (.joblib) into session state.
    Returns: (model_object, model_type_str)
    """
    try:
        model = joblib.load(file_buffer)
        
        # Detect model type
        if hasattr(model, "predict_interval") or hasattr(model, "predict_quantiles"):
            st.session_state["trained_quantile_model"] = model
            st.session_state["last_model_type"] = "quantile"
            model_type = "Quantile Regressor"
        elif hasattr(model, "get_feature_importance") or hasattr(model, "predict"):
            st.session_state["trained_point_model"] = model
            st.session_state["last_model_type"] = "point"
            model_type = "Point Regressor"
        else:
            st.session_state["trained_custom_model"] = model
            model_type = "Custom Model"
            
        return model, model_type
    except Exception as e:
        raise ValueError(f"Failed to load model artifact: {e}")


def render_sidebar_dataset_selector():
    """
    Renders persistent dataset selection and custom upload controls in the sidebar.
    Visible across all pages of the application.
    """
    df, info = get_active_dataset()
    
    st.sidebar.markdown("### 📁 Dataset Management")
    
    # Status Badge
    if info.get("is_custom", False):
        st.sidebar.success(f"🟢 **Active:** {info['name']}\n\n*{info['records']:,} records*")
    else:
        st.sidebar.info(f"🔵 **Active:** {info['name']}\n\n*{info['records']:,} records*")
        
    with st.sidebar.expander("⬆️ Upload Custom Energy Dataset", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload CSV File:",
            type=["csv"],
            key="sidebar_dataset_uploader",
            help="Upload any CSV with timestamp and consumption/MW columns. Column names will be auto-detected."
        )
        if uploaded_file is not None:
            if st.button("Apply Uploaded Dataset", key="btn_apply_sidebar_upload", type="primary"):
                try:
                    with st.spinner("Parsing, validating, and preprocessing custom dataset..."):
                        clean_df, audit, new_info = process_custom_csv_upload(
                            uploaded_file, custom_name=f"Custom: {uploaded_file.name}"
                        )
                        st.sidebar.success(f"Loaded {new_info['records']:,} clean hourly records!")
                        st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error processing CSV: {e}")
                    
        if info.get("is_custom", False):
            if st.button("🔄 Reset to PJM East Benchmark", key="btn_reset_dataset"):
                reset_to_default_dataset()
                st.sidebar.info("Reset to PJM East Benchmark dataset.")
                st.rerun()

