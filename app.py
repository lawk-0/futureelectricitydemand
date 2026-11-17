"""
Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction
Main Streamlit Application Entrypoint
Member 6: System Integration Lead
"""

import os
import sys

# Ensure project root is in system path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Intelligent Energy Demand Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for academic/professional presentation & dark-mode safety
st.markdown("""
<style>
    /* Main layout container */
    .main .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 96%;
    }

    /* Metric Cards - Sleek Slate Dark Mode with High Contrast */
    [data-testid="stMetric"], .stMetric {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 0.9rem 1.15rem !important;
        border-radius: 10px !important;
        border-left: 5px solid #38bdf8 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.2rem !important;
    }
    [data-testid="stMetricValue"] div, [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
    }
    [data-testid="stMetricDelta"] div, [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* Custom Cards */
    .info-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Expander header styling */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border-radius: 8px !important;
    }

    /* Sidebar navigation aesthetics */
    div[data-testid="stSidebarNav"] {
        padding-top: 0.5rem;
    }

    /* Headers crisp contrast */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# Import session state & dataset manager
from src.utils.session_state import (
    get_active_dataset,
    render_sidebar_dataset_selector,
)

# Import dashboard pages
from dashboard.overview import render_overview
from dashboard.data_explorer import render_data_explorer
from dashboard.model_training import render_model_training
from dashboard.forecasting import render_forecasting
from dashboard.model_comparison import render_model_comparison
from dashboard.uncertainty import render_uncertainty
from dashboard.peak_analytics import render_peak_analytics
from dashboard.technical_info import render_technical_info


def main():
    st.sidebar.title("⚡ Energy Forecaster")
    st.sidebar.caption("B.Tech Capstone Project — University CSE")
    st.sidebar.markdown("---")

    # Persistent Dataset Selector & Custom Uploader in Sidebar
    render_sidebar_dataset_selector()

    st.sidebar.markdown("---")

    # Navigation menu
    page = st.sidebar.radio(
        "Navigation Menu:",
        [
            "🏠 Project Overview",
            "📊 Data Explorer",
            "⚙️ Model Training",
            "🔮 Demand Forecasting",
            "📈 Model Comparison",
            "🛡️ Uncertainty Analysis",
            "⚡ Peak Demand Analytics",
            "📚 Technical Information",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    
    # Dynamic System & Model Status
    _, active_info = get_active_dataset()
    has_custom_model = "trained_quantile_model" in st.session_state or "trained_point_model" in st.session_state
    model_status_label = "🟢 Custom / Active in Session" if has_custom_model else "🔵 Pretrained Baseline Loaded"

    st.sidebar.markdown(
        f"**System Status:** 🟢 All Modules Active\n\n"
        f"**Active Data:** {active_info['name']}\n\n"
        f"**Records:** {active_info['records']:,} Hours\n\n"
        f"**Model Engine:** {model_status_label}"
    )

    # Dispatch to selected page
    if page == "🏠 Project Overview":
        render_overview()
    elif page == "📊 Data Explorer":
        render_data_explorer()
    elif page == "⚙️ Model Training":
        render_model_training()
    elif page == "🔮 Demand Forecasting":
        render_forecasting()
    elif page == "📈 Model Comparison":
        render_model_comparison()
    elif page == "🛡️ Uncertainty Analysis":
        render_uncertainty()
    elif page == "⚡ Peak Demand Analytics":
        render_peak_analytics()
    elif page == "📚 Technical Information":
        render_technical_info()


if __name__ == "__main__":
    main()
