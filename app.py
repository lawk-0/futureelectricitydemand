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

# Custom styling for academic/professional presentation
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    div[data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
    st.sidebar.markdown(
        "**System Status:** 🟢 All Modules Operational\n\n"
        "**Dataset:** PJM East (145k Records)\n\n"
        "**Quantile Models:** Loaded & Calibrated"
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

