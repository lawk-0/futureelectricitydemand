"""
Page 1: Project Overview
Member 6: System Integration Lead
"""

import streamlit as st
import plotly.graph_objects as go
from src.utils.helpers import load_config


def render_overview():
    st.title("⚡ Intelligent Energy Demand Forecasting & Peak Load Prediction")
    st.markdown(
        "**University B.Tech Final Year Capstone Project in Machine Learning & Time-Series Engineering**"
    )

    st.markdown("---")

    # Executive Summary Card
    st.subheader("📌 Project Overview & Problem Statement")
    st.markdown("""
    Modern electrical grids and power distribution operators face severe operational challenges due to
    large fluctuations in electricity consumption driven by diurnal human activity, business cycles,
    weather variations, and industrial shifts.

    Traditional point forecasts only predict a single expected demand number (e.g., *32,500 MW*), providing
    grid operators zero visibility into forecast uncertainty. This project implements an **end-to-end,
    production-grade machine learning platform** that forecasts future hourly electricity consumption while
    delivering **rigorous probabilistic prediction intervals (10th, 50th, and 90th percentiles)** and
    **automated peak-demand overload risk detection**.
    """)

    # Key Metrics Highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Primary Dataset", "PJM East (PJME)", "145,392 Hours")
    with col2:
        st.metric("Primary ML Model", "LightGBM Quantile", "0.994 R² Score")
    with col3:
        st.metric("Uncertainty Interval", "80% Nominal", "82.2% Empirical PICP")
    with col4:
        st.metric("Prediction Horizons", "24h / 48h / 7d", "Multi-Step")

    st.markdown("---")

    # Core Differentiators
    st.subheader("🎯 Core Technical Differentiators")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Probabilistic Uncertainty")
        st.write(
            "Instead of naive single-value estimates, models generate asymmetric 10th and 90th percentile bounds "
            "using quantile regression loss with non-crossing monotonicity and conformal prediction calibration."
        )
    with c2:
        st.markdown("#### 2. Temporal Leakage Prevention")
        st.write(
            "Features (lags, rolling averages, rolling standard deviations) strictly use shifted operators (t-1). "
            "Evaluations enforce strict chronological train-val-test boundaries without random splitting."
        )
    with c3:
        st.markdown("#### 3. Peak Demand Analytics")
        st.write(
            "Detects high-demand spikes exceeding configurable historical percentiles (90th/95th), "
            "providing multi-tier operational alerts to assist grid reserve scheduling."
        )

    st.markdown("---")

    # Architecture Pipeline Visualization
    st.subheader("🏗️ System Architecture & Workflow Pipeline")
    pipeline_steps = [
        ("1. Data Ingestion", "Raw PJM Hourly CSV / Uploaded Stream, Schema Detection, UTC Alignment"),
        ("2. Validation & Preprocessing", "Duplicate Resolution (DST), Gap Imputation, Quality Auditing"),
        ("3. Feature Engineering", "Cyclical Trig Encodings, Shifted Lags (1..168h), Non-leaking Rolling Stats"),
        ("4. Temporal Validation", "Chronological Train-Validation-Test Splitting, Rolling-Origin Backtesting"),
        ("5. Multi-Model Forecasting", "Seasonal Naive (168h), Prophet Fourier GAM, LightGBM Regressor"),
        ("6. Probabilistic Quantiles", "LightGBM Pinball Loss (q=0.10, 0.50, 0.90) + Conformal Calibration"),
        ("7. Peak Detection & Alerts", "90th/95th Percentile Thresholding, Peak Lead Hours, Demand Alerts"),
        ("8. Interactive Dashboard", "Real-time Multi-Horizon Forecasts, Uncertainty Shading, CSV Exports"),
    ]

    for title, desc in pipeline_steps:
        st.markdown(f"**{title}** — {desc}")

    st.info(
        "💡 **Quick Navigation:** Use the sidebar on the left to explore raw data, train models, generate "
        "interactive forecasts, inspect prediction intervals, and review formal technical documentation."
    )

