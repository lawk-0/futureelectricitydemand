"""
Page 2: Data Explorer & Exploratory Data Analysis
Member 1 & Member 6: Data Engineering & Integration Leads
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import DataLoader
from src.data.validation import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.analytics.peak_detection import PeakDemandAnalyzer


def render_data_explorer():
    st.title("📊 Data Explorer & Time-Series Diagnostics")
    st.markdown("Comprehensive exploratory analysis, quality auditing, and seasonality profiling.")

    # Data Source Selection
    st.sidebar.subheader("Dataset Ingestion")
    data_source_mode = st.sidebar.radio(
        "Select Data Source:",
        ["Preloaded PJM East Benchmark", "Upload Custom CSV Dataset"],
    )

    df = None
    if data_source_mode == "Preloaded PJM East Benchmark":
        try:
            df = pd.read_csv("data/processed/clean_consumption.csv")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.sidebar.success(f"Loaded {len(df):,} hourly records from PJM East.")
        except Exception as e:
            st.error(f"Error loading preprocessed dataset: {e}")
            return
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Energy Time-Series CSV:", type=["csv"])
        if uploaded_file is not None:
            loader = DataLoader()
            try:
                raw_df = loader.load_dataset(uploaded_file)
                preprocessor = DataPreprocessor()
                df, _ = preprocessor.preprocess(raw_df)
                st.sidebar.success(f"Parsed {len(df):,} clean records from upload.")
            except Exception as e:
                st.sidebar.error(f"Upload error: {e}")
                return
        else:
            st.info("Please upload a CSV file with timestamp and consumption columns, or switch to the Preloaded PJM dataset.")
            return

    # 1. Dataset Overview Metrics
    st.subheader("1. Dataset Overview & Data Quality Audit")
    validator = DataValidator()
    audit_report = validator.validate(df)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Records", f"{len(df):,}")
    with m2:
        st.metric("Time Span", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
    with m3:
        st.metric("Data Integrity Score", f"{audit_report.get('integrity_score', 100):.2f}%")
    with m4:
        st.metric("Duplicate / Gap Fixes", f"{audit_report.get('duplicate_timestamps_count', 0)} dups / {audit_report.get('missing_timestamps_count', 0)} gaps")

    with st.expander("🔍 View Complete Data Quality Audit Report"):
        st.json(audit_report)

    st.markdown("---")

    # 2. Interactive Time Series Viewer
    st.subheader("2. Interactive Electricity Consumption Trajectory")
    st.markdown("Select a date range to inspect macro cycles, heatwaves, cold snaps, and seasonal volatility.")

    # Sampling for responsive plotting if dataset is huge (>20k rows)
    subsample_slider = st.slider(
        "Display Sampling Window (Most recent hours):",
        min_value=720,
        max_value=min(30000, len(df)),
        value=min(8760, len(df)),
        step=720,
        help="Adjust the visible window length (e.g. 8,760 hours = 1 full year).",
    )
    plot_df = df.tail(subsample_slider).copy()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["consumption"],
        mode="lines",
        name="Consumption (MW)",
        line=dict(color="#1f77b4", width=1.2),
    ))
    # Add rolling average line
    plot_df["rolling_7d"] = plot_df["consumption"].rolling(168).mean()
    fig_ts.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["rolling_7d"],
        mode="lines",
        name="7-Day Moving Average",
        line=dict(color="#ff7f0e", width=2.0),
    ))
    fig_ts.update_layout(
        title="Hourly Electricity Demand (MW) with 7-Day Trend",
        xaxis_title="Timestamp",
        yaxis_title="Demand (MW)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # 3. Seasonality & Diurnal Pattern Breakdown
    st.subheader("3. Seasonality & Diurnal Profile Analysis")
    c_left, c_right = st.columns(2)

    df_patterns = df.copy()
    df_patterns["hour"] = df_patterns["timestamp"].dt.hour
    df_patterns["dayofweek"] = df_patterns["timestamp"].dt.day_name()
    df_patterns["month"] = df_patterns["timestamp"].dt.month_name()
    df_patterns["is_weekend"] = df_patterns["timestamp"].dt.dayofweek.isin([5, 6]).map({True: "Weekend", False: "Weekday"})

    with c_left:
        # Diurnal Hourly Demand Profile
        hourly_stats = df_patterns.groupby(["hour", "is_weekend"])["consumption"].mean().reset_index()
        fig_hourly = px.line(
            hourly_stats,
            x="hour",
            y="consumption",
            color="is_weekend",
            markers=True,
            title="Diurnal Hourly Demand Profile (Weekday vs. Weekend)",
            labels={"hour": "Hour of Day (0-23)", "consumption": "Average Demand (MW)", "is_weekend": "Day Type"},
            template="plotly_white",
            color_discrete_map={"Weekday": "#2ca02c", "Weekend": "#9467bd"},
        )
        fig_hourly.update_layout(height=380)
        st.plotly_chart(fig_hourly, use_container_width=True)

    with c_right:
        # Day of Week Distribution
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_stats = df_patterns.groupby("dayofweek")["consumption"].mean().reindex(day_order).reset_index()
        fig_dow = px.bar(
            dow_stats,
            x="dayofweek",
            y="consumption",
            title="Weekly Pattern: Mean Consumption by Day of Week",
            labels={"dayofweek": "Day", "consumption": "Mean Demand (MW)"},
            template="plotly_white",
            color="consumption",
            color_continuous_scale="Blues",
        )
        fig_dow.update_layout(height=380)
        st.plotly_chart(fig_dow, use_container_width=True)

    # Monthly Seasonality
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    monthly_stats = df_patterns.groupby("month")["consumption"].mean().reindex(month_order).reset_index()
    fig_month = px.bar(
        monthly_stats,
        x="month",
        y="consumption",
        title="Annual Seasonality: Mean Consumption by Month (Summer & Winter Bi-Modal Peaks)",
        labels={"month": "Month", "consumption": "Average Demand (MW)"},
        template="plotly_white",
        color="consumption",
        color_continuous_scale="Oranges",
    )
    fig_month.update_layout(height=350)
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("---")

    # 4. Historical Peak Statistics
    st.subheader("4. Historical Peak Benchmarks & Percentile Distribution")
    analyzer = PeakDemandAnalyzer()
    peak_stats = analyzer.analyze_historical_peaks(df)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("All-Time Historical Peak", f"{peak_stats['all_time_peak_mw']:,.1f} MW", f"Hour: {peak_stats['all_time_peak_timestamp']}")
    with p2:
        st.metric("90th Percentile (Alert Threshold)", f"{peak_stats['threshold_p90_mw']:,.1f} MW")
    with p3:
        st.metric("95th Percentile (Critical Risk)", f"{peak_stats['threshold_p95_mw']:,.1f} MW")
    with p4:
        st.metric("Peak-to-Average Ratio", f"{peak_stats['peak_to_average_ratio']}x")

