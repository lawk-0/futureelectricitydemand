"""
Page 2: Data Explorer & Exploratory Data Analysis
Member 1 & Member 6: Data Engineering & Integration Leads
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data.validation import DataValidator
from src.analytics.peak_detection import PeakDemandAnalyzer
from src.utils.helpers import apply_dark_theme
from src.utils.session_state import (
    get_active_dataset,
    process_custom_csv_upload,
    reset_to_default_dataset,
)


def render_data_explorer():
    st.title("📊 Data Explorer & Time-Series Diagnostics")
    st.markdown("Comprehensive exploratory analysis, quality auditing, and seasonality profiling for active or uploaded datasets.")

    # In-page Dataset Management & Custom Upload Section
    st.subheader("1. Ingest & Manage Dataset")
    col_src1, col_src2 = st.columns([2, 1])

    with col_src1:
        uploaded_file = st.file_uploader(
            "Upload Custom Energy Consumption CSV (Auto-detects timestamp and load columns):",
            type=["csv"],
            key="page_data_explorer_uploader",
            help="Supports any time-series CSV. Schema detection automatically maps timestamp and demand columns."
        )
        if uploaded_file is not None:
            if st.button("Apply Uploaded CSV as Active Dataset", type="primary", key="btn_apply_page_upload"):
                try:
                    with st.spinner("Processing, validating, and regularizing uploaded dataset..."):
                        clean_df, audit, info = process_custom_csv_upload(
                            uploaded_file, custom_name=f"Custom: {uploaded_file.name}"
                        )
                        st.success(f"Successfully processed {len(clean_df):,} records! Active across all pages.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")

    with col_src2:
        df, info = get_active_dataset()
        st.markdown(f"**Currently Active Dataset:**")
        st.info(f"🏷️ **{info['name']}**\n\n🕒 **Rows:** {info['records']:,} hours\n\n📅 **Span:** {info['start'][:10]} to {info['end'][:10]}")
        if info.get("is_custom", False):
            if st.button("🔄 Reset to Default PJM East Benchmark", key="btn_page_reset_pjm"):
                reset_to_default_dataset()
                st.info("Reset to PJM East Benchmark.")
                st.rerun()

    st.markdown("---")

    # 2. Data Quality Audit & Overview
    st.subheader("2. Dataset Summary & Quality Audit Report")
    validator = DataValidator()
    audit_report = validator.validate(df)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Records", f"{len(df):,}")
    with m2:
        st.metric("Time Horizon", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
    with m3:
        st.metric("Data Integrity Score", f"{audit_report.get('integrity_score', 100):.2f}%")
    with m4:
        st.metric("Duplicate / Gap Fixes", f"{audit_report.get('duplicate_timestamps_count', 0)} dups / {audit_report.get('missing_timestamps_count', 0)} gaps")

    with st.expander("🔍 View Complete Data Quality Audit Breakdown"):
        st.json(audit_report)

    st.markdown("---")

    # 3. Interactive Time Series Viewer
    st.subheader("3. Interactive Electricity Consumption Trajectory")
    st.markdown("Inspect historical cycles, seasonal demand swings, and multi-year patterns.")

    # Responsive sampling window slider
    max_slider = len(df)
    subsample_slider = st.slider(
        "Display Sampling Window (Most recent hours):",
        min_value=min(168, max_slider),
        max_value=min(35000, max_slider),
        value=min(8760, max_slider),
        step=168,
        help="Adjust window length (8,760 hours = 1 full year).",
    )
    plot_df = df.tail(subsample_slider).copy()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["consumption"],
        mode="lines",
        name="Consumption (MW)",
        line=dict(color="#38bdf8", width=1.3),
    ))
    # 7-day moving average
    plot_df["rolling_7d"] = plot_df["consumption"].rolling(168, min_periods=24).mean()
    fig_ts.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["rolling_7d"],
        mode="lines",
        name="7-Day Moving Average",
        line=dict(color="#f59e0b", width=2.2),
    ))
    fig_ts.update_layout(
        title=f"Hourly Electricity Demand (MW) with 7-Day Trend — {info['name']}",
        xaxis_title="Timestamp",
        yaxis_title="Demand (MW)",
        height=450,
        hovermode="x unified",
    )
    apply_dark_theme(fig_ts)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # 4. Seasonality & Diurnal Pattern Breakdown
    st.subheader("4. Seasonality & Diurnal Profile Analysis")
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
            color_discrete_map={"Weekday": "#34d399", "Weekend": "#a78bfa"},
        )
        fig_hourly.update_layout(height=380)
        apply_dark_theme(fig_hourly)
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
            color="consumption",
            color_continuous_scale="Blues",
        )
        fig_dow.update_layout(height=380)
        apply_dark_theme(fig_dow)
        st.plotly_chart(fig_dow, use_container_width=True)

    # Monthly Seasonality
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    present_months = [m for m in month_order if m in df_patterns["month"].values]
    if present_months:
        monthly_stats = df_patterns.groupby("month")["consumption"].mean().reindex(present_months).reset_index()
        fig_month = px.bar(
            monthly_stats,
            x="month",
            y="consumption",
            title="Annual Seasonality: Mean Consumption by Month",
            labels={"month": "Month", "consumption": "Average Demand (MW)"},
            color="consumption",
            color_continuous_scale="Oranges",
        )
        fig_month.update_layout(height=350)
        apply_dark_theme(fig_month)
        st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("---")

    # 5. Historical Peak Benchmarks
    st.subheader("5. Historical Peak Benchmarks & Percentile Distribution")
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
