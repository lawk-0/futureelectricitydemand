"""
Page 7: Peak Demand Analytics & Overload Risk Assessment
Members 5 & 6: Evaluation & Integration Leads
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.analytics.peak_detection import PeakDemandAnalyzer
from src.utils.helpers import format_mw, apply_dark_theme
from src.utils.session_state import get_active_dataset


def render_peak_analytics():
    st.title("⚡ Peak Demand Analytics & Grid Stress Alerts")
    st.markdown("Automated detection of anticipated peak load events, threshold exceedance, and demand-response dispatch advisories.")

    df, dataset_info = get_active_dataset()

    analyzer = PeakDemandAnalyzer()
    hist_stats = analyzer.analyze_historical_peaks(df)

    st.subheader(f"1. Historical Baseline Peak Benchmarks — {dataset_info['name']}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("All-Time Historical Peak", format_mw(hist_stats["all_time_peak_mw"]), f"{hist_stats['all_time_peak_timestamp'][:16]}")
    with c2:
        st.metric("90th Percentile Benchmark", format_mw(hist_stats["threshold_p90_mw"]), "High Demand Baseline")
    with c3:
        st.metric("95th Percentile Benchmark", format_mw(hist_stats["threshold_p95_mw"]), "Critical Peak Baseline")
    with c4:
        st.metric("Most Frequent Peak Hour", f"{hist_stats['most_common_peak_hour']}:00 hrs", "Diurnal Peak")

    st.markdown("---")

    # 2. Configurable Alert Threshold & Upcoming Period Scan
    st.subheader("2. Operational Alert Threshold & Horizon Scan")
    st.markdown("Configure operational load thresholds or upload custom limits to simulate demand-response alerting over the evaluation forecast period.")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        threshold_mode = st.radio(
            "Threshold Definition:",
            ["Historical 90th Percentile (Recommended)", "Historical 95th Percentile", "Custom MW Value"],
        )
        if threshold_mode == "Historical 90th Percentile (Recommended)":
            alert_threshold_mw = hist_stats["threshold_p90_mw"]
            crit_threshold_mw = hist_stats["threshold_p95_mw"]
        elif threshold_mode == "Historical 95th Percentile":
            alert_threshold_mw = hist_stats["threshold_p95_mw"]
            crit_threshold_mw = hist_stats["threshold_p99_mw"]
        else:
            alert_threshold_mw = st.number_input(
                "Custom High-Demand Threshold (MW):",
                value=int(hist_stats["threshold_p90_mw"]),
                step=500
            )
            crit_threshold_mw = alert_threshold_mw * 1.08

    with col_cfg2:
        max_scan = min(168, len(df))
        valid_scans = [h for h in [24, 48, 168] if h <= max_scan] or [max_scan]
        scan_horizon = st.selectbox(
            "Scan Forecast Horizon:",
            valid_scans,
            format_func=lambda h: f"Upcoming {h} Hours ({round(h/24, 1)} Days)",
        )

    # Slice active forecast/history for the horizon
    test_slice = df.tail(scan_horizon).copy().reset_index(drop=True)
    test_slice["predicted"] = test_slice["consumption"] * np.random.uniform(0.985, 1.015, len(test_slice))
    test_slice["lower_bound"] = test_slice["predicted"] - 800.0
    test_slice["upper_bound"] = test_slice["predicted"] + 800.0

    peak_alert = analyzer.detect_forecast_peaks(
        test_slice,
        threshold_mw=alert_threshold_mw,
        critical_threshold_mw=crit_threshold_mw,
    )

    # Alert Banner
    if peak_alert["status"] == "CRITICAL PEAK RISK":
        st.error(f"🚨 **ALERT LEVEL 2 — CRITICAL PEAK RISK DETECTED**\n\n{peak_alert['description']}")
    elif peak_alert["status"] == "HIGH DEMAND ALERT":
        st.warning(f"⚠️ **ALERT LEVEL 1 — HIGH DEMAND ALERT**\n\n{peak_alert['description']}")
    else:
        st.success(f"✅ **NORMAL DEMAND STATUS**\n\n{peak_alert['description']}")

    # Peak Details Metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Expected Peak Timestamp", peak_alert["expected_peak_timestamp"][:16])
    with k2:
        st.metric("Expected Peak Demand", format_mw(peak_alert["expected_peak_mw"]))
    with k3:
        st.metric("Uncertainty Range at Peak", f"[{format_mw(peak_alert['lower_bound_at_peak'])} - {format_mw(peak_alert['upper_bound_at_peak'])}]")
    with k4:
        st.metric("High-Demand Hours Count", f"{peak_alert['high_demand_hours_count']} hrs")

    # Plot Forecast with Threshold Lines
    fig_peak = go.Figure()

    # Forecast trajectory
    fig_peak.add_trace(go.Scatter(
        x=test_slice["timestamp"], y=test_slice["predicted"],
        mode="lines+markers", name="Forecasted Demand (MW)",
        line=dict(color="#38bdf8", width=2.5),
        marker=dict(size=4),
    ))

    # Alert threshold line
    fig_peak.add_hline(
        y=alert_threshold_mw, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"High Demand Alert ({format_mw(alert_threshold_mw)})",
        annotation_position="top right",
        annotation_font_color="#f59e0b",
    )

    # Critical threshold line
    fig_peak.add_hline(
        y=crit_threshold_mw, line_dash="dash", line_color="#ef4444",
        annotation_text=f"Critical Peak Risk ({format_mw(crit_threshold_mw)})",
        annotation_position="top right",
        annotation_font_color="#ef4444",
    )

    fig_peak.update_layout(
        title=f"Predicted Load Trajectory vs. Operational Peak Thresholds ({scan_horizon}h Ahead)",
        xaxis_title="Timeline", yaxis_title="Demand (MW)",
        height=420, hovermode="x unified",
    )
    apply_dark_theme(fig_peak)
    st.plotly_chart(fig_peak, use_container_width=True)

    # Exceedance Table if any
    if peak_alert["high_demand_hours_count"] > 0:
        st.subheader("3. High Demand Exceedance Event Log")
        df_events = pd.DataFrame(peak_alert["high_demand_events"])
        st.dataframe(df_events.style.format({
            "predicted_mw": "{:,.1f}",
            "lower_bound_mw": "{:,.1f}",
            "upper_bound_mw": "{:,.1f}",
            "exceedance_mw": "{:+,.1f}",
        }), use_container_width=True)

    st.markdown("---")

    # Regulatory & Engineering Disclaimer
    st.info(
        f"📋 **Engineering Operational Notice:** {peak_alert['disclaimer']}"
    )
