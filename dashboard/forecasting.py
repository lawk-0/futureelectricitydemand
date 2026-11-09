"""
Page 4: Demand Forecasting & Multi-Horizon Inference
Members 4 & 6: ML & Integration Leads
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.forecasting.inference import EnergyForecastingEngine
from src.evaluation.metrics import evaluate_forecast
from src.analytics.peak_detection import PeakDemandAnalyzer
from src.utils.helpers import format_mw


def render_forecasting():
    st.title("🔮 Multi-Step Demand Forecasting & Uncertainty Intervals")
    st.markdown("Generate real-time multi-horizon electricity consumption forecasts with probabilistic upper and lower prediction bounds.")

    try:
        df = pd.read_csv("data/processed/clean_consumption.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as e:
        st.error(f"Please ensure clean dataset is available: {e}")
        return

    # Configuration Controls
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        model_choice = st.selectbox(
            "Forecasting Model:",
            ["LightGBM Quantile (10th, 50th, 90th)", "Seasonal Naive (168h Baseline)", "Prophet / Statistical"],
        )
    with col_c2:
        horizon_choice = st.selectbox(
            "Forecast Horizon:",
            [24, 48, 168],
            format_func=lambda h: f"Next {h} Hours ({round(h/24, 1)} Days)",
        )
    with col_c3:
        eval_mode = st.radio(
            "Evaluation Scenario:",
            ["Holdout Test Evaluation (Compare vs Ground Truth)", "Future Out-of-Sample Horizon"],
        )

    engine = EnergyForecastingEngine()
    analyzer = PeakDemandAnalyzer()

    # Define training and testing datasets
    if eval_mode == "Holdout Test Evaluation (Compare vs Ground Truth)":
        # Reserve last `horizon_choice` hours for testing
        train_data = df.iloc[:-horizon_choice].tail(15000).reset_index(drop=True)
        actual_test = df.iloc[-horizon_choice:].reset_index(drop=True)
    else:
        # Train on recent data and forecast into the true future
        train_data = df.tail(15000).reset_index(drop=True)
        actual_test = None

    forecast_btn = st.button("⚡ Generate Forecast", type="primary")

    if forecast_btn or "last_forecast" in st.session_state:
        with st.spinner("Generating multi-step time series forecast..."):
            if forecast_btn:
                if "LightGBM" in model_choice:
                    forecast_df = engine.forecast_quantile_lightgbm(
                        train_df=train_data,
                        horizon=horizon_choice,
                        actual_df=actual_test,
                        apply_conformal=True,
                    )
                elif "Seasonal Naive" in model_choice:
                    forecast_df = engine.forecast_seasonal_naive(
                        train_df=train_data,
                        horizon=horizon_choice,
                        season_lag=168,
                        actual_df=actual_test,
                    )
                else:  # Prophet
                    forecast_df = engine.forecast_prophet(
                        train_df=train_data.tail(6000),
                        horizon=horizon_choice,
                        actual_df=actual_test,
                    )
                st.session_state["last_forecast"] = forecast_df
            else:
                forecast_df = st.session_state["last_forecast"]

        # Summary KPIs
        peak_idx = forecast_df["predicted"].idxmax()
        peak_val = forecast_df.loc[peak_idx, "predicted"]
        peak_time = forecast_df.loc[peak_idx, "timestamp"]
        mean_forecast = forecast_df["predicted"].mean()
        avg_mpiw = (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Expected Peak Demand", format_mw(peak_val), f"{peak_time.strftime('%b %d, %H:00')}")
        with m2:
            st.metric("Mean Forecasted Load", format_mw(mean_forecast))
        with m3:
            st.metric("Mean Interval Width (MPIW)", format_mw(avg_mpiw), "80% Uncertainty Band")
        with m4:
            if "actual" in forecast_df.columns and not forecast_df["actual"].isna().all():
                m_eval = evaluate_forecast(forecast_df["actual"], forecast_df["predicted"], forecast_df["lower_bound"], forecast_df["upper_bound"])
                st.metric("Forecast Horizon MAE", f"{m_eval['MAE']:,.1f} MW", f"PICP: {m_eval['PICP']:.1f}%")
            else:
                st.metric("Coverage Target", "80% Nominal", "Conformalized")

        # Interactive Forecast Visualization
        st.subheader("Forecast Trajectory & Uncertainty Envelope")
        fig = go.Figure()

        # Context historical tail (last 48 hours for continuity)
        context_tail = train_data.tail(48)
        fig.add_trace(go.Scatter(
            x=context_tail["timestamp"], y=context_tail["consumption"],
            mode="lines", name="Historical Context",
            line=dict(color="#7f7f7f", width=1.5, dash="dot"),
        ))

        # Upper bound (invisible line to anchor fill)
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"], y=forecast_df["upper_bound"],
            mode="lines", line=dict(width=0), showlegend=False,
            name="Upper Bound (90th)",
        ))

        # Lower bound with shaded region to upper bound
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"], y=forecast_df["lower_bound"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.22)", name="80% Prediction Interval [10th-90th]",
        ))

        # Median / Central Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"], y=forecast_df["predicted"],
            mode="lines+markers", name="Predicted Demand (50th / Median)",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=4),
        ))

        # Actual observations if available
        if "actual" in forecast_df.columns and not forecast_df["actual"].isna().all():
            fig.add_trace(go.Scatter(
                x=forecast_df["timestamp"], y=forecast_df["actual"],
                mode="lines+markers", name="Actual Demand (Ground Truth)",
                line=dict(color="#2ca02c", width=2.0),
                marker=dict(size=4, symbol="circle-open"),
            ))

        # Peak indicator marker
        fig.add_trace(go.Scatter(
            x=[peak_time], y=[peak_val],
            mode="markers+text", name="Predicted Peak",
            text=[f"Peak: {format_mw(peak_val)}"],
            textposition="top center",
            marker=dict(color="#d62728", size=11, symbol="star"),
        ))

        fig.update_layout(
            title=f"Electricity Demand Forecast: {model_choice} ({horizon_choice} Hours Ahead)",
            xaxis_title="Timeline", yaxis_title="Demand (Megawatts)",
            template="plotly_white", height=480, hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabular View and CSV Export
        st.subheader("Forecast Tabular Records & Data Export")
        st.dataframe(forecast_df.style.format({
            "predicted": "{:,.1f}",
            "lower_bound": "{:,.1f}",
            "upper_bound": "{:,.1f}",
            "actual": "{:,.1f}",
        }), use_container_width=True, height=240)

        # Download CSV
        csv_data = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv_data,
            file_name=f"electricity_demand_forecast_{horizon_choice}h.csv",
            mime="text/csv",
        )

