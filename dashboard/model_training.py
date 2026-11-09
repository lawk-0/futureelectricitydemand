"""
Page 3: Model Training & Hyperparameter Configuration
Members 4 & 6: ML & Integration Leads
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.features.feature_engineering import FeatureEngineer
from src.evaluation.backtesting import chronological_split
from src.models.lightgbm_model import LightGBMDemandModel
from src.models.quantile_model import LightGBMQuantileRegressor
from src.evaluation.metrics import evaluate_forecast
from src.utils.helpers import save_artifact, timed_block


def render_model_training():
    st.title("⚙️ Model Training & Hyperparameter Configuration")
    st.markdown("Train production gradient boosted decision tree regressors with strict temporal partitioning.")

    # Load preprocessed dataset
    try:
        df = pd.read_csv("data/processed/clean_consumption.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as e:
        st.error(f"Please run data preprocessing first: {e}")
        return

    st.subheader("1. Chronological Partitioning Configuration")
    st.info("ℹ️ **Temporal Split Rule:** Future observations are strictly excluded from the training and validation partitions.")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        history_window = st.selectbox(
            "Training History Horizon:",
            [10000, 20000, 25000, len(df)],
            index=2,
            format_func=lambda x: f"{x:,} hours (~{round(x/8760, 1)} years)",
            help="Selecting a recent multi-year window ensures fast convergence and high seasonal relevancy.",
        )
    with col_t2:
        val_hours = st.slider("Validation Set Horizon (Hours):", min_value=168, max_value=1440, value=720, step=168)
    with col_t3:
        test_hours = st.slider("Holdout Test Horizon (Hours):", min_value=168, max_value=1440, value=720, step=168)

    df_subset = df.tail(history_window).reset_index(drop=True)
    train_df, val_df, test_df = chronological_split(df_subset, test_hours=test_hours, val_hours=val_hours)

    # Visualization of chronological partitions
    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(
        x=train_df["timestamp"], y=train_df["consumption"],
        mode="lines", name=f"Train Partition ({len(train_df):,} hrs)",
        line=dict(color="#1f77b4", width=1),
    ))
    fig_split.add_trace(go.Scatter(
        x=val_df["timestamp"], y=val_df["consumption"],
        mode="lines", name=f"Validation Partition ({len(val_df):,} hrs)",
        line=dict(color="#ff7f0e", width=1.5),
    ))
    fig_split.add_trace(go.Scatter(
        x=test_df["timestamp"], y=test_df["consumption"],
        mode="lines", name=f"Test Partition ({len(test_df):,} hrs)",
        line=dict(color="#2ca02c", width=1.5),
    ))
    fig_split.update_layout(
        title="Chronological Train / Validation / Test Boundary Alignment",
        xaxis_title="Timeline", yaxis_title="Demand (MW)",
        template="plotly_white", height=320, hovermode="x unified",
    )
    st.plotly_chart(fig_split, use_container_width=True)

    st.markdown("---")

    st.subheader("2. Model Architecture & Hyperparameters")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        model_type = st.selectbox(
            "Select Model Architecture:",
            ["LightGBM Quantile Regressor (10th, 50th, 90th)", "LightGBM Point Regressor (L2 Loss)"],
        )
        learning_rate = st.select_slider("Learning Rate (eta):", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
        n_estimators = st.slider("Number of Boosting Iterations (n_estimators):", 50, 500, 300, step=50)

    with col_m2:
        max_depth = st.slider("Maximum Tree Depth:", 4, 16, 8)
        num_leaves = st.slider("Number of Leaves (num_leaves):", 15, 127, 63)
        apply_conformal = st.checkbox("Enable Split Conformal Calibration", value=True, help="Guarantees finite-sample coverage at nominal confidence.")

    train_btn = st.button("🚀 Train Selected Model", type="primary")

    if train_btn:
        with st.spinner("Generating features and training gradient boosted trees..."):
            fe = FeatureEngineer()
            train_feats = fe.create_features(train_df, min_lag=1, drop_na=True)
            feature_cols = fe.get_feature_names(train_feats)
            X_train = train_feats[feature_cols]
            y_train = train_feats["consumption"]

            # Val features
            comb_val = pd.concat([train_df.tail(200), val_df], ignore_index=True)
            val_feats = fe.create_features(comb_val, min_lag=1, drop_na=True)
            val_feats = val_feats[val_feats["timestamp"].isin(val_df["timestamp"])].reset_index(drop=True)
            X_val = val_feats[feature_cols]
            y_val = val_feats["consumption"]

            # Test features
            comb_test = pd.concat([val_df.tail(200), test_df], ignore_index=True)
            test_feats = fe.create_features(comb_test, min_lag=1, drop_na=True)
            test_feats = test_feats[test_feats["timestamp"].isin(test_df["timestamp"])].reset_index(drop=True)
            X_test = test_feats[feature_cols]
            y_test = test_feats["consumption"]

            params = {
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
            }

            if "Quantile" in model_type:
                q_model = LightGBMQuantileRegressor(quantiles=[0.10, 0.50, 0.90], params=params)
                q_model.fit(X_train, y_train, X_val, y_val)
                if apply_conformal:
                    offset = q_model.calibrate_conformal(X_val, y_val, target_coverage=0.80)
                else:
                    offset = 0.0

                low_test, med_test, upp_test = q_model.predict_interval(X_test, apply_conformal=apply_conformal)
                metrics = evaluate_forecast(y_test, med_test, low_test, upp_test)
                save_artifact(q_model, "models/lightgbm_quantile.joblib")
                st.session_state["trained_quantile_model"] = q_model
                st.session_state["last_model_type"] = "quantile"

                st.success("✅ Quantile LightGBM Model successfully trained and cached!")
                st.subheader("Training & Evaluation Summary on Holdout Test Set")
                c_res1, c_res2, c_res3, c_res4 = st.columns(4)
                with c_res1:
                    st.metric("Test MAE", f"{metrics['MAE']:,.2f} MW")
                with c_res2:
                    st.metric("Test RMSE", f"{metrics['RMSE']:,.2f} MW")
                with c_res3:
                    st.metric("Empirical PICP (Coverage)", f"{metrics['PICP']:.2f}%", "Nominal: 80%")
                with c_res4:
                    st.metric("MPIW (Sharpness)", f"{metrics['MPIW']:,.2f} MW")

            else:
                pt_model = LightGBMDemandModel(params=params)
                pt_model.fit(X_train, y_train, X_val, y_val)
                preds = pt_model.predict(X_test)
                metrics = evaluate_forecast(y_test, preds)
                save_artifact(pt_model, "models/lightgbm_point.joblib")
                st.session_state["trained_point_model"] = pt_model
                st.session_state["last_model_type"] = "point"

                st.success("✅ Point LightGBM Model successfully trained and cached!")
                c_res1, c_res2, c_res3 = st.columns(3)
                with c_res1:
                    st.metric("Test MAE", f"{metrics['MAE']:,.2f} MW")
                with c_res2:
                    st.metric("Test RMSE", f"{metrics['RMSE']:,.2f} MW")
                with c_res3:
                    st.metric("Test R²", f"{metrics['R2']:.4f}")

                # Feature Importance Table & Chart
                st.subheader("Feature Importance Ranking (Total Gain)")
                df_imp = pt_model.get_feature_importance(importance_type="gain").head(15)
                fig_imp = px.bar(
                    df_imp, x="importance_pct", y="feature", orientation="h",
                    title="Top 15 Informative Features (LightGBM Split Gain %)",
                    labels={"importance_pct": "Relative Information Gain (%)", "feature": "Feature"},
                    template="plotly_white", color="importance_pct", color_continuous_scale="Viridis",
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=420)
                st.plotly_chart(fig_imp, use_container_width=True)

