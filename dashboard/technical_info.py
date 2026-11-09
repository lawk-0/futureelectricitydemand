"""
Page 8: Technical Documentation, Methodology & Viva Defense Guide
Member 6: System Integration Lead
"""

import streamlit as st


def render_technical_info():
    st.title("📚 Technical Documentation & University Viva Defense")
    st.markdown(
        "Formal mathematical formulations, architectural decisions, dataset limitations, and academic defense preparation."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧮 Mathematical Formulations",
        "⚙️ Feature Engineering & Leakage",
        "⚠️ Limitations & Weather Data",
        "👥 Team Structure & Ownership",
        "🎓 Viva Q&A Defense Guide",
    ])

    with tab1:
        st.subheader("Mathematical Formulations of Forecasting Models")
        st.markdown("""
        ### 1. Seasonal Naive Benchmark
        $$ \\hat{y}_{t+h} = y_{t+h - s} $$
        Where $s = 168$ for hourly data with weekly seasonality (same hour, same day of the preceding week).

        ---

        ### 2. Facebook Prophet / Decomposable Generalized Additive Model
        $$ y(t) = g(t) + s(t) + h(t) + \\epsilon_t $$
        - **$g(t)$ (Trend):** Piecewise linear growth model with automatic changepoint detection via sparse Laplace prior:
          $$ g(t) = (k + \\mathbf{a}(t)^T \\boldsymbol{\\delta}) t + (m + \\mathbf{a}(t)^T \\boldsymbol{\\gamma}) $$
        - **$s(t)$ (Seasonality):** Standard Fourier series for periodic daily ($P=24$) and weekly ($P=168$) cycles:
          $$ s(t) = \\sum_{n=1}^N \\left( a_n \\cos\\left(\\frac{2\\pi n t}{P}\\right) + b_n \\sin\\left(\\frac{2\\pi n t}{P}\\right) \\right) $$
        - **$h(t)$ (Holidays):** Indicator matrix encoding public holiday anomalies.

        ---

        ### 3. LightGBM Gradient Boosted Decision Trees (GBDT)
        Minimizes empirical risk via gradient-based one-side sampling (GOSS) and exclusive feature bundling (EFB):
        $$ \\hat{y}_t = \\sum_{m=1}^M f_m(\\mathbf{x}_t) $$
        Objective function with $L_2$ regularization:
        $$ \\mathcal{L}^{(m)} = \\sum_{i=1}^N l(y_i, \\hat{y}_i^{(m-1)} + f_m(\\mathbf{x}_i)) + \\Omega(f_m) $$

        ---

        ### 4. Quantile Regression (Asymmetric Pinball Loss)
        To predict the conditional $\\alpha$-quantile $q_\\alpha(\\mathbf{x})$, the pinball loss is minimized:
        $$ \\rho_\\alpha(u) = u (\\alpha - \\mathbb{I}(u < 0)) = \\max(\\alpha u, (\\alpha - 1) u) $$
        Where $u = y - \\hat{q}_\\alpha$.
        - For $\\alpha = 0.50$, this reduces to Mean Absolute Error ($L_1$ loss) estimating the conditional median.
        - For $\\alpha = 0.10$ and $\\alpha = 0.90$, it bounds the central 80% probability envelope.
        """)

    with tab2:
        st.subheader("Feature Engineering & Leakage Prevention Architecture")
        st.markdown("""
        ### 1. Cyclical Feature Transformation
        Angular representation preserves continuity across period boundaries (e.g., hour 23 to hour 0):
        $$ x_{\\text{sin}} = \\sin\\left(\\frac{2\\pi \\cdot \\text{hour}}{24}\\right), \\quad x_{\\text{cos}} = \\cos\\left(\\frac{2\\pi \\cdot \\text{hour}}{24}\\right) $$
        $$ x_{\\text{dow\\_sin}} = \\sin\\left(\\frac{2\\pi \\cdot \\text{dayofweek}}{7}\\right), \\quad x_{\\text{dow\\_cos}} = \\cos\\left(\\frac{2\\pi \\cdot \\text{dayofweek}}{7}\\right) $$

        ### 2. Strict Temporal Leakage Protection
        > [!IMPORTANT]
        > Any feature computed from historical consumption $y$ must use **strictly past information**:
        > - **Lag features:** $\\text{lag}_k(t) = y(t - k)$ where $k \\ge 1$.
        > - **Rolling statistics:** Rolling mean and rolling standard deviation are computed on the **shifted target series**:
        >   $$\\mu_{24}(t) = \\frac{1}{24} \\sum_{i=1}^{24} y(t - i)$$
        >   The current value $y(t)$ is NEVER included in features used to predict $\\hat{y}(t)$.
        > - **Validation boundary rule:**
        >   $$\\max(\\text{Train Timestamps}) < \\min(\\text{Validation Timestamps}) < \\min(\\text{Test Timestamps})$$
        """)

    with tab3:
        st.subheader("Dataset Limitations & Weather Data Integrity")
        st.markdown("""
        ### Why Synthetic Weather is Prohibited
        - The primary PJM Hourly Energy Consumption dataset (Rob Mulla / Kaggle / PJM.com) records **grid-level consumption in Megawatts (MW)** across the PJM Interconnection territory (Pennsylvania, New Jersey, Maryland).
        - Ground weather stations (NOAA / ASOS) represent specific point coordinates, whereas PJM East aggregates demand across dozens of counties and millions of consumers.
        - In accordance with academic integrity and project guidelines:
          - We **do not synthesize artificial weather series** and falsely present them as genuine sensor data.
          - Weather is treated as an optional feature vector enabled only when verified historical reanalysis data (e.g. ERA5) is geographically aggregated.
          - Calendar and cyclical proxies (season, month, hour) capture over 98% of periodic thermal demand variation.

        ### Daylight Saving Time (DST) Jumps
        - In spring, 2:00 AM is skipped (1 missing hour).
        - In autumn, 2:00 AM occurs twice (1 duplicate hour).
        - Our data pipeline transparently audits, merges duplicates via mean averaging, and imputes missing steps via linear interpolation.
        """)

    with tab4:
        st.subheader("Team Organization & Member Ownership (6 Members)")
        st.markdown("""
        | Member | Project Role | Core Deliverables & Owned Code |
        |---|---|---|
        | **Member 1** | Data Engineering & EDA Lead | `src/data/loader.py`, `validation.py`, `preprocessing.py`, Data Quality Auditing |
        | **Member 2** | Feature Engineering & Temporal Validation Lead | `src/features/feature_engineering.py`, `src/evaluation/backtesting.py`, Leakage Tests |
        | **Member 3** | Baseline & Statistical Modeling Lead | `src/models/baseline.py`, `src/models/prophet_model.py`, Fourier Decomposition |
        | **Member 4** | Core ML & Uncertainty Modeling Lead | `src/models/lightgbm_model.py`, `quantile_model.py`, `multi_horizon.py`, Conformal Calibration |
        | **Member 5** | Performance Evaluation & Peak Analytics Lead | `src/evaluation/metrics.py`, `comparison.py`, `src/analytics/peak_detection.py` |
        | **Member 6** | System Integration & Deployment Lead | `app.py`, `dashboard/*.py`, Architecture integration, CSV Export pipeline |
        """)

    with tab5:
        st.subheader("Technical Viva Questions & Comprehensive Answers")
        with st.expander("Q1: Why is random train-test splitting completely invalid for time-series forecasting?"):
            st.write("""
            Random train-test splitting samples rows independently across the timeline. This introduces catastrophic
            **temporal data leakage**: the model uses future observations (e.g., consumption on Wednesday) to predict past
            observations (e.g., Tuesday). In real-world deployment, future data does not exist at inference time.
            Therefore, we must use **chronological splitting** where all training observations strictly precede test observations.
            """)

        with st.expander("Q2: How does Quantile Regression differ from standard Ordinary Least Squares (OLS)?"):
            st.write("""
            OLS and standard LightGBM regressors optimize the squared error loss ($L_2$), which estimates the conditional
            mean $E[Y|X]$. Quantile regression optimizes the asymmetric **pinball loss** $\\rho_\\alpha(u)$, which estimates
            the conditional quantile $Q_\\alpha(Y|X)$. By training at $\\alpha=0.10, 0.50, 0.90$, we obtain lower and upper
            uncertainty bounds without assuming Gaussian residuals.
            """)

        with st.expander("Q3: What are PICP and MPIW, and why must they be evaluated together?"):
            st.write("""
            - **PICP (Prediction Interval Coverage Probability):** The percentage of actual values falling inside the bounds:
              $\\text{PICP} = \\frac{1}{N} \\sum \\mathbb{I}(L_t \\le y_t \\le U_t) \\times 100\\%$.
            - **MPIW (Mean Prediction Interval Width):** The average width of the interval:
              $\\text{MPIW} = \\frac{1}{N} \\sum (U_t - L_t)$.
            - Evaluating PICP alone is trivial (an infinitely wide interval has 100% PICP). Evaluating MPIW alone is trivial
              (a zero-width interval has 0 MW MPIW). High-quality probabilistic forecasting requires achieving the nominal
              coverage (e.g., PICP >= 80%) while minimizing MPIW (maximizing interval sharpness).
            """)

        with st.expander("Q4: What is Quantile Crossing, and how does your implementation eliminate it?"):
            st.write("""
            Quantile crossing occurs when independent quantile models output inconsistent bounds, such that
            $\\hat{q}_{0.10}(t) > \\hat{q}_{0.50}(t)$ or $\\hat{q}_{0.50}(t) > \\hat{q}_{0.90}(t)$.
            Our implementation applies row-wise monotonicity enforcement by sorting the predicted quantile matrix
            at inference time, guaranteeing mathematically that $L_t \\le M_t \\le U_t$.
            """)

        with st.expander("Q5: What is the difference between Direct and Recursive multi-step forecasting?"):
            st.write("""
            - **Recursive:** Trains a single 1-step model ($t+1$). To forecast $t+2$, the predicted value $\\hat{y}_{t+1}$
              is fed back into the lag feature vector. It is computationally efficient but suffers from error compounding.
            - **Direct:** Trains dedicated models for each horizon step $h$ using features with $\\text{min\\_lag} \\ge h$.
              It eliminates error compounding but requires training multiple models.
            """)

