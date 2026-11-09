# Team Structure & Individual Member Contributions

**Project Title:** Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction Using Machine Learning  
**Team Size:** 6 Members  
**Academic Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering  
**Department:** Computer Science & Engineering  

---

## Member Ownership Matrix

| Member | Primary Role | Code Modules Owned | Documentation & Test Assets Owned |
|---|---|---|---|
| **Member 1** | Data Engineering & EDA Lead | `src/data/loader.py`<br>`src/data/validation.py`<br>`src/data/preprocessing.py` | `docs/dataset_documentation.md`<br>`notebooks/01_data_exploration.ipynb`<br>`tests/test_data.py` |
| **Member 2** | Feature Engineering & Temporal Validation Lead | `src/features/feature_engineering.py`<br>`src/evaluation/backtesting.py` | `docs/methodology.md`<br>`notebooks/02_feature_engineering.ipynb`<br>`tests/test_features.py`<br>`tests/test_leakage.py` |
| **Member 3** | Baseline & Statistical Modeling Lead | `src/models/baseline.py`<br>`src/models/prophet_model.py` | `notebooks/03_baseline_models.ipynb`<br>Statistical Model Documentation |
| **Member 4** | Core ML & Uncertainty Modeling Lead | `src/models/lightgbm_model.py`<br>`src/models/quantile_model.py`<br>`src/models/chronos_model.py`<br>`src/forecasting/multi_horizon.py` | `tests/test_forecasting.py`<br>Quantile Regression Formulations |
| **Member 5** | Evaluation & Energy Analytics Lead | `src/evaluation/metrics.py`<br>`src/evaluation/comparison.py`<br>`src/analytics/peak_detection.py` | `reports/model_evaluation.md`<br>`reports/benchmark_metrics.json`<br>`tests/test_metrics.py` |
| **Member 6** | System Integration & Deployment Lead | `app.py`<br>`dashboard/*.py`<br>`src/forecasting/inference.py`<br>`src/utils/helpers.py` | `README.md`<br>`docs/architecture.md`<br>`docs/viva_questions.md`<br>`docs/project_report.md` |

---

## Detailed Task Allocations

### Member 1: Data Engineering & EDA Lead
- Ingestion of 145,366 records from PJM East dataset.
- Automatic schema column detector for arbitrary uploaded CSVs.
- Validation suite: duplicate detection (DST transitions), hourly time-grid continuity auditing, outlier IQR screening.
- Implemented multi-tier gap imputation (linear interpolation for short gaps, hour-of-week profile median for larger gaps).

### Member 2: Feature Engineering & Temporal Validation Lead
- Constructed cyclical trigonometric transforms (hour sin/cos, day of week sin/cos, month sin/cos).
- Configured non-leaking lag features ($t-1, t-2, t-24, t-48, t-168$).
- Implemented shifted rolling statistics (24h and 168h rolling mean/std shifted by $t-1$).
- Architected chronological train-validation-test splitter and rolling-origin backtesting generator.
- Authored automated tests verifying zero future leakage.

### Member 3: Baseline & Statistical Modeling Lead
- Developed Seasonal Naive benchmark forecaster with configurable daily (24h) and weekly (168h) seasonal lags.
- Implemented Facebook Prophet Generalized Additive Model with trend changepoints and daily/weekly Fourier harmonics.
- Engineered environment-robust harmonic regression fallback for sandboxed Windows execution.

### Member 4: Core ML & Uncertainty Modeling Lead
- Developed primary LightGBM point regressor with GBDT optimization and early stopping.
- Developed LightGBM Quantile Regressors for percentiles $\alpha \in \{0.10, 0.50, 0.90\}$.
- Implemented row-wise quantile monotonicity post-processing to eliminate quantile crossing.
- Implemented Split Conformal Prediction calibration to guarantee finite-sample empirical coverage.
- Structured the foundation model (Chronos-T5) experimental wrapper.

### Member 5: Evaluation & Energy Analytics Lead
- Developed forecasting error metrics: MAE, RMSE, sMAPE (zero-safe), and $R^2$.
- Implemented uncertainty metrics: Pinball Loss, PICP (coverage probability), and MPIW (sharpness).
- Developed Peak Demand Analyzer calculating all-time benchmarks and multi-tier operational load alerts (Normal, High Demand Alert, Critical Peak Risk).
- Formulated residual diagnostic analytics.

### Member 6: System Integration & Deployment Lead
- Architected Streamlit multipage application (`app.py` and 8 modular dashboard pages).
- Integrated interactive Plotly visualizations for multi-horizon forecast envelopes.
- Built one-click CSV export engine for generated predictions.
- Coordinated cross-module integration and authored comprehensive technical documentation.

