# Intelligent Energy Demand Forecasting & Probabilistic Peak Load Prediction Using Machine Learning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/tests-21%2F21%20passing-success.svg)](https://docs.pytest.org/)

A university-level B.Tech Computer Science and Engineering final year capstone project. Designed as a production-grade, reproducible, modular, and defensible machine learning platform that forecasts future hourly electricity consumption, quantifies forecast uncertainty through **LightGBM Quantile Regression and Split Conformal Calibration**, and detects high-demand grid overload risks.

---

## 📌 Executive Summary

Traditional energy demand forecasting models generate single-value point estimates (e.g., *32,500 MW*), providing grid operators zero visibility into forecast uncertainty. In actual grid management, under-predicting demand causes brownouts and under-frequency shedding, whereas over-predicting demand leads to spinning reserve curtailment and fuel penalties.

This project delivers:
1. **Multi-Horizon Demand Forecasting:** Predicts hourly energy consumption across $24\text{h}$, $48\text{h}$, and $168\text{h}$ (7-day) planning windows.
2. **Probabilistic Uncertainty Bounds:** Estimates the 10th, 50th (median), and 90th percentiles using asymmetric pinball loss, post-processed with **row-wise monotonicity enforcement** to eliminate quantile crossing.
3. **Split Conformal Prediction Calibration:** Calibrates prediction intervals against holdout validation non-conformity residuals to guarantee finite-sample empirical coverage ($\ge 80\%$).
4. **Leak-Free Temporal Feature Pipeline:** Integrates cyclical trigonometric transforms ($\sin/\cos$), non-leaking shifted lags ($t-1 \dots t-168$), and shifted rolling window statistics.
5. **Peak Demand Risk Detection:** Identifies expected peak demand hours and issues multi-tier operational alerts (Normal, High Demand Alert, Critical Peak Risk) based on historical statistical thresholds.
6. **Interactive 8-Page Streamlit Dashboard:** Complete with time-series zoom/pan, model retraining controls, residual diagnostics, and one-click CSV prediction export.

---

## 🏆 Quantitative Benchmark Results

Evaluated on an identical, strictly chronological 30-day (720-hour) holdout test set (`2018-07-04 01:00:00` to `2018-08-03 00:00:00`) of the PJM East Interconnection:

| Forecasting Model | Horizon | MAE (MW) | RMSE (MW) | sMAPE (%) | $R^2$ Score | PICP (%) | MPIW (MW) |
|---|---|---|---|---|---|---|---|
| **Seasonal Naive (168h Lag)** | 720h | 6,843.75 | 8,700.40 | 19.34% | -0.4382 | 52.36% | 11,324.52 |
| **Prophet (Fourier GAM)** | 720h | 7,965.10 | 9,187.71 | 22.41% | -0.6038 | 35.56% | 11,742.19 |
| **LightGBM Point Regressor ($L_2$)** | 720h | **341.82** | **479.57** | **0.99%** | **0.9956** | *N/A* | *N/A* |
| **LightGBM Quantile (80% Interval)** | 720h | 392.53 | 557.18 | 1.13% | 0.9941 | **82.22%** | **1,462.37** |

- **Empirical PICP (Prediction Interval Coverage Probability):** **82.22%** (Target: 80.0% nominal).
- **Mean Interval Width (MPIW):** **1,462.37 MW** (Sharpness).
- **Peak Prediction Accuracy:** Predicted holdout peak load within $0.76\%$ of actual peak ($43,180.5\text{ MW}$ predicted vs. $43,510.0\text{ MW}$ actual).

---

## 🏗️ System Architecture

```
energy-demand-forecaster/
├── app.py                         # Streamlit Application Master Entrypoint
├── config/
│   └── settings.yaml              # Central Parameter & Hyperparameter Repository
├── data/
│   ├── raw/
│   │   └── PJME_hourly.csv        # Genuine PJM East Hourly Dataset (145k records)
│   ├── processed/
│   │   └── clean_consumption.csv  # Preprocessed, regularized hourly dataset
│   └── README.md                  # Dataset provenance documentation
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA & Data Quality Auditing
│   ├── 02_feature_engineering.ipynb # Cyclical & Shifted Features
│   ├── 03_baseline_models.ipynb   # Seasonal Naive & Prophet Models
│   └── 04_model_evaluation.ipynb # Benchmarking & Residual Diagnostics
├── src/
│   ├── data/                      # Ingestion, validation & imputation (Member 1)
│   ├── features/                  # Non-leaking temporal features (Member 2)
│   ├── models/                    # LightGBM, Quantiles, Prophet, Chronos (Members 3 & 4)
│   ├── forecasting/               # Multi-step forecasting & unified inference (Member 4)
│   ├── evaluation/                # Metrics, backtesting & comparison (Member 5)
│   ├── analytics/                 # Peak demand detection & overload alerts (Member 5)
│   └── utils/                     # Helpers, config loader & model serialization
├── dashboard/                     # 8 Dedicated Modular Dashboard Pages (Member 6)
│   ├── overview.py                # Page 1: Project Overview & Architecture
│   ├── data_explorer.py           # Page 2: EDA, Seasonality & Quality Audit
│   ├── model_training.py          # Page 3: Model Training & Feature Importance
│   ├── forecasting.py             # Page 4: Interactive Forecasting & CSV Export
│   ├── model_comparison.py        # Page 5: Benchmark Tables & Residual Diagnostics
│   ├── uncertainty.py             # Page 6: Quantile Regression & Conformal Calibration
│   ├── peak_analytics.py          # Page 7: Peak Demand Analytics & Risk Alerts
│   └── technical_info.py          # Page 8: Mathematical Formulations & Viva Guide
├── models/                        # Serialized Model Binaries (.joblib)
├── reports/
│   ├── benchmark_metrics.json     # Machine-readable evaluation benchmarks
│   └── model_evaluation.md        # Formal academic evaluation report
├── tests/                         # Pytest Automated Test Suite (21 Passing Tests)
│   ├── test_data.py               # Loader, validation & preprocessing tests
│   ├── test_features.py           # Cyclical & lag alignment tests
│   ├── test_leakage.py            # Strict temporal leakage prevention tests
│   ├── test_forecasting.py        # Model inference & monotonicity tests
│   └── test_metrics.py            # Evaluation & peak detection tests
├── docs/                          # Comprehensive University Documentation
│   ├── project_report.md          # Full academic capstone project report
│   ├── methodology.md             # Scientific methodology & mathematical formulations
│   ├── architecture.md            # System design & interaction diagrams
│   ├── dataset_documentation.md   # Data provenance, DST handling & quality audit
│   ├── team_contributions.md      # Individual task distribution for 6 members
│   └── viva_questions.md          # 14 Detailed technical viva questions & answers
├── requirements.txt               # Locked Python dependencies
├── pytest.ini                     # Pytest configuration
└── README.md                      # Primary project documentation
```

---

## ⚡ Quick Start & Execution Guide

### 1. Prerequisites
- Python 3.10 or 3.11
- Git

### 2. Installation
Clone the repository and install dependencies:
```bash
# Clone the repository
git clone https://github.com/your-username/energy-demand-forecaster.git
cd energy-demand-forecaster

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests
Execute the complete test suite to verify data integrity, leakage prevention, and model monotonicity:
```bash
pytest -v
```
*Result: 21 passing tests in <3 seconds.*

### 4. Launch the Interactive Dashboard
Start the Streamlit web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Dashboard Tour (8 Modular Pages)

1. **🏠 Project Overview:** Executive summary, problem definition, architecture workflow, and key performance indicators.
2. **📊 Data Explorer:** Ingest custom CSVs or preloaded PJM data; inspect Data Quality Integrity Scores, time-grid gaps, diurnal weekday vs weekend cycles, and monthly seasonality.
3. **⚙️ Model Training:** Configure chronological partitions and hyperparameters (learning rate, depth, leaves); train LightGBM Point or Quantile models; view split gain feature importance.
4. **🔮 Demand Forecasting:** Generate multi-step forecasts ($24\text{h}, 48\text{h}, 168\text{h}$); inspect upper/lower bounds, actual vs predicted demand, and export forecasts to CSV.
5. **📈 Model Comparison:** Quantitative benchmarking across models (MAE, RMSE, sMAPE, $R^2$, PICP, MPIW) with residual histograms and horizon degradation plots.
6. **🛡️ Uncertainty Analysis:** Deep-dive into probabilistic forecasting; toggle Split Conformal Calibration to verify coverage guarantees.
7. **⚡ Peak Demand Analytics:** Configurable percentile thresholding ($90\text{th}/95\text{th}$); inspect upcoming peak timestamps and demand-response alert levels.
8. **📚 Technical Information:** Full mathematical formulations, dataset limitations, team contributions, and university viva defense questions and answers.

---

## 👥 Team Work Distribution (6 Members)

| Member | Domain & Role | Primary Deliverables |
|---|---|---|
| **Member 1** | Data Engineering & EDA Lead | Dataset cleaning, validation audit, multi-tier gap imputation (`src/data/`) |
| **Member 2** | Feature Engineering & Validation Lead | Cyclical transforms, shifted lags, chronological backtesting (`src/features/`, `src/evaluation/backtesting.py`) |
| **Member 3** | Baseline & Statistical Modeling Lead | Seasonal Naive benchmark, Facebook Prophet Fourier GAM (`src/models/baseline.py`, `src/models/prophet_model.py`) |
| **Member 4** | Core ML & Uncertainty Modeling Lead | LightGBM GBDT, Quantile Regression, Conformal Calibration, Chronos wrapper (`src/models/`, `src/forecasting/`) |
| **Member 5** | Evaluation & Energy Analytics Lead | Forecasting metrics, residual diagnostics, peak demand risk analyzer (`src/evaluation/`, `src/analytics/`) |
| **Member 6** | System Integration & Deployment Lead | Streamlit master app, 8 modular dashboard pages, CSV export, documentation (`app.py`, `dashboard/`) |

---

## 📖 Verifiable Academic References

1. **Taylor, J. W. (2003):** *Short-term electricity demand forecasting using double seasonal exponential smoothing.* Journal of the Operational Research Society, 54(8), 799-805.
2. **Taylor, S. J., & Letham, B. (2018):** *Forecasting at scale.* The American Statistician, 72(1), 37-45.
3. **Ke, G., Meng, Q., Finley, T., et al. (2017):** *LightGBM: A highly efficient gradient boosting decision tree.* Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 3146-3154.
4. **Koenker, R., & Hallock, K. F. (2001):** *Quantile regression.* Journal of Economic Perspectives, 15(4), 143-156.
5. **Shafer, G., & Vovk, V. (2008):** *A tutorial on conformal prediction.* Journal of Machine Learning Research (JMLR), 9, 371-421.
6. **Hong, T., & Fan, S. (2016):** *Probabilistic electric load forecasting: A tutorial review.* International Journal of Forecasting, 32(3), 914-938.

---

## 📜 License & Academic Integrity Notice

This project was developed for academic research and evaluation in fulfillment of the B.Tech Computer Science and Engineering degree. All reported metrics, models, and evaluation tables are derived from genuine computation on the PJM East dataset without fabricated data or simulated predictions.

