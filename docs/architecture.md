# System Architecture & Design Specification

**Project:** Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction  
**System Architecture Lead:** Member 6 (System Integration & Architecture Lead)

---

## 1. Architectural Philosophy

The application follows a **Domain-Driven, Decoupled Modular Architecture** separating data engineering, feature transformation, machine learning inference, uncertainty quantification, evaluation diagnostics, and presentation.

```
energy-demand-forecaster/
├── app.py                     # Streamlit User Interface Dispatcher
├── config/settings.yaml       # Central Parameter Repository
├── data/
│   ├── raw/                   # Raw unmodified datasets (PJME_hourly.csv)
│   └── processed/             # Cleaned, continuous hourly datasets
├── src/
│   ├── data/                  # Ingestion, validation, preprocessing (Member 1)
│   ├── features/              # Non-leaking temporal feature engineering (Member 2)
│   ├── models/                # Baseline, Prophet, LightGBM, Quantiles (Members 3 & 4)
│   ├── forecasting/           # Multi-horizon strategies & inference engine (Member 4)
│   ├── evaluation/            # Metrics, backtesting, model comparison (Member 5)
│   ├── analytics/             # Peak demand detection & overload alerts (Member 5)
│   └── utils/                 # Configuration, logging, persistence helpers
├── dashboard/                 # Modular Streamlit page renderers (Member 6)
├── tests/                     # Automated Pytest suite (21 unit/integration tests)
└── docs/                      # University academic defense documentation
```

---

## 2. Component Interaction & Data Flow

```mermaid
flowchart TD
    A["Raw Dataset Ingestion\n(data/raw/PJME_hourly.csv)"] --> B["DataLoader\n(src/data/loader.py)"]
    B --> C["DataValidator\n(src/data/validation.py)"]
    C --> D["DataPreprocessor\n(src/data/preprocessing.py)"]
    D --> E["Clean Time-Grid Dataset\n(data/processed/clean_consumption.csv)"]
    
    E --> F["FeatureEngineer\n(src/features/feature_engineering.py)"]
    F --> G["Temporal Backtester\n(src/evaluation/backtesting.py)"]
    
    G --> H1["Seasonal Naive Model\n(src/models/baseline.py)"]
    G --> H2["Prophet Fourier GAM\n(src/models/prophet_model.py)"]
    G --> H3["LightGBM Quantile Regressor\n(src/models/quantile_model.py)"]
    
    H3 --> I["Conformal Calibration Layer\n(Alpha=0.80)"]
    
    H1 --> J["Inference Engine\n(src/forecasting/inference.py)"]
    H2 --> J
    I --> J
    
    J --> K["Peak Demand Analyzer\n(src/analytics/peak_detection.py)"]
    J --> L["Model Comparator & Metrics\n(src/evaluation/metrics.py)"]
    
    K --> M["Interactive Streamlit Dashboard\n(app.py & dashboard/*.py)"]
    L --> M
```

---

## 3. Core Software Engineering Patterns

1. **Strategy Pattern:** Multi-step forecasting dynamically supports **Direct Multi-Horizon** and **Recursive Multi-Step** strategies via common interfaces.
2. **Facade Pattern:** `EnergyForecastingEngine` encapsulates feature engineering, temporal lag updating, recursive inference, and ground-truth merging behind a single high-level API.
3. **Immutability & Pure Functions:** Feature transformations return new DataFrames without mutating historical input data in-place.
4. **Defensive Serialization:** Model states, fitted preprocessors, and benchmark evaluation outputs are serialized via `joblib` and `JSON` with schema versioning.

