"""
Generator for project research notebooks.
"""

import json
import os


def make_notebook(cells, out_path):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {out_path}")


# Notebook 1: EDA
nb1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01. Exploratory Data Analysis & Quality Auditing\n",
            "**Project:** Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction\n",
            "**Lead:** Member 1 (Data Engineering & EDA Lead)\n",
            "\n",
            "This notebook inspects the PJM Hourly Energy Consumption dataset, checks time regularity, identifies missing and duplicate timestamps, and analyzes diurnal, weekly, and annual seasonality."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from src.data.loader import DataLoader\n",
            "from src.data.validation import DataValidator\n",
            "\n",
            "loader = DataLoader()\n",
            "df = loader.load_dataset('data/raw/PJME_hourly.csv')\n",
            "print('Dataset Shape:', df.shape)\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "validator = DataValidator()\n",
            "audit = validator.validate(df)\n",
            "print('Data Integrity Score:', audit['integrity_score'], '%')\n",
            "print('Duplicate Timestamps:', audit['duplicate_timestamps_count'])\n",
            "print('Missing Grid Timestamps:', audit['missing_timestamps_count'])"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Diurnal Hourly Pattern\n",
            "df['hour'] = df['timestamp'].dt.hour\n",
            "hourly_mean = df.groupby('hour')['consumption'].mean()\n",
            "plt.figure(figsize=(10, 4))\n",
            "hourly_mean.plot(kind='line', marker='o', color='#1f77b4')\n",
            "plt.title('Diurnal Electricity Demand Profile - Mean MW by Hour')\n",
            "plt.xlabel('Hour of Day (0-23)')\n",
            "plt.ylabel('Consumption (MW)')\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.show()"
        ]
    }
]
make_notebook(nb1_cells, "notebooks/01_data_exploration.ipynb")

# Notebook 2: Feature Engineering
nb2_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 02. Feature Engineering & Temporal Leakage Prevention\n",
            "**Project:** Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction\n",
            "**Lead:** Member 2 (Feature Engineering & Validation Lead)\n",
            "\n",
            "Demonstrates cyclical trigonometric encoding, shifted lag construction, and rolling statistical features with zero future leakage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "from src.features.feature_engineering import FeatureEngineer\n",
            "\n",
            "df = pd.read_csv('data/processed/clean_consumption.csv')\n",
            "df['timestamp'] = pd.to_datetime(df['timestamp'])\n",
            "\n",
            "fe = FeatureEngineer()\n",
            "feats = fe.create_features(df.tail(5000), min_lag=1, drop_na=True)\n",
            "feature_names = fe.get_feature_names(feats)\n",
            "print(f'Generated {len(feature_names)} features:', feature_names)\n",
            "feats.head()"
        ]
    }
]
make_notebook(nb2_cells, "notebooks/02_feature_engineering.ipynb")

# Notebook 3: Baseline Models
nb3_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03. Baseline & Statistical Forecasting Models\n",
            "**Project:** Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction\n",
            "**Lead:** Member 3 (Baseline & Statistical Modeling Lead)\n",
            "\n",
            "Evaluates Seasonal Naive benchmarks (24h daily and 168h weekly lags) and Facebook Prophet Generalized Additive Models."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "from src.models.baseline import SeasonalNaiveModel\n",
            "from src.models.prophet_model import ProphetDemandModel\n",
            "from src.evaluation.metrics import evaluate_forecast\n",
            "\n",
            "df = pd.read_csv('data/processed/clean_consumption.csv')\n",
            "df['timestamp'] = pd.to_datetime(df['timestamp'])\n",
            "\n",
            "train_df = df.iloc[-1000:-168]\n",
            "test_df = df.iloc[-168:]\n",
            "\n",
            "naive = SeasonalNaiveModel(season_lag=168).fit(train_df)\n",
            "forecast_naive = naive.predict(horizon=168, start_timestamp=test_df['timestamp'].iloc[0])\n",
            "metrics_naive = evaluate_forecast(test_df['consumption'], forecast_naive['predicted'])\n",
            "print('Seasonal Naive 168h MAE:', metrics_naive['MAE'], 'MW')\n",
            "print('Seasonal Naive 168h RMSE:', metrics_naive['RMSE'], 'MW')"
        ]
    }
]
make_notebook(nb3_cells, "notebooks/03_baseline_models.ipynb")

# Notebook 4: Model Evaluation
nb4_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 04. Model Evaluation & Quantile Uncertainty Benchmarks\n",
            "**Project:** Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction\n",
            "**Lead:** Member 4 & 5 (Core ML & Analytics Leads)\n",
            "\n",
            "Compares LightGBM point and quantile regression against baselines on a 30-day (720-hour) holdout test set."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "import pandas as pd\n",
            "\n",
            "with open('reports/benchmark_metrics.json', 'r') as f:\n",
            "    benchmarks = json.load(f)\n",
            "\n",
            "df_comp = pd.DataFrame(benchmarks['models_evaluated'])\n",
            "display_cols = ['model', 'MAE', 'RMSE', 'sMAPE', 'R2', 'PICP', 'MPIW']\n",
            "print(df_comp[[c for c in display_cols if c in df_comp.columns]])"
        ]
    }
]
make_notebook(nb4_cells, "notebooks/04_model_evaluation.ipynb")

