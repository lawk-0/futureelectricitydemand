# Dataset Directory

## Structure
- `raw/`: Contains original, unmodified datasets.
  - `PJME_hourly.csv`: PJM East (Pennsylvania-New Jersey-Maryland Interconnection) hourly energy consumption dataset (MW) from 2002 to 2018 (145,366 records).
- `processed/`: Contains preprocessed, chronologically ordered, gap-filled, and feature-engineered datasets ready for model training and evaluation.

## Provenance
- Source: PJM Interconnection LLC via Kaggle Hourly Energy Consumption repository (Rob Mulla).
- Sampling Frequency: 1 Hour (hourly timestamps).
- Units: Megawatts (MW).

