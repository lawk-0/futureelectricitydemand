# Dataset Provenance & Data Engineering Documentation

**Project:** Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction  
**Dataset:** PJM Hourly Energy Consumption (PJM East / PJME)  
**Lead Author:** Member 1 (Data Engineering & EDA Lead)

---

## 1. Dataset Origin & Description

The primary dataset utilized in this project is the **PJM East (PJME) Hourly Energy Consumption Dataset**, originally collected and published by **PJM Interconnection LLC**, a regional transmission organization (RTO) coordinating the movement of wholesale electricity in all or parts of 13 states in the eastern United States (including Pennsylvania, New Jersey, and Maryland).

- **Primary Repository:** Kaggle Hourly Energy Consumption (Rob Mulla)  
  *URL:* `https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption`
- **Original Source Authority:** PJM Interconnection LLC  
  *URL:* `https://www.pjm.com/`
- **Measurement Unit:** Megawatts (MW)
- **Time Frequency:** Hourly (1-hour discrete sampling intervals)
- **Time Horizon:** January 1, 2002 01:00:00 to August 3, 2018 00:00:00 (16.6 years)
- **Total Raw Records:** 145,366 rows

---

## 2. Schema Definition

| Raw Column Name | Standardized Column Name | Data Type | Description |
|---|---|---|---|
| `Datetime` | `timestamp` | `datetime64[ns]` | Date and hourly timestamp of the observation |
| `PJME_MW` | `consumption` | `float64` | Estimated net electricity load in Megawatts (MW) |

---

## 3. Data Quality Audit & Artifact Findings

During data validation (`src/data/validation.py`), the following empirical characteristics were detected:

1. **Duplicate Timestamps (8 instances / 4 unique events):**
   - *Cause:* Daylight Saving Time (DST) autumnal clock fall-back in the US Eastern Time Zone. At 2:00 AM on the first Sunday of November, clocks fall back to 1:00 AM, creating two consecutive 1:00 AM readings.
   - *Resolution:* Preprocessing groups duplicates by timestamp and aggregates via arithmetic mean (`mean()`).
2. **Missing Time Grid Gaps (30 hours over 16.6 years):**
   - *Cause:* DST spring forward transitions (1 missing hour each March) and minor utility telemetry dropouts.
   - *Resolution:* Imputed transparently using linear interpolation for isolated gaps ($\le 4\text{ hours}$) and hour-of-week seasonal profile medians for extended intervals. No silent alterations occur.
3. **Data Integrity Score:** **99.97%**

---

## 4. Key Limitations & Environmental Constraints

- **Lack of Co-Located Weather:**
  The dataset does not contain weather observations. Because PJM East aggregates electricity demand across a vast geographical multi-state region, introducing point-source temperature data without regional population weighting would introduce geographic bias. In accordance with university research ethics, **no artificial weather data is fabricated**.
- **Non-Stationarity & Macro Trends:**
  Electricity demand exhibits long-term demographic growth, energy efficiency improvements (LED lighting, inverter HVAC), and rooftop solar self-generation.

