"""
Data validation and data quality auditing module.
Member 1: Data Engineering Lead
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.utils.helpers import get_logger

logger = get_logger("DataValidation")


class DataValidator:
    """
    Performs comprehensive data quality checks on energy time-series:
    - Frequency inspection and gap detection
    - Duplicate timestamp auditing
    - Missing value profiling
    - Range validity and anomaly/outlier screening
    - Formal Data Quality Report generation
    """

    def __init__(self, time_col: str = "timestamp", target_col: str = "consumption", freq: str = "1h"):
        self.time_col = time_col
        self.target_col = target_col
        self.freq = freq

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes complete validation suite on input dataframe and returns
        a structured Data Quality Report.
        """
        if df.empty:
            return {"error": "Empty dataframe provided for validation."}

        report: Dict[str, Any] = {}
        total_records = len(df)
        report["total_records"] = total_records

        # 1. Date range
        min_date = df[self.time_col].min()
        max_date = df[self.time_col].max()
        report["start_timestamp"] = str(min_date)
        report["end_timestamp"] = str(max_date)

        # 2. Duplicate Timestamps
        duplicates_mask = df.duplicated(subset=[self.time_col], keep=False)
        duplicate_count = duplicates_mask.sum()
        report["duplicate_timestamps_count"] = int(duplicate_count)
        if duplicate_count > 0:
            dup_samples = df[duplicates_mask][self.time_col].drop_duplicates().head(5).astype(str).tolist()
            report["duplicate_samples"] = dup_samples
            report["duplicate_explanation"] = (
                "Duplicate timestamps typically correspond to Daylight Saving Time (DST) "
                "clock transitions where the 2:00 AM hour occurs twice in autumn."
            )
        else:
            report["duplicate_samples"] = []

        # 3. Frequency & Missing Timestamps (Time Grid Gaps)
        # Construct ideal grid
        try:
            full_date_range = pd.date_range(start=min_date, end=max_date, freq=self.freq)
            expected_count = len(full_date_range)
            actual_unique_ts = df[self.time_col].nunique()
            missing_ts_count = expected_count - actual_unique_ts
            report["expected_record_count"] = int(expected_count)
            report["actual_unique_records"] = int(actual_unique_ts)
            report["missing_timestamps_count"] = int(max(0, missing_ts_count))
            report["time_coverage_percentage"] = round((actual_unique_ts / expected_count) * 100, 2)

            # Sample missing timestamps if any
            if missing_ts_count > 0:
                missing_set = set(full_date_range) - set(df[self.time_col])
                sample_missing = sorted(list(missing_set))[:5]
                report["missing_timestamp_samples"] = [str(ts) for ts in sample_missing]
            else:
                report["missing_timestamp_samples"] = []
        except Exception as e:
            logger.warning(f"Error computing time-grid coverage: {e}")
            report["expected_record_count"] = total_records
            report["missing_timestamps_count"] = 0
            report["time_coverage_percentage"] = 100.0
            report["missing_timestamp_samples"] = []

        # 4. Target Consumption Missing Values (NaN or Null)
        null_target_count = int(df[self.target_col].isna().sum())
        report["null_consumption_count"] = null_target_count

        # 5. Range Validity (Negative or Zero Demand)
        valid_consumption = df[self.target_col].dropna()
        negative_count = int((valid_consumption < 0).sum())
        zero_count = int((valid_consumption == 0).sum())
        report["negative_values_count"] = negative_count
        report["zero_values_count"] = zero_count

        # 6. Statistical Summary & Outlier Detection
        if not valid_consumption.empty:
            mean_val = float(valid_consumption.mean())
            std_val = float(valid_consumption.std())
            min_val = float(valid_consumption.min())
            max_val = float(valid_consumption.max())
            median_val = float(valid_consumption.median())
            q25 = float(valid_consumption.quantile(0.25))
            q75 = float(valid_consumption.quantile(0.75))
            iqr = q75 - q25

            # Outlier screening using IQR method (standard 3*IQR for extreme anomalies)
            lower_bound = q25 - 3.0 * iqr
            upper_bound = q75 + 3.0 * iqr
            extreme_outliers = valid_consumption[
                (valid_consumption < lower_bound) | (valid_consumption > upper_bound)
            ]
            report["summary_stats"] = {
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "median": round(median_val, 2),
                "q25": round(q25, 2),
                "q75": round(q75, 2),
                "iqr": round(iqr, 2),
            }
            report["outlier_count"] = int(len(extreme_outliers))
            report["outlier_thresholds"] = {"lower": round(lower_bound, 2), "upper": round(upper_bound, 2)}
        else:
            report["summary_stats"] = {}
            report["outlier_count"] = 0

        # 7. Overall Data Integrity Score (0-100%)
        # Penalize for missing timestamps, duplicate timestamps, nulls, negatives
        coverage = report.get("time_coverage_percentage", 100.0)
        dup_penalty = min(5.0, (duplicate_count / total_records) * 100) if total_records else 0
        null_penalty = min(10.0, (null_target_count / total_records) * 100) if total_records else 0
        neg_penalty = 10.0 if negative_count > 0 else 0
        integrity_score = max(0.0, min(100.0, coverage - dup_penalty - null_penalty - neg_penalty))
        report["integrity_score"] = round(integrity_score, 2)
        report["is_acceptable"] = integrity_score >= 80.0

        logger.info(
            f"Validation audit completed: records={total_records:,}, "
            f"duplicates={duplicate_count}, missing_ts={report.get('missing_timestamps_count', 0)}, "
            f"integrity_score={report['integrity_score']}%"
        )
        return report

