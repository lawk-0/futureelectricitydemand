"""
Helper utilities: configuration loading, logging, serialization, and formatting.
"""

import os
import yaml
import logging
import joblib
from contextlib import contextmanager
import time
from typing import Any, Dict, Optional


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load project configuration YAML file safely."""
    if not os.path.exists(config_path):
        # Fallback to relative path if run from subdirectory
        alt_path = os.path.join(os.path.dirname(__file__), "..", "..", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_logger(name: str = "EnergyForecaster", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


@contextmanager
def timed_block(task_name: str, logger: Optional[logging.Logger] = None):
    """Context manager to measure and log execution time of code blocks."""
    start_time = time.perf_counter()
    if logger:
        logger.info(f"Starting {task_name}...")
    yield
    elapsed = time.perf_counter() - start_time
    msg = f"Completed {task_name} in {elapsed:.3f} seconds"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def save_artifact(obj: Any, file_path: str) -> str:
    """Safely saves Python object / model artifact using joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)
    return file_path


def load_artifact(file_path: str) -> Any:
    """Loads a previously saved artifact using joblib."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact not found: {file_path}")
    return joblib.load(file_path)


def format_mw(val: float) -> str:
    """Formats numeric energy demand in Megawatts with commas."""
    return f"{val:,.1f} MW"

