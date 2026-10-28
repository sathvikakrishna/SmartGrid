"""Shared cleaning + feature engineering.

Use the SAME functions for: training data prep, predict.py, and the web backend.
Deliberately NO month / day-of-year feature: the test period (mid-Jun to Jul) lies
after the training period, so calendar-position features would not generalise.
"""
import numpy as np
import pandas as pd

WEATHER_COLS = ["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
# Fallback values (training medians) used only if a value is missing and cannot be interpolated
FALLBACKS = {"ambient_temp_c": 18.5, "humidity_pct": 70.7, "solar_irradiance_wm2": 8.0}

FEATURE_COLS = WEATHER_COLS + [
    "hour", "day_of_week", "is_weekend", "is_evening",
    "hour_sin", "hour_cos", "cooling_degrees",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and fill missing values. Keeps row order and the original timestamp."""
    out = df.copy()
    for c in WEATHER_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].interpolate(limit_direction="both").fillna(FALLBACKS[c])
    out["humidity_pct"] = out["humidity_pct"].clip(0, 100)
    out["solar_irradiance_wm2"] = out["solar_irradiance_wm2"].clip(lower=0)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the model features added (timestamp column is kept)."""
    out = clean(df)
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_evening"] = out["hour"].between(18, 22).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["cooling_degrees"] = (out["ambient_temp_c"] - 24.0).clip(lower=0)
    return out
