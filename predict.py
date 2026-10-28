"""HackSphere 3.0 - predict.py

Calls the published Zoho Catalyst QuickML endpoint to predict demand (kW).

    from predict import predict          # predict(df) -> numpy array, one value per row
    python predict.py --input sample_test_input.csv --output predictions.csv

Self-contained on purpose (feature code is inlined) so it works if only this file is copied.
Configuration comes from environment variables or a .env file next to this script
(see .env.example). Secrets are NEVER hard-coded.
If anything fails it raises a clear error instead of returning fake predictions.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

try:  # optional: load a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Paste your published endpoint URL here (it is not a secret), or set QUICKML_ENDPOINT_URL.
QUICKML_ENDPOINT_URL = "PASTE_YOUR_PUBLISHED_QUICKML_ENDPOINT_URL_HERE"

WEATHER = ["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
FALLBACKS = {"ambient_temp_c": 18.5, "humidity_pct": 70.7, "solar_irradiance_wm2": 8.0}
# Must match the columns (and names) of train_features.csv used to train the QuickML model
FEATURES = WEATHER + [
    "hour", "day_of_week", "is_weekend", "is_evening",
    "hour_sin", "hour_cos", "cooling_degrees",
]


# ----------------------------------------------------------------- features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in WEATHER:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].interpolate(limit_direction="both").fillna(FALLBACKS[c])
    out["humidity_pct"] = out["humidity_pct"].clip(0, 100)
    out["solar_irradiance_wm2"] = out["solar_irradiance_wm2"].clip(lower=0)
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_evening"] = out["hour"].between(18, 22).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["cooling_degrees"] = (out["ambient_temp_c"] - 24.0).clip(lower=0)
    return out[FEATURES]


# ------------------------------------------------------------- QuickML client
def _need(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing environment variable {name} (see .env.example)")
    return val


class _BatchUnsupported(Exception):
    pass


class _Client:
    def __init__(self):
        self.url = os.getenv("QUICKML_ENDPOINT_URL", QUICKML_ENDPOINT_URL)
        if "PASTE_" in self.url:
            raise RuntimeError("Set QUICKML_ENDPOINT_URL (env var or constant in predict.py)")
        self.key = _need("QUICKML_ENDPOINT_KEY")
        self.org = _need("CATALYST_ORG")
        self.env = os.getenv("QUICKML_ENVIRONMENT", "Development")
        self.accounts = os.getenv("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.in").rstrip("/")
        self._lock = threading.Lock()
        self._token = os.getenv("ZOHO_ACCESS_TOKEN") or None
        if self._token is None:
            self._refresh(None)

    def _refresh(self, old_token):
        with self._lock:
            if self._token is not None and self._token != old_token:
                return  # another thread already refreshed
            r = requests.post(
                f"{self.accounts}/oauth/v2/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": _need("ZOHO_CLIENT_ID"),
                    "client_secret": _need("ZOHO_CLIENT_SECRET"),
                    "refresh_token": _need("ZOHO_REFRESH_TOKEN"),
                },
                timeout=30,
            )
            body = r.json() if r.content else {}
            if "access_token" not in body:
                raise RuntimeError(f"Could not refresh Zoho access token: {body.get('error', r.status_code)}")
            self._token = body["access_token"]

    def call(self, payload, attempts=3):
        refreshed = False
        last_err = None
        for i in range(attempts + 1):
            token = self._token
            headers = {
                "X-QUICKML-ENDPOINT-KEY": self.key,
                "Authorization": f"Zoho-oauthtoken {token}",
                "CATALYST-ORG": self.org,
                "Environment": self.env,
                "Content-Type": "application/json",
            }
            try:
                r = requests.post(self.url, headers=headers, json=payload, timeout=120)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                time.sleep(1.5 * (i + 1))
                continue
            if r.status_code == 401 and not refreshed:
                refreshed = True
                self._refresh(token)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last_err = requests.HTTPError(f"HTTP {r.status_code}", response=r)
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            return _parse(r.json())
        raise RuntimeError(f"QuickML call failed after retries: {last_err}")


def _to_float(x):
    if isinstance(x, dict):  # tolerate {"prediction": 123.4}-style items
        for v in x.values():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        raise ValueError(f"No numeric value in {x}")
    return float(x)


def _parse(body):
    res = body.get("result", body.get("predictions")) if isinstance(body, dict) else body
    if res is None:
        raise ValueError(f"Unexpected response format: {str(body)[:200]}")
    if not isinstance(res, list):
        res = [res]
    return [_to_float(x) for x in res]


def _predict_batch(client, rows, chunk=200):
    out = []
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        vals = client.call({"data": part})
        if len(vals) != len(part):
            raise _BatchUnsupported(f"sent {len(part)} rows, got {len(vals)} values")
        out.extend(vals)
    return out


def _predict_per_row(client, rows):
    def one(row):
        vals = client.call({"data": row})
        if len(vals) != 1:
            raise RuntimeError(f"Expected 1 value for one row, got {len(vals)}")
        return vals[0]

    workers = int(os.getenv("QUICKML_WORKERS", "6"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, rows))  # map keeps the input order


# ------------------------------------------------------------------- public
def predict(input_df: pd.DataFrame):
    """input_df columns: timestamp, ambient_temp_c, humidity_pct, solar_irradiance_wm2.
    Returns a numpy array with one predicted demand (kW) per input row, in order."""
    missing = [c for c in ["timestamp"] + WEATHER if c not in input_df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    if len(input_df) == 0:
        return np.array([], dtype=float)

    feats = build_features(input_df.reset_index(drop=True))
    rows = json.loads(feats.to_json(orient="records"))  # plain Python numbers, ints stay ints
    client = _Client()
    try:
        vals = _predict_batch(client, rows)
    except (_BatchUnsupported, ValueError, requests.HTTPError) as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            raise RuntimeError(f"QuickML rejected the credentials (HTTP {status}). Check the .env values.")
        print(f"[predict.py] Batch request not accepted ({e}); sending one row at a time.", file=sys.stderr)
        vals = _predict_per_row(client, rows)

    preds = np.array(vals, dtype=float)
    if len(preds) != len(input_df) or not np.isfinite(preds).all():
        raise RuntimeError("QuickML returned a wrong number of values or NaN/inf values")
    return preds


def main():
    ap = argparse.ArgumentParser(description="Predict demand with the published QuickML model")
    ap.add_argument("--input", default="sample_test_input.csv")
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    preds = predict(df)
    out = pd.DataFrame({"timestamp": df["timestamp"], "target_demand_kw": np.round(preds, 2)})
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}. First 3:\n{out.head(3).to_string(index=False)}")


if __name__ == "__main__":
    main()