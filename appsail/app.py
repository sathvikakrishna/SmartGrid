"""AppSail version of the dashboard backend.

Pure Python on purpose (no pandas/numpy): compiled packages are the usual reason Python apps
break when uploaded to Catalyst from Windows. Same rules, battery model and QuickML calls
as the local app.py + predict.py. All secrets come from environment variables.
"""
import csv
import io
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
from flask import Flask, jsonify, render_template, request

try:
    from dotenv import load_dotenv

    _here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_here, "..", ".env"))  # local runs: the project-level .env
    # runtime.env is uploaded with the app (keep it OUT of git). Its values win over console variables.
    load_dotenv(os.path.join(_here, "runtime.env"), override=True)
except ImportError:
    pass

app = Flask(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(HERE, "sample_test_input.csv")
WEATHER = ["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
REQUIRED = ["timestamp"] + WEATHER
FALLBACKS = {"ambient_temp_c": 18.5, "humidity_pct": 70.7, "solar_irradiance_wm2": 8.0}

# Transition rules (problem statement)
STORAGE_THRESHOLD_KW, OVERLOAD_KW = 160, 200
EVENING_START, EVENING_END = 18, 22  # inclusive
# Battery simulation assumptions (not defined by the problem; listed in the README)
CAPACITY_KWH, MAX_DISCHARGE_KW, CHARGE_KW, SOC_FLOOR_PCT = 1000.0, 100.0, 100.0, 10.0


# ------------------------------------------------------------ data cleaning
def _num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _fill(vals, fallback):
    """Fill None by linear interpolation (nearest value at the edges). Returns (values, n_filled)."""
    idx = [i for i, v in enumerate(vals) if v is not None]
    if not idx:
        return [fallback] * len(vals), len(vals)
    out = list(vals)
    for i, v in enumerate(vals):
        if v is None:
            left = max((j for j in idx if j < i), default=None)
            right = min((j for j in idx if j > i), default=None)
            if left is None:
                out[i] = vals[right]
            elif right is None:
                out[i] = vals[left]
            else:
                out[i] = vals[left] + (vals[right] - vals[left]) * (i - left) / (right - left)
    return out, len(vals) - len(idx)


def _parse_ts(s):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot read timestamp '{s}' (use YYYY-MM-DD HH:MM:SS)")


def _features(t, temp, hum, sol):
    """Must match the training columns of train_features.csv exactly."""
    hum, sol = min(max(hum, 0.0), 100.0), max(sol, 0.0)
    dow = t.weekday()
    return {
        "ambient_temp_c": temp, "humidity_pct": hum, "solar_irradiance_wm2": sol,
        "hour": t.hour, "day_of_week": dow, "is_weekend": int(dow >= 5),
        "is_evening": int(EVENING_START <= t.hour <= EVENING_END),
        "hour_sin": math.sin(2 * math.pi * t.hour / 24), "hour_cos": math.cos(2 * math.pi * t.hour / 24),
        "cooling_degrees": max(temp - 24.0, 0.0),
    }


# ------------------------------------------------------------ QuickML client
_lock = threading.Lock()
_tok = {"v": os.getenv("ZOHO_ACCESS_TOKEN") or None, "t": time.time()}


# Catalyst's console rejects some variable names ("reserved keywords"), so every setting can also
# be provided under a plain alternative name. The original names still work (local .env).
ALIASES = {
    "QUICKML_ENDPOINT_URL": ("QML_ENDPOINT_URL",),
    "QUICKML_ENDPOINT_KEY": ("QML_ENDPOINT_KEY",),
    "CATALYST_ORG": ("QML_ORG_ID",),
    "QUICKML_ENVIRONMENT": ("QML_STAGE",),
    "ZOHO_ACCOUNTS_URL": ("OAUTH_ACCOUNTS_URL",),
    "ZOHO_CLIENT_ID": ("OAUTH_CLIENT_ID",),
    "ZOHO_CLIENT_SECRET": ("OAUTH_CLIENT_SECRET",),
    "ZOHO_REFRESH_TOKEN": ("OAUTH_REFRESH_TOKEN",),
}


def _get(name, default=None):
    for n in (name,) + ALIASES.get(name, ()):
        v = os.getenv(n)
        if v:
            return v.strip()
    return default


def _need(name):
    v = _get(name)
    if not v:
        alt = ", ".join(ALIASES.get(name, ()))
        raise RuntimeError(f"Missing environment variable {alt or name}")
    return v


def _refresh(old=None):
    with _lock:
        if _tok["v"] and _tok["v"] != old and time.time() - _tok["t"] < 3000:
            return _tok["v"]
        accounts = _get("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.in").rstrip("/")
        r = requests.post(f"{accounts}/oauth/v2/token", timeout=30, data={
            "grant_type": "refresh_token", "client_id": _need("ZOHO_CLIENT_ID"),
            "client_secret": _need("ZOHO_CLIENT_SECRET"), "refresh_token": _need("ZOHO_REFRESH_TOKEN")})
        body = r.json() if r.content else {}
        if "access_token" not in body:
            raise RuntimeError(f"Could not refresh Zoho token: {body.get('error', r.status_code)}")
        _tok.update(v=body["access_token"], t=time.time())
        return _tok["v"]


def _parse(body):
    res = body.get("result", body.get("predictions")) if isinstance(body, dict) else body
    if res is None:
        raise ValueError(f"Unexpected QuickML response: {str(body)[:150]}")
    return [float(x) for x in (res if isinstance(res, list) else [res])]


def _call(payload):
    url, refreshed = _need("QUICKML_ENDPOINT_URL"), False
    for attempt in range(4):
        token = _tok["v"] or _refresh()
        headers = {"X-QUICKML-ENDPOINT-KEY": _need("QUICKML_ENDPOINT_KEY"),
                   "Authorization": f"Zoho-oauthtoken {token}", "CATALYST-ORG": _need("CATALYST_ORG"),
                   "Environment": _get("QUICKML_ENVIRONMENT", "Development"),
                   "Content-Type": "application/json"}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 401 and not refreshed:
            refreshed = True
            _refresh(token)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return _parse(r.json())
    raise RuntimeError("QuickML call failed after retries")


def predict_rows(rows):
    """One predicted kW per feature row. Batch first; falls back to one row at a time."""
    try:
        vals = _call({"data": rows})
        if len(vals) == len(rows):
            return vals
    except (requests.HTTPError, ValueError) as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            raise RuntimeError(f"QuickML rejected the credentials (HTTP {code})")
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(lambda r: _call({"data": r})[0], rows))


# ------------------------------------------------------------ rules + battery
def classify(demand_kw, hour):
    high = demand_kw > STORAGE_THRESHOLD_KW
    evening = EVENING_START <= hour <= EVENING_END
    storage = high or evening
    if storage:
        reason = " + ".join(r for r, on in (("demand above 160 kW", high), ("evening window 18:00-22:00", evening)) if on)
    else:
        reason = "demand within limit and outside evening window"
    return storage, demand_kw >= OVERLOAD_KW, reason


def simulate_battery(rows, initial_soc_pct):
    soc = max(0.0, min(100.0, initial_soc_pct)) / 100 * CAPACITY_KWH
    floor = SOC_FLOOR_PCT / 100 * CAPACITY_KWH
    for r in rows:
        if r["storage"]:
            kwh = min(r["demand_kw"], MAX_DISCHARGE_KW, max(0.0, soc - floor))
            soc -= kwh
            r["battery_kw"] = round(kwh, 1)  # positive = discharging
        else:
            kwh = min(CHARGE_KW, CAPACITY_KWH - soc)
            soc += kwh
            r["battery_kw"] = -round(kwh, 1)  # negative = charging
        r["soc_pct"] = round(soc / CAPACITY_KWH * 100, 1)
        r["battery_low"] = bool(r["storage"] and soc <= floor + 1e-6)


def run_forecast(records, initial_soc):
    if not records:
        raise ValueError("The CSV has no rows")
    notes = []
    if len(records) > 24:
        notes.append(f"CSV has {len(records)} rows; using the first 24 (one 24-hour window).")
        records = records[:24]
    times = [_parse_ts(r["timestamp"]) for r in records]
    cols, filled = {}, 0
    for c in WEATHER:
        cols[c], n = _fill([_num(r.get(c)) for r in records], FALLBACKS[c])
        filled += n
    if filled:
        notes.append(f"Cleaned {filled} missing/invalid weather value(s) by interpolation.")
    feats = [_features(t, cols[WEATHER[0]][i], cols[WEATHER[1]][i], cols[WEATHER[2]][i]) for i, t in enumerate(times)]
    preds = predict_rows(feats)

    rows = []
    for t, p in zip(times, preds):
        demand = round(float(p), 2)
        storage, overload, reason = classify(demand, t.hour)
        rows.append({"ts": t.strftime("%Y-%m-%d %H:%M:%S"), "hh": t.strftime("%H:%M"), "hour": t.hour,
                     "demand_kw": demand, "storage": storage, "overload": overload, "reason": reason,
                     "mode": "Renewable Storage Mode" if storage else "Main Grid Mode"})
    simulate_battery(rows, initial_soc)
    peak = max(rows, key=lambda r: r["demand_kw"])
    kpis = {"peak_kw": peak["demand_kw"], "peak_time": peak["hh"],
            "avg_kw": round(sum(r["demand_kw"] for r in rows) / len(rows), 1),
            "storage_hours": sum(r["storage"] for r in rows), "overload_hours": sum(r["overload"] for r in rows),
            "min_soc_pct": min(r["soc_pct"] for r in rows)}
    config = {"capacity_kwh": CAPACITY_KWH, "max_discharge_kw": MAX_DISCHARGE_KW,
              "charge_kw": CHARGE_KW, "soc_floor_pct": SOC_FLOOR_PCT}
    return {"rows": rows, "kpis": kpis, "notes": notes, "config": config}


def _read_csv(text):
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    reader.fieldnames = [(h or "").strip() for h in (reader.fieldnames or [])]
    missing = [c for c in REQUIRED if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    return list(reader)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/forecast")
def forecast():
    try:
        initial_soc = float(request.form.get("initial_soc", 90))
        upload = request.files.get("file")
        if upload and upload.filename:
            text = upload.read().decode("utf-8", errors="replace")
        else:
            with open(SAMPLE_CSV, encoding="utf-8") as f:
                text = f.read()
        return jsonify(run_forecast(_read_csv(text), initial_soc))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # QuickML / network / auth problems
        return jsonify({"error": f"Forecast failed: {e}"}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("X_ZOHO_CATALYST_LISTEN_PORT", os.getenv("PORT", "5000"))))