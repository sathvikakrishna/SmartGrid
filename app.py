"""Smart Energy Microgrid dashboard backend (Flask).

Run locally:  python app.py   ->  http://127.0.0.1:5000
Forecasts come from the published QuickML endpoint through predict.py (same .env / env vars).
"""
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request

from predict import predict

app = Flask(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(HERE, "sample_test_input.csv")
REQUIRED = ["timestamp", "ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]

# --- Transition rules (from the problem statement) ---
STORAGE_THRESHOLD_KW = 160
OVERLOAD_KW = 200
EVENING_START, EVENING_END = 18, 22  # inclusive

# --- Battery simulation assumptions (the problem does not define them; stated in README) ---
CAPACITY_KWH = 1000.0
MAX_DISCHARGE_KW = 100.0
CHARGE_KW = 100.0
SOC_FLOOR_PCT = 10.0


def classify(demand_kw, hour):
    """Return (storage_mode, overload, reason) for one forecast hour."""
    high = demand_kw > STORAGE_THRESHOLD_KW
    evening = EVENING_START <= hour <= EVENING_END
    storage = high or evening
    if storage:
        reason = " + ".join(
            r for r, on in (("demand above 160 kW", high), ("evening window 18:00-22:00", evening)) if on
        )
    else:
        reason = "demand within limit and outside evening window"
    return storage, demand_kw >= OVERLOAD_KW, reason


def simulate_battery(rows, initial_soc_pct):
    soc = max(0.0, min(100.0, initial_soc_pct)) / 100 * CAPACITY_KWH
    floor = SOC_FLOOR_PCT / 100 * CAPACITY_KWH
    for r in rows:
        if r["storage"]:  # battery supplies the load, limited by max power and remaining charge
            kwh = min(r["demand_kw"], MAX_DISCHARGE_KW, max(0.0, soc - floor))
            soc -= kwh
            r["battery_kw"] = round(kwh, 1)  # positive = discharging
        else:  # main grid mode: battery recharges
            kwh = min(CHARGE_KW, CAPACITY_KWH - soc)
            soc += kwh
            r["battery_kw"] = -round(kwh, 1)  # negative = charging
        r["soc_pct"] = round(soc / CAPACITY_KWH * 100, 1)
        r["battery_low"] = bool(r["storage"] and soc <= floor + 1e-6)


def run_forecast(df, initial_soc):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    notes = []
    if len(df) == 0:
        raise ValueError("The CSV has no rows")
    if len(df) > 24:
        notes.append(f"CSV has {len(df)} rows; using the first 24 (one 24-hour window).")
        df = df.head(24)
    df = df[REQUIRED].reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError("Some timestamps could not be read (use YYYY-MM-DD HH:MM:SS)")
    n_missing = int(df[REQUIRED[1:]].apply(pd.to_numeric, errors="coerce").isna().sum().sum())
    if n_missing:
        notes.append(f"Cleaned {n_missing} missing/invalid weather value(s) by interpolation.")

    preds = predict(df)  # QuickML via predict.py
    rows = []
    for t, p in zip(ts, preds):
        demand = round(float(p), 2)
        storage, overload, reason = classify(demand, int(t.hour))
        rows.append({
            "ts": t.strftime("%Y-%m-%d %H:%M:%S"), "hh": t.strftime("%H:%M"), "hour": int(t.hour),
            "demand_kw": demand, "storage": storage, "overload": overload, "reason": reason,
            "mode": "Renewable Storage Mode" if storage else "Main Grid Mode",
        })
    simulate_battery(rows, initial_soc)

    peak = max(rows, key=lambda r: r["demand_kw"])
    kpis = {
        "peak_kw": peak["demand_kw"], "peak_time": peak["hh"],
        "avg_kw": round(sum(r["demand_kw"] for r in rows) / len(rows), 1),
        "storage_hours": sum(r["storage"] for r in rows),
        "overload_hours": sum(r["overload"] for r in rows),
        "min_soc_pct": min(r["soc_pct"] for r in rows),
    }
    config = {"capacity_kwh": CAPACITY_KWH, "max_discharge_kw": MAX_DISCHARGE_KW,
              "charge_kw": CHARGE_KW, "soc_floor_pct": SOC_FLOOR_PCT}
    return {"rows": rows, "kpis": kpis, "notes": notes, "config": config}


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/forecast")
def forecast():
    try:
        initial_soc = float(request.form.get("initial_soc", 90))
        upload = request.files.get("file")
        df = pd.read_csv(upload) if upload and upload.filename else pd.read_csv(SAMPLE_CSV)
        return jsonify(run_forecast(df, initial_soc))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # QuickML / network / auth problems
        return jsonify({"error": f"Forecast failed: {e}"}), 502


if __name__ == "__main__":
    port = int(os.getenv("X_ZOHO_CATALYST_LISTEN_PORT", os.getenv("PORT", "5000")))
    app.run(host="0.0.0.0", port=port)