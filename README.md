# Smart Energy Microgrid and Demand Forecasting Engine

HackSphere 3.0 submission by **<TEAM NAME>**.

- **Live app:** <LIVE_APP_URL>
- **Model:** Zoho Catalyst QuickML (XGBoost regression), published as a REST endpoint
- **Hosting:** Zoho Catalyst AppSail (Python 3.13, Flask)

The app forecasts hourly energy demand (kW) for a 24-hour window, then turns each forecast hour into a power dispatch signal: **Main Grid Mode** or **Renewable Storage Mode**, plus a **Critical Overload Alert**. A dashboard shows the load curve, hourly grid states, a simulated battery state of charge, alerts and a system log.

## Architecture

```
CSV upload / sample data
        |
   Flask backend (AppSail) -- clean + validate -- build features
        |
   QuickML endpoint (OAuth, REST)  -->  predicted demand (kW) per hour
        |
   Transition controller (rules) + battery simulation
        |
   Dashboard (Chart.js): load curve, 24 mode blocks, battery chart, alerts, log
```

`predict.py` (repo root) calls the same QuickML endpoint for batch evaluation.

## Model

- **Algorithm:** XGB Regressor trained in QuickML (Prediction pipeline) on `train_telemetry.csv` (4000 hourly rows).
- **Features (10):** `ambient_temp_c`, `humidity_pct`, `solar_irradiance_wm2`, `hour`, `day_of_week`, `is_weekend`, `is_evening` (hour 18-22), `hour_sin`, `hour_cos`, `cooling_degrees` (max(0, temp - 24)).
  Built from the timestamp and weather columns, so no past demand is needed at inference. No month/day-of-year feature, because the test period follows the training period in time.
- **QuickML evaluation metrics:** RMSE 8.64 kW, MAE 6.88 kW, R2 0.970, MAPE 4.9%.
- **Stricter check:** an independent local gradient-boosting model on a chronological holdout (last 20% of the training data) gave about 15 kW RMSE, versus 72.8 kW for predicting the mean.
- Feature code lives in `features.py` and is inlined in `predict.py` and `appsail/app.py`. Keep them identical to the training columns in `train_features.csv` (made by `prepare_data.py`).

## Transition controller rules

| Condition | Result |
|---|---|
| Demand > 160 kW, or hour between 18:00 and 22:00 inclusive | Renewable Storage Mode |
| Otherwise | Main Grid Mode |
| Demand >= 200 kW (independent of mode) | Critical Overload Alert |

## Battery simulation (assumptions)

The problem statement does not define the battery, so the dashboard simulates one (constants at the top of `appsail/app.py`): 1000 kWh capacity, max discharge 100 kW in Storage Mode, recharge 100 kW in Main Grid Mode, 10% floor, adjustable start charge (default 90%). This is an illustration, not real battery data.

## Repository contents

| Path | Purpose |
|---|---|
| `predict.py` | Batch prediction through the QuickML endpoint (`predict(df)` and CLI) |
| `test_eval.py` | Organizer self-test |
| `features.py`, `prepare_data.py` | Feature engineering, data check, builds `train_features.csv` for QuickML |
| `get_tokens.py` | Helper to exchange a Zoho grant code for a refresh token |
| `appsail/app.py` | Flask backend: cleaning, QuickML call, rules, battery simulation |
| `appsail/templates/index.html` | Dashboard |
| `appsail/app-config.json` | AppSail config (start command, no secrets) |
| `.env.example` | Names of the required environment variables |

## Environment variables (names only; never commit values)

| Variable | Meaning |
|---|---|
| `QUICKML_ENDPOINT_URL` | Published QuickML endpoint URL |
| `QUICKML_ENDPOINT_KEY` | Endpoint key (header `X-QUICKML-ENDPOINT-KEY`) |
| `CATALYST_ORG` | Catalyst org id (header `CATALYST-ORG`) |
| `QUICKML_ENVIRONMENT` | `Development` or `Production` |
| `ZOHO_ACCOUNTS_URL` | `https://accounts.zoho.in` (India data center) |
| `ZOHO_CLIENT_ID`, `ZOHO_CLIENT_SECRET` | Zoho API Console Self Client |
| `ZOHO_REFRESH_TOKEN` | Refresh token with scope `QuickML.deployment.READ` |

Copy `.env.example` to `.env` and fill it in. Access tokens are refreshed automatically.

## Run locally

```
python -m pip install pandas numpy requests flask python-dotenv scikit-learn
python prepare_data.py          # checks data, writes train_features.csv
python predict.py --input sample_test_input.csv --output predictions.csv
python test_eval.py             # should print SUCCESS
cd appsail && python app.py     # dashboard at http://127.0.0.1:5000 (needs sample_test_input.csv in appsail/)
```

## Reproduce the model

1. Upload `train_features.csv` as a QuickML dataset.
2. Create a **Prediction** pipeline, target `target_demand_kw`, regression, add an XGB Regressor, execute.
3. Create an endpoint for the model and publish it. Create a Self Client in the Zoho API Console (scope `QuickML.deployment.READ`) and run `python get_tokens.py` for the refresh token.

## Deploy to Catalyst AppSail

```
cd appsail
python -m pip install -r requirements.txt -t . --only-binary=:all: --platform manylinux2014_aarch64 --platform manylinux_2_17_aarch64 --platform manylinux_2_28_aarch64 --python-version 3.13 --implementation cp --abi cp313
cd ..
catalyst login
catalyst appsail:add      # source dir: appsail, runtime Python 3.13, start command: python3 -u app.py
catalyst deploy
```

Provide the environment variables to the running app either as AppSail environment variables in the Catalyst console, or through an `appsail/runtime.env` file (same names as above; it is git-ignored and uploaded with the app).

## Limitations and next steps

- One model with default QuickML settings; tuning or AutoML comparison could lower the error further.
- Battery behavior is simulated; real battery telemetry would replace the assumptions.
- Weather inputs come from CSV; a live forecast-weather feed is a natural extension.
