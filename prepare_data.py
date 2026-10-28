"""Run from the folder containing train_telemetry.csv and sample_test_input.csv:
    python prepare_data.py
Creates train_features.csv (upload this one to QuickML) and prints baseline RMSEs.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS, WEATHER_COLS, add_features

TARGET = "target_demand_kw"


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


train = pd.read_csv("train_telemetry.csv")
test = pd.read_csv("sample_test_input.csv")

print("Train rows:", len(train), "| range:", train["timestamp"].iloc[0], "->", train["timestamp"].iloc[-1])
print("Test rows :", len(test), "| range:", test["timestamp"].iloc[0], "->", test["timestamp"].iloc[-1])
print("\nColumn ranges (train vs test) - if test goes outside train, tree models can't extrapolate:")
for c in WEATHER_COLS:
    print(f"  {c:22s} train [{train[c].min():7.2f}, {train[c].max():7.2f}]  test [{test[c].min():7.2f}, {test[c].max():7.2f}]")

df = add_features(train)
X, y = df[FEATURE_COLS], df[TARGET]

# Chronological validation: train on the first 80%, validate on the last 20% (mimics the real test split)
cut = int(len(df) * 0.8)
Xtr, Xva, ytr, yva = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

print("\nValidation RMSE (last 20% of train, chronological):")
print("  predict-the-mean        :", round(rmse(yva, np.full(len(yva), ytr.mean())), 2))
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr)
print("  Ridge (linear)          :", round(rmse(yva, ridge.predict(Xva)), 2))
gbm = HistGradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
print("  Gradient boosting (tree):", round(rmse(yva, gbm.predict(Xva)), 2))

# QuickML upload file: engineered features + target, no raw timestamp
out = df[FEATURE_COLS + [TARGET]]
out.to_csv("train_features.csv", index=False)
print("\nSaved train_features.csv with columns:", list(out.columns))
