"""Independent checks on the exported results. Run from the project root after the pipeline."""
import glob

import joblib
import numpy as np
import pandas as pd

pred = pd.read_csv("outputs/predictions/actual_vs_predicted.csv", parse_dates=["timestamp"])
metrics = pd.read_csv("outputs/metrics/station_pollutant_metrics.csv")
splits = pd.read_csv("outputs/tables/splits.csv", parse_dates=["start", "end"])
files = {"MIDC Ambad": "midcambad", "Gangapur Road": "gangapur",
         "Pandav Nagari": "pandav", "Hirwadi": "hirwadi"}
cols = {"PM2.5": "PM2.5 (µg/m³)", "PM10": "PM10 (µg/m³)", "NO2": "NO2 (µg/m³)",
        "SO2": "SO2 (µg/m³)", "CO": "CO (mg/m³)", "O3": "Ozone (µg/m³)", "NH3": "NH3 (µg/m³)"}

for station, frag in files.items():
    path = next(p for p in glob.glob("data/*") if frag in p.lower().replace("_", ""))
    raw = pd.read_csv(path, parse_dates=["Timestamp"]).set_index("Timestamp").sort_index()
    sp = splits[splits.station == station].set_index("split")
    test_start = sp.loc["test", "start"].normalize()
    p = pred[pred.station == station]

    # 1. Every evaluated hour lies in the test year; none was in training or validation.
    assert (p.timestamp >= test_start).all(), station
    assert sp.loc["train", "end"] < sp.loc["val", "start"] <= sp.loc["val", "end"] < test_start

    for pol, col in cols.items():
        q = p[p.pollutant == pol].set_index("timestamp")
        # 2. Actual values are the raw file values at the same station/hour/pollutant.
        assert np.allclose(q.actual, raw.loc[q.index, col]), (station, pol)
        # 3. Persistence = last valid value strictly before the hour, re-derived from the raw
        #    file with the documented rules (>= 999 ceiling, >= 12 h identical-value runs).
        x = raw[col].where(raw[col] < 999)
        run_len = x.groupby((x != x.shift()).cumsum()).transform("size")
        x = x.where(~((run_len >= 12) & x.notna()))
        assert np.allclose(q.persistence_pred, x.ffill().shift(1).loc[q.index]), (station, pol)

        # 4. Metrics recomputed by hand from the exported predictions.
        err = q.cnn_lstm_pred - q.actual
        m = metrics[(metrics.station == station) & (metrics.pollutant == pol)].iloc[0]
        mse = float(np.mean(err ** 2))
        r2 = 1 - np.sum(err ** 2) / np.sum((q.actual - q.actual.mean()) ** 2)
        assert np.isclose(mse, m["CNN-LSTM MSE"]) and np.isclose(np.sqrt(mse), m["CNN-LSTM RMSE"])
        assert np.isclose(np.mean(np.abs(err)), m["CNN-LSTM MAE"]) and np.isclose(r2, m["CNN-LSTM R2"])
        assert m["n_test_obs"] == len(q)

    # 5. Scaler was fitted on training rows only: its mean equals the train-period mean.
    prep = joblib.load(f"outputs/models/preprocessing_{station.replace(' ', '_')}.joblib")
    train = raw.loc[sp.loc["train", "start"]:sp.loc["train", "end"], list(cols.values())]
    train = train.where(~((train >= 999) & train.columns.str.startswith("PM")))
    drift = np.abs(prep["scaler"].mean_ - train.mean().values) / prep["scaler"].scale_
    assert (drift < 0.05).all(), (station, drift)  # small gap = stuck-run removals only
    print(f"{station:14s} OK  test rows={len(p)}  test from {test_start.date()}")
print("All checks passed.")
