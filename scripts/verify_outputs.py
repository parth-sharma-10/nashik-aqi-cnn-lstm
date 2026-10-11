"""Independent checks on the exported results. Run from the project root after the pipeline."""
import glob

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

pred = pd.read_csv("outputs/predictions/actual_vs_predicted.csv", parse_dates=["timestamp"])
metrics = pd.read_csv("outputs/metrics/station_pollutant_metrics.csv")
splits = pd.read_csv("outputs/tables/splits.csv", parse_dates=["start", "end"])
files = {"MIDC Ambad": "midcambad", "Gangapur Road": "gangapur",
         "Pandav Nagari": "pandav", "Hirwadi": "hirwadi"}
cols = {"PM2.5": "PM2.5 (µg/m³)", "PM10": "PM10 (µg/m³)", "NO2": "NO2 (µg/m³)",
        "SO2": "SO2 (µg/m³)", "CO": "CO (mg/m³)", "O3": "Ozone (µg/m³)", "NH3": "NH3 (µg/m³)"}
START = pd.Timestamp("2025-01-01")
REPLAY_HOURS = 48


def documented_cleaning(raw):
    """Re-derive cleaned values from the raw file: >= 999 PM ceiling, >= 12 h stuck runs."""
    out = {}
    for pol, col in cols.items():
        x = raw[col].where(raw[col] >= 0)
        if pol in ("PM2.5", "PM10"):
            x = x.where(x < 999)
        run_len = x.groupby((x != x.shift()).cumsum()).transform("size")
        out[pol] = x.where(~((run_len >= 12) & x.notna()))
    return pd.DataFrame(out)


def calendar(ts):
    h, dow, mon = ts.hour.values, ts.dayofweek.values, ts.month.values - 1
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                            np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)])


def replay_forecast(station, design, hist):
    """Re-run the first hours of the 2025 forecast from data before 2025 only."""
    tag = station.replace(" ", "_")
    prep = joblib.load(f"outputs/models/preprocessing_{tag}.joblib")
    model = tf.keras.models.load_model(f"outputs/models/cnn_lstm_{design}_{tag}.keras",
                                       compile=False)
    seed_ts = pd.date_range(end=START - pd.Timedelta(hours=1), periods=24, freq="h")
    seed = hist.reindex(seed_ts)
    present = seed.notna().any(axis=1).values
    z = np.zeros((24, len(cols)))
    z[present] = prep["imputer"].transform(prep["scaler"].transform(seed[present]))
    window = np.column_stack([z, calendar(seed_ts)]).astype("float32")
    future = pd.date_range(START, periods=REPLAY_HOURS, freq="h")
    cal = calendar(future)
    floor = -prep["scaler"].mean_ / prep["scaler"].scale_
    out = []
    for i in range(REPLAY_HOURS):
        step = model(window[None], training=False).numpy()[0]
        if design == "change":
            step = step + window[-1, :len(cols)]
        step = np.maximum(step, floor)
        out.append(step)
        window = np.vstack([window[1:], np.concatenate([step, cal[i]])[None]]).astype("float32")
    return pd.DataFrame(np.maximum(np.array(out) * prep["scaler"].scale_ + prep["scaler"].mean_, 0),
                        index=future, columns=list(cols))


for station, frag in files.items():
    path = next(p for p in glob.glob("data/*") if frag in p.lower().replace("_", ""))
    raw = pd.read_csv(path, parse_dates=["Timestamp"]).set_index("Timestamp").sort_index()
    clean = documented_cleaning(raw)
    hist = clean[clean.index < START]
    sp = splits[splits.station == station].set_index("split")
    p = pred[pred.station == station]

    # 1. Every evaluated hour is in 2025; training and validation end before 2025.
    assert (p.timestamp.dt.year == 2025).all(), station
    assert sp.loc["train", "end"] < sp.loc["val", "start"] <= sp.loc["val", "end"] < START

    # 2. The CNN-LSTM forecast depends only on pre-2025 data: replaying it from the raw
    #    file truncated at 2024-12-31 reproduces the exported forecast.
    design = p.selected_design.iloc[0]
    replay = replay_forecast(station, design, hist)
    early = p[p.timestamp < START + pd.Timedelta(hours=REPLAY_HOURS)]
    expected = [replay.loc[t, pol] for t, pol in zip(early.timestamp, early.pollutant)]
    assert np.allclose(early.cnn_lstm_pred, expected, rtol=1e-3, atol=1e-3), station

    by_month_hour = hist.groupby([hist.index.month, hist.index.hour]).mean()
    by_month = hist.groupby(hist.index.month).mean()
    for pol, col in cols.items():
        q = p[p.pollutant == pol].set_index("timestamp")
        # 3. Actual values are the raw file values at the same station/hour/pollutant.
        assert np.allclose(q.actual, raw.loc[q.index, col]), (station, pol)
        # 4. Baselines re-derived from the raw file, using data before 2025 only.
        assert np.allclose(q.persistence_pred, hist[pol].dropna().iloc[-1]), (station, pol)
        clim = by_month_hour[pol].reindex(list(zip(q.index.month, q.index.hour))).values
        clim = np.where(np.isnan(clim), by_month[pol].reindex(q.index.month).values, clim)
        assert np.allclose(q.climatology_pred, clim), (station, pol)
        last_year = hist[pol].reindex(q.index - pd.Timedelta(days=364)).values
        assert np.allclose(q.same_hour_last_year_pred, np.where(np.isnan(last_year), clim, last_year))

        # 5. Metrics recomputed by hand from the exported predictions.
        m = metrics[(metrics.station == station) & (metrics.pollutant == pol)].iloc[0]
        for name, column in (("CNN-LSTM", "cnn_lstm_pred"), ("Climatology", "climatology_pred")):
            err = q[column] - q.actual
            mse = float(np.mean(err ** 2))
            r2 = 1 - np.sum(err ** 2) / np.sum((q.actual - q.actual.mean()) ** 2)
            assert np.isclose(mse, m[f"{name} MSE"]) and np.isclose(np.sqrt(mse), m[f"{name} RMSE"])
            assert np.isclose(np.mean(np.abs(err)), m[f"{name} MAE"])
            # float32 predictions round-trip through CSV at ~7 significant digits; an absolute
            # tolerance keeps near-zero R2 values from failing on that alone.
            assert np.isclose(r2, m[f"{name} R2"], rtol=0, atol=1e-5), (station, pol, name, r2)
        assert m["n_test_obs"] == len(q)

    # 6. Scaler was fitted on training rows only: its mean equals the train-period mean.
    prep = joblib.load(f"outputs/models/preprocessing_{station.replace(' ', '_')}.joblib")
    train = clean.loc[sp.loc["train", "start"]:sp.loc["train", "end"]]
    assert np.allclose(prep["scaler"].mean_, train.mean().values, rtol=1e-6), station
    print(f"{station:14s} OK  2025 rows={len(p)}  design={design}  "
          f"forecast replayed for {REPLAY_HOURS} h from pre-2025 data")
print("All checks passed.")
