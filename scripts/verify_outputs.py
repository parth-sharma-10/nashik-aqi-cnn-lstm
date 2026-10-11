"""Independent checks on the exported results of one mode.

Run from the project root after the pipeline:
    python scripts/verify_outputs.py backtest_2025
    python scripts/verify_outputs.py forecast_2026
Everything is re-derived from the raw station CSVs and the saved models, not from the
pipeline's in-memory state.
"""
import glob
import json
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

MODE = sys.argv[1] if len(sys.argv) > 1 else "forecast_2026"
OUT = f"outputs/{MODE}"
meta = json.load(open(f"{OUT}/run_metadata.json"))
START, END = pd.Timestamp(meta["horizon_start"]), pd.Timestamp(meta["horizon_end"])
HORIZON = pd.date_range(START, END, freq="h")
assert meta["horizon_hours"] == len(HORIZON) == 8760, (meta["horizon_hours"], len(HORIZON))

fc = pd.read_csv(f"{OUT}/predictions/recursive_forecast.csv", parse_dates=["timestamp"])
one = pd.read_csv(f"{OUT}/predictions/rolling_one_step.csv",
                  parse_dates=["timestamp", "input_window_end"])
rec_m = pd.read_csv(f"{OUT}/metrics/recursive_station_pollutant_metrics.csv")
one_m = pd.read_csv(f"{OUT}/metrics/one_step_station_pollutant_metrics.csv")
splits = pd.read_csv(f"{OUT}/tables/splits.csv", parse_dates=["start", "end"])
files = {"MIDC Ambad": "midcambad", "Gangapur Road": "gangapur",
         "Pandav Nagari": "pandav", "Hirwadi": "hirwadi"}
cols = {"PM2.5": "PM2.5 (µg/m³)", "PM10": "PM10 (µg/m³)", "NO2": "NO2 (µg/m³)",
        "SO2": "SO2 (µg/m³)", "CO": "CO (mg/m³)", "O3": "Ozone (µg/m³)", "NH3": "NH3 (µg/m³)"}
POLL = list(cols)
FORECAST_COLUMNS = ["station", "timestamp", "pollutant", "unit", "cnn_lstm_pred",
                    "selected_design", "climatology_pred", "same_hour_last_year_pred",
                    "persistence_pred", "cnn_lstm_level_pred", "cnn_lstm_change_pred", "actual",
                    "ground_truth", "training_cutoff", "month"]
ONE_STEP_COLUMNS = ["station", "timestamp", "input_window_end", "pollutant", "unit", "actual",
                    "cnn_lstm_pred", "design", "persistence_pred"]
REPLAY_HOURS = 48
ONE_STEP_SAMPLES = 20
assert list(fc.columns) == FORECAST_COLUMNS, list(fc.columns)
assert list(one.columns) == ONE_STEP_COLUMNS, list(one.columns)
assert meta["target_columns"] == cols


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


def load(station, design):
    tag = station.replace(" ", "_")
    prep = joblib.load(f"{OUT}/models/preprocessing_{tag}.joblib")
    model = tf.keras.models.load_model(f"{OUT}/models/cnn_lstm_{design}_{tag}.keras",
                                       compile=False)
    return prep, model


def scaled_imputed(prep, rows):
    present = rows.notna().any(axis=1).values
    z = np.zeros((len(rows), len(cols)))
    # Same spurious macOS Accelerate FP flags as in the pipeline; finiteness is asserted.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        z[present] = prep["imputer"].transform(prep["scaler"].transform(rows[present]))
    assert np.isfinite(z).all()
    return z


def replay_forecast(station, design, hist):
    """Re-run the first hours of the recursive forecast from pre-origin data only."""
    prep, model = load(station, design)
    seed_ts = pd.date_range(end=START - pd.Timedelta(hours=1), periods=24, freq="h")
    window = np.column_stack([scaled_imputed(prep, hist.reindex(seed_ts)),
                              calendar(seed_ts)]).astype("float32")
    cal = calendar(HORIZON[:REPLAY_HOURS])
    floor = -prep["scaler"].mean_ / prep["scaler"].scale_
    out = []
    for i in range(REPLAY_HOURS):
        step = model(window[None], training=False).numpy()[0]
        if design == "change":
            step = step + window[-1, :len(cols)]
        step = np.maximum(step, floor)
        out.append(step)
        window = np.vstack([window[1:], np.concatenate([step, cal[i]])[None]]).astype("float32")
    level = np.maximum(np.array(out) * prep["scaler"].scale_ + prep["scaler"].mean_, 0)
    return pd.DataFrame(level, index=HORIZON[:REPLAY_HOURS], columns=POLL)


def replay_one_step(station, design, clean, ts):
    """Rebuild one input window from the raw file and predict the hour `ts`."""
    prep, model = load(station, design)
    hours = pd.date_range(end=ts - pd.Timedelta(hours=1), periods=24, freq="h")
    assert hours.isin(clean.index).all(), (station, ts, "window crosses a gap")
    x = np.column_stack([scaled_imputed(prep, clean.loc[hours]), calendar(hours)])
    z = model(x[None].astype("float32"), training=False).numpy()[0]
    if design == "change":
        last = clean.loc[:hours[-1]].ffill().iloc[-1].values
        z = z + (last - prep["scaler"].mean_) / prep["scaler"].scale_
    return np.maximum(z * prep["scaler"].scale_ + prep["scaler"].mean_, 0)


def check_metrics(m, q, methods):
    assert m["n_valid"] == len(q), (m["station"], m["pollutant"], m["n_valid"], len(q))
    if q.empty:
        assert np.isnan(m["CNN-LSTM RMSE"]) and m["evaluation_period"] == "no measured hours"
        return
    for name, column in methods:
        err = q[column] - q.actual
        mse = float(np.mean(err ** 2))
        assert np.isclose(mse, m[f"{name} MSE"]) and np.isclose(np.sqrt(mse), m[f"{name} RMSE"])
        assert np.isclose(np.mean(np.abs(err)), m[f"{name} MAE"])
        if len(q) > 1 and q.actual.var() > 0:
            r2 = 1 - np.sum(err ** 2) / np.sum((q.actual - q.actual.mean()) ** 2)
            assert np.isclose(r2, m[f"{name} R2"], rtol=0, atol=1e-6), (m["station"], name, r2)
        else:
            assert np.isnan(m[f"{name} R2"]) and m["r2_flag"] == "undefined"


rng = np.random.default_rng(0)
for station, frag in files.items():
    path = next(p for p in glob.glob("data/*") if frag in p.lower().replace("_", ""))
    raw = pd.read_csv(path, parse_dates=["Timestamp"]).set_index("Timestamp").sort_index()
    clean = documented_cleaning(raw)
    hist = clean[clean.index < START]
    sp = splits[splits.station == station].set_index("split")
    f = fc[fc.station == station]

    # 1. Training and validation end before the origin.
    assert sp.loc["train", "end"] < sp.loc["val", "start"] <= sp.loc["val", "end"] < START

    # 2. Full horizon per pollutant: every hour once, in order, through the last hour.
    for pol in POLL:
        ts = f[f.pollutant == pol].timestamp
        expected_ts = pd.Series(HORIZON, name="timestamp")
        assert ts.reset_index(drop=True).equals(expected_ts), (station, pol)

    # 3. Ground truth: measured hours carry the cleaned raw value, all other hours are
    #    empty and labelled by why; no measured hour is dropped.
    in_file = HORIZON.isin(raw.index)
    for pol in POLL:
        q = f[f.pollutant == pol].set_index("timestamp")
        truth = clean[pol].reindex(HORIZON)
        assert np.allclose(q.actual, truth, equal_nan=True), (station, pol)
        expected = np.where(truth.notna(), "observed",
                            np.where(in_file, "missing_value", "no_row_in_file"))
        assert (q.ground_truth.values == expected).all(), (station, pol)

    # 4. The recursive forecast depends only on pre-origin data: replaying it from the raw
    #    file truncated at the cutoff reproduces the export.
    design = f.selected_design.iloc[0]
    replay = replay_forecast(station, design, hist)
    early = f[f.timestamp < START + pd.Timedelta(hours=REPLAY_HOURS)]
    expected = [replay.loc[t, pol] for t, pol in zip(early.timestamp, early.pollutant)]
    assert np.allclose(early.cnn_lstm_pred, expected, rtol=1e-3, atol=1e-3), station

    # 5. Baselines re-derived from the raw file, pre-origin data only.
    by_month_hour = hist.groupby([hist.index.month, hist.index.hour]).mean()
    by_month = hist.groupby(hist.index.month).mean()
    for pol in POLL:
        q = f[f.pollutant == pol].set_index("timestamp")
        last_value = hist[pol].dropna().iloc[-1]
        assert np.allclose(q.persistence_pred, last_value, atol=1e-4), (station, pol)
        clim = by_month_hour[pol].reindex(list(zip(q.index.month, q.index.hour))).values
        clim = np.where(np.isnan(clim), by_month[pol].reindex(q.index.month).values, clim)
        assert np.allclose(q.climatology_pred, clim, atol=1e-4), (station, pol)
        last_year = hist[pol].reindex(q.index - pd.Timedelta(days=364)).values
        assert np.allclose(q.same_hour_last_year_pred,
                           np.where(np.isnan(last_year), clim, last_year), atol=1e-4)

    # 6. Recursive metrics recomputed from the exported rows with ground truth only.
    for pol in POLL:
        q = f[(f.pollutant == pol) & (f.ground_truth == "observed")]
        m = rec_m[(rec_m.station == station) & (rec_m.pollutant == pol)].iloc[0]
        check_metrics(m, q, (("CNN-LSTM", "cnn_lstm_pred"), ("Climatology", "climatology_pred")))

    # 7. One-step rows: only measured horizon hours, input window ends one hour earlier,
    #    and a sample of predictions is reproduced from windows rebuilt from the raw file.
    o = one[one.station == station]
    assert (o.timestamp - o.input_window_end == pd.Timedelta(hours=1)).all()
    assert o.timestamp.between(START, END).all() and o.actual.notna().all()
    assert np.allclose(o.actual, [clean.loc[t, p] for t, p in zip(o.timestamp, o.pollutant)])
    if len(o):
        o_design = o.design.iloc[0]
        for t in rng.choice(o.timestamp.unique(), ONE_STEP_SAMPLES, replace=False):
            t = pd.Timestamp(t)
            pred = replay_one_step(station, o_design, clean, t)
            rows = o[o.timestamp == t].set_index("pollutant")
            exp = [pred[POLL.index(p)] for p in rows.index]
            assert np.allclose(rows.cnn_lstm_pred, exp, rtol=1e-3, atol=1e-3), (station, t)
    for pol in POLL:
        m = one_m[(one_m.station == station) & (one_m.pollutant == pol)].iloc[0]
        check_metrics(m, o[o.pollutant == pol],
                      (("CNN-LSTM", "cnn_lstm_pred"), ("Persistence", "persistence_pred")))

    # 8. Scaler was fitted on training rows only: its mean equals the train-period mean.
    prep = joblib.load(f"{OUT}/models/preprocessing_{station.replace(' ', '_')}.joblib")
    train = clean.loc[sp.loc["train", "start"]:sp.loc["train", "end"]]
    assert np.allclose(prep["scaler"].mean_, train.mean().values, rtol=1e-6), station

    measured = int((f.ground_truth == "observed").sum())
    print(f"{station:14s} OK  horizon {f.timestamp.min()} -> {f.timestamp.max()}  "
          f"measured cells={measured}  one-step rows={len(o)}  design={design}")
print(f"{MODE}: all checks passed.")
