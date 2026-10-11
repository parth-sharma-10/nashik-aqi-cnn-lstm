"""Multi-horizon forecasting experiments: baselines, direct CNN-LSTM, hybrid, rolling origins.

Run from the project root (see experiments/README.md for the full sequence):
    python experiments/exp_multihorizon.py fold F2024
    python experiments/exp_multihorizon.py fold F2025
    python experiments/exp_multihorizon.py select
    python experiments/exp_multihorizon.py fold F2026
    python experiments/exp_multihorizon.py final
    python experiments/exp_multihorizon.py report

Everything is written under experiments/outputs/. The baseline pipeline's outputs/ folder is
only read (its saved CNN-LSTM checkpoints are reused for the 2025 and 2026 cutoffs).

Folds: a fold is a training cutoff (the forecast origin). Validation = the 6 months before
it; training = everything earlier. Every fitted object (scaler, imputer, networks, blend
weights, climatologies, AR coefficients) uses data before the cutoff only. F2024 and F2025
are used to select the method; F2026 is a holdout scored only after selection.
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
BASELINE_OUT = os.path.join(ROOT, "outputs")
EXP_OUT = os.path.join(ROOT, "experiments", "outputs")

SEED = 42
TARGETS = {"PM2.5": "PM2.5 (µg/m³)", "PM10": "PM10 (µg/m³)", "NO2": "NO2 (µg/m³)",
           "SO2": "SO2 (µg/m³)", "CO": "CO (mg/m³)", "O3": "Ozone (µg/m³)",
           "NH3": "NH3 (µg/m³)"}
POLL = list(TARGETS)
UNITS = {p: ("mg/m³" if p == "CO" else "µg/m³") for p in POLL}
STATION_PATTERNS = {"midcambad": "MIDC Ambad", "gangapur": "Gangapur Road",
                    "pandav": "Pandav Nagari", "hirwadi": "Hirwadi"}
STATIONS = list(STATION_PATTERNS.values())

# Baseline pipeline settings, reproduced exactly (tests compare against the baseline).
STUCK_HOURS = 12
PM_CEILING = 999
K_NEIGHBORS = 5
MAX_IMPUTED_FRACTION = 0.5
REC_LOOKBACK = 24

DIR_LOOKBACK = 168
DIR_HORIZON = 168
VAL_MONTHS = 6
MIN_TRAIN_DAYS = 180  # station must have observations >= 180 days before validation starts
MAX_TRAIN_WINDOWS = 20000
BATCH, PATIENCE = 256, 6
FOLDS = {"F2024": "2024-01-01", "F2025": "2025-01-01", "F2026": "2026-01-01"}
SELECTION_FOLDS = ("F2024", "F2025")
BASELINE_MODE = {"F2025": "backtest_2025", "F2026": "forecast_2026"}
SHORT_BUCKETS = [(1, 6), (7, 24), (25, 72), (73, 168)]
YEAR_BUCKETS = [(1, 6), (7, 24), (25, 72), (73, 168), (169, 720), (721, 2160), (2161, 4416),
                (4417, 8784)]
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
           5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "Post-monsoon", 11: "Post-monsoon"}  # IMD seasons
SHORT_METHODS = ["persistence", "daily_naive", "weekly_naive", "same_hour_last_year",
                 "clim_month_hour", "clim_doy_hour", "damped_anomaly", "cnn_lstm_recursive",
                 "direct", "hybrid_direct"]
YEAR_METHODS = ["persistence", "same_hour_last_year", "clim_month_hour", "clim_doy_hour",
                "damped_anomaly", "cnn_lstm_recursive", "direct_chained",
                "hybrid_direct_chained", "hybrid_recursive"]
NN_METHODS = {"cnn_lstm_recursive", "direct", "direct_chained"}


# ---------------------------------------------------------------- data

def find_station_files(data_dir=DATA_DIR):
    found = {}
    for path in glob.glob(f"{data_dir}/*"):
        key = os.path.basename(path).lower().replace("_", "").replace(" ", "")
        for frag, station in STATION_PATTERNS.items():
            if frag in key and path.lower().endswith(".csv"):
                found[station] = path
    missing = set(STATIONS) - set(found)
    if missing:
        raise FileNotFoundError(f"No CSV in {data_dir} for {sorted(missing)}")
    return found


def clean_station(df):
    """Baseline cleaning rules: sort, drop duplicates and leading empty rows, negatives,
    PM >= 999 and >= 12-hour identical runs set to missing."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp", kind="stable").drop_duplicates("Timestamp", keep="first")
    df = df.rename(columns={v: k for k, v in TARGETS.items()})[["Timestamp", *POLL]]
    df = df.reset_index(drop=True)
    df = df.iloc[int(df[POLL].notna().any(axis=1).idxmax()):].reset_index(drop=True)
    for c in POLL:
        neg = df[c] < 0
        ceiling = (df[c] >= PM_CEILING) if c in ("PM2.5", "PM10") else pd.Series(False, df.index)
        run_id = (df[c] != df[c].shift()).cumsum()
        stuck = (df[c].groupby(run_id).transform("size") >= STUCK_HOURS) & df[c].notna()
        df.loc[neg | ceiling | stuck, c] = np.nan
    return df


def load_clean(data_dir=DATA_DIR):
    return {s: clean_station(pd.read_csv(p)) for s, p in find_station_files(data_dir).items()}


def fold_bounds(fold):
    origin = pd.Timestamp(FOLDS[fold])
    val_start = origin - pd.DateOffset(months=VAL_MONTHS)
    end = origin + pd.DateOffset(years=1) - pd.Timedelta(hours=1)
    return origin, val_start, end


def eligible(d, fold):
    """Enough history: first observation at least MIN_TRAIN_DAYS before validation starts."""
    _, val_start, _ = fold_bounds(fold)
    return (val_start - d.Timestamp.iloc[0]) >= pd.Timedelta(days=MIN_TRAIN_DAYS)


def calendar(ts):
    ts = pd.DatetimeIndex(ts)
    h, dow, mon = ts.hour.values, ts.dayofweek.values, ts.month.values - 1
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                            np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)])


def prepare(d, fold):
    """Regular hourly grid up to the end of the fold's horizon. Scaler and KNN imputer are
    fitted on rows before validation starts; post-cutoff rows are transformed, never fitted.
    Rows absent from the file stay absent (`present` False) and are never imputed."""
    origin, val_start, end = fold_bounds(fold)
    d = d[d.Timestamp <= end]
    grid = pd.date_range(d.Timestamp.iloc[0], end, freq="h")
    raw = d.set_index("Timestamp")[POLL].reindex(grid)
    present = grid.isin(d.Timestamp)
    train = present & (grid < val_start)
    scaler = StandardScaler().fit(raw[train])
    zs = scaler.transform(raw[present])
    imputer = KNNImputer(n_neighbors=K_NEIGHBORS, weights="distance").fit(zs[train[present]])
    # macOS Accelerate BLAS raises spurious FP flags in sklearn's nan-euclidean matmul;
    # finiteness is asserted instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        zi = imputer.transform(zs)
    assert np.isfinite(zi).all()
    z = np.full((len(grid), len(POLL)), np.nan, dtype="float32")
    z[present] = zi
    obs = raw.values.astype("float64")
    imputed_row = np.where(present, np.isnan(obs).mean(axis=1), 1.0)
    any_obs = ~np.isnan(obs).all(axis=1)
    return {"grid": grid, "obs": obs, "z": z, "present": present, "scaler": scaler,
            "imputer": imputer, "cal": calendar(grid).astype("float32"),
            "cs_present": np.r_[0, np.cumsum(present)],
            "cs_imputed": np.r_[0, np.cumsum(imputed_row)],
            "cs_observed": np.r_[0, np.cumsum(any_obs)],
            "idx": lambda ts: int((pd.Timestamp(ts) - grid[0]) / pd.Timedelta(hours=1)),
            "origin": origin, "val_start": val_start, "end": end}


def valid_input(p, i, lookback):
    """Input rows [i - lookback, i) all present in the file and at most half imputed."""
    i = np.asarray(i)
    lo = i - lookback
    ok = lo >= 0
    lo = np.where(ok, lo, 0)
    full = (p["cs_present"][i] - p["cs_present"][lo]) == lookback
    frac = (p["cs_imputed"][i] - p["cs_imputed"][lo]) / lookback
    return ok & full & (frac <= MAX_IMPUTED_FRACTION)


def scaled(p, values):
    return (values - p["scaler"].mean_) / p["scaler"].scale_


def unscaled(p, values):
    return values * p["scaler"].scale_ + p["scaler"].mean_


def seed_window(p, i, lookback):
    """Input window ending before index i; hours absent from the file -> 0 (training mean),
    as in the baseline recursive forecast."""
    rows = np.arange(i - lookback, i)
    z = np.where(rows[:, None] >= 0, p["z"][np.clip(rows, 0, None)], np.nan)
    ts = p["grid"][0] + pd.to_timedelta(rows, unit="h")
    return np.column_stack([np.nan_to_num(z, nan=0.0), calendar(ts)]).astype("float32")


# ---------------------------------------------------------------- baselines

def history(p, cutoff):
    """Observed values strictly before `cutoff`, indexed by timestamp."""
    k = p["idx"](cutoff)
    return pd.DataFrame(p["obs"][:k], index=p["grid"][:k], columns=POLL)


def clim_month_hour(hist, ts):
    ts = pd.DatetimeIndex(ts)
    mh = hist.groupby([hist.index.month, hist.index.hour]).mean()
    m = hist.groupby(hist.index.month).mean()
    out = mh.reindex(pd.MultiIndex.from_arrays([ts.month, ts.hour])).set_axis(ts)
    return out.fillna(m.reindex(ts.month).set_axis(ts)).fillna(hist.mean())


def clim_doy_hour(hist, ts, half_window=15, min_years=2):
    """Mean of observations within +-15 days of the same day of year, same hour. Used only
    where those 31 days are covered by >= 2 distinct years; elsewhere month x hour."""
    ts = pd.DatetimeIndex(ts)
    fallback = clim_month_hour(hist, ts)
    doy = np.minimum(hist.index.dayofyear.values, 365) - 1
    hour = hist.index.hour.values
    years = hist.index.year.values
    out = fallback.copy()
    offsets = np.arange(-half_window, half_window + 1)
    year_ids = {y: k for k, y in enumerate(np.unique(years))}
    yk = np.array([year_ids[y] for y in years])
    t_doy = np.minimum(ts.dayofyear.values, 365) - 1
    for j, pol in enumerate(POLL):
        x = hist[pol].values
        ok = ~np.isnan(x)
        s = np.zeros((365, 24))
        c = np.zeros((365, 24))
        cover = np.zeros((365, len(year_ids)), dtype=bool)
        np.add.at(s, (doy[ok], hour[ok]), x[ok])
        np.add.at(c, (doy[ok], hour[ok]), 1)
        cover[doy[ok], yk[ok]] = True
        ss = sum(np.roll(s, -o, axis=0) for o in offsets)
        cc = sum(np.roll(c, -o, axis=0) for o in offsets)
        nyears = sum(np.roll(cover, -o, axis=0) for o in offsets).astype(bool).sum(axis=1)
        mean = np.where(cc > 0, ss / np.maximum(cc, 1), np.nan)
        val = mean[t_doy, ts.hour.values]
        use = (nyears[t_doy] >= min_years) & ~np.isnan(val)
        out[pol] = np.where(use, val, fallback[pol].values)
    return out


def anomaly_phi(hist, clim):
    """Lag-1 autocorrelation of hourly anomalies from climatology, per pollutant."""
    a = hist - clim
    phi = {}
    for pol in POLL:
        x, y = a[pol].values[1:], a[pol].values[:-1]
        ok = ~np.isnan(x) & ~np.isnan(y)
        phi[pol] = float(np.clip(np.corrcoef(x[ok], y[ok])[0, 1], 0, 0.999)) if ok.sum() > 2 else 0.0
    return phi


# ---------------------------------------------------------------- networks

def masked_mse(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    err = tf.where(mask, y_true - y_pred, tf.zeros_like(y_pred))
    return tf.reduce_sum(tf.square(err)) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)


def build_cnn_lstm():
    """Baseline one-step network (unchanged)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(REC_LOOKBACK, len(POLL) + 6)),
        tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(POLL)),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=masked_mse)
    return model


def build_direct():
    """Same CNN-LSTM family, 168-hour input, all 168 x 7 future values in one output."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(DIR_LOOKBACK, len(POLL) + 6)),
        tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(DIR_HORIZON * len(POLL)),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=masked_mse)
    return model


def input_windows(p, origins, lookback):
    rows = np.asarray(origins)[:, None] + np.arange(-lookback, 0)[None, :]
    feats = np.concatenate([p["z"], p["cal"]], axis=1)
    return np.nan_to_num(feats[rows], nan=0.0).astype("float32")


def target_windows(p, origins, horizon):
    """Scaled observed targets for hours [i, i + horizon); NaN when missing or past the grid."""
    rows = np.asarray(origins)[:, None] + np.arange(horizon)[None, :]
    y = np.full(rows.shape + (len(POLL),), np.nan)
    inside = rows < len(p["grid"])
    y[inside] = scaled(p, p["obs"][rows[inside]])
    return y.astype("float32")


def direct_origins(p, start, stop, stride=1):
    """Origins i with a valid 168-h input window whose whole 168-h target window lies in
    [start, stop) and holds at least one observation."""
    i = np.arange(max(p["idx"](start), DIR_LOOKBACK), p["idx"](stop) - DIR_HORIZON + 1, stride)
    i = i[valid_input(p, i, DIR_LOOKBACK)]
    has_target = p["cs_observed"][i + DIR_HORIZON] - p["cs_observed"][i] > 0
    return i[has_target]


def fit(model, x, y, xv, yv, epochs):
    hist = model.fit(x, y, validation_data=(xv, yv), epochs=epochs, batch_size=BATCH, verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE,
                                                                 restore_best_weights=True)])
    return hist.history


def train_direct(p, epochs):
    tf.keras.utils.set_random_seed(SEED)
    n_all = len(direct_origins(p, p["grid"][0], p["val_start"]))
    stride = max(1, int(np.ceil(n_all / MAX_TRAIN_WINDOWS)))
    tr = direct_origins(p, p["grid"][0], p["val_start"], stride)
    va = direct_origins(p, p["val_start"], p["origin"], 3)
    flat = lambda o: target_windows(p, o, DIR_HORIZON).reshape(len(o), -1)
    model = build_direct()
    hist = fit(model, input_windows(p, tr, DIR_LOOKBACK), flat(tr),
               input_windows(p, va, DIR_LOOKBACK), flat(va), epochs)
    return model, {"train_windows": len(tr), "train_stride": stride, "val_windows": len(va),
                   "epochs_run": len(hist["loss"]),
                   "best_epoch": int(np.argmin(hist["val_loss"])) + 1,
                   "best_val_loss": float(np.min(hist["val_loss"]))}


def floor_zero(z, floor, stats=None, key=None):
    """Physical constraint: no negative concentration. `stats` counts the network outputs
    that fell below zero before the floor was applied."""
    if stats is not None:
        stats[key] = stats.get(key, 0) + int((z < floor).sum())
        stats[f"{key}_outputs"] = stats.get(f"{key}_outputs", 0) + int(z.size)
    return np.maximum(z, floor)


def predict_direct(model, p, origins, stats=None):
    """Scaled predictions (N, 168, 7), floored at zero concentration."""
    out = model(input_windows(p, origins, DIR_LOOKBACK), training=False).numpy()
    out = out.reshape(len(origins), DIR_HORIZON, len(POLL))
    return floor_zero(out, scaled(p, 0.0), stats, "direct")


def recursive_batch(model, design, p, windows, start_ts, horizon, stats=None):
    """Baseline recursive feedback for many origins at once. windows: (N, 24, 13) seeds;
    start_ts: first forecast hour of each origin. Returns scaled (N, horizon, 7)."""
    step = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    floor = scaled(p, 0.0).astype("float32")
    starts = pd.DatetimeIndex(start_ts)
    out = np.empty((len(windows), horizon, len(POLL)), dtype="float32")
    w = windows.copy()
    for h in range(horizon):
        z = step(w).numpy()
        if design == "change":
            z = z + w[:, -1, :len(POLL)]
        z = floor_zero(z, floor, stats, "cnn_lstm_recursive")
        out[:, h] = z
        cal = calendar(starts + pd.Timedelta(hours=h)).astype("float32")
        w = np.concatenate([w[:, 1:], np.concatenate([z, cal], axis=1)[:, None]], axis=1)
    return out


def chained_direct(model, p, i, horizon, stats=None):
    """Year-ahead by chaining 168-h blocks: each block's predictions become the next input."""
    w = seed_window(p, i, DIR_LOOKBACK)
    start = p["grid"][0] + pd.Timedelta(hours=int(i))
    floor = scaled(p, 0.0)
    blocks = []
    for b in range(int(np.ceil(horizon / DIR_HORIZON))):
        z = model(w[None], training=False).numpy().reshape(DIR_HORIZON, len(POLL))
        z = floor_zero(z, floor, stats, "direct_chained").astype("float32")
        blocks.append(z)
        ts = start + pd.to_timedelta(b * DIR_HORIZON + np.arange(DIR_HORIZON), unit="h")
        w = np.column_stack([z, calendar(ts)]).astype("float32")
    return np.concatenate(blocks)[:horizon]


def baseline_cnn_lstm(fold, station):
    """Saved baseline checkpoint and its recursively selected design (read-only)."""
    mode = BASELINE_MODE[fold]
    sel = pd.read_csv(f"{BASELINE_OUT}/{mode}/metrics/model_selection_validation.csv",
                      index_col="station")
    design = sel.loc[station, "selected_design"]
    path = f"{BASELINE_OUT}/{mode}/models/cnn_lstm_{design}_{station.replace(' ', '_')}.keras"
    return tf.keras.models.load_model(path, compile=False), design


def train_cnn_lstm(p, epochs):
    """Baseline recipe for folds without a saved checkpoint: 24-h windows, both target
    designs, design chosen by a recursive forecast of the validation half-year."""
    t = np.arange(REC_LOOKBACK, len(p["grid"]))
    t = t[p["present"][t] & valid_input(p, t, REC_LOOKBACK)
          & ((p["cs_present"][t + 1] - p["cs_present"][t - REC_LOOKBACK]) == REC_LOOKBACK + 1)]
    last_obs = pd.DataFrame(p["obs"]).ffill().values
    t = t[~np.isnan(p["obs"][t]).all(axis=1) & np.isfinite(last_obs[t - 1]).all(axis=1)]
    ts = p["grid"][t]
    tr, va = t[ts < p["val_start"]], t[(ts >= p["val_start"]) & (ts < p["origin"])]
    x = input_windows(p, t, REC_LOOKBACK)
    y = scaled(p, p["obs"][t]).astype("float32")
    last = scaled(p, last_obs[t - 1]).astype("float32")
    k_tr, k_va = np.isin(t, tr), np.isin(t, va)
    n_val = p["idx"](p["origin"]) - p["idx"](p["val_start"])
    best = None
    for design in ("level", "change"):
        tf.keras.utils.set_random_seed(SEED)
        target = y - last if design == "change" else y
        model = build_cnn_lstm()
        fit(model, x[k_tr], target[k_tr], x[k_va], target[k_va], epochs)
        vi = p["idx"](p["val_start"])
        pred = recursive_batch(model, design, p, seed_window(p, vi, REC_LOOKBACK)[None],
                               [p["val_start"]], n_val)[0]
        truth = scaled(p, p["obs"][vi:vi + n_val])
        mse = float(np.nanmean(np.square(truth - pred)))
        if best is None or mse < best[2]:
            best = (model, design, mse)
    return best[0], best[1]


# ---------------------------------------------------------------- blending

def blend_weight(y, nn, clim):
    """Least-squares weight w in [0, 1] for w * nn + (1 - w) * clim."""
    ok = ~np.isnan(y)
    d = (nn - clim)[ok]
    den = float(np.sum(d * d))
    return float(np.clip(np.sum((y - clim)[ok] * d) / den, 0, 1)) if den > 0 else 0.0


def bucket_of(lead, buckets):
    lead = np.asarray(lead)
    out = np.full(lead.shape, len(buckets) - 1)
    for k, (lo, hi) in reversed(list(enumerate(buckets))):
        out[(lead >= lo) & (lead <= hi)] = k
    return out


def fit_weights(actual, nn, clim, leads, buckets):
    """actual/nn/clim: (..., 7) aligned arrays in original units; leads: same leading shape.
    Returns weights (n_buckets, 7). Buckets with no validation data inherit the previous
    bucket's weight (longest validated lead)."""
    b = bucket_of(leads, buckets)
    w = np.zeros((len(buckets), len(POLL)))
    for k in range(len(buckets)):
        m = b == k
        for j in range(len(POLL)):
            if m.any() and (~np.isnan(actual[m][..., j])).any():
                w[k, j] = blend_weight(actual[m][..., j], nn[m][..., j], clim[m][..., j])
            elif k > 0:
                w[k, j] = w[k - 1, j]
    return w


def apply_weights(w, nn, clim, leads, buckets):
    wk = w[bucket_of(leads, buckets)]
    return wk * nn + (1 - wk) * clim


# ---------------------------------------------------------------- per-station fold

def baselines_for(p, cutoff, ts):
    """All non-network forecasts for target timestamps `ts` from data before `cutoff`."""
    hist = history(p, cutoff)
    cmh = clim_month_hour(hist, ts)
    cdoy = clim_doy_hour(hist, ts)
    last_year = hist.reindex(pd.DatetimeIndex(ts) - pd.Timedelta(days=364)).set_axis(
        pd.DatetimeIndex(ts)).fillna(cmh)
    return hist, cmh, cdoy, last_year


def damped(p, hist, ts, cmh_hist_fn):
    """clim(t) + phi^(t - t_last) * (y_last - clim(t_last)), from data before cutoff_idx."""
    clim_hist = cmh_hist_fn(hist.index)
    phi = anomaly_phi(hist, clim_hist)
    out = cmh_hist_fn(ts).copy()
    tsi = ((pd.DatetimeIndex(ts) - p["grid"][0]) / pd.Timedelta(hours=1)).astype(int).values
    for pol in POLL:
        s = hist[pol]
        j = s.last_valid_index()
        a0 = s.loc[j] - clim_hist.loc[j, pol]
        lag = tsi - p["idx"](j)
        out[pol] = out[pol].values + (phi[pol] ** lag) * a0
    return out, phi


def year_ahead(p, models, origin_ts, n_hours, stats=None):
    """Every method's forecast for n_hours from origin_ts, using data before origin_ts.
    Returns dict method -> (n_hours, 7) in original units (clipped at 0 where noted)."""
    i = p["idx"](origin_ts)
    ts = pd.date_range(origin_ts, periods=n_hours, freq="h")
    hist, cmh, cdoy, last_year = baselines_for(p, origin_ts, ts)
    fn = lambda t: clim_month_hour(hist, t)
    damp, phi = damped(p, hist, ts, fn)
    rec_model, design = models["cnn_lstm"]
    rec = recursive_batch(rec_model, design, p, seed_window(p, i, REC_LOOKBACK)[None],
                          [origin_ts], n_hours, stats)[0]
    chain = chained_direct(models["direct"], p, i, n_hours, stats)
    out = {"persistence": np.tile(hist.ffill().iloc[-1].values, (n_hours, 1)),
           "same_hour_last_year": last_year.values, "clim_month_hour": cmh.values,
           "clim_doy_hour": cdoy.values, "damped_anomaly": damp.values,
           "cnn_lstm_recursive": unscaled(p, rec), "direct_chained": unscaled(p, chain)}
    return ts, out, phi


def short_horizon(p, models, start, stop, stats=None):
    """Rolling origins at 00:00 each day in [start, stop - 168 h]; all methods for leads
    1..168. Inputs are the measured hours before each origin; models stay frozen."""
    days = pd.date_range(start, pd.Timestamp(stop) - pd.Timedelta(hours=DIR_HORIZON), freq="D")
    oi = np.array([p["idx"](d) for d in days], dtype=int)
    keep = valid_input(p, oi, DIR_LOOKBACK)
    n_candidates = len(oi)
    oi = oi[keep]
    if len(oi) == 0:
        return None, n_candidates
    leads = np.arange(1, DIR_HORIZON + 1)
    rows = oi[:, None] + leads[None, :] - 1
    ts_grid = p["grid"][rows.ravel()]
    hist, cmh, cdoy, last_year = baselines_for(p, start, ts_grid)
    shape = rows.shape + (len(POLL),)
    obs = p["obs"]
    last = pd.DataFrame(obs).ffill().values
    persistence = np.repeat(last[oi - 1][:, None], DIR_HORIZON, axis=1)
    daily = obs[oi[:, None] - 24 + (leads[None, :] - 1) % 24]
    weekly = obs[oi[:, None] - 168 + (leads[None, :] - 1) % 168]
    cmh_a = cmh.values.reshape(shape)
    fallback = lambda x: np.where(np.isnan(x), cmh_a, x)
    # damped anomaly per origin: anomaly of the last observation before the origin
    clim_hist = clim_month_hour(hist, p["grid"][:len(obs)])
    phi = anomaly_phi(hist, clim_hist.iloc[:len(hist)])
    last_idx = pd.DataFrame(np.where(np.isnan(obs), np.nan, np.arange(len(obs))[:, None])).ffill().values
    damp = np.empty(shape)
    for j, pol in enumerate(POLL):
        jl = last_idx[oi - 1, j]
        okj = ~np.isnan(jl)
        jl_i = np.where(okj, jl, 0).astype(int)
        a0 = np.where(okj, obs[jl_i, j] - clim_hist[pol].values[jl_i], 0.0)
        lag = rows - jl_i[:, None]
        damp[..., j] = cmh_a[..., j] + (phi[pol] ** lag) * a0[:, None]
    rec_model, design = models["cnn_lstm"]
    rec = recursive_batch(rec_model, design, p, input_windows(p, oi, REC_LOOKBACK),
                          p["grid"][oi], DIR_HORIZON, stats)
    direct = predict_direct(models["direct"], p, oi, stats)
    out = {"persistence": fallback(persistence), "daily_naive": fallback(daily),
           "weekly_naive": fallback(weekly), "same_hour_last_year": last_year.values.reshape(shape),
           "clim_month_hour": cmh_a, "clim_doy_hour": cdoy.values.reshape(shape),
           "damped_anomaly": damp, "cnn_lstm_recursive": unscaled(p, rec),
           "direct": unscaled(p, direct)}
    return {"origins": p["grid"][oi], "rows": rows, "leads": np.broadcast_to(leads, rows.shape),
            "actual": obs[rows], "pred": out}, n_candidates


def negatives_and_clip(pred):
    """Physical constraint: concentrations are >= 0. Count values below zero, then clip."""
    counts = {}
    for k, v in pred.items():
        counts[k] = int((v < 0).sum())
        pred[k] = np.maximum(v, 0.0)
    return counts


def long_format(fold, station, ts, actual, pred, extra):
    """One row per target hour x pollutant; `extra` holds same-shape columns (lead, origin)."""
    n = len(ts)
    cols = {"fold": fold, "station": station,
            "timestamp": np.repeat(np.asarray(ts), len(POLL)),
            "pollutant": np.tile(POLL, n), "actual": actual.reshape(-1)}
    for k, v in extra.items():
        cols[k] = np.repeat(np.asarray(v), len(POLL))
    for k, v in pred.items():
        cols[k] = np.round(v.reshape(-1), 4)
    return pd.DataFrame(cols)


def ground_truth(p, rows, actual):
    in_file = p["present"][rows]
    return np.where(~np.isnan(actual), "observed",
                    np.where(np.repeat(in_file, len(POLL)).reshape(actual.shape),
                             "missing_value", "no_row_in_file"))


def run_station(fold, station, d, epochs, out_dir):
    p = prepare(d, fold)
    meta = {"station": station, "fold": fold, "first_observation": str(d.Timestamp.iloc[0]),
            "val_start": str(p["val_start"]), "cutoff": str(p["origin"])}
    direct, meta["direct"] = train_direct(p, epochs)
    tag = station.replace(" ", "_")
    direct.save(f"{out_dir}/models/direct_{tag}.keras")
    joblib.dump({"scaler": p["scaler"], "imputer": p["imputer"]},
                f"{out_dir}/models/preprocessing_{tag}.joblib")
    if fold in BASELINE_MODE:
        cnn = baseline_cnn_lstm(fold, station)
        meta["cnn_lstm_source"] = "baseline checkpoint"
    else:
        cnn = train_cnn_lstm(p, epochs)
        cnn[0].save(f"{out_dir}/models/cnn_lstm_{cnn[1]}_{tag}.keras")
        meta["cnn_lstm_source"] = "trained with the baseline recipe"
    meta["cnn_lstm_design"] = cnn[1]
    models = {"direct": direct, "cnn_lstm": cnn}

    # Blend weights from validation only: models trained before val_start, climatology from
    # data before val_start, forecasts issued inside the validation half-year.
    vs = short_horizon(p, models, p["val_start"], p["origin"])[0]
    w_short = fit_weights(vs["actual"], vs["pred"]["direct"], vs["pred"]["clim_month_hour"],
                          vs["leads"], SHORT_BUCKETS)
    n_val = p["idx"](p["origin"]) - p["idx"](p["val_start"])
    vts, vy, _ = year_ahead(p, models, p["val_start"], n_val)
    vrows = np.arange(p["idx"](p["val_start"]), p["idx"](p["val_start"]) + n_val)
    vleads = np.arange(1, n_val + 1)
    w_chain = fit_weights(p["obs"][vrows], vy["direct_chained"], vy["clim_month_hour"],
                          vleads, YEAR_BUCKETS)
    w_rec = fit_weights(p["obs"][vrows], vy["cnn_lstm_recursive"], vy["clim_month_hour"],
                        vleads, YEAR_BUCKETS)
    # Leads <= 168 h: one long validation run gives only a handful of hours per bucket, so
    # use the weights fitted on all daily validation origins instead (the chained forecast's
    # first block is exactly the direct forecast).
    n_short = len(SHORT_BUCKETS)
    assert YEAR_BUCKETS[:n_short] == SHORT_BUCKETS
    w_chain[:n_short] = w_short
    w_rec[:n_short] = fit_weights(vs["actual"], vs["pred"]["cnn_lstm_recursive"],
                                  vs["pred"]["clim_month_hour"], vs["leads"], SHORT_BUCKETS)
    weights = []
    for name, w, buckets in (("hybrid_direct", w_short, SHORT_BUCKETS),
                             ("hybrid_direct_chained", w_chain, YEAR_BUCKETS),
                             ("hybrid_recursive", w_rec, YEAR_BUCKETS)):
        for k, (lo, hi) in enumerate(buckets):
            weights.append({"fold": fold, "station": station, "method": name,
                            "lead_from": lo, "lead_to": hi,
                            **{pol: round(w[k, j], 4) for j, pol in enumerate(POLL)}})

    # Short horizons on the test year.
    sh, n_cand = short_horizon(p, models, p["origin"], p["end"] + pd.Timedelta(hours=1))
    short_df = None
    meta["short_origins_candidates"] = n_cand
    meta["short_origins_scored"] = 0 if sh is None else len(sh["origins"])
    if sh is not None:
        sh["pred"]["hybrid_direct"] = apply_weights(w_short, sh["pred"]["direct"],
                                                    sh["pred"]["clim_month_hour"], sh["leads"],
                                                    SHORT_BUCKETS)
        meta["short_negatives_clipped"] = negatives_and_clip(sh["pred"])
        origin_rep = np.broadcast_to(np.asarray(sh["origins"])[:, None], sh["rows"].shape)
        short_df = long_format(fold, station, p["grid"][sh["rows"].ravel()],
                               sh["actual"].reshape(-1, len(POLL)),
                               {k: v.reshape(-1, len(POLL)) for k, v in sh["pred"].items()},
                               {"origin": origin_rep.ravel(), "lead": sh["leads"].ravel()})
        short_df = short_df[short_df.actual.notna()]

    # Year ahead from the cutoff over the whole horizon.
    n_year = p["idx"](p["end"]) - p["idx"](p["origin"]) + 1
    ts, yp, phi = year_ahead(p, models, p["origin"], n_year)
    leads = np.arange(1, n_year + 1)
    yp["hybrid_direct_chained"] = apply_weights(w_chain, yp["direct_chained"],
                                                yp["clim_month_hour"], leads, YEAR_BUCKETS)
    yp["hybrid_recursive"] = apply_weights(w_rec, yp["cnn_lstm_recursive"],
                                           yp["clim_month_hour"], leads, YEAR_BUCKETS)
    meta["year_negatives_clipped"] = negatives_and_clip(yp)
    hist = history(p, p["origin"])
    p999 = np.nanpercentile(hist.values, 99.9, axis=0)
    meta["year_above_2x_p999"] = {k: int((v > 2 * p999).sum()) for k, v in yp.items()}
    meta["anomaly_phi"] = phi
    rows = np.arange(p["idx"](p["origin"]), p["idx"](p["origin"]) + n_year)
    actual = p["obs"][rows]
    year_df = long_format(fold, station, ts, actual, yp, {"lead": leads})
    year_df["ground_truth"] = ground_truth(p, rows, actual).reshape(-1)
    return short_df, year_df, pd.DataFrame(weights), meta


def load_models(fold, station, out_root=EXP_OUT):
    tag = station.replace(" ", "_")
    direct = tf.keras.models.load_model(f"{out_root}/folds/{fold}/models/direct_{tag}.keras",
                                        compile=False)
    if fold in BASELINE_MODE:
        return {"direct": direct, "cnn_lstm": baseline_cnn_lstm(fold, station)}
    path = glob.glob(f"{out_root}/folds/{fold}/models/cnn_lstm_*_{tag}.keras")[0]
    design = os.path.basename(path).split("_")[2]
    return {"direct": direct, "cnn_lstm": (tf.keras.models.load_model(path, compile=False), design)}


def diagnose(out_root=EXP_OUT):
    """Re-run test-period inference from the saved models (no training) to count network
    outputs below zero before the floor, and confirm the saved forecasts are reproduced."""
    data = load_clean()
    rows = []
    for fold in FOLDS:
        meta = json.load(open(f"{out_root}/folds/{fold}/fold_meta.json"))
        saved = pd.read_csv(f"{out_root}/folds/{fold}/year_ahead.csv.gz", parse_dates=["timestamp"])
        for st in meta["stations"]:
            if st.get("excluded"):
                continue
            station = st["station"]
            p = prepare(data[station], fold)
            models = load_models(fold, station, out_root)
            n_year = p["idx"](p["end"]) - p["idx"](p["origin"]) + 1
            ys, ss = {}, {}
            ts, yp, _ = year_ahead(p, models, p["origin"], n_year, ys)
            short_horizon(p, models, p["origin"], p["end"] + pd.Timedelta(hours=1), ss)
            g = saved[saved.station == station]
            for k in ("cnn_lstm_recursive", "direct_chained"):
                assert np.allclose(g[k].values, np.round(np.maximum(yp[k], 0), 4).reshape(-1),
                                   atol=2e-4), (fold, station, k)
            for kind, stats in (("year_ahead", ys), ("short_horizon", ss)):
                for k in [k for k in stats if not k.endswith("_outputs")]:
                    rows.append({"fold": fold, "station": station, "protocol": kind,
                                 "method": k, "outputs": stats[f"{k}_outputs"],
                                 "below_zero_before_floor": stats[k]})
            rows.append({"fold": fold, "station": station, "protocol": "year_ahead",
                         "method": "damped_anomaly", "outputs": int(yp["damped_anomaly"].size),
                         "below_zero_before_floor": int((yp["damped_anomaly"] < 0).sum())})
            print(f"diagnose {fold} {station}: saved forecasts reproduced")
    df = pd.DataFrame(rows)
    df["share"] = df.below_zero_before_floor / df.outputs
    os.makedirs(f"{out_root}/metrics", exist_ok=True)
    df.to_csv(f"{out_root}/metrics/floor_diagnostics.csv", index=False)


def run_fold(fold, epochs, stations=None, out_root=EXP_OUT):
    out_dir = f"{out_root}/folds/{fold}"
    os.makedirs(f"{out_dir}/models", exist_ok=True)
    data = load_clean()
    shorts, years, weights, metas = [], [], [], []
    for station in stations or STATIONS:
        d = data[station]
        if not eligible(d, fold):
            metas.append({"station": station, "fold": fold, "excluded": True,
                          "reason": f"first observation {d.Timestamp.iloc[0]:%Y-%m-%d} is less "
                                    f"than {MIN_TRAIN_DAYS} days before validation starts"})
            print(f"{fold} {station}: excluded (insufficient history)")
            continue
        s, y, w, m = run_station(fold, station, d, epochs, out_dir)
        print(f"{fold} {station}: direct {m['direct']}; short origins "
              f"{m['short_origins_scored']}/{m['short_origins_candidates']}")
        if s is not None:
            shorts.append(s)
        years.append(y)
        weights.append(w)
        metas.append(m)
    if shorts:
        pd.concat(shorts).to_csv(f"{out_dir}/short_horizon.csv.gz", index=False)
    pd.concat(years).to_csv(f"{out_dir}/year_ahead.csv.gz", index=False)
    pd.concat(weights).to_csv(f"{out_dir}/blend_weights.csv", index=False)
    fold_meta = {"fold": fold, "cutoff": FOLDS[fold], "epochs_max": epochs, "stations": metas}
    if fold not in SELECTION_FOLDS:  # holdout: record the selection that existed beforehand
        sel = json.load(open(f"{out_root}/selection.json"))
        fold_meta["selection_before_run"] = {k: sel[k] for k in ("selected_method", "created")}
    with open(f"{out_dir}/fold_meta.json", "w") as fh:
        json.dump(fold_meta, fh, indent=2, default=str)


# ---------------------------------------------------------------- metrics and selection

def metric_row(y, yhat):
    err = yhat - y
    mse = float(np.mean(err ** 2))
    var = float(np.sum((y - y.mean()) ** 2))
    return {"n": len(y), "RMSE": np.sqrt(mse), "MAE": float(np.mean(np.abs(err))),
            "R2": 1 - float(np.sum(err ** 2)) / var if len(y) > 1 and var > 0 else np.nan}


def quality_flags(df):
    distinct = df.groupby(["fold", "station", "pollutant"]).actual.nunique()
    return set(distinct[distinct < 100].index)


def scores(df, methods, keys):
    """Metrics for every method on identical rows, grouped by `keys`."""
    rows = []
    for k, g in df.groupby(keys, sort=False):
        base = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        y = g.actual.values
        for m in methods:
            rows.append({**base, "method": m, **metric_row(y, g[m].values)})
    return pd.DataFrame(rows)


def skill_table(sc, keys, ref="clim_month_hour"):
    """RMSE skill of each method against `ref` within each group."""
    r = sc[sc.method == ref].set_index(keys)["RMSE"]
    sc = sc.copy()
    sc["skill_vs_clim"] = 1 - sc.RMSE / r.reindex(pd.MultiIndex.from_frame(sc[keys])).values
    return sc


def load_fold(fold, kind, root=EXP_OUT):
    path = f"{root}/folds/{fold}/{'short_horizon' if kind == 'short' else 'year_ahead'}.csv.gz"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if kind == "year":
        df = df[df.ground_truth == "observed"]
    df["season"] = df.timestamp.dt.month.map(SEASONS)
    return df


def selection_scores(folds=SELECTION_FOLDS, root=EXP_OUT):
    """Year-ahead station x pollutant skill on the selection folds (unflagged series)."""
    df = pd.concat([load_fold(f, "year", root) for f in folds])
    flagged = quality_flags(df)
    df = df[[(f, s, p) not in flagged for f, s, p in zip(df.fold, df.station, df.pollutant)]]
    keys = ["fold", "station", "pollutant"]
    return skill_table(scores(df, YEAR_METHODS, keys), keys), flagged


def select(root=EXP_OUT):
    """Pre-registered rule: the year-ahead method with the highest mean RMSE skill against
    month x hour climatology over all (fold, station, pollutant) series of the selection
    folds. Climatology itself (skill 0) is a candidate."""
    sc, flagged = selection_scores(root=root)
    summary = sc.groupby("method").agg(mean_skill=("skill_vs_clim", "mean"),
                                       median_skill=("skill_vs_clim", "median"),
                                       series_better=("skill_vs_clim", lambda s: int((s > 0).sum())),
                                       series=("skill_vs_clim", "size")).sort_values(
        "mean_skill", ascending=False)
    chosen = summary.index[0]
    result = {"selected_method": chosen, "rule": select.__doc__.split("\n\n")[0].strip(),
              "folds": list(SELECTION_FOLDS), "flagged_series_excluded": sorted(map(list, flagged)),
              "summary": summary.round(4).reset_index().to_dict(orient="records"),
              "created": pd.Timestamp.now().isoformat(timespec="seconds")}
    with open(f"{root}/selection.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(summary.round(3))
    print("selected:", chosen)
    return result


def final(root=EXP_OUT):
    """2026 deliverable: the selected method's forecast from the F2026 run (trained and
    weighted on data up to 2025-12-31), every hour of 2026, all four stations."""
    sel = json.load(open(f"{root}/selection.json"))
    fold_meta = json.load(open(f"{root}/folds/F2026/fold_meta.json"))
    assert fold_meta["selection_before_run"]["created"] == sel["created"], \
        "selection.json changed after the F2026 holdout run; re-run fold F2026"
    method = sel["selected_method"]
    y = pd.read_csv(f"{root}/folds/F2026/year_ahead.csv.gz", parse_dates=["timestamp"])
    out = y[["station", "timestamp", "pollutant", "lead", "actual", "ground_truth"]].copy()
    out.insert(3, "unit", out.pollutant.map(UNITS))
    out["forecast"] = y[method]
    out["method"] = method
    out["clim_month_hour"] = y["clim_month_hour"]
    out["baseline_cnn_lstm_recursive"] = y["cnn_lstm_recursive"]
    out.to_csv(f"{root}/forecast_2026_selected.csv", index=False)
    with open(f"{root}/forecast_2026_metadata.json", "w") as fh:
        json.dump({"method": method, "training_cutoff": "2025-12-31 23:00",
                   "horizon": ["2026-01-01 00:00", "2026-12-31 23:00"],
                   "rows": len(out), "selection": sel["rule"],
                   "stations": [m["station"] for m in fold_meta["stations"]]}, fh, indent=2)
    print(f"forecast_2026_selected.csv: {len(out)} rows, method {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fold", "select", "final", "diagnose", "report"])
    ap.add_argument("fold", nargs="?", choices=list(FOLDS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--stations", nargs="*")
    ap.add_argument("--out", default=EXP_OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "fold":
        if a.fold == "F2026" and not os.path.exists(f"{a.out}/selection.json"):
            raise SystemExit("Run `select` on the selection folds before the F2026 holdout.")
        run_fold(a.fold, a.epochs, a.stations, a.out)
    elif a.stage == "select":
        select(a.out)
    elif a.stage == "final":
        final(a.out)
    elif a.stage == "diagnose":
        diagnose(a.out)
    else:
        from report import build  # experiments/report.py (script directory is on sys.path)
        build(a.out)


if __name__ == "__main__":
    main()
