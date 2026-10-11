# %% [markdown]
# # Nashik Air-Quality Prediction with KNN Imputation and CNN-LSTM
#
# Hourly forecasts of seven criteria pollutants at four Nashik monitoring stations. Each
# station gets its own CNN-LSTM, trained only on that station's data before the forecast
# origin. The pipeline runs in one of two modes (environment variable `NASHIK_MODE`):
#
# - `backtest_2025`: train on data up to 2024-12-31, forecast every hour of 2025, score
#   against the measured 2025 data.
# - `forecast_2026` (default): train on data up to 2025-12-31, forecast every hour of 2026,
#   score only the 2026 hours that were actually measured.
#
# Each mode produces two kinds of prediction, reported separately:
# 1. **Recursive forecast** of the whole horizon from the last 24 hours before the origin:
#    predictions are fed back as inputs, no measurement after the origin is ever used.
# 2. **Rolling one-step evaluation**: the frozen model predicts each measured hour from
#    the measured previous 24 hours (operational next-hour forecasting). Post-origin
#    measurements are inputs here, never training data.
#
# Run top to bottom. In Colab, the upload cell asks for the four station CSVs
# (or the zip they came in).

# %% [markdown]
# ## 1. Environment setup and imports

# %%
import glob
import json
import os
import platform
import subprocess
import sys
import zipfile

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from aqi_cpcb import AQI_EDGES, CATEGORIES, daily_aqi

SEED = 42
tf.keras.utils.set_random_seed(SEED)  # seeds Python, NumPy and TensorFlow

MODES = {  # forecast horizon per mode; training cutoff = one hour before forecast_start
    "backtest_2025": {"forecast_start": "2025-01-01 00:00", "forecast_end": "2025-12-31 23:00"},
    "forecast_2026": {"forecast_start": "2026-01-01 00:00", "forecast_end": "2026-12-31 23:00"},
}
MODE = os.environ.get("NASHIK_MODE", "forecast_2026")
if MODE not in MODES:
    raise ValueError(f"NASHIK_MODE={MODE!r}; expected one of {sorted(MODES)}")
FORECAST_START = pd.Timestamp(MODES[MODE]["forecast_start"])
FORECAST_END = pd.Timestamp(MODES[MODE]["forecast_end"])
TRAINING_CUTOFF = FORECAST_START - pd.Timedelta(hours=1)
HORIZON = pd.date_range(FORECAST_START, FORECAST_END, freq="h")
N_FORECAST = len(HORIZON)
print(f"Mode {MODE}: training cutoff {TRAINING_CUTOFF}, horizon {HORIZON[0]} -> "
      f"{HORIZON[-1]} ({N_FORECAST} hours)")

OUT = f"outputs/{MODE}"
for sub in ("figures", "metrics", "models", "predictions", "tables"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)

VERSIONS = {
    "python": platform.python_version(), "tensorflow": tf.__version__,
    "pandas": pd.__version__, "numpy": np.__version__,
    "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
}
print(VERSIONS)

# %% [markdown]
# ## 2. Dataset upload and loading
#
# Station identity comes from the file name: the four files share one schema and none
# contains a station column. Hirwadi's file carries an extra `Source_File` column whose
# labels do not match its timestamps (e.g. `2023.csv.csv` holds 2026 rows), so it is
# dropped and the `Timestamp` column is treated as authoritative.

# %%
try:
    from google.colab import files  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
if IN_COLAB and not glob.glob(f"{DATA_DIR}/*.[cC][sS][vV]"):
    print("Select the four station CSVs (or their zip), plus build_report.py if you want "
          "the DOCX report generated in Colab.")
    os.makedirs("scripts", exist_ok=True)
    for name in files.upload():
        os.replace(name, f"scripts/{name}" if name.endswith(".py") else f"{DATA_DIR}/{name}")
for z in glob.glob(f"{DATA_DIR}/*.zip"):
    with zipfile.ZipFile(z) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".csv"):
                with open(f"{DATA_DIR}/{os.path.basename(member)}", "wb") as fh:
                    fh.write(zf.read(member))

STATION_PATTERNS = {  # file-name fragment -> station label used everywhere below
    "midcambad": "MIDC Ambad", "gangapur": "Gangapur Road",
    "pandav": "Pandav Nagari", "hirwadi": "Hirwadi",
}


def find_station_files(data_dir):
    found = {}
    for path in glob.glob(f"{data_dir}/*"):
        key = os.path.basename(path).lower().replace("_", "").replace(" ", "")
        for frag, station in STATION_PATTERNS.items():
            if frag in key and path.lower().endswith(".csv"):
                found[station] = path
    missing = set(STATION_PATTERNS.values()) - set(found)
    if missing:
        raise FileNotFoundError(f"No CSV found in '{data_dir}/' for: {sorted(missing)}. "
                                "Upload the four Nashik station files.")
    return {station: found[station] for station in STATION_PATTERNS.values()}


STATION_FILES = find_station_files(DATA_DIR)
raw = {s: pd.read_csv(p) for s, p in STATION_FILES.items()}
for s, d in raw.items():
    print(f"{s:14s} {STATION_FILES[s]:45s} rows={len(d):6d} cols={d.shape[1]}")

# %% [markdown]
# ## 3. Dataset inspection and data dictionary

# %%
TARGETS = {  # standardised name -> original column (CPCB-AQI pollutants present in the data)
    "PM2.5": "PM2.5 (µg/m³)", "PM10": "PM10 (µg/m³)", "NO2": "NO2 (µg/m³)",
    "SO2": "SO2 (µg/m³)", "CO": "CO (mg/m³)", "O3": "Ozone (µg/m³)", "NH3": "NH3 (µg/m³)",
}
UNITS = {"PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³",
         "CO": "mg/m³", "O3": "µg/m³", "NH3": "µg/m³"}
POLL = list(TARGETS)
EXPECTED = ["Timestamp", *TARGETS.values()]
for s, d in raw.items():
    absent = [c for c in EXPECTED if c not in d.columns]
    if absent:
        raise KeyError(f"{s}: expected columns not found: {absent}")
    if d.empty:
        raise ValueError(f"{s}: file is empty")


def column_role(col):
    if col == "Timestamp":
        return "timestamp"
    if col == "Source_File":
        return "metadata (dropped: labels inconsistent with timestamps)"
    if col in TARGETS.values():
        return "pollutant - model input and prediction target"
    if col.split(" (")[0] in {"AT", "RH", "WS", "WD", "RF", "TOT-RF", "SR", "BP", "VWS"}:
        return "meteorology - not used (see report)"
    return "pollutant (VOC/NOx species) - not used"


rows = []
for col in raw["Gangapur Road"].columns.union(raw["Hirwadi"].columns, sort=False):
    entry = {"original_column": col, "role": column_role(col),
             "unit": col.split("(")[-1].rstrip(")") if "(" in col else ""}
    for s, d in raw.items():
        entry[f"missing_%_{s}"] = round(100 * d[col].isna().mean(), 1) if col in d else None
    rows.append(entry)
data_dictionary = pd.DataFrame(rows)
data_dictionary.to_csv(f"{OUT}/tables/data_dictionary.csv", index=False)
data_dictionary

# %%
inspection = []
for s, d in raw.items():
    ts = pd.to_datetime(d["Timestamp"], errors="raise")
    steps = ts.sort_values().diff().value_counts()
    inspection.append({
        "station": s, "rows": len(d), "first": ts.min(), "last": ts.max(),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "sorted_in_file": bool(ts.is_monotonic_increasing),
        "modal_step": str(steps.index[0]), "irregular_steps": int(steps.iloc[1:].sum()),
        "years_present": ", ".join(str(y) for y in sorted(ts.dt.year.unique())),
        "all_pollutants_missing_rows": int(d[list(TARGETS.values())].isna().all(axis=1).sum()),
    })
inspection = pd.DataFrame(inspection)
inspection.to_csv(f"{OUT}/tables/dataset_inspection.csv", index=False)
inspection

# %% [markdown]
# ## 4. Data-quality checks, cleaning and exploratory analysis
#
# Cleaning rules (applied per station, counts logged):
# 1. Sort by timestamp; duplicate timestamps would keep the first row (none occur).
# 2. Drop rows before the station's first pollutant observation (the 2017 placeholder
#    blocks at three stations are entirely empty).
# 3. Negative concentrations -> missing (none occur).
# 4. PM2.5 / PM10 >= 999 -> missing: the analyser ceiling, not a measurement.
# 5. A value repeated unchanged for >= 12 consecutive hours -> missing (stuck sensor,
#    mostly CO flat at 0.00 mg/m³).
#
# No resampling or aggregation: one observation = one hourly average as reported.
# Time gaps (rows absent from the file) are not filled; sequences never span them.

# %%
STUCK_HOURS = 12
PM_CEILING = 999


def clean_station(df):
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    log = {"duplicates_removed": int(df["Timestamp"].duplicated().sum())}
    df = df.drop_duplicates("Timestamp", keep="first").reset_index(drop=True)
    df = df.rename(columns={v: k for k, v in TARGETS.items()})[["Timestamp", *POLL]]
    first_obs = int(df[POLL].notna().any(axis=1).idxmax())
    log["leading_empty_rows_dropped"] = first_obs
    df = df.iloc[first_obs:].reset_index(drop=True)
    for c in POLL:
        neg = df[c] < 0
        ceiling = (df[c] >= PM_CEILING) if c in ("PM2.5", "PM10") else pd.Series(False, df.index)
        run_id = (df[c] != df[c].shift()).cumsum()
        stuck = (df[c].groupby(run_id).transform("size") >= STUCK_HOURS) & df[c].notna()
        log[f"{c}_negative"] = int(neg.sum())
        log[f"{c}_ceiling"] = int(ceiling.sum())
        log[f"{c}_stuck"] = int(stuck.sum())
        df.loc[neg | ceiling | stuck, c] = np.nan
    return df, log


clean, cleaning_log = {}, {}
for s, d in raw.items():
    clean[s], cleaning_log[s] = clean_station(d)
    # Rows after the horizon play no part. Rows inside the horizon are ground truth and
    # one-step inputs only; every fitted object below filters to rows before the origin.
    clean[s] = clean[s][clean[s]["Timestamp"] <= FORECAST_END].reset_index(drop=True)
cleaning_log = pd.DataFrame(cleaning_log).T
cleaning_log.to_csv(f"{OUT}/tables/cleaning_log.csv")
print("Retained period per station:")
for s, d in clean.items():
    gaps = d["Timestamp"].diff()
    big = gaps[gaps > pd.Timedelta(hours=1)]
    print(f"  {s:14s} {d.Timestamp.min()} -> {d.Timestamp.max()}  rows={len(d)}  "
          f"absent-row gaps: {[str(g) for g in big]}")
cleaning_log

# %%
STATION_COLORS = {"MIDC Ambad": "#2a78d6", "Gangapur Road": "#eb6834",
                  "Pandav Nagari": "#1baf7a", "Hirwadi": "#eda100"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def savefig(fig, name):
    fig.savefig(f"{OUT}/figures/{name}.png", bbox_inches="tight")
    if "ipykernel" in sys.modules:
        plt.show()
    else:
        plt.close(fig)


miss = pd.DataFrame({s: 100 * d[POLL].isna().mean() for s, d in clean.items()}).T
fig, ax = plt.subplots(figsize=(8, 3.2))
im = ax.imshow(miss.values, cmap="Blues", vmin=0, vmax=max(30, miss.values.max()))
ax.set_xticks(range(len(POLL)), POLL)
ax.set_yticks(range(len(miss)), miss.index)
for i in range(miss.shape[0]):
    for j in range(miss.shape[1]):
        v = miss.values[i, j]
        ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=8,
                color="white" if v > 0.6 * im.get_clim()[1] else "black")
ax.grid(False)
ax.set_xlabel("Pollutant")
ax.set_ylabel("Station")
ax.set_title("Missing hourly values before imputation (after cleaning, retained period)")
fig.colorbar(im, ax=ax, label="Missing (%)")
savefig(fig, "fig01_missing_values")
miss.round(2).to_csv(f"{OUT}/tables/missing_pct_before_imputation.csv")

# %%
fig, axes = plt.subplots(len(POLL), 1, figsize=(10, 15), sharex=True)
for ax, p in zip(axes, POLL):
    for s, d in clean.items():
        m = d.set_index("Timestamp")[p].resample("MS").mean()
        ax.plot(m.index, m.values, lw=1.4, marker="o", ms=2, color=STATION_COLORS[s], label=s)
    ax.set_ylabel(f"{p} ({UNITS[p]})")
axes[0].legend(ncol=4, loc="upper right", frameon=False)
axes[-1].set_xlabel("Month")
fig.suptitle("Historical pollutant trends: monthly mean of hourly observations", y=0.995)
fig.tight_layout()
savefig(fig, "fig02_historical_trends")

# %%
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for ax, p in zip(axes.flat, POLL):
    data = [clean[s][p].dropna().values for s in clean]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, s in zip(bp["boxes"], clean):
        patch.set_facecolor(STATION_COLORS[s])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(clean) + 1), [s.replace(" ", "\n") for s in clean])
    ax.set_title(p)
    ax.set_ylabel(UNITS[p])
axes.flat[-1].axis("off")
fig.suptitle("Station comparison: distribution of hourly concentrations "
             "(box = IQR, whiskers = 1.5 IQR, outliers hidden)")
fig.tight_layout()
savefig(fig, "fig03_station_comparison")

summary_stats = pd.concat({s: d[POLL].describe().T for s, d in clean.items()})
summary_stats.round(2).to_csv(f"{OUT}/tables/descriptive_statistics.csv")

# %% [markdown]
# ## 5. Chronological train / validation / test split
#
# Test = the forecast horizon (calendar year 2025 or 2026, depending on the mode).
# Validation = the six months before the horizon; training = everything earlier. Scaler,
# imputer, model weights, early stopping and the choice between the two model designs all
# use data up to the training cutoff only. Test rows exist only where the station file
# has them (2026: Jan-Jun and Aug at three stations, none at Pandav Nagari).

# %%
def split_bounds(df):
    test_year = FORECAST_START.year
    test_start = FORECAST_START
    val_start = test_start - pd.DateOffset(months=6)
    return test_year, val_start, test_start


def split_label(ts, val_start, test_start):
    return np.where(ts >= test_start, "test", np.where(ts >= val_start, "val", "train"))


split_rows = []
for s, d in clean.items():
    test_year, val_start, test_start = split_bounds(d)
    lab = split_label(d["Timestamp"], val_start, test_start)
    for part in ("train", "val", "test"):
        sub = d[lab == part]
        split_rows.append({
            "station": s, "split": part, "start": sub.Timestamp.min(),
            "end": sub.Timestamp.max(), "hourly_rows": len(sub),
            "test_year": test_year,
            "months_present": sub.Timestamp.dt.to_period("M").nunique(),
            "pm25_observed": int(sub["PM2.5"].notna().sum()),
        })
splits = pd.DataFrame(split_rows)
splits.to_csv(f"{OUT}/tables/splits.csv", index=False)
splits

# %% [markdown]
# ## 6. KNN imputation and feature scaling
#
# Per station: a `StandardScaler` (pollutants differ in units and range, e.g. CO in
# mg/m³ vs PM10 in hundreds of µg/m³) and a `KNNImputer` (k=5, distance-weighted,
# NaN-Euclidean distance) are fitted on that station's **training rows only**, then
# applied to validation and test rows. Donors are therefore always training-period
# hours; no future observation influences any imputed value.
#
# Distance uses only the seven pollutant concentrations of the same hour - no station
# ID, no timestamp. A missing value at hour s is filled from the other pollutants
# observed at hour s, which are known by the time a forecast is issued for any hour
# after s. Imputed values are used **only as model inputs**; prediction targets and
# all evaluation use observed values only. Hours with every pollutant missing get the
# training mean (KNN has no distance to work with); windows dominated by such hours
# are excluded in step 7.

# %%
K_NEIGHBORS = 5
prep = {}
imputation_counts = []
for s, d in clean.items():
    test_year, val_start, test_start = split_bounds(d)
    lab = split_label(d["Timestamp"], val_start, test_start)
    train_mask = lab == "train"
    scaler = StandardScaler().fit(d.loc[train_mask, POLL])
    Xs = scaler.transform(d[POLL])
    imputer = KNNImputer(n_neighbors=K_NEIGHBORS, weights="distance").fit(Xs[train_mask])
    # macOS Accelerate BLAS raises spurious FP flags in sklearn's nan-euclidean matmul;
    # distances were checked against a direct computation, and the assert guards results.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Xi = imputer.transform(Xs)
    assert np.isfinite(Xi).all(), f"{s}: imputation left non-finite values"
    observed = d[POLL].notna().values
    prep[s] = {"df": d, "split": lab, "scaler": scaler, "imputer": imputer,
               "X_imputed_scaled": Xi, "observed": observed}
    for part in ("train", "val", "test"):
        m = lab == part
        before = (~observed[m]).sum(axis=0)
        after = np.isnan(Xi[m]).sum(axis=0)
        for p, b, a in zip(POLL, before, after):
            imputation_counts.append({"station": s, "split": part, "pollutant": p,
                                      "missing_before": int(b), "missing_after": int(a)})
    joblib.dump({"scaler": scaler, "imputer": imputer, "features": POLL},
                f"{OUT}/models/preprocessing_{s.replace(' ', '_')}.joblib")
imputation_counts = pd.DataFrame(imputation_counts)
imputation_counts.to_csv(f"{OUT}/tables/imputation_counts.csv", index=False)
imputation_counts.pivot_table(index=["station", "split"], columns="pollutant",
                              values="missing_before").astype(int)

# %% [markdown]
# ## 7. Sliding-window sequence generation
#
# One observation = one hourly average. The network is trained as a one-step model:
# input = the previous **24 hours** (one diurnal cycle) of 13 features - 7 scaled,
# KNN-imputed pollutants plus sine/cosine encodings of hour-of-day, day-of-week and month
# (deterministic, so known for every future hour); target = the 7 pollutants at the
# **next hour**. Training/validation windows are kept only if
# - their 25 timestamps are consecutive hours (no window crosses a gap in the file),
# - at least one target pollutant is observed at the predicted hour,
# - at most 50% of the input pollutant cells were imputed, and
# - every pollutant has at least one earlier observation.
#
# The year-long forecast in step 10 applies this one-step model recursively.

# %%
LOOKBACK = 24
MAX_IMPUTED_FRACTION = 0.5


def calendar_features(ts):
    h, dow, mon = ts.dt.hour.values, ts.dt.dayofweek.values, ts.dt.month.values - 1
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                            np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)])


def make_windows(p):
    d = p["df"]
    feats = np.column_stack([p["X_imputed_scaled"], calendar_features(d["Timestamp"])])
    feats = feats.astype("float32")
    y_raw = d[POLL].values.astype("float32")  # NaN where not observed
    y_scaled = ((y_raw - p["scaler"].mean_) / p["scaler"].scale_).astype("float32")
    hours = (d["Timestamp"].values.astype("datetime64[h]").astype("int64"))
    t = np.arange(LOOKBACK, len(d))
    contiguous = hours[t] - hours[t - LOOKBACK] == LOOKBACK
    imputed_frac = np.lib.stride_tricks.sliding_window_view(
        (~p["observed"]).mean(axis=1), LOOKBACK)[:-1].mean(axis=1)
    last_obs = d[POLL].ffill().values  # most recent observed value at or before each hour
    keep = (contiguous & ~np.isnan(y_raw[t]).all(axis=1)
            & (imputed_frac <= MAX_IMPUTED_FRACTION) & np.isfinite(last_obs[t - 1]).all(axis=1))
    t = t[keep]
    X = np.stack([feats[i - LOOKBACK:i] for i in t])
    last_scaled = ((last_obs[t - 1] - p["scaler"].mean_) / p["scaler"].scale_).astype("float32")
    return {"X": X, "y_scaled": y_scaled[t], "y_raw": y_raw[t], "t": t,
            "timestamp": d["Timestamp"].values[t], "split": p["split"][t],
            "persistence": last_obs[t - 1], "last_scaled": last_scaled}


windows = {s: make_windows(p) for s, p in prep.items()}
seq_rows = []
for s, w in windows.items():
    for part in ("train", "val"):
        n = int((w["split"] == part).sum())
        if n == 0:
            raise ValueError(f"{s}: no {part} sequences - check the data and split dates")
        seq_rows.append({"station": s, "split": part, "sequences": n})
    print(f"{s:14s} X shape {w['X'].shape}  (samples, lookback hours, features; "
          "test windows are used only for the rolling one-step evaluation)")
pd.DataFrame(seq_rows).pivot(index="station", columns="split", values="sequences")

# %% [markdown]
# ## 8. CNN-LSTM model definition
#
# Conv1D layers learn short local temporal motifs (e.g. rush-hour rises) across the 24
# hourly steps; max-pooling halves the sequence; an LSTM reads the pooled feature
# sequence for longer-range dependencies; dropout regularises; a dense head outputs all
# 7 pollutants (multi-output). Missing targets are masked out of the MSE loss.
#
# Two target parameterisations of the same network are trained:
# - `level`: predict the standardised concentration at the next hour;
# - `change`: predict the change from the last value, then add it back.
#
# A design that is better one hour ahead is not necessarily better over months of
# feeding its own output back (small biases accumulate). So the design used for the
# recursive forecast is chosen per station by a **recursive forecast of the validation
# half-year** (started from the 24 hours before validation), scored against the observed
# validation values. The design used for the rolling one-step evaluation is the one with
# the lower one-step validation loss. The forecast horizon plays no part in either choice.

# %%
N_FEATURES = len(POLL) + 6


def masked_mse(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    err = tf.where(mask, y_true - y_pred, tf.zeros_like(y_pred))
    return tf.reduce_sum(tf.square(err)) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)


def build_cnn_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, N_FEATURES)),
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


build_cnn_lstm().summary()

# %% [markdown]
# ## 9. Model training and validation
#
# Adam (lr 1e-3), batch 256, up to 60 epochs, early stopping on validation loss with
# patience 6 and best-weights restore. No hyperparameter search: one fixed
# configuration for all stations.

# %%
BATCH, PATIENCE = 256, 6
EPOCHS = int(os.environ.get("NASHIK_EPOCHS", 60))  # lower only for smoke tests
DESIGNS = ("level", "change")


def design_target(w, design):
    return w["y_scaled"] - w["last_scaled"] if design == "change" else w["y_scaled"]


# NASHIK_REUSE_MODELS=1 loads this mode's saved checkpoints and training histories instead
# of retraining (same data and seed give the same weights; use it to re-run later steps).
REUSE = os.environ.get("NASHIK_REUSE_MODELS") == "1"
if REUSE:
    with open(f"{OUT}/metrics/training_history.json") as fh:
        saved_histories = json.load(fh)
models, histories, selection = {}, {}, []
for s, w in windows.items():
    tr, va = w["split"] == "train", w["split"] == "val"
    row = {"station": s, "train_sequences": int(tr.sum()), "val_sequences": int(va.sum()),
           "val_loss_persistence": float(np.nanmean(
               np.square(w["y_scaled"][va] - w["last_scaled"][va])))}
    for design in DESIGNS:
        path = f"{OUT}/models/cnn_lstm_{design}_{s.replace(' ', '_')}.keras"
        if REUSE:
            model = tf.keras.models.load_model(path, compile=False)
            history = saved_histories[f"{s}|{design}"]
        else:
            tf.keras.utils.set_random_seed(SEED)
            y = design_target(w, design)
            model = build_cnn_lstm()
            history = model.fit(
                w["X"][tr], y[tr], validation_data=(w["X"][va], y[va]),
                epochs=EPOCHS, batch_size=BATCH, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE,
                                                            restore_best_weights=True)]).history
            model.save(path)
        models[(s, design)], histories[f"{s}|{design}"] = model, history
        best = int(np.argmin(history["val_loss"]))
        row.update({f"val_loss_{design}": history["val_loss"][best],
                    f"train_loss_{design}": history["loss"][best],
                    f"best_epoch_{design}": best + 1,
                    f"epochs_run_{design}": len(history["loss"])})
    selection.append(row)
selection = pd.DataFrame(selection).set_index("station")
with open(f"{OUT}/metrics/training_history.json", "w") as fh:
    json.dump(histories, fh)
selection.round(4)

# %% [markdown]
# ## 10. Recursive forecasting and design selection
#
# `recursive_forecast` starts from the 24 hours before the first forecast hour (the
# KNN-imputed, scaled values the model was trained on; an hour absent from the file is
# set to the training mean). It predicts the next hour, appends that prediction (with the
# known calendar features of that hour) to the window, drops the oldest hour, and
# repeats. Predictions below zero concentration are clipped to zero before being fed
# back. Only the starting 24 hours are real data.

# %%
def recursive_forecast(model, design, p, start, n_hours):
    sc = p["scaler"]
    hist_ts = pd.date_range(end=start - pd.Timedelta(hours=1), periods=LOOKBACK, freq="h")
    known = pd.DataFrame(p["X_imputed_scaled"], index=p["df"]["Timestamp"])
    seed = known.reindex(hist_ts)
    seed_missing = int(seed.isna().all(axis=1).sum())
    window = np.column_stack([seed.fillna(0.0).values,  # 0 = training mean after scaling
                              calendar_features(pd.Series(hist_ts))]).astype("float32")
    future = pd.date_range(start, periods=n_hours, freq="h")
    cal = calendar_features(pd.Series(future)).astype("float32")
    zero_scaled = (-sc.mean_ / sc.scale_).astype("float32")
    step = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    out = np.empty((n_hours, len(POLL)), dtype="float32")
    for i in range(n_hours):
        z = step(window[None])[0].numpy()
        if design == "change":
            z = z + window[-1, :len(POLL)]
        z = np.maximum(z, zero_scaled)
        out[i] = z
        window = np.vstack([window[1:], np.concatenate([z, cal[i]])[None]])
    return future, np.maximum(out * sc.scale_ + sc.mean_, 0), seed_missing


def scaled_mse(p, future, pred):
    obs = p["df"].set_index("Timestamp")[POLL].reindex(future).values
    z_obs = (obs - p["scaler"].mean_) / p["scaler"].scale_
    z_pred = (pred - p["scaler"].mean_) / p["scaler"].scale_
    return float(np.nanmean(np.square(z_obs - z_pred)))


for s, p in prep.items():
    _, val_start, test_start = split_bounds(p["df"])
    n_val = int((test_start - val_start) / pd.Timedelta(hours=1))
    for design in DESIGNS:
        fut, pred, _ = recursive_forecast(models[(s, design)], design, p, val_start, n_val)
        selection.loc[s, f"val_recursive_mse_{design}"] = scaled_mse(p, fut, pred)
selection["selected_design"] = [
    min(DESIGNS, key=lambda d: selection.loc[s, f"val_recursive_mse_{d}"]) for s in selection.index]
selection["one_step_design"] = [
    min(DESIGNS, key=lambda d: selection.loc[s, f"val_loss_{d}"]) for s in selection.index]
selection.to_csv(f"{OUT}/metrics/model_selection_validation.csv")
selection[["val_loss_level", "val_loss_change", "val_recursive_mse_level",
           "val_recursive_mse_change", "selected_design", "one_step_design"]].round(4)

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, s in zip(axes, windows):
    h = histories[f"{s}|{selection.loc[s, 'selected_design']}"]
    ax.plot(range(1, len(h["loss"]) + 1), h["loss"], lw=2, color="#2a78d6", label="Training")
    ax.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], lw=2, color="#eb6834",
            label="Validation")
    ax.axvline(int(np.argmin(h["val_loss"])) + 1, color="#52514e", ls=":", lw=1,
               label="Best epoch")
    ax.axhline(selection.loc[s, "val_loss_persistence"], color="#3b3b3b", lw=1, ls="--",
               label="One-step persistence (val.)")
    ax.set_title(f"{s} ({selection.loc[s, 'selected_design']} target)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE (standardised units)")
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("CNN-LSTM one-step training and validation loss (selected design per station)")
fig.tight_layout()
savefig(fig, "fig04_loss_curves")

# %% [markdown]
# ## 11. Recursive forecast of the whole horizon, and baselines
#
# For each station both designs forecast every hour of the horizon (8,760 hours) from the
# last 24 hours before the origin; the design selected on validation is the reported
# CNN-LSTM forecast. Three baselines use the same information (data before the origin):
# - **Climatology**: mean of the observed values for the same calendar month and hour of
#   day over all data before the origin (month mean, then overall mean, if a cell is empty).
# - **Same hour last year**: the observed value 364 days earlier (52 weeks, so the weekday
#   matches); where that hour was not measured, the climatology value is used.
# - **Persistence**: the last observed value before the origin, held for the year.
#
# Every hour of the horizon is exported, measured or not. `ground_truth` says whether the
# station file holds a measurement for that hour and pollutant (`observed`), holds the row
# but no valid value (`missing_value`), or has no row at all (`no_row_in_file`, e.g. July
# and Sep-Dec 2026, and all of 2026 at Pandav Nagari). `actual` is empty unless observed;
# metrics use observed rows only. Forecasts are rounded to 4 decimals, so the exported
# file reproduces the exported metrics exactly.

# %%
def climatology(d, ts):
    hist = d[d["Timestamp"] < FORECAST_START]
    by_month_hour = hist.groupby([hist.Timestamp.dt.month, hist.Timestamp.dt.hour])[POLL].mean()
    by_month = hist.groupby(hist.Timestamp.dt.month)[POLL].mean()
    out = by_month_hour.reindex(pd.MultiIndex.from_arrays([ts.month, ts.hour])).set_axis(ts)
    return out.fillna(by_month.reindex(ts.month).set_axis(ts)).fillna(hist[POLL].mean())


def same_hour_last_year(d, ts, clim):
    hist = d[d["Timestamp"] < FORECAST_START].set_index("Timestamp")[POLL]
    return hist.reindex(ts - pd.Timedelta(days=364)).set_axis(ts).fillna(clim)


FORECAST_COLS = ["cnn_lstm_pred", "climatology_pred", "same_hour_last_year_pred",
                 "persistence_pred", "cnn_lstm_level_pred", "cnn_lstm_change_pred"]
forecast_rows, coverage_rows = [], []
for s, p in prep.items():
    d = p["df"]
    yhat = {}
    for design in DESIGNS:
        future, yhat[design], seed_missing = recursive_forecast(
            models[(s, design)], design, p, FORECAST_START, N_FORECAST)
    assert future.equals(HORIZON), s
    clim = climatology(d, future)
    last_year = same_hour_last_year(d, future, clim)
    persist = d[d["Timestamp"] < FORECAST_START][POLL].ffill().iloc[-1]
    indexed = d.set_index("Timestamp")
    actual = indexed[POLL].reindex(future)
    in_file = future.isin(indexed.index)
    chosen = selection.loc[s, "selected_design"]
    measured = actual.notna().any(axis=1)
    coverage_rows.append({
        "station": s, "horizon_hours": N_FORECAST, "rows_in_file": int(in_file.sum()),
        "hours_any_pollutant_measured": int(measured.sum()),
        "first_measured": future[measured.values].min() if measured.any() else None,
        "last_measured": future[measured.values].max() if measured.any() else None,
        "months_without_rows": ", ".join(
            m.strftime("%b") for m in pd.period_range(FORECAST_START, FORECAST_END, freq="M")
            if not in_file[future.to_period("M") == m].any()) or "none",
        "seed_hours_absent_from_file": seed_missing})
    print(f"{s:14s} forecast {len(future)} hours, {future[0]} -> {future[-1]}; "
          f"rows in file {int(in_file.sum())}; starting-window hours absent: {seed_missing}")
    for j, pol in enumerate(POLL):
        obs = actual[pol].notna().values
        forecast_rows.append(pd.DataFrame({
            "station": s, "timestamp": future, "pollutant": pol, "unit": UNITS[pol],
            "cnn_lstm_pred": yhat[chosen][:, j], "selected_design": chosen,
            "climatology_pred": clim[pol].values,
            "same_hour_last_year_pred": last_year[pol].values,
            "persistence_pred": persist[pol],
            "cnn_lstm_level_pred": yhat["level"][:, j],
            "cnn_lstm_change_pred": yhat["change"][:, j],
            "actual": actual[pol].values,
            "ground_truth": np.where(obs, "observed",
                                     np.where(in_file, "missing_value", "no_row_in_file")),
        }))
forecast = pd.concat(forecast_rows, ignore_index=True)
forecast[FORECAST_COLS] = forecast[FORECAST_COLS].astype("float64").round(4)
forecast["training_cutoff"] = TRAINING_CUTOFF
forecast["month"] = forecast.timestamp.dt.month
assert forecast[FORECAST_COLS].notna().all().all()
assert (forecast.groupby(["station", "pollutant"]).size() == N_FORECAST).all()
assert forecast.actual.isna().eq(forecast.ground_truth != "observed").all()
forecast.to_csv(f"{OUT}/predictions/recursive_forecast.csv", index=False)
coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(f"{OUT}/tables/horizon_coverage.csv", index=False)
predictions = forecast[forecast.ground_truth == "observed"].reset_index(drop=True)
coverage

# %% [markdown]
# ## 12. Rolling one-step evaluation
#
# The models stay frozen at the training cutoff. For every measured hour of the horizon
# that has a valid window (24 consecutive hours in the file, at most 50% imputed input
# cells), the model predicts that hour from the measured previous 24 hours - the task the
# network was trained for. This uses post-origin measurements as *inputs*, never as
# training data. Persistence (last observed value) is scored on exactly the same rows.
# Stations with no measured hours in the horizon get no rows.

# %%
ONE_STEP_COLS = ["station", "timestamp", "input_window_end", "pollutant", "unit", "actual",
                 "cnn_lstm_pred", "design", "persistence_pred"]
one_step_rows = []
for s, w in windows.items():
    te = w["split"] == "test"
    design = selection.loc[s, "one_step_design"]
    if not te.any():
        print(f"{s:14s} no measured windows in the horizon - no one-step evaluation")
        continue
    sc = prep[s]["scaler"]
    z = models[(s, design)](w["X"][te], training=False).numpy()
    if design == "change":
        z = z + w["last_scaled"][te]
    yhat = np.maximum(z * sc.scale_ + sc.mean_, 0).astype("float64").round(4)
    ts = pd.to_datetime(w["timestamp"][te])
    y = prep[s]["df"][POLL].values[w["t"][te]]
    for j, pol in enumerate(POLL):
        obs = ~np.isnan(y[:, j])
        one_step_rows.append(pd.DataFrame({
            "station": s, "timestamp": ts[obs], "input_window_end": ts[obs] - pd.Timedelta(hours=1),
            "pollutant": pol, "unit": UNITS[pol], "actual": y[obs, j],
            "cnn_lstm_pred": yhat[obs, j], "design": design,
            "persistence_pred": w["persistence"][te][obs, j]}))
    print(f"{s:14s} one-step windows: {int(te.sum())} ({design} design)")
one_step = (pd.concat(one_step_rows, ignore_index=True) if one_step_rows
            else pd.DataFrame(columns=ONE_STEP_COLS))
one_step.to_csv(f"{OUT}/predictions/rolling_one_step.csv", index=False)

# %% [markdown]
# ## 13. Evaluation metrics
#
# MSE, RMSE, MAE and R² (scikit-learn), in original units, per station and pollutant, on
# measured hours only, for every method on exactly the same hours. Each row reports the
# number of valid samples, the evaluated date range and the share of the 8,760 horizon
# hours that were measured. R² is `undefined` with fewer than 2 samples or zero variance,
# and flagged `unreliable` below one week (168 samples). A station with no measured hours
# still gets a row, with n_valid = 0 and empty metrics.
#
# Recursive skill = 1 - RMSE_CNN-LSTM / RMSE_climatology; one-step skill =
# 1 - RMSE_CNN-LSTM / RMSE_persistence (> 0: the CNN-LSTM is better).
#
# Data-quality flag: a series is flagged when its measured horizon values take fewer than
# 100 distinct values (a stuck or coarsely quantised sensor). Flagged series are reported
# but left out of the pooled all-station summary, which concatenates the hourly rows of
# the unflagged stations for one pollutant. Pollutants are never pooled together.

# %%
RECURSIVE_METHODS = {"CNN-LSTM": "cnn_lstm_pred", "Climatology": "climatology_pred",
                     "Same hour last year": "same_hour_last_year_pred",
                     "Persistence": "persistence_pred",
                     "CNN-LSTM level design": "cnn_lstm_level_pred",
                     "CNN-LSTM change design": "cnn_lstm_change_pred"}
ONE_STEP_METHODS = {"CNN-LSTM": "cnn_lstm_pred", "Persistence": "persistence_pred"}
MIN_DISTINCT_VALUES = 100
R2_MIN_SAMPLES = 168


def metrics(y, yhat):
    mse = mean_squared_error(y, yhat)
    r2 = r2_score(y, yhat) if len(y) > 1 and np.var(y) > 0 else np.nan
    return {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(y, yhat), "R2": r2}


def r2_flag(y):
    if len(y) < 2 or np.var(y) == 0:
        return "undefined"
    return "unreliable (n < 168)" if len(y) < R2_MIN_SAMPLES else "ok"


def score(base, g, methods, skill_ref, n_series=1):
    row = {**base, "unit": UNITS[base["pollutant"]], "n_valid": len(g),
           "horizon_hours": N_FORECAST * n_series,
           "coverage_pct": 100 * len(g) / (N_FORECAST * n_series)}
    if g.empty:
        row["evaluation_period"], row["r2_flag"] = "no measured hours", "undefined"
        return row
    ts = pd.to_datetime(g.timestamp)
    row["evaluation_period"] = f"{ts.min():%Y-%m-%d %H:%M} to {ts.max():%Y-%m-%d %H:%M}"
    row["r2_flag"] = r2_flag(g.actual)
    for name, col in methods.items():
        row.update({f"{name} {k}": v for k, v in metrics(g.actual, g[col]).items()})
    row[f"RMSE skill vs {skill_ref.lower()}"] = 1 - row["CNN-LSTM RMSE"] / row[f"{skill_ref} RMSE"]
    return row


def metric_tables(df, methods, skill_ref):
    per_station = pd.DataFrame([
        score({"station": s, "pollutant": pol},
              df[(df.station == s) & (df.pollutant == pol)], methods, skill_ref)
        for s in prep for pol in POLL])
    per_station["quality_flag"] = [(s, pol) in flagged for s, pol in
                                   zip(per_station.station, per_station.pollutant)]
    pooled_rows = []
    for pol in POLL:
        ok = [s for s in prep if (s, pol) not in flagged]
        g = df[(df.pollutant == pol) & df.station.isin(ok)]
        row = score({"station": "All stations (pooled)", "pollutant": pol}, g, methods,
                    skill_ref, n_series=len(ok))
        row["stations_included"] = ", ".join(s for s in ok if (g.station == s).any())
        pooled_rows.append(row)
    return per_station, pd.DataFrame(pooled_rows)


distinct = predictions.groupby(["station", "pollutant"]).actual.nunique()
flagged = set(distinct[distinct < MIN_DISTINCT_VALUES].index)
print("Flagged series (fewer than 100 distinct measured values):", sorted(flagged))
station_metrics, overall_metrics = metric_tables(predictions, RECURSIVE_METHODS, "Climatology")
one_step_metrics, one_step_overall = metric_tables(one_step, ONE_STEP_METHODS, "Persistence")
station_metrics["distinct_measured_values"] = [
    distinct.get((s, pol), 0) for s, pol in zip(station_metrics.station, station_metrics.pollutant)]
station_metrics.to_csv(f"{OUT}/metrics/recursive_station_pollutant_metrics.csv", index=False)
overall_metrics.to_csv(f"{OUT}/metrics/recursive_overall_pollutant_metrics.csv", index=False)
one_step_metrics.to_csv(f"{OUT}/metrics/one_step_station_pollutant_metrics.csv", index=False)
one_step_overall.to_csv(f"{OUT}/metrics/one_step_overall_pollutant_metrics.csv", index=False)

skill_summary = station_metrics.pivot(index="station", columns="pollutant",
                                      values="RMSE skill vs climatology").reindex(
    index=list(prep), columns=POLL)
skill_summary["mean skill (unflagged)"] = station_metrics[~station_metrics.quality_flag] \
    .groupby("station")["RMSE skill vs climatology"].mean()
skill_summary.round(3).to_csv(f"{OUT}/metrics/rmse_skill_summary.csv")

station_metrics[["station", "pollutant", "evaluation_period", "n_valid", "coverage_pct",
                 "CNN-LSTM MSE", "CNN-LSTM RMSE", "CNN-LSTM MAE", "CNN-LSTM R2", "r2_flag",
                 "quality_flag"]].round(3)

# %%
one_step_metrics[["station", "pollutant", "n_valid", "CNN-LSTM RMSE", "Persistence RMSE",
                  "CNN-LSTM R2", "RMSE skill vs persistence", "r2_flag",
                  "quality_flag"]].round(3)

# %%
monthly = predictions.groupby(["station", "pollutant", "month"], sort=False)[
    ["actual", *RECURSIVE_METHODS.values()]].mean().reset_index()
monthly_rows = []
for (s, p), g in monthly.groupby(["station", "pollutant"], sort=False):
    row = {"station": s, "pollutant": p, "months": len(g)}
    for name, col in RECURSIVE_METHODS.items():
        m = metrics(g.actual, g[col])
        row[f"{name} monthly RMSE"], row[f"{name} monthly R2"] = m["RMSE"], m["R2"]
    monthly_rows.append(row)
monthly_metrics = pd.DataFrame(monthly_rows)
monthly_metrics["quality_flag"] = [(s, p) in flagged for s, p in
                                   zip(monthly_metrics.station, monthly_metrics.pollutant)]
monthly_metrics.to_csv(f"{OUT}/metrics/recursive_monthly_mean_metrics.csv", index=False)

unflagged = predictions[[(st, po) not in flagged
                         for st, po in zip(predictions.station, predictions.pollutant)]]
MAIN_METHODS = {k: RECURSIVE_METHODS[k] for k in
                ("CNN-LSTM", "Climatology", "Same hour last year", "Persistence")}
rmse_by_month = unflagged.groupby(["pollutant", "month"], sort=False).apply(
    lambda g: pd.Series({name: np.sqrt(mean_squared_error(g.actual, g[col]))
                         for name, col in MAIN_METHODS.items()}),
    include_groups=False).reset_index()
rmse_by_month.to_csv(f"{OUT}/metrics/recursive_rmse_by_month.csv", index=False)
overall_metrics[["pollutant", "n_valid", "CNN-LSTM RMSE", "Climatology RMSE",
                 "RMSE skill vs climatology", "stations_included"]].round(3)

# %% [markdown]
# ## 14. Predicted AQI (CPCB)
#
# The model's outputs are pollutant concentrations; AQI is derived from them afterwards
# with the CPCB National AQI method (`aqi_cpcb.py`): 24-hour means for PM2.5, PM10, NO2,
# SO2 and NH3, maximum 8-hour mean for CO and O3, the AQI being the largest sub-index, and
# only when at least three sub-indices including PM2.5 or PM10 are available.
# - **Predicted AQI**: from the recursive CNN-LSTM forecast, every day of the horizon.
# - **Measured AQI**: from the station measurements, only on days where a valid AQI can be
#   computed (>= 16 measured hours per 24-hour mean, >= 6 of 8 hours per 8-hour mean).
#
# Neither is an official CPCB AQI bulletin. Flagged sensor series (e.g. a quantised CO
# series) still enter the measured AQI as measured.

# %%
aqi_rows = []
for s in prep:
    f = forecast[forecast.station == s]
    pred_aqi = daily_aqi(f.pivot(index="timestamp", columns="pollutant", values="cnn_lstm_pred"))
    meas_aqi = daily_aqi(f.pivot(index="timestamp", columns="pollutant", values="actual"))
    aqi_rows.append(pd.DataFrame({
        "station": s, "date": pred_aqi.index,
        "predicted_aqi": pred_aqi.aqi.round(1).values,
        "predicted_category": pred_aqi.category.values,
        "predicted_dominant": pred_aqi.dominant.values,
        "measured_aqi": meas_aqi.aqi.round(1).values,
        "measured_category": meas_aqi.category.values,
        "measured_dominant": meas_aqi.dominant.values,
        "measured_subindices": meas_aqi.n_subindices.values}))
aqi_daily = pd.concat(aqi_rows, ignore_index=True)
aqi_daily.to_csv(f"{OUT}/metrics/daily_aqi.csv", index=False)
both = aqi_daily.dropna(subset=["measured_aqi"])
aqi_summary = pd.DataFrame([{
    "station": s, "forecast_days": int((aqi_daily.station == s).sum()),
    "days_with_measured_aqi": len(g),
    "category_agreement_pct": 100 * (g.predicted_category == g.measured_category).mean()
    if len(g) else np.nan,
    "aqi_rmse": np.sqrt(np.mean((g.predicted_aqi - g.measured_aqi) ** 2)) if len(g) else np.nan,
    "predicted_aqi_mean": aqi_daily.predicted_aqi[aqi_daily.station == s].mean(),
    "measured_aqi_mean": g.measured_aqi.mean() if len(g) else np.nan,
} for s in prep for g in [both[both.station == s]]])
aqi_summary.to_csv(f"{OUT}/metrics/aqi_summary.csv", index=False)
aqi_summary.round(2)

# %% [markdown]
# ## 15. Visualisations
#
# Grey shading marks periods with no measurement, where only the forecast exists.

# %%
METHOD_COLORS = {"CNN-LSTM": "#2a78d6", "Climatology": "#eb6834",
                 "Same hour last year": "#1baf7a", "Persistence": "#eda100"}
YEAR = FORECAST_START.year


def shade_unmeasured(ax, measured_daily):
    """Shade runs of days with no measurement."""
    gap = ~measured_daily.values
    run_id = np.cumsum(np.r_[True, gap[1:] != gap[:-1]])
    for r in np.unique(run_id[gap]):
        days = measured_daily.index[run_id == r]
        ax.axvspan(days[0], days[-1] + pd.Timedelta(days=1), color="#bdbdbd", alpha=0.35,
                   lw=0, label="No measurement")


for s in prep:
    sub = forecast[forecast.station == s]
    fig, axes = plt.subplots(len(POLL), 1, figsize=(12, 16), sharex=True)
    for ax, p in zip(axes, POLL):
        g = sub[sub.pollutant == p].set_index("timestamp").sort_index()
        daily = g[["actual", "cnn_lstm_pred", "climatology_pred"]].resample("D").mean()
        shade_unmeasured(ax, daily.actual.notna())
        ax.plot(daily.index, daily.actual, lw=1.2, color="#3b3b3b", label="Measured")
        ax.plot(daily.index, daily.cnn_lstm_pred, lw=1.2, color=METHOD_COLORS["CNN-LSTM"],
                label="CNN-LSTM recursive forecast")
        ax.plot(daily.index, daily.climatology_pred, lw=1.2, ls="--",
                color=METHOD_COLORS["Climatology"], label="Climatology")
        ax.set_ylabel(f"{p} ({UNITS[p]})")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0].legend(uniq.values(), uniq.keys(), ncol=4, loc="upper right", frameon=False)
    axes[-1].set_xlabel(f"{YEAR} (daily mean of hourly values; grey = no measurement)")
    note = "" if sub.actual.notna().any() else f" - no {YEAR} measurements exist"
    fig.suptitle(f"{s}: recursive {YEAR} forecast from data up to {TRAINING_CUTOFF:%Y-%m-%d}"
                 f" vs measured{note}", y=0.995)
    fig.tight_layout()
    savefig(fig, f"fig05_forecast_{s.replace(' ', '_')}")

# %%
fig, axes = plt.subplots(len(prep), 1, figsize=(12, 9), sharex=True)
for ax, s in zip(axes, prep):
    g = forecast[(forecast.station == s) & (forecast.pollutant == "PM2.5")]
    g = g[g.timestamp < FORECAST_START + pd.Timedelta(days=14)].set_index("timestamp")
    ax.plot(g.index, g.actual, lw=1, color="#3b3b3b", label="Measured")
    ax.plot(g.index, g.cnn_lstm_pred, lw=1.4, color=METHOD_COLORS["CNN-LSTM"],
            label="CNN-LSTM recursive forecast")
    ax.plot(g.index, g.climatology_pred, lw=1, ls="--",
            color=METHOD_COLORS["Climatology"], label="Climatology")
    ax.set_title(s if g.actual.notna().any() else f"{s} (no measurements)", fontsize=9)
    ax.set_ylabel("PM2.5 (µg/m³)")
axes[0].legend(ncol=3, loc="upper right", frameon=False)
axes[-1].set_xlabel(f"First 14 days of {YEAR} (hourly)")
fig.suptitle(f"How the recursive forecast evolves: PM2.5, 1-14 January {YEAR}", y=0.995)
fig.tight_layout()
savefig(fig, "fig05b_first_two_weeks_pm25")


# %%
def grid_plot(kind, name, title, df, label):
    fig, axes = plt.subplots(len(POLL), 4, figsize=(14, 22))
    for j, s in enumerate(prep):
        for i, p in enumerate(POLL):
            ax = axes[i, j]
            ax.set_title(f"{s} - {p}", fontsize=9)
            g = df[(df.station == s) & (df.pollutant == p)]
            if g.empty:
                ax.text(0.5, 0.5, "No measured hours", ha="center", va="center",
                        transform=ax.transAxes, color="#52514e")
                continue
            if kind == "scatter":
                ax.scatter(g.actual, g.cnn_lstm_pred, s=3, alpha=0.25, color=STATION_COLORS[s],
                           edgecolors="none", label=label)
                lim = [0, max(g.actual.max(), g.cnn_lstm_pred.max())]
                ax.plot(lim, lim, color="#3b3b3b", lw=1, ls="--", label="y = x")
                ax.set_xlabel(f"Measured ({UNITS[p]})")
                ax.set_ylabel(f"Predicted ({UNITS[p]})")
            else:
                ax.hist(g.cnn_lstm_pred - g.actual, bins=60, color=STATION_COLORS[s], alpha=0.85)
                ax.axvline(0, color="#3b3b3b", lw=1)
                ax.set_xlabel(f"Predicted - measured ({UNITS[p]})")
                ax.set_ylabel("Hours")
    if kind == "scatter":
        axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.997)
    fig.tight_layout()
    savefig(fig, name)


grid_plot("scatter", "fig06_scatter_recursive",
          f"CNN-LSTM recursive {YEAR} forecast vs measured (hourly). Dashed: perfect forecast",
          predictions, f"Measured {YEAR} hours")
grid_plot("hist", "fig07_residuals_recursive",
          f"CNN-LSTM recursive {YEAR} forecast error distribution (hourly)",
          predictions, "")
grid_plot("scatter", "fig10_scatter_one_step",
          f"Rolling one-step CNN-LSTM vs measured, {YEAR} (hourly). Dashed: perfect prediction",
          one_step, f"Measured {YEAR} hours")

# %%
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
x = np.arange(len(prep))
for ax, p in zip(axes.flat, POLL):
    g = station_metrics[station_metrics.pollutant == p].set_index("station").loc[list(prep)]
    for k, name in enumerate(MAIN_METHODS):
        ax.bar(x + (k - 1.5) * 0.2, g[f"{name} RMSE"], 0.19, color=METHOD_COLORS[name],
               label=name)
    ax.set_xticks(x, [s.replace(" ", "\n") for s in prep])
    ax.set_title(p)
    ax.set_ylabel(f"RMSE ({UNITS[p]})")
axes[0, 0].legend(frameon=False, fontsize=7)
ax = axes.flat[-1]
SKILL_YLIM = (-1.0, 0.5)
for k, s in enumerate(prep):
    vals = skill_summary.loc[s, POLL]
    ax.bar(np.arange(len(POLL)) + (k - 1.5) * 0.2, vals.clip(lower=SKILL_YLIM[0]), 0.19,
           color=STATION_COLORS[s], label=s)
    for j, v in enumerate(vals):
        if v < SKILL_YLIM[0]:  # label bars clipped by the axis instead of letting one dwarf all
            ax.annotate(f"{v:.1f}", (j + (k - 1.5) * 0.2, SKILL_YLIM[0]), ha="center",
                        va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
ax.set_ylim(*SKILL_YLIM)
ax.axhline(0, color="#3b3b3b", lw=1)
ax.set_xticks(range(len(POLL)), POLL, rotation=45)
ax.set_ylabel("1 - RMSE_model / RMSE_climatology")
ax.set_title("CNN-LSTM skill vs climatology")
ax.legend(frameon=False, fontsize=7)
fig.suptitle(f"Recursive {YEAR} forecast: hourly RMSE on measured hours "
             "(missing bars = no measurements)")
fig.tight_layout()
savefig(fig, "fig08_station_metrics")

# %%
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, p in zip(axes.flat, POLL):
    g = rmse_by_month[rmse_by_month.pollutant == p]
    for name in MAIN_METHODS:
        ax.plot(g.month, g[name], lw=2 if name == "CNN-LSTM" else 1.2, marker="o", ms=3,
                color=METHOD_COLORS[name], label=name)
    ax.set_xticks(range(1, 13))
    ax.set_title(p)
    ax.set_xlabel(f"Month of {YEAR} (forecast lead)")
    ax.set_ylabel(f"RMSE ({UNITS[p]})")
axes.flat[-1].axis("off")
axes[0, 0].legend(frameon=False, fontsize=7)
fig.suptitle(f"Recursive forecast: hourly RMSE by month of {YEAR}, unflagged stations pooled "
             "(months without measurements are absent)")
fig.tight_layout()
savefig(fig, "fig09_rmse_by_month")

# %%
AQI_COLORS = ["#009865", "#a3c853", "#fff833", "#f29c33", "#e93f33", "#af2d24"]
fig, axes = plt.subplots(len(prep), 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, prep):
    g = aqi_daily[aqi_daily.station == s]
    for lo, hi, colour, cat in zip(AQI_EDGES[:-1], AQI_EDGES[1:], AQI_COLORS, CATEGORIES):
        ax.axhspan(lo, hi, color=colour, alpha=0.15, lw=0)
    ax.plot(g.date, g.predicted_aqi, lw=1.4, color=METHOD_COLORS["CNN-LSTM"],
            label="Predicted AQI (from recursive forecast)")
    ax.scatter(g.date, g.measured_aqi, s=6, color="#3b3b3b", label="Measured AQI (valid days)")
    top = np.nanmax([g.predicted_aqi.max(), g.measured_aqi.max(), 100])
    ax.set_ylim(0, min(500, top * 1.1))
    ax.set_ylabel("AQI")
    n_meas = int(g.measured_aqi.notna().sum())
    ax.set_title(f"{s} ({n_meas} days with a valid measured AQI)", fontsize=9)
axes[0].legend(ncol=2, loc="upper right", frameon=False, fontsize=7)
axes[-1].set_xlabel(f"{YEAR}. Background bands: CPCB categories "
                    + ", ".join(f"{c} ({lo}-{hi})" for c, lo, hi in
                                zip(CATEGORIES, AQI_EDGES[:-1], AQI_EDGES[1:])))
fig.suptitle(f"Daily CPCB AQI derived from the {YEAR} pollutant forecast "
             "(not an official AQI bulletin)", y=0.995)
fig.tight_layout()
savefig(fig, "fig11_daily_aqi")


# %% [markdown]
# ## 16. Run metadata and export
#
# All tables, metrics, predictions, figures, models and preprocessing objects are in
# `outputs/<mode>/`. When both modes have run and `scripts/build_report.py` is present, it
# builds the DOCX report from them (every number in the report is read from these files).

# %%
def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # Colab: no repository


metadata = {
    "mode": MODE, "model_version": f"cnn_lstm_{MODE}", "git_commit": git_commit(),
    "run_timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
    "training_cutoff": str(TRAINING_CUTOFF), "forecast_origin": str(TRAINING_CUTOFF),
    "horizon_start": str(FORECAST_START), "horizon_end": str(FORECAST_END),
    "horizon_hours": N_FORECAST,
    "forecast_types": {
        "recursive": "every horizon hour from the last 24 h before the origin; predictions "
                     "fed back as inputs; no post-origin measurement used",
        "rolling_one_step": "frozen model; each measured horizon hour predicted from the "
                            "measured previous 24 h"},
    "target_columns": {p: TARGETS[p] for p in POLL},
    "features": POLL + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"],
    "non_negativity": "predictions clipped at 0 concentration (before feedback and on output)",
    "splits": {s: {"train_end": str(split_bounds(p["df"])[1] - pd.Timedelta(hours=1)),
                   "val_start": str(split_bounds(p["df"])[1]),
                   "val_end": str(TRAINING_CUTOFF)} for s, p in prep.items()},
    "selected_design_recursive": selection["selected_design"].to_dict(),
    "selected_design_one_step": selection["one_step_design"].to_dict(),
    "models": {s: {d: f"{OUT}/models/cnn_lstm_{d}_{s.replace(' ', '_')}.keras" for d in DESIGNS}
               for s in prep},
    "preprocessing": {s: f"{OUT}/models/preprocessing_{s.replace(' ', '_')}.joblib" for s in prep},
    "evaluation": {r.station: {"evaluated_from": str(r.first_measured),
                               "evaluated_to": str(r.last_measured),
                               "hours_measured": r.hours_any_pollutant_measured,
                               "months_without_rows": r.months_without_rows}
                   for r in coverage.itertuples()},
    "flagged_series": sorted(f"{s}|{p}" for s, p in flagged),
    "versions": VERSIONS, "seed": SEED, "lookback_hours": LOOKBACK, "k_neighbors": K_NEIGHBORS,
    "batch_size": BATCH, "max_epochs": EPOCHS, "patience": PATIENCE,
    "max_imputed_fraction": MAX_IMPUTED_FRACTION, "stuck_hours": STUCK_HOURS,
    "epochs_run": {s: len(h["loss"]) for s, h in histories.items()},
    "best_epoch": {s: int(np.argmin(h["val_loss"])) + 1 for s, h in histories.items()},
    "station_files": STATION_FILES,
}
with open(f"{OUT}/run_metadata.json", "w") as fh:
    json.dump(metadata, fh, indent=2, default=str)

both_modes_done = all(os.path.exists(f"outputs/{m}/run_metadata.json") for m in MODES)
if os.path.exists("scripts/build_report.py") and both_modes_done:
    try:
        import docx  # noqa: F401  (python-docx)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-docx"], check=True)
    subprocess.run([sys.executable, "scripts/build_report.py"], check=True)
if IN_COLAB:
    import shutil
    shutil.make_archive("nashik_outputs", "zip", "outputs")
    files.download("nashik_outputs.zip")
print(sorted(glob.glob(f"{OUT}/**/*.*", recursive=True)))
