# %% [markdown]
# # Nashik Air-Quality Prediction with KNN Imputation and CNN-LSTM
#
# Blind forecast of every hour of 2025 for seven criteria pollutants at four Nashik
# monitoring stations, compared against the measured 2025 data. Each station gets its own
# CNN-LSTM, trained on that station's data up to the end of 2024 only. Starting from the
# last 24 hours of 2024, the model predicts one hour, feeds that prediction back as input,
# and repeats for all 8,760 hours of 2025 - no 2025 measurement is ever used as input.
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

SEED = 42
tf.keras.utils.set_random_seed(SEED)  # seeds Python, NumPy and TensorFlow

OUT = "outputs"
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


TEST_YEAR = 2025
clean, cleaning_log = {}, {}
for s, d in raw.items():
    clean[s], cleaning_log[s] = clean_station(d)
    # Data after the forecast year (Jan-Aug 2026) plays no part in this study.
    clean[s] = clean[s][clean[s]["Timestamp"].dt.year <= TEST_YEAR].reset_index(drop=True)
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
    plt.show()


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
# Test = calendar year 2025 at all four stations (all 12 months are present everywhere).
# Validation = Jul-Dec 2024; training = everything before Jul 2024. Data from 2026 is
# dropped. Scaler, imputer, model weights, early stopping and the choice between the two
# model designs all use data up to 2024-12-31 only.

# %%
def split_bounds(df):
    test_year = TEST_YEAR
    test_start = pd.Timestamp(f"{test_year}-01-01")
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
          "2025 windows are built but never used)")
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
# feeding its own output back (small biases accumulate). So the design used for 2025 is
# chosen per station by a **recursive forecast of the validation half-year**
# (Jul-Dec 2024, started from the last 24 hours of June 2024), scored against the
# observed validation values. 2025 plays no part in this choice.

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
BATCH, EPOCHS, PATIENCE = 256, 60, 6
DESIGNS = ("level", "change")


def design_target(w, design):
    return w["y_scaled"] - w["last_scaled"] if design == "change" else w["y_scaled"]


models, histories, selection = {}, {}, []
for s, w in windows.items():
    tr, va = w["split"] == "train", w["split"] == "val"
    row = {"station": s, "train_sequences": int(tr.sum()), "val_sequences": int(va.sum()),
           "val_loss_persistence": float(np.nanmean(
               np.square(w["y_scaled"][va] - w["last_scaled"][va])))}
    for design in DESIGNS:
        tf.keras.utils.set_random_seed(SEED)
        y = design_target(w, design)
        model = build_cnn_lstm()
        hist = model.fit(
            w["X"][tr], y[tr], validation_data=(w["X"][va], y[va]),
            epochs=EPOCHS, batch_size=BATCH, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE,
                                                        restore_best_weights=True)])
        models[(s, design)], histories[f"{s}|{design}"] = model, hist.history
        best = int(np.argmin(hist.history["val_loss"]))
        row.update({f"val_loss_{design}": hist.history["val_loss"][best],
                    f"train_loss_{design}": hist.history["loss"][best],
                    f"best_epoch_{design}": best + 1,
                    f"epochs_run_{design}": len(hist.history["loss"])})
        model.save(f"{OUT}/models/cnn_lstm_{design}_{s.replace(' ', '_')}.keras")
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
selection.to_csv(f"{OUT}/metrics/model_selection_validation.csv")
selection[["val_loss_level", "val_loss_change", "val_recursive_mse_level",
           "val_recursive_mse_change", "selected_design"]].round(4)

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
# ## 11. Forecast of 2025 and baselines
#
# For each station both designs forecast all 8,760 hours of 2025 from the last 24 hours
# of 2024; the design selected on validation is the reported CNN-LSTM forecast. Three
# baselines use the same information (data up to 2024-12-31 only):
# - **Climatology**: mean of the observed values for the same calendar month and hour of
#   day over all data before 2025 (month mean, then overall mean, if a cell is empty).
# - **Same hour last year**: the observed value 364 days earlier (52 weeks, so the weekday
#   matches); where that hour was not measured, the climatology value is used.
# - **Persistence**: the last observed 2024 value of each pollutant, held for the year.
#
# Every forecast is compared with the measured (cleaned) 2025 value of the same station,
# hour and pollutant; hours without a measurement are skipped.

# %%
def climatology(d, ts):
    hist = d[d["Timestamp"] < pd.Timestamp(f"{TEST_YEAR}-01-01")]
    by_month_hour = hist.groupby([hist.Timestamp.dt.month, hist.Timestamp.dt.hour])[POLL].mean()
    by_month = hist.groupby(hist.Timestamp.dt.month)[POLL].mean()
    out = by_month_hour.reindex(pd.MultiIndex.from_arrays([ts.month, ts.hour])).set_axis(ts)
    return out.fillna(by_month.reindex(ts.month).set_axis(ts)).fillna(hist[POLL].mean())


def same_hour_last_year(d, ts, clim):
    hist = d[d["Timestamp"] < pd.Timestamp(f"{TEST_YEAR}-01-01")].set_index("Timestamp")[POLL]
    return hist.reindex(ts - pd.Timedelta(days=364)).set_axis(ts).fillna(clim)


FORECAST_START = pd.Timestamp(f"{TEST_YEAR}-01-01")
N_FORECAST = int((pd.Timestamp(f"{TEST_YEAR + 1}-01-01") - FORECAST_START) / pd.Timedelta(hours=1))
pred_rows = []
for s, p in prep.items():
    d = p["df"]
    yhat = {}
    for design in DESIGNS:
        future, yhat[design], seed_missing = recursive_forecast(
            models[(s, design)], design, p, FORECAST_START, N_FORECAST)
    print(f"{s:14s} forecast {len(future)} hours, {future[0]} -> {future[-1]}; "
          f"starting-window hours absent from the file: {seed_missing}")
    clim = climatology(d, future)
    last_year = same_hour_last_year(d, future, clim)
    persist = d[d["Timestamp"] < FORECAST_START][POLL].ffill().iloc[-1]
    actual = d.set_index("Timestamp")[POLL].reindex(future)
    chosen = selection.loc[s, "selected_design"]
    for j, pol in enumerate(POLL):
        obs = actual[pol].notna().values
        pred_rows.append(pd.DataFrame({
            "station": s, "timestamp": future[obs], "pollutant": pol, "unit": UNITS[pol],
            "actual": actual[pol].values[obs],
            "cnn_lstm_pred": yhat[chosen][obs, j], "selected_design": chosen,
            "climatology_pred": clim[pol].values[obs],
            "same_hour_last_year_pred": last_year[pol].values[obs],
            "persistence_pred": persist[pol],
            "cnn_lstm_level_pred": yhat["level"][obs, j],
            "cnn_lstm_change_pred": yhat["change"][obs, j],
        }))
predictions = pd.concat(pred_rows, ignore_index=True)
predictions["test_year"] = TEST_YEAR
predictions["month"] = pd.to_datetime(predictions["timestamp"]).dt.month
assert predictions.drop(columns="selected_design").notna().all().all()
predictions.to_csv(f"{OUT}/predictions/actual_vs_predicted.csv", index=False)
predictions.head()

# %% [markdown]
# ## 12-13. Evaluation metrics and baselines
#
# MSE, RMSE, MAE and R² from scikit-learn, in original units, per station and pollutant,
# for the CNN-LSTM and each baseline on exactly the same hours. Skill =
# 1 - RMSE_CNN-LSTM / RMSE_climatology (> 0 means the CNN-LSTM beats climatology, the
# strongest standard baseline for a forecast this far ahead).
#
# Hourly scores judge whether the forecast gets each individual hour right. A year-ahead
# forecast can only realistically capture the seasonal level, so monthly means of the
# forecast are also compared with monthly means of the measurements (12 points per
# series), and the hourly RMSE is broken down by month of 2025 to show how error changes
# with lead time.
#
# Data-quality flag: a 2025 series is flagged when it takes fewer than 100 distinct values
# over the year (a stuck or coarsely quantised sensor). Flagged series are reported but left
# out of the pooled all-station summary, which concatenates the hourly rows of the unflagged
# stations for one pollutant. Pollutants are never pooled together.

# %%
METHODS = {"CNN-LSTM": "cnn_lstm_pred", "Climatology": "climatology_pred",
           "Same hour last year": "same_hour_last_year_pred", "Persistence": "persistence_pred"}


def metrics(y, yhat):
    mse = mean_squared_error(y, yhat)
    r2 = r2_score(y, yhat) if len(y) > 1 and np.var(y) > 0 else np.nan
    return {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(y, yhat), "R2": r2}


def evaluate(groups):
    rows = []
    for keys, g in groups:
        base = dict(zip(["station", "pollutant"], keys)) if isinstance(keys, tuple) \
            else {"station": "All stations (pooled)", "pollutant": keys}
        start, end = pd.to_datetime(g.timestamp).min(), pd.to_datetime(g.timestamp).max()
        base.update({"evaluation_period": f"{start:%Y-%m-%d} to {end:%Y-%m-%d}",
                     "unit": g["unit"].iloc[0], "n_test_obs": len(g)})
        for name, col in METHODS.items():
            base.update({f"{name} {k}": v for k, v in metrics(g.actual, g[col]).items()})
        base["RMSE skill vs climatology"] = 1 - base["CNN-LSTM RMSE"] / base["Climatology RMSE"]
        base["level-design RMSE"] = metrics(g.actual, g.cnn_lstm_level_pred)["RMSE"]
        base["change-design RMSE"] = metrics(g.actual, g.cnn_lstm_change_pred)["RMSE"]
        rows.append(base)
    return pd.DataFrame(rows)


MIN_DISTINCT_VALUES = 100

station_metrics = evaluate(predictions.groupby(["station", "pollutant"], sort=False))
distinct = predictions.groupby(["station", "pollutant"], sort=False).actual.nunique()
station_metrics["distinct_test_values"] = distinct.values
station_metrics["quality_flag"] = station_metrics.distinct_test_values < MIN_DISTINCT_VALUES
flagged = set(zip(station_metrics.station[station_metrics.quality_flag],
                  station_metrics.pollutant[station_metrics.quality_flag]))
print("Flagged 2025 series:", flagged)
unflagged = predictions[[(st, po) not in flagged
                         for st, po in zip(predictions.station, predictions.pollutant)]]
overall_metrics = evaluate(unflagged.groupby("pollutant", sort=False))
overall_metrics["stations_included"] = [
    ", ".join(unflagged[unflagged.pollutant == p].station.unique())
    for p in overall_metrics.pollutant]
station_metrics.to_csv(f"{OUT}/metrics/station_pollutant_metrics.csv", index=False)
overall_metrics.to_csv(f"{OUT}/metrics/overall_pollutant_metrics.csv", index=False)

skill_summary = station_metrics.pivot(index="station", columns="pollutant",
                                      values="RMSE skill vs climatology")[POLL]
skill_summary["mean skill (unflagged)"] = station_metrics[~station_metrics.quality_flag] \
    .groupby("station")["RMSE skill vs climatology"].mean()
skill_summary.round(3).to_csv(f"{OUT}/metrics/rmse_skill_summary.csv")

results_table = station_metrics[["station", "pollutant", "evaluation_period", "n_test_obs",
                                 "CNN-LSTM MSE", "CNN-LSTM RMSE", "CNN-LSTM MAE", "CNN-LSTM R2",
                                 "quality_flag"]]
results_table.round(3)

# %%
monthly = predictions.groupby(["station", "pollutant", "month"], sort=False)[
    ["actual", *METHODS.values()]].mean().reset_index()
monthly_rows = []
for (s, p), g in monthly.groupby(["station", "pollutant"], sort=False):
    row = {"station": s, "pollutant": p, "months": len(g)}
    for name, col in METHODS.items():
        m = metrics(g.actual, g[col])
        row[f"{name} monthly RMSE"], row[f"{name} monthly R2"] = m["RMSE"], m["R2"]
    monthly_rows.append(row)
monthly_metrics = pd.DataFrame(monthly_rows)
monthly_metrics["quality_flag"] = [(s, p) in flagged for s, p in
                                   zip(monthly_metrics.station, monthly_metrics.pollutant)]
monthly_metrics.to_csv(f"{OUT}/metrics/monthly_mean_metrics.csv", index=False)

rmse_by_month = unflagged.groupby(["pollutant", "month"], sort=False).apply(
    lambda g: pd.Series({name: np.sqrt(mean_squared_error(g.actual, g[col]))
                         for name, col in METHODS.items()}), include_groups=False).reset_index()
rmse_by_month.to_csv(f"{OUT}/metrics/rmse_by_month.csv", index=False)
monthly_metrics.round(3)

# %%
overall_metrics.round(3)

# %%
skill_summary.round(3)

# %% [markdown]
# ## 14. Visualisations of the 2025 forecast

# %%
METHOD_COLORS = {"CNN-LSTM": "#2a78d6", "Climatology": "#eb6834",
                 "Same hour last year": "#1baf7a", "Persistence": "#eda100"}

for s in prep:
    sub = predictions[predictions.station == s]
    fig, axes = plt.subplots(len(POLL), 1, figsize=(12, 16), sharex=True)
    for ax, p in zip(axes, POLL):
        g = sub[sub.pollutant == p].set_index("timestamp").sort_index()
        daily = g[["actual", "cnn_lstm_pred", "climatology_pred"]].resample("D").mean()
        ax.plot(daily.index, daily.actual, lw=1.2, color="#3b3b3b", label="Measured")
        ax.plot(daily.index, daily.cnn_lstm_pred, lw=1.2, color=METHOD_COLORS["CNN-LSTM"],
                label="CNN-LSTM forecast")
        ax.plot(daily.index, daily.climatology_pred, lw=1.2, ls="--",
                color=METHOD_COLORS["Climatology"], label="Climatology")
        ax.set_ylabel(f"{p} ({UNITS[p]})")
    axes[0].legend(ncol=3, loc="upper right", frameon=False)
    axes[-1].set_xlabel("2025 (daily mean of hourly values; blank = no measurement)")
    fig.suptitle(f"{s}: 2025 forecast from data up to 2024 vs measured 2025", y=0.995)
    fig.tight_layout()
    savefig(fig, f"fig05_actual_vs_pred_{s.replace(' ', '_')}")

# %%
fig, axes = plt.subplots(len(prep), 1, figsize=(12, 9), sharex=True)
for ax, s in zip(axes, prep):
    g = predictions[(predictions.station == s) & (predictions.pollutant == "PM2.5")]
    g = g[pd.to_datetime(g.timestamp) < FORECAST_START + pd.Timedelta(days=14)]
    g = g.set_index("timestamp").sort_index()
    full = g.reindex(pd.date_range(FORECAST_START, periods=14 * 24, freq="h"))
    ax.plot(full.index, full.actual, lw=1, color="#3b3b3b", label="Measured")
    ax.plot(full.index, full.cnn_lstm_pred, lw=1.4, color=METHOD_COLORS["CNN-LSTM"],
            label="CNN-LSTM forecast")
    ax.plot(full.index, full.climatology_pred, lw=1, ls="--",
            color=METHOD_COLORS["Climatology"], label="Climatology")
    ax.set_title(s, fontsize=9)
    ax.set_ylabel("PM2.5 (µg/m³)")
axes[0].legend(ncol=3, loc="upper right", frameon=False)
axes[-1].set_xlabel("First 14 days of 2025 (hourly)")
fig.suptitle("How the recursive forecast evolves: PM2.5, 1-14 January 2025", y=0.995)
fig.tight_layout()
savefig(fig, "fig05b_first_two_weeks_pm25")


# %%
def grid_plot(kind, name, title):
    fig, axes = plt.subplots(len(POLL), 4, figsize=(14, 22))
    for j, s in enumerate(prep):
        for i, p in enumerate(POLL):
            ax = axes[i, j]
            g = predictions[(predictions.station == s) & (predictions.pollutant == p)]
            if kind == "scatter":
                ax.scatter(g.actual, g.cnn_lstm_pred, s=3, alpha=0.25, color=STATION_COLORS[s],
                           edgecolors="none", label="Hourly 2025 obs.")
                lim = [0, max(g.actual.max(), g.cnn_lstm_pred.max())]
                ax.plot(lim, lim, color="#3b3b3b", lw=1, ls="--", label="y = x")
                ax.set_xlabel(f"Measured ({UNITS[p]})")
                ax.set_ylabel(f"Forecast ({UNITS[p]})")
            else:
                res = g.cnn_lstm_pred - g.actual
                ax.hist(res, bins=60, color=STATION_COLORS[s], alpha=0.85)
                ax.axvline(0, color="#3b3b3b", lw=1)
                ax.set_xlabel(f"Forecast - measured ({UNITS[p]})")
                ax.set_ylabel("Hours")
            ax.set_title(f"{s} - {p}", fontsize=9)
    if kind == "scatter":
        axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=0.997)
    fig.tight_layout()
    savefig(fig, name)


grid_plot("scatter", "fig06_scatter_actual_vs_pred",
          "CNN-LSTM 2025 forecast vs measured (hourly). Dashed line: perfect forecast")
grid_plot("hist", "fig07_residuals", "CNN-LSTM 2025 forecast error distribution (hourly)")

# %%
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
x = np.arange(len(prep))
for ax, p in zip(axes.flat, POLL):
    g = station_metrics[station_metrics.pollutant == p].set_index("station").loc[list(prep)]
    for k, name in enumerate(METHODS):
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
fig.suptitle("2025 hourly RMSE per station: CNN-LSTM vs baselines")
fig.tight_layout()
savefig(fig, "fig08_station_metrics")

# %%
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, p in zip(axes.flat, POLL):
    g = rmse_by_month[rmse_by_month.pollutant == p]
    for name in METHODS:
        ax.plot(g.month, g[name], lw=2 if name == "CNN-LSTM" else 1.2, marker="o", ms=3,
                color=METHOD_COLORS[name], label=name)
    ax.set_xticks(range(1, 13))
    ax.set_title(p)
    ax.set_xlabel("Month of 2025 (forecast lead)")
    ax.set_ylabel(f"RMSE ({UNITS[p]})")
axes.flat[-1].axis("off")
axes[0, 0].legend(frameon=False, fontsize=7)
fig.suptitle("Hourly RMSE by month of 2025, all unflagged stations pooled")
fig.tight_layout()
savefig(fig, "fig09_rmse_by_month")


# %% [markdown]
# ## 15. Export summary
#
# All tables, metrics, predictions, figures, models and preprocessing objects are in
# `outputs/`. If `scripts/build_report.py` is present it builds the DOCX report from them
# (every number in the report is read from these files).

# %%
with open(f"{OUT}/metrics/run_info.json", "w") as fh:
    json.dump({"versions": VERSIONS, "seed": SEED, "lookback_hours": LOOKBACK,
               "test_year": TEST_YEAR, "forecast": "recursive, from data up to 2024-12-31",
               "forecast_hours": N_FORECAST, "k_neighbors": K_NEIGHBORS, "batch_size": BATCH,
               "max_epochs": EPOCHS, "patience": PATIENCE,
               "max_imputed_fraction": MAX_IMPUTED_FRACTION, "stuck_hours": STUCK_HOURS,
               "epochs_run": {s: len(h["loss"]) for s, h in histories.items()},
               "best_epoch": {s: int(np.argmin(h["val_loss"])) + 1 for s, h in histories.items()},
               "station_files": STATION_FILES}, fh, indent=2, default=str)
if os.path.exists("scripts/build_report.py"):
    import subprocess
    import sys
    try:
        import docx  # noqa: F401  (python-docx)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-docx"], check=True)
    subprocess.run([sys.executable, "scripts/build_report.py"], check=True)
if IN_COLAB:
    import shutil
    shutil.make_archive("nashik_outputs", "zip", OUT)
    files.download("nashik_outputs.zip")
print(sorted(glob.glob(f"{OUT}/**/*.*", recursive=True)))
