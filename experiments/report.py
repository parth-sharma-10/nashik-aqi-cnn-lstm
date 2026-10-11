"""Metrics tables, figures and REPORT.md for the multi-horizon experiments.

Called by `python experiments/exp_multihorizon.py report`. Reads only experiments/outputs/.
Every number in REPORT.md comes from the metric files written here.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp_multihorizon as E  # noqa: E402

LABELS = {"persistence": "Persistence", "daily_naive": "Daily naive (24 h)",
          "weekly_naive": "Weekly naive (168 h)", "same_hour_last_year": "Same hour last year",
          "clim_month_hour": "Climatology month x hour", "clim_doy_hour": "Climatology day-of-year",
          "damped_anomaly": "Damped anomaly", "cnn_lstm_recursive": "CNN-LSTM recursive (baseline)",
          "direct": "Direct CNN-LSTM", "hybrid_direct": "Hybrid: direct + climatology",
          "direct_chained": "Direct CNN-LSTM, chained", "hybrid_direct_chained":
          "Hybrid: chained direct + climatology", "hybrid_recursive":
          "Hybrid: recursive + climatology"}
COLORS = {"persistence": "#eda100", "daily_naive": "#9b6a00", "weekly_naive": "#c9a24a",
          "same_hour_last_year": "#1baf7a", "clim_month_hour": "#eb6834",
          "clim_doy_hour": "#a8431c", "damped_anomaly": "#8a5cc2",
          "cnn_lstm_recursive": "#52514e", "direct": "#2a78d6", "direct_chained": "#2a78d6",
          "hybrid_direct": "#0b3d91", "hybrid_direct_chained": "#0b3d91",
          "hybrid_recursive": "#7d7d7d"}
STATION_ORDER = E.STATIONS


def unflagged(df, flagged):
    return df[[(f, s, p) not in flagged for f, s, p in zip(df.fold, df.station, df.pollutant)]]


def station_scores(df, methods, extra_keys=()):
    keys = ["fold", "station", "pollutant", *extra_keys]
    return E.skill_table(E.scores(df, methods, keys), keys)


def summarise(sc, keys):
    """Mean over unflagged station x pollutant series of RMSE skill vs month x hour clim."""
    return sc.groupby([*keys, "method"], sort=False).agg(
        series=("skill_vs_clim", "size"), mean_skill=("skill_vs_clim", "mean"),
        median_skill=("skill_vs_clim", "median"),
        series_better=("skill_vs_clim", lambda s: int((s > 0).sum())),
        n=("n", "sum")).reset_index()


def md_table(df, floatfmt=3):
    def fmt(v):
        if isinstance(v, float):
            return "-" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def pivot_skill(summary, col, methods, pct=True):
    t = summary.pivot(index="method", columns=col, values="mean_skill").reindex(methods)
    t = (100 * t).round(1) if pct else t
    t.insert(0, "Method", [LABELS[m] for m in t.index])
    return t.reset_index(drop=True)


def build(root=E.EXP_OUT):
    mdir, fdir = f"{root}/metrics", f"{root}/figures"
    os.makedirs(mdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    folds = [f for f in E.FOLDS if os.path.exists(f"{root}/folds/{f}/year_ahead.csv.gz")]
    year = pd.concat([E.load_fold(f, "year", root) for f in folds])
    short = pd.concat([E.load_fold(f, "short", root) for f in folds
                       if os.path.exists(f"{root}/folds/{f}/short_horizon.csv.gz")])
    flagged = E.quality_flags(year)
    metas = {f: json.load(open(f"{root}/folds/{f}/fold_meta.json")) for f in folds}
    sel = json.load(open(f"{root}/selection.json"))

    # ---- year ahead
    ysc = station_scores(year, E.YEAR_METHODS)
    ysc["quality_flag"] = [(f, s, p) in flagged for f, s, p in zip(ysc.fold, ysc.station,
                                                                   ysc.pollutant)]
    ysc.to_csv(f"{mdir}/year_ahead_station_metrics.csv", index=False)
    yu = unflagged(year, flagged)
    ysum = summarise(ysc[~ysc.quality_flag], ["fold"])
    ysum.to_csv(f"{mdir}/year_ahead_summary.csv", index=False)
    year_month = summarise(station_scores(yu.assign(month=yu.timestamp.dt.month),
                                          E.YEAR_METHODS, ["month"]), ["fold", "month"])
    year_month.to_csv(f"{mdir}/year_ahead_by_month.csv", index=False)
    year_season = summarise(station_scores(yu, E.YEAR_METHODS, ["season"]), ["fold", "season"])
    year_season.to_csv(f"{mdir}/year_ahead_by_season.csv", index=False)

    # ---- short horizons: 24 h = leads 1-24, 168 h = leads 1-168
    short["horizon"] = "168h"
    sh = pd.concat([short[short.lead <= 24].assign(horizon="24h"), short])
    ssc = station_scores(sh, E.SHORT_METHODS, ["horizon"])
    ssc["quality_flag"] = [(f, s, p) in flagged for f, s, p in zip(ssc.fold, ssc.station,
                                                                   ssc.pollutant)]
    ssc.to_csv(f"{mdir}/short_horizon_station_metrics.csv", index=False)
    ssum = summarise(ssc[~ssc.quality_flag], ["fold", "horizon"])
    ssum.to_csv(f"{mdir}/short_horizon_summary.csv", index=False)
    su = unflagged(short, flagged)
    su = su.assign(bucket=[f"{lo}-{hi}" for lo, hi in
                           np.array(E.SHORT_BUCKETS)[E.bucket_of(su.lead, E.SHORT_BUCKETS)]])
    sb = summarise(station_scores(su, E.SHORT_METHODS, ["bucket"]), ["fold", "bucket"])
    sb.to_csv(f"{mdir}/short_horizon_by_lead.csv", index=False)
    ss = summarise(station_scores(su[su.lead <= 24], E.SHORT_METHODS, ["season"]),
                   ["fold", "season"])
    ss.to_csv(f"{mdir}/short_horizon_24h_by_season.csv", index=False)

    # ---- drift: monthly mean forecast / monthly mean climatology (all hours, not just
    # measured ones), plus the extreme-value and negative counts recorded by the folds.
    allyear = pd.concat([pd.read_csv(f"{root}/folds/{f}/year_ahead.csv.gz",
                                     parse_dates=["timestamp"]) for f in folds])
    allyear["month"] = allyear.timestamp.dt.month
    nn_cols = ["cnn_lstm_recursive", "direct_chained", "hybrid_direct_chained", "hybrid_recursive"]
    mm = allyear.groupby(["fold", "station", "pollutant", "month"])[
        nn_cols + ["clim_month_hour"]].mean()
    ratio = mm[nn_cols].div(mm["clim_month_hour"], axis=0)
    drift = pd.DataFrame({
        "share of station-pollutant-months outside 1/3-3x climatology":
            ((ratio > 3) | (ratio < 1 / 3)).groupby("fold").mean().stack(),
        "max ratio": ratio.groupby("fold").max().stack(),
        "min ratio": ratio.groupby("fold").min().stack()}).reset_index()
    drift.columns = ["fold", "method", *drift.columns[2:]]
    extreme = []
    for f, m in metas.items():
        for st in m["stations"]:
            if not st.get("excluded"):
                extreme += [{"fold": f, "method": k, "above_2x_p999": st["year_above_2x_p999"][k]}
                            for k in nn_cols]
    extreme = pd.DataFrame(extreme).groupby(["fold", "method"]).sum().reset_index()
    floors = pd.read_csv(f"{root}/metrics/floor_diagnostics.csv")
    floors = floors[floors.protocol == "year_ahead"].groupby(["fold", "method"])[
        ["outputs", "below_zero_before_floor"]].sum().reset_index()
    floors["pct_floored_at_zero"] = 100 * floors.below_zero_before_floor / floors.outputs
    drift = drift.merge(extreme, on=["fold", "method"]).merge(
        floors[["fold", "method", "pct_floored_at_zero"]], on=["fold", "method"], how="left")
    drift.to_csv(f"{mdir}/drift_diagnostics.csv", index=False)

    # ---- coverage
    cov = []
    for f, m in metas.items():
        for st in m["stations"]:
            yrows = allyear[(allyear.fold == f) & (allyear.station == st["station"])]
            cov.append({"fold": f, "station": st["station"],
                        "status": "excluded: " + st["reason"] if st.get("excluded") else "used",
                        "short_origins_scored": st.get("short_origins_scored", 0),
                        "short_origins_possible": st.get("short_origins_candidates", 0),
                        "year_cells_measured": int((yrows.ground_truth == "observed").sum()),
                        "year_cells_total": len(yrows)})
    cov = pd.DataFrame(cov)
    cov.to_csv(f"{mdir}/coverage.csv", index=False)

    figures(root, fdir, sb, year_month, allyear, sel)
    write_report(root, sel, ysum, ssum, sb, year_season, ss, drift, cov, flagged, metas, ysc)


def figures(root, fdir, sb, year_month, allyear, sel):
    sel_folds = list(E.SELECTION_FOLDS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, f in zip(axes, sel_folds):
        g = sb[sb.fold == f]
        order = [f"{lo}-{hi}" for lo, hi in E.SHORT_BUCKETS]
        for m in E.SHORT_METHODS:
            if m == "clim_month_hour":
                continue
            gm = g[g.method == m].set_index("bucket").reindex(order)
            ax.plot(order, 100 * gm.mean_skill, marker="o", lw=2 if "direct" in m else 1.2,
                    color=COLORS[m], ls="--" if m in ("clim_doy_hour", "weekly_naive") else "-",
                    label=LABELS[m])
        ax.axhline(0, color="#3b3b3b", lw=1)
        ax.set_ylim(-60, 60)
        ax.set_title(f"{f}: rolling daily origins")
        ax.set_xlabel("Lead time (hours)")
    axes[0].set_ylabel("Mean RMSE skill vs month x hour climatology (%)")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Short horizons: skill against climatology by lead (unflagged series; "
                 "values below -60% clipped by the axis)")
    fig.tight_layout()
    fig.savefig(f"{fdir}/short_horizon_skill_by_lead.png", dpi=140)
    plt.close(fig)

    folds = sorted(year_month.fold.unique())
    fig, axes = plt.subplots(1, len(folds), figsize=(5 * len(folds), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, folds):
        g = year_month[year_month.fold == f]
        for m in E.YEAR_METHODS:
            if m in ("clim_month_hour", "persistence", "same_hour_last_year"):
                continue
            gm = g[g.method == m].set_index("month")
            ax.plot(gm.index, 100 * gm.mean_skill, marker="o", ms=3, color=COLORS[m],
                    lw=2 if m == sel["selected_method"] else 1.2,
                    ls=":" if m == "hybrid_recursive" else "-", label=LABELS[m])
        ax.axhline(0, color="#3b3b3b", lw=1)
        ax.set_ylim(-150, 40)
        ax.set_xticks(range(1, 13))
        ax.set_title(f"{f}{' (holdout)' if f not in E.SELECTION_FOLDS else ''}")
        ax.set_xlabel("Month of the forecast year")
    axes[0].set_ylabel("Mean RMSE skill vs climatology (%)")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Year-ahead forecasts from 1 January: skill by month (measured hours, "
                 "unflagged series; values below -150% clipped by the axis)")
    fig.tight_layout()
    fig.savefig(f"{fdir}/year_ahead_skill_by_month.png", dpi=140)
    plt.close(fig)

    if os.path.exists(f"{root}/forecast_2026_selected.csv"):
        f26 = pd.read_csv(f"{root}/forecast_2026_selected.csv", parse_dates=["timestamp"])
        fig, axes = plt.subplots(len(STATION_ORDER), 1, figsize=(12, 11), sharex=True)
        for ax, s in zip(axes, STATION_ORDER):
            g = f26[(f26.station == s) & (f26.pollutant == "PM2.5")].set_index("timestamp")
            daily = g[["actual", "forecast", "clim_month_hour",
                       "baseline_cnn_lstm_recursive"]].resample("D").mean()
            nomeas = daily.actual.isna()
            ax.fill_between(daily.index, 0, 1, where=nomeas.values, color="#bdbdbd", alpha=0.35,
                            transform=ax.get_xaxis_transform(), lw=0, label="No measurement")
            ax.plot(daily.index, daily.actual, color="#3b3b3b", lw=1.1, label="Measured")
            ax.plot(daily.index, daily.baseline_cnn_lstm_recursive, color="#52514e", lw=1,
                    ls=":", label="Baseline recursive CNN-LSTM")
            ax.plot(daily.index, daily.forecast, color="#2a78d6", lw=1.6,
                    label=f"Selected: {LABELS[sel['selected_method']]}")
            ax.set_title(s, fontsize=9)
            ax.set_ylabel("PM2.5 (µg/m³)")
        axes[0].legend(ncol=4, fontsize=7, frameon=False, loc="upper right")
        axes[-1].set_xlabel("2026 (daily means)")
        fig.suptitle("2026 PM2.5 forecast from data up to 2025-12-31", y=0.995)
        fig.tight_layout()
        fig.savefig(f"{fdir}/forecast_2026_pm25.png", dpi=140)
        plt.close(fig)


def write_report(root, sel, ysum, ssum, sb, year_season, ss, drift, cov, flagged, metas, ysc):
    from report_text import render  # narrative kept separate from the computations
    text = render(root=root, sel=sel, ysum=ysum, ssum=ssum, sb=sb, year_season=year_season,
                  ss=ss, drift=drift, cov=cov, flagged=flagged, metas=metas, ysc=ysc,
                  md_table=md_table, pivot_skill=pivot_skill, LABELS=LABELS)
    with open(f"{root}/REPORT.md", "w") as fh:
        fh.write(text)
    print(f"{root}/REPORT.md written")
