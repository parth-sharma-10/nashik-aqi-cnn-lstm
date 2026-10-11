"""Narrative for experiments/outputs/REPORT.md. Numbers come from the metric tables;
statements that depend on the results are guarded by asserts."""
import numpy as np
import pandas as pd

import exp_multihorizon as E


def pct(x):
    return f"{100 * x:+.1f}%"


def render(root, sel, ysum, ssum, sb, year_season, ss, drift, cov, flagged, metas, ysc,
           md_table, pivot_skill, LABELS):
    chosen = sel["selected_method"]
    s_sel = pd.DataFrame(sel["summary"]).set_index("method")
    y = ysum.set_index(["fold", "method"])
    s = ssum.set_index(["fold", "horizon", "method"])

    def ys(fold, m):
        return y.loc[(fold, m), "mean_skill"]

    def sh(fold, h, m):
        return s.loc[(fold, h, m), "mean_skill"]

    holdout_clim_wins = max(ys("F2026", m) for m in ("clim_doy_hour", "damped_anomaly")) \
        > ys("F2026", chosen)
    # Claims made in the text below.
    assert chosen == "hybrid_direct_chained"
    assert s_sel.loc[chosen, "mean_skill"] - s_sel.loc["clim_doy_hour", "mean_skill"] < 0.01
    assert ys("F2025", chosen) < 0 and ys("F2026", chosen) < 0 and holdout_clim_wins
    assert ys("F2026", chosen) > ys("F2026", "cnn_lstm_recursive") + 0.3
    assert all(sh(f, "168h", "hybrid_direct") > 0 for f in E.FOLDS)
    best168 = {f: ssum[(ssum.fold == f) & (ssum.horizon == "168h")].set_index("method")
               .mean_skill.idxmax() for f in E.FOLDS}
    best24 = {f: ssum[(ssum.fold == f) & (ssum.horizon == "24h")].set_index("method")
              .mean_skill.idxmax() for f in E.FOLDS}
    assert all(m in ("direct", "hybrid_direct") for m in best168.values())
    assert best168["F2025"] == best168["F2026"] == "hybrid_direct"

    n_flag = {f: sum(1 for k in flagged if k[0] == f) for f in E.FOLDS}
    w = pd.read_csv(f"{root}/folds/F2026/blend_weights.csv")
    w_long = w[(w.method == chosen) & (w.lead_from >= 169)][E.POLL].values
    d26 = drift[drift.fold == "F2026"].set_index("method")

    out = []
    add = out.append
    add("# Multi-horizon forecasting experiments: Nashik air quality\n")
    add("Branch `exp-multihorizon`, built on the baseline commit 7649c2d. Baseline code and "
        "`outputs/` are unchanged; every file here is under `experiments/`.\n")

    add("## Answer\n")
    add(f"- **Year ahead: the revised method does not beat climatology.** The method chosen by "
        f"the pre-registered rule on the 2024 and 2025 backtests, "
        f"*{LABELS[chosen]}*, had a mean RMSE skill of "
        f"{pct(s_sel.loc[chosen, 'mean_skill'])} against month x hour climatology on those "
        f"folds. That is a tie with day-of-year climatology "
        f"({pct(s_sel.loc['clim_doy_hour', 'mean_skill'])}). The whole margin came from "
        f"Gangapur Road in 2024 ({pct(ys('F2024', chosen))}). In 2025 it was "
        f"{pct(ys('F2025', chosen))}, and on the untouched 2026 holdout "
        f"{pct(ys('F2026', chosen))}, while day-of-year climatology scored "
        f"{pct(ys('F2026', 'clim_doy_hour'))}. For a forecast months ahead, climatology "
        "remains the best available estimate.")
    add(f"- **The revised method does remove the drift.** The baseline recursive CNN-LSTM "
        f"scored {pct(ys('F2026', 'cnn_lstm_recursive'))} on the 2026 holdout. "
        f"{pct(d26.loc['cnn_lstm_recursive', 'share of station-pollutant-months outside 1/3-3x climatology'])[1:]}"
        " of its station-pollutant-months had a mean more than 3 times above or below "
        "climatology. The selected hybrid scored "
        f"{pct(ys('F2026', chosen))}, with "
        f"{pct(d26.loc[chosen, 'share of station-pollutant-months outside 1/3-3x climatology'])[1:]}"
        " of months outside that range.")
    add(f"- **Up to a week ahead, the direct multi-horizon model helps.** At the 168-hour "
        "horizon the best method in every fold was the direct model or its blend with "
        "climatology. The hybrid scored "
        + ", ".join(f"{f} {pct(sh(f, '168h', 'hybrid_direct'))}" for f in E.FOLDS)
        + f" RMSE skill against climatology (best in F2025 and F2026; in F2024 the plain "
        f"direct model was slightly ahead at {pct(sh('F2024', '168h', 'direct'))}). "
        "At 24 hours, simpler baselines are as good or "
        "better: the best method was "
        + ", ".join(f"{LABELS[m]} in {f} ({pct(sh(f, '24h', m))})" for f, m in best24.items())
        + ". The hybrid direct model scored "
        + ", ".join(f"{pct(sh(f, '24h', 'hybrid_direct'))}" for f in E.FOLDS) + " there.\n")

    add("## Setup\n")
    add("**Folds.** A fold is a training cutoff on 1 January. Validation is the 6 months "
        "before it (early stopping and blend weights), and training is everything earlier. "
        "Every scaler, imputer, network, climatology, AR coefficient and blend weight in a "
        "fold uses data before its cutoff only. F2024 and F2025 select the method. F2026 is "
        "a holdout, run only after `selection.json` was written; its run records the "
        "selection it saw, and `final` refuses to proceed if the selection changed "
        "afterwards. Measured 2026 values were never used to tune, select or weight anything.")
    add(f"\nA station takes part in a fold only if its first observation is at least "
        f"{E.MIN_TRAIN_DAYS} days before validation starts. MIDC Ambad, Hirwadi and Pandav "
        "Nagari start in May-June 2023, so F2024 is Gangapur Road only. Pandav Nagari has no "
        "2026 measurements: it has a 2026 forecast but no 2026 score.\n")
    c = cov.copy()
    c["status"] = c.status.str.replace("excluded: first observation", "excluded (starts", regex=False)
    c["status"] = c.status.str.replace(" is less than 180 days before validation starts", ")",
                                       regex=False)
    c.columns = ["Fold", "Station", "Status", "Short-horizon origins with valid input",
                 "Possible origins", "Year-ahead cells measured", "Year-ahead cells"]
    add(md_table(c) + "\n")
    add("**Two protocols, scored on identical rows.**\n")
    add("1. *Short horizons.* A forecast is issued at 00:00 every day of the test year. "
        "Each needs a valid 168-hour input window: every hour present in the file, and at "
        "most half of the pollutant cells imputed. Inputs are measurements before the "
        "origin; models stay frozen at the cutoff. Leads 1-168 h are scored. The 24 h "
        "horizon is leads 1-24.")
    add("2. *Year ahead.* One forecast from 1 January 00:00 for every hour of the year, "
        "scored on measured hours.\n")
    add("Within a protocol, every method is scored on exactly the same station, hour and "
        "pollutant rows. Skill = 1 - RMSE_method / RMSE_climatology(month x hour), "
        "computed per station-pollutant series and then averaged. Series whose measured "
        "values take fewer than 100 distinct values are flagged as sensor faults and "
        f"excluded from the averages (F2026: {n_flag['F2026']} series; none in F2024 or F2025).\n")
    add("**Methods.**\n")
    methods = [
        ("Climatology month x hour", "mean for the same month and hour, all data before "
         "the cutoff (the reference)"),
        ("Climatology day-of-year", "mean for the same hour within +-15 days of the same day "
         "of year, used only where 2 or more years cover that window, otherwise month x hour"),
        ("Persistence", "last observed value"),
        ("Daily / weekly naive", "repeat the last 24 or 168 measured hours (falls back to "
         "climatology where those hours are missing)"),
        ("Same hour last year", "the value 364 days earlier, else climatology"),
        ("Damped anomaly", "climatology + phi^h x (last anomaly from climatology); phi is the "
         "lag-1 autocorrelation of training anomalies"),
        ("CNN-LSTM recursive (baseline)", "the existing one-step network, fed its own "
         "output. Saved baseline checkpoints for F2025 and F2026, retrained with the baseline "
         "recipe for F2024"),
        ("Direct CNN-LSTM", "same network family with a 168 h input, predicting all "
         "168 h x 7 pollutants in one pass (masked MSE, early stopping on validation). "
         "For the full year it is chained: each 168 h block becomes the next input"),
        ("Hybrids", "w x network + (1 - w) x climatology, with least-squares w in [0, 1] "
         "per station, pollutant and lead bucket, fitted on validation forecasts made by "
         "the models trained before validation, with climatology from data before "
         "validation. Leads up to 168 h use the daily validation origins; longer leads use "
         "one validation forecast run from the start of validation (July-December), and "
         "leads beyond 4,416 h reuse the last bucket"),
    ]
    add(md_table(pd.DataFrame(methods, columns=["Method", "Definition"])) + "\n")
    add("**Physical constraint.** Concentrations cannot be negative, so network outputs "
        "are floored at zero before being fed back and on output, as in the baseline. The "
        "share of outputs that hit the floor is reported below. No upper cap is applied. "
        "Predictions above twice the training-period 99.9th percentile are counted: there "
        "were none in any fold.\n")

    add("## Year-ahead results\n")
    t = pivot_skill(ysum, "fold", E.YEAR_METHODS)
    t.columns = ["Method", *[f"{c} skill %" for c in t.columns[1:]]]
    nb = ysum.pivot(index="method", columns="fold", values="series_better").reindex(E.YEAR_METHODS)
    nser = ysum.pivot(index="method", columns="fold", values="series").reindex(E.YEAR_METHODS)
    for f in nb.columns:
        t[f"{f} series better"] = [f"{int(a)}/{int(b)}" for a, b in zip(nb[f], nser[f])]
    add("Mean RMSE skill against month x hour climatology, and the number of "
        "station-pollutant series where the method beats it. F2026 is the holdout.\n")
    add(md_table(t, 1) + "\n")
    add("![Year-ahead skill by month](figures/year_ahead_skill_by_month.png)\n")
    seas = year_season[year_season.method.isin([chosen, "clim_doy_hour", "direct_chained",
                                                "cnn_lstm_recursive"])].copy()
    seas["skill %"] = (100 * seas.mean_skill).round(1)
    seas = seas.pivot_table(index=["fold", "season"], columns="method", values="skill %")
    seas.columns = [LABELS[m] for m in seas.columns]
    add("By IMD season (mean skill %, measured hours only; F2026 has no post-monsoon "
        "measurements):\n")
    add(md_table(seas.reset_index(), 1) + "\n")
    hold = ysc[(ysc.fold == "F2026") & ~ysc.quality_flag & ysc.method.isin(
        [chosen, "clim_doy_hour", "cnn_lstm_recursive"])].pivot_table(
        index=["station", "pollutant"], columns="method", values="skill_vs_clim")
    hold = (100 * hold).round(1)
    hold.columns = [LABELS[m] + " skill %" for m in hold.columns]
    nrow = ysc[(ysc.fold == "F2026") & (ysc.method == chosen) & ~ysc.quality_flag].set_index(
        ["station", "pollutant"]).n
    hold.insert(0, "n", nrow.reindex(hold.index).values)
    add("2026 holdout per series (unflagged):\n")
    add(md_table(hold.reset_index(), 1) + "\n")
    add(f"Why the hybrid does not generalise: its long-lead weights are fitted on a single "
        f"validation trajectory (July-December), and they stay high. In F2026, the weights "
        f"for leads over 168 h average {w_long.mean():.2f} (range {w_long.min():.2f}-"
        f"{w_long.max():.2f}), so the network still carries most of the forecast where it "
        "has no skill. Weights fitted over several origins or years, or shrunk towards "
        "zero with lead, are the obvious next step. That step would need a new "
        "selection fold, because 2026 can no longer be used for it.\n")

    add("## Short-horizon results\n")
    t = pivot_skill(ssum[ssum.horizon == "24h"], "fold", E.SHORT_METHODS)
    t2 = pivot_skill(ssum[ssum.horizon == "168h"], "fold", E.SHORT_METHODS)
    t.columns = ["Method", *[f"{c} 24h %" for c in t.columns[1:]]]
    for c in t2.columns[1:]:
        t[f"{c} 168h %"] = t2[c].values
    add("Mean RMSE skill against month x hour climatology, rolling daily origins:\n")
    add(md_table(t, 1) + "\n")
    lead = sb[sb.fold == "F2025"].pivot(index="method", columns="bucket", values="mean_skill")
    lead_raw = lead.copy()
    lead = (100 * lead.reindex(E.SHORT_METHODS)[[f"{a}-{b}" for a, b in E.SHORT_BUCKETS]]).round(1)
    lead.insert(0, "Method", [LABELS[m] for m in lead.index])
    lead.columns = ["Method", *[f"{c} h" for c in lead.columns[1:]]]
    add("F2025, skill % by lead bucket (all four stations):\n")
    add(md_table(lead.reset_index(drop=True), 1) + "\n")
    add("![Short-horizon skill by lead](figures/short_horizon_skill_by_lead.png)\n")
    add("The best method changes with lead. For the first hours, the last measurement "
        "carries most of the information (damped anomaly, recursive CNN-LSTM). Within a "
        "day, the daily naive forecast is hard to beat. From 3 to 7 days ahead, the hybrid "
        "direct model leads. 24 h results by season are in "
        "`metrics/short_horizon_24h_by_season.csv`.\n")

    add("## Drift and constraint diagnostics\n")
    dd = drift.copy()
    dd["method"] = dd.method.map(LABELS)
    dd = dd.rename(columns={"share of station-pollutant-months outside 1/3-3x climatology":
                            "months outside 1/3-3x clim (share)",
                            "max ratio": "max monthly ratio", "min ratio": "min monthly ratio",
                            "above_2x_p999": "values > 2x p99.9",
                            "pct_floored_at_zero": "% outputs floored at 0"})
    add("Monthly mean forecast divided by monthly mean climatology, over every hour of the "
        "year (measured or not); the hybrids are blends of floored outputs, so they have no "
        "floor count of their own.\n")
    add(md_table(dd, 3) + "\n")

    add("## Deliverable: 2026 forecast\n")
    add(f"`forecast_2026_selected.csv` has 8,760 hours x 7 pollutants x 4 stations = 245,280 "
        f"rows. Its forecast column is the selected method, {LABELS[chosen]}: trained, "
        "early-stopped and weighted on data up to 2025-12-31. Alongside it are the "
        "month x hour climatology and the baseline recursive CNN-LSTM, plus `actual` and "
        "`ground_truth` (`observed` / `missing_value` / `no_row_in_file`). July and "
        "September-December 2026, and all of Pandav Nagari, have no ground truth.\n")
    add("![2026 PM2.5 forecast](figures/forecast_2026_pm25.png)\n")
    f26 = pd.read_csv(f"{root}/forecast_2026_selected.csv", parse_dates=["timestamp"])
    f26["month"] = f26.timestamp.dt.month
    mm = f26.groupby(["station", "pollutant", "month"])[["forecast", "clim_month_hour"]].mean()
    mm["ratio"] = mm.forecast / mm.clim_month_hour
    odd = mm[(mm.ratio < 0.5) | (mm.ratio > 2)].reset_index()
    assert len(odd) > 0 and ("Hirwadi", "PM2.5") in set(zip(odd.station, odd.pollutant))
    odd_summary = odd.groupby(["station", "pollutant"]).agg(
        months=("month", lambda m: ", ".join(pd.Timestamp(2026, x, 1).strftime("%b") for x in m)),
        lowest_ratio=("ratio", "min"), highest_ratio=("ratio", "max")).reset_index()
    add(f"Plausibility check: {len(odd)} of {len(mm)} station-pollutant-months of the "
        "deliverable have a monthly mean below half or above twice climatology. Most fall "
        "in months with no measurement to check against:\n")
    add(md_table(odd_summary, 2) + "\n")
    hp = mm.loc[("Hirwadi", "PM2.5", 11)]
    add(f"Hirwadi PM2.5 is the clearest case. The forecast stays near {hp.forecast:.0f} "
        f"µg/m³ in November against a climatology of {hp.clim_month_hour:.0f}, because that "
        "series' long-lead blend weight is high and the chained network shows no winter "
        "rise.\n")
    add("Given the holdout result, use the `clim_month_hour` column (or day-of-year "
        "climatology) for months-ahead planning. The selected hybrid is reported because it "
        "is what the pre-registered rule chose, not because it won.\n")

    add("## Limitations\n")
    lim = [
        "Few independent years: method selection rests on 35 station-pollutant series "
        "from two folds, one of which is Gangapur Road alone. Margins under about 1% are "
        "not distinguishable from noise; no confidence intervals or multiple seeds.",
        "Three stations have about 1.5 years of training data in F2025, and F2024 cannot "
        "use them at all. Day-of-year climatology falls back to month x hour wherever fewer "
        "than two years cover the window; for those stations in F2025 that is roughly "
        "January-May (only 2024 covers it).",
        "Validation is always July-December, so blend weights and early stopping are tuned "
        "on monsoon and post-monsoon behaviour only.",
        "The direct model's best validation epoch was between 1 and 7 in every fold, so it "
        "is lightly trained. Its skill decays with lead (F2025: "
        f"{pct(lead_raw.loc['direct', '1-6'])} at 1-6 h, "
        f"{pct(lead_raw.loc['direct', '73-168'])} at 73-168 h).",
        "2026 holdout covers January-June and August at three stations; flagged sensor "
        "series (Gangapur Road NO2, CO, NH3; MIDC Ambad CO) are excluded from its averages.",
        "Meteorological inputs remain unused (missing or inconsistent sensors), which caps "
        "what any multi-day forecast can learn.",
    ]
    add("\n".join(f"- {x}" for x in lim) + "\n")

    add("## Reproduce\n")
    add("```bash\n"
        "cd ~/nashik-aqi-cnn-lstm   # branch exp-multihorizon; data/ holds the four CSVs\n"
        "# the baseline checkpoints in outputs/backtest_2025 and outputs/forecast_2026 must exist\n"
        ".venv/bin/python experiments/exp_multihorizon.py fold F2024\n"
        ".venv/bin/python experiments/exp_multihorizon.py fold F2025\n"
        ".venv/bin/python experiments/exp_multihorizon.py select\n"
        ".venv/bin/python experiments/exp_multihorizon.py fold F2026   # holdout, after select\n"
        ".venv/bin/python experiments/exp_multihorizon.py final\n"
        ".venv/bin/python experiments/exp_multihorizon.py diagnose\n"
        ".venv/bin/python experiments/exp_multihorizon.py report\n"
        ".venv/bin/python -m unittest discover tests -v\n"
        "```\n")
    add("Each fold takes about 2.5-3.5 minutes on a 10-core laptop CPU; diagnose about "
        "2 minutes. Seeds are fixed (42). Retraining on another TensorFlow build can change "
        "the networks slightly, and with margins this small that could change the "
        "selection.\n")
    return "\n".join(out)
