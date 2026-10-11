"""CPCB National AQI (2014) from hourly pollutant concentrations.

Breakpoints: CPCB National Air Quality Index (2014). PM2.5, PM10, NO2, SO2 and NH3 use the
24-hour mean; CO and O3 use the maximum 8-hour rolling mean of the day. Units: µg/m³,
except CO in mg/m³. The AQI is the largest sub-index, and is reported only when at least
three pollutants have a sub-index and one of them is PM2.5 or PM10. Pb is not measured at
these stations, so it never contributes.
"""
import numpy as np
import pandas as pd

AQI_EDGES = [0, 50, 100, 200, 300, 400, 500]
CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
# Concentration edges between categories; Severe starts at the last value.
BREAKPOINTS = {
    "PM2.5": [0, 30, 60, 90, 120, 250],
    "PM10": [0, 50, 100, 250, 350, 430],
    "NO2": [0, 40, 80, 180, 280, 400],
    "SO2": [0, 40, 80, 380, 800, 1600],
    "NH3": [0, 200, 400, 800, 1200, 1800],
    "CO": [0, 1.0, 2.0, 10, 17, 34],
    "O3": [0, 50, 100, 168, 208, 748],
}
EIGHT_HOUR = ("CO", "O3")
MIN_HOURS_PER_DAY = 16  # a measured daily value needs >= 16 valid hours
MIN_HOURS_PER_8H = 6    # a measured 8-hour mean needs >= 6 of its 8 hours


def sub_index(pollutant, conc):
    """Linear sub-index. The Severe band has no published upper concentration; it is
    extended with the width of the Very Poor band and capped at 500."""
    bp = BREAKPOINTS[pollutant]
    edges = bp + [bp[-1] + (bp[-1] - bp[-2])]
    c = np.asarray(conc, dtype=float)
    idx = np.interp(c, edges, AQI_EDGES)
    return np.where(np.isnan(c), np.nan, np.minimum(idx, 500))


def category(aqi):
    a = np.asarray(aqi, dtype=float)
    band = np.searchsorted(AQI_EDGES[1:-1], np.nan_to_num(a), side="left")
    cats = np.array(CATEGORIES, dtype=object)[band]
    return np.where(np.isnan(a), None, cats)


def daily_aqi(hourly):
    """hourly: DataFrame indexed by hourly timestamps, one column per pollutant (NaN = not
    measured). Returns one row per day with each sub-index, the AQI, its category, the
    dominant pollutant and the number of sub-indices available."""
    day = hourly.index.floor("D")
    subs = {}
    for pol in BREAKPOINTS:
        if pol not in hourly:
            continue
        x = hourly[pol]
        if pol in EIGHT_HOUR:
            rolled = x.rolling(8, min_periods=MIN_HOURS_PER_8H).mean()
            conc = rolled.groupby(day).max()
        else:
            conc = x.groupby(day).mean()
        enough = x.notna().groupby(day).sum() >= MIN_HOURS_PER_DAY
        subs[pol] = pd.Series(sub_index(pol, conc.where(enough)), index=conc.index)
    subs = pd.DataFrame(subs).reindex(columns=list(BREAKPOINTS))
    n = subs.notna().sum(axis=1)
    has_pm = subs[["PM2.5", "PM10"]].notna().any(axis=1)
    valid = (n >= 3) & has_pm
    aqi = subs.max(axis=1).where(valid)
    out = subs.add_prefix("si_")
    out["aqi"] = aqi
    out["category"] = category(aqi)
    out["dominant"] = subs.fillna(-1).idxmax(axis=1).where(valid)
    out["n_subindices"] = n
    out.index.name = "date"
    return out
