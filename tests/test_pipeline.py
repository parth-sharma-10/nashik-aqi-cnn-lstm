"""Run from the project root:  python -m unittest discover tests -v

The AQI tests run anywhere. The output tests read outputs/<mode>/ and are skipped, with
the reason printed, when that mode has not been run yet.
"""
import os
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aqi_cpcb import category, daily_aqi, sub_index  # noqa: E402

MODES = {"backtest_2025": 2025, "forecast_2026": 2026}
POLL = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
STATIONS = ["MIDC Ambad", "Gangapur Road", "Pandav Nagari", "Hirwadi"]


class CpcbAqi(unittest.TestCase):
    def test_breakpoints(self):
        np.testing.assert_allclose(sub_index("PM2.5", [0, 30, 45, 60, 90, 120, 250]),
                                   [0, 50, 75, 100, 200, 300, 400])
        np.testing.assert_allclose(sub_index("CO", [1.0, 2.0, 10]), [50, 100, 200])
        np.testing.assert_allclose(sub_index("O3", [168, 208]), [200, 300])
        self.assertEqual(sub_index("PM10", [5000])[0], 500)
        self.assertTrue(np.isnan(sub_index("NO2", [np.nan])[0]))

    def test_categories(self):
        self.assertEqual(list(category([0, 50, 51, 100, 200, 300, 400, 401, 500])),
                         ["Good", "Good", "Satisfactory", "Satisfactory", "Moderate", "Poor",
                          "Very Poor", "Severe", "Severe"])
        self.assertIsNone(category([np.nan])[0])

    @staticmethod
    def day(**cols):
        idx = pd.date_range("2026-01-01", periods=24, freq="h")
        return pd.DataFrame({k: np.full(24, v, dtype=float) for k, v in cols.items()}, index=idx)

    def test_aqi_is_max_subindex_with_pm(self):
        out = daily_aqi(self.day(**{"PM2.5": 45.0, "NO2": 120.0, "SO2": 10.0}))
        self.assertAlmostEqual(out.aqi.iloc[0], 140.0)
        self.assertEqual(out.dominant.iloc[0], "NO2")
        self.assertEqual(out.category.iloc[0], "Moderate")

    def test_needs_three_subindices(self):
        self.assertTrue(np.isnan(daily_aqi(self.day(**{"PM2.5": 45.0, "NO2": 20.0})).aqi.iloc[0]))

    def test_needs_pm(self):
        out = daily_aqi(self.day(NO2=20.0, SO2=10.0, NH3=50.0))
        self.assertTrue(np.isnan(out.aqi.iloc[0]))

    def test_needs_16_measured_hours(self):
        h = self.day(**{"PM2.5": 45.0, "NO2": 20.0, "SO2": 10.0})
        h.iloc[15:, 0] = np.nan  # 15 PM2.5 hours left
        self.assertTrue(np.isnan(daily_aqi(h).aqi.iloc[0]))

    def test_ozone_uses_max_8h_mean(self):
        h = self.day(**{"PM2.5": 10.0, "NO2": 10.0, "O3": 20.0})
        h.iloc[10:18, 2] = 208.0  # one 8-hour block at 208 µg/m³ -> sub-index 300
        out = daily_aqi(h)
        self.assertAlmostEqual(out.si_O3.iloc[0], 300.0)
        self.assertAlmostEqual(out.aqi.iloc[0], 300.0)


class Outputs(unittest.TestCase):
    def load(self, mode, name, **kw):
        path = f"outputs/{mode}/{name}"
        if not os.path.exists(path):
            self.skipTest(f"{path} missing - run NASHIK_MODE={mode} python nashik_aq_pipeline.py")
        return pd.read_csv(path, **kw)

    def test_full_hourly_horizon_for_every_station(self):
        for mode, year in MODES.items():
            with self.subTest(mode=mode):
                fc = self.load(mode, "predictions/recursive_forecast.csv",
                               parse_dates=["timestamp"])
                horizon = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
                self.assertEqual(len(horizon), 8760)
                self.assertEqual(sorted(fc.station.unique()), sorted(STATIONS))
                for (s, p), g in fc.groupby(["station", "pollutant"]):
                    self.assertTrue(g.timestamp.reset_index(drop=True).equals(
                        pd.Series(horizon, name="timestamp")), (mode, s, p))
                    self.assertTrue(g.cnn_lstm_pred.notna().all())
                    self.assertTrue((g.cnn_lstm_pred >= 0).all(), "non-negativity constraint")

    def test_2026_forecast_continues_past_last_measurement(self):
        fc = self.load("forecast_2026", "predictions/recursive_forecast.csv",
                       parse_dates=["timestamp"])
        late = fc[fc.timestamp >= "2026-09-01"]
        self.assertEqual(len(late), 4 * 7 * 122 * 24)  # Sep-Dec, every station and pollutant
        self.assertTrue(late.actual.isna().all())
        self.assertTrue(late.cnn_lstm_pred.notna().all())

    def test_missing_ground_truth_is_not_fabricated(self):
        fc = self.load("forecast_2026", "predictions/recursive_forecast.csv",
                       parse_dates=["timestamp"])
        july = fc[fc.timestamp.dt.month == 7]
        self.assertTrue((july.ground_truth == "no_row_in_file").all())
        self.assertTrue(july.actual.isna().all())
        pandav = fc[fc.station == "Pandav Nagari"]
        self.assertTrue((pandav.ground_truth == "no_row_in_file").all())
        self.assertTrue(fc.actual.isna().eq(fc.ground_truth != "observed").all())

    def test_metrics_count_only_observed_hours(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                fc = self.load(mode, "predictions/recursive_forecast.csv")
                m = self.load(mode, "metrics/recursive_station_pollutant_metrics.csv")
                observed = fc[fc.ground_truth == "observed"].groupby(
                    ["station", "pollutant"]).size()
                for r in m.itertuples():
                    self.assertEqual(r.n_valid, observed.get((r.station, r.pollutant), 0))
                self.assertEqual(len(m), len(STATIONS) * len(POLL))

    def test_one_step_alignment(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                o = self.load(mode, "predictions/rolling_one_step.csv",
                              parse_dates=["timestamp", "input_window_end"])
                self.assertTrue((o.timestamp - o.input_window_end == pd.Timedelta(hours=1)).all())
                self.assertTrue(o.actual.notna().all())
                self.assertFalse(o.duplicated(["station", "timestamp", "pollutant"]).any())

    def test_independent_verifier(self):
        """Replays forecasts from data truncated at the cutoff (leakage), rebuilds one-step
        windows from the raw files (alignment) and recomputes every metric."""
        for mode in MODES:
            with self.subTest(mode=mode):
                self.load(mode, "run_metadata.json", nrows=0)
                r = subprocess.run([sys.executable, "scripts/verify_outputs.py", mode],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
