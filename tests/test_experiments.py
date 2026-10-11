"""Tests for experiments/exp_multihorizon.py. Run from the project root:
    python -m unittest discover tests -v

Synthetic-data tests always run. Tests on the real station files skip when data/ is absent;
tests on experiment outputs skip until the corresponding stage has run.
"""
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import exp_multihorizon as E  # noqa: E402

OUT = E.EXP_OUT


def synthetic_station(start="2023-07-01", end="2025-12-31 23:00", gap=None):
    """Hourly frame with distinct, non-repeating values (no stuck runs)."""
    ts = pd.date_range(start, end, freq="h")
    k = np.arange(len(ts), dtype=float)
    df = pd.DataFrame({"Timestamp": ts})
    for j, p in enumerate(E.POLL):
        df[p] = 10 + j + 5 * np.sin(2 * np.pi * k / 24) + (k % 97) / 97.0
    if gap is not None:
        df = df[(df.Timestamp < gap[0]) | (df.Timestamp > gap[1])]
    return df.reset_index(drop=True)


def real_station(name="Pandav Nagari"):
    if not os.path.isdir(E.DATA_DIR) or not any(
            "pandav" in f.lower() for f in os.listdir(E.DATA_DIR)):
        raise unittest.SkipTest("station CSVs not in data/")
    return E.load_clean()[name]


class Alignment(unittest.TestCase):
    def setUp(self):
        self.p = E.prepare(synthetic_station(), "F2025")

    def test_input_and_target_windows(self):
        p = self.p
        i = p["idx"]("2024-03-10 05:00")
        x = E.input_windows(p, [i], E.DIR_LOOKBACK)[0]
        y = E.target_windows(p, [i], E.DIR_HORIZON)[0]
        np.testing.assert_allclose(x[:, :7], p["z"][i - 168:i], rtol=1e-6)
        np.testing.assert_allclose(x[-1, 7:], E.calendar([p["grid"][i - 1]])[0], atol=1e-6)
        np.testing.assert_allclose(y, E.scaled(p, p["obs"][i:i + 168]), rtol=1e-5)

    def test_targets_beyond_grid_are_missing(self):
        p = self.p
        i = len(p["grid"]) - 10
        y = E.target_windows(p, [i], E.DIR_HORIZON)[0]
        self.assertTrue(np.isnan(y[10:]).all())
        self.assertFalse(np.isnan(y[:10]).any())

    def test_training_targets_end_before_validation(self):
        p = self.p
        o = E.direct_origins(p, p["grid"][0], p["val_start"])
        self.assertLessEqual(p["grid"][o.max() + E.DIR_HORIZON - 1], p["val_start"]
                             - pd.Timedelta(hours=1))
        v = E.direct_origins(p, p["val_start"], p["origin"])
        self.assertGreaterEqual(p["grid"][v.min()], p["val_start"])
        self.assertLessEqual(p["grid"][v.max() + E.DIR_HORIZON - 1],
                             p["origin"] - pd.Timedelta(hours=1))

    def test_bucket_assignment(self):
        b = E.bucket_of([1, 6, 7, 24, 25, 168, 169, 8760], E.YEAR_BUCKETS)
        self.assertEqual(list(b), [0, 0, 1, 1, 2, 3, 4, 7])


class MissingData(unittest.TestCase):
    def test_windows_never_cross_absent_rows(self):
        gap = (pd.Timestamp("2024-05-01 00:00"), pd.Timestamp("2024-05-01 05:00"))
        p = E.prepare(synthetic_station(gap=gap), "F2025")
        first_after = p["idx"]("2024-05-01 06:00")
        i = np.arange(first_after - 5, first_after + 200)
        ok = E.valid_input(p, i, E.DIR_LOOKBACK)
        self.assertFalse(ok[i < first_after + E.DIR_LOOKBACK].any())
        self.assertTrue(ok[i >= first_after + E.DIR_LOOKBACK].all())
        self.assertTrue(np.isnan(p["z"][p["idx"]("2024-05-01 02:00")]).all())

    def test_mostly_imputed_windows_are_rejected(self):
        df = synthetic_station()
        bad = (df.Timestamp >= "2024-05-01") & (df.Timestamp < "2024-05-06")
        df.loc[bad, E.POLL[1:]] = np.nan  # 6 of 7 missing for 120 of 168 h: 61% imputed
        p = E.prepare(df, "F2025")
        self.assertFalse(E.valid_input(p, [p["idx"]("2024-05-06")], E.DIR_LOOKBACK)[0])
        self.assertTrue(E.valid_input(p, [p["idx"]("2024-05-20")], E.DIR_LOOKBACK)[0])

    def test_blend_weight_ignores_missing_actuals(self):
        y = np.array([1.0, np.nan, 3.0])
        w = E.blend_weight(y, np.array([1.0, 100.0, 3.0]), np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(w, 1.0)

    def test_doy_climatology_falls_back_with_one_year(self):
        one = synthetic_station("2024-01-01", "2024-12-31 23:00").set_index("Timestamp")[E.POLL]
        ts = pd.date_range("2025-03-01", periods=48, freq="h")
        pd.testing.assert_frame_equal(E.clim_doy_hour(one, ts), E.clim_month_hour(one, ts))
        two = synthetic_station("2022-01-01", "2024-12-31 23:00").set_index("Timestamp")[E.POLL]
        two.loc[two.index.day > 15] += 20  # second half of each month differs
        self.assertFalse(np.allclose(E.clim_doy_hour(two, ts), E.clim_month_hour(two, ts)))


class Metrics(unittest.TestCase):
    def test_metric_row_matches_sklearn(self):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        rng = np.random.default_rng(0)
        y, yhat = rng.normal(size=200), rng.normal(size=200)
        m = E.metric_row(y, yhat)
        self.assertAlmostEqual(m["RMSE"], np.sqrt(mean_squared_error(y, yhat)))
        self.assertAlmostEqual(m["MAE"], mean_absolute_error(y, yhat))
        self.assertAlmostEqual(m["R2"], r2_score(y, yhat))
        self.assertTrue(np.isnan(E.metric_row(np.ones(5), np.zeros(5))["R2"]))

    def test_blend_weight_is_least_squares_optimum(self):
        rng = np.random.default_rng(1)
        y, nn, clim = rng.normal(size=500), rng.normal(size=500), rng.normal(size=500)
        nn = 0.6 * y + 0.4 * nn
        w = E.blend_weight(y, nn, clim)
        grid = np.linspace(0, 1, 1001)
        best = grid[np.argmin([np.mean((g * nn + (1 - g) * clim - y) ** 2) for g in grid])]
        self.assertAlmostEqual(w, best, places=2)


class Leakage(unittest.TestCase):
    """Changing every observation at or after a cutoff must not change anything issued
    before it: preprocessing, baselines and network forecasts."""

    @classmethod
    def setUpClass(cls):
        cls.d = real_station()
        cls.fold = "F2025"
        origin, _, _ = E.fold_bounds(cls.fold)
        cls.perturbed = cls.d.copy()
        after = cls.perturbed.Timestamp >= origin
        cls.perturbed.loc[after, E.POLL] = cls.perturbed.loc[after, E.POLL] * 7 + 50
        cls.p0 = E.prepare(cls.d, cls.fold)
        cls.p1 = E.prepare(cls.perturbed, cls.fold)
        tf.keras.utils.set_random_seed(0)
        cls.models = {"direct": E.build_direct(), "cnn_lstm": (E.build_cnn_lstm(), "change")}

    def test_preprocessing_fitted_before_cutoff(self):
        np.testing.assert_array_equal(self.p0["scaler"].mean_, self.p1["scaler"].mean_)
        k = self.p0["idx"](self.p0["origin"])
        np.testing.assert_array_equal(self.p0["z"][:k], self.p1["z"][:k])

    def test_year_ahead_forecasts_ignore_post_cutoff_data(self):
        ts0, f0, _ = E.year_ahead(self.p0, self.models, self.p0["origin"], 400)
        ts1, f1, _ = E.year_ahead(self.p1, self.models, self.p1["origin"], 400)
        for k in f0:
            np.testing.assert_array_equal(f0[k], f1[k], err_msg=k)

    def test_rolling_origin_uses_only_hours_before_each_origin(self):
        p0 = self.p0
        cut = pd.Timestamp("2025-06-01")
        later = self.d.copy()
        after = later.Timestamp >= cut
        later.loc[after, E.POLL] = later.loc[after, E.POLL] * 3 + 9
        p2 = E.prepare(later, self.fold)
        a, _ = E.short_horizon(p0, self.models, p0["origin"], cut + pd.Timedelta(days=7))
        b, _ = E.short_horizon(p2, self.models, p2["origin"], cut + pd.Timedelta(days=7))
        before = np.asarray(a["origins"]) <= cut
        self.assertTrue(before.any())
        for k in a["pred"]:
            np.testing.assert_array_equal(a["pred"][k][before], b["pred"][k][before], err_msg=k)


class SelectionIsolation(unittest.TestCase):
    def test_selection_reads_only_selection_folds(self):
        seen = []
        original = E.load_fold

        def spy(fold, kind, root=OUT):
            seen.append(fold)
            raise RuntimeError("stop")
        E.load_fold = spy
        try:
            with self.assertRaises(RuntimeError):
                E.selection_scores()
        finally:
            E.load_fold = original
        self.assertNotIn("F2026", seen)
        self.assertNotIn("F2026", E.SELECTION_FOLDS)


class ExperimentOutputs(unittest.TestCase):
    def need(self, path):
        if not os.path.exists(path):
            self.skipTest(f"{path} missing - run the experiment stages first")
        return path

    def test_short_horizon_alignment_and_completeness(self):
        for fold in E.FOLDS:
            path = f"{OUT}/folds/{fold}/short_horizon.csv.gz"
            if not os.path.exists(path):
                continue
            s = pd.read_csv(path, parse_dates=["timestamp", "origin"], nrows=300000)
            with self.subTest(fold=fold):
                self.assertTrue((s.timestamp == s.origin + pd.to_timedelta(s.lead - 1, "h")).all())
                self.assertTrue(s.actual.notna().all())
                self.assertTrue(s[E.SHORT_METHODS].notna().all().all())
                self.assertTrue((s[E.SHORT_METHODS] >= 0).all().all())
                self.assertTrue((s.origin >= pd.Timestamp(E.FOLDS[fold])).all())

    def test_year_ahead_coverage_and_labels(self):
        for fold in E.FOLDS:
            path = f"{OUT}/folds/{fold}/year_ahead.csv.gz"
            if not os.path.exists(path):
                continue
            y = pd.read_csv(path, parse_dates=["timestamp"])
            origin, _, end = E.fold_bounds(fold)
            n = len(pd.date_range(origin, end, freq="h"))
            with self.subTest(fold=fold):
                for _, g in y.groupby(["station", "pollutant"]):
                    self.assertEqual(len(g), n)
                    self.assertEqual(g.timestamp.min(), origin)
                    self.assertEqual(g.timestamp.max(), end)
                self.assertTrue(y.actual.isna().eq(y.ground_truth != "observed").all())
                self.assertTrue(y[E.YEAR_METHODS].notna().all().all())

    def test_2026_deliverable(self):
        f = pd.read_csv(self.need(f"{OUT}/forecast_2026_selected.csv"), parse_dates=["timestamp"])
        self.assertEqual(sorted(f.station.unique()), sorted(E.STATIONS))
        self.assertEqual(len(f), 8760 * 7 * 4)
        self.assertEqual(f.timestamp.max(), pd.Timestamp("2026-12-31 23:00"))
        self.assertTrue(f.forecast.notna().all() and (f.forecast >= 0).all())
        july = f[f.timestamp.dt.month == 7]
        self.assertTrue((july.ground_truth == "no_row_in_file").all() and july.actual.isna().all())
        self.assertTrue((f[f.station == "Pandav Nagari"].ground_truth == "no_row_in_file").all())

    def test_holdout_ran_after_selection(self):
        sel = json.load(open(self.need(f"{OUT}/selection.json")))
        meta = json.load(open(self.need(f"{OUT}/folds/F2026/fold_meta.json")))
        self.assertEqual(meta["selection_before_run"]["created"], sel["created"])
        self.assertEqual(sel["folds"], list(E.SELECTION_FOLDS))

    def test_exported_metrics_match_predictions(self):
        m = pd.read_csv(self.need(f"{OUT}/metrics/year_ahead_station_metrics.csv"))
        self.need(f"{OUT}/folds/F2025/year_ahead.csv.gz")  # git-ignored; exists after a run
        y = E.load_fold("F2025", "year")
        for _, r in m[(m.fold == "F2025")].sample(25, random_state=0).iterrows():
            g = y[(y.station == r.station) & (y.pollutant == r.pollutant)]
            ref = E.metric_row(g.actual.values, g[r.method].values)
            self.assertEqual(ref["n"], r.n)
            self.assertAlmostEqual(ref["RMSE"], r.RMSE, places=6)
            self.assertAlmostEqual(ref["MAE"], r.MAE, places=6)


if __name__ == "__main__":
    unittest.main()
