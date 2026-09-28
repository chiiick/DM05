import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forecast import read_data
from horizon import TrajectoryModel, choose_trajectory, forecast_trajectory, training_mask, trajectory_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "okm_augumented_2021.csv"


class HorizonTimingTest(unittest.TestCase):
    def test_training_purges_all_targets_crossing_the_cutoff(self):
        anchors = pd.date_range("2021-06-29", "2021-07-01", freq="h")
        mask = training_mask(anchors, "2021-07-01")
        self.assertEqual(anchors[mask].max(), pd.Timestamp("2021-06-30 00:00"))
        self.assertTrue(((anchors[mask] + pd.Timedelta(hours=23)) < pd.Timestamp("2021-07-01")).all())

    def test_daily_forecast_and_all_baselines_ignore_future_rows(self):
        clean, _ = read_data(DATA)
        frame, features, means, peaks, base, peak_base = trajectory_data(clean)
        train = training_mask(frame.index, "2021-02-01")
        model = TrajectoryModel(8, features, residual=True).fit(frame.loc[train], means[train], peaks[train])
        origin = pd.Timestamp("2021-02-04 23:00")
        anchor = origin + pd.Timedelta(hours=1)
        history = clean.loc[clean.index <= origin]
        extended = history.reindex(history.index.union(pd.DatetimeIndex([anchor])))
        unknown, _, _, _, historical_base, historical_peak = trajectory_data(extended, require_targets=False)
        np.testing.assert_allclose(unknown.loc[anchor, features], frame.loc[anchor, features])
        i, j = frame.index.get_loc(anchor), unknown.index.get_loc(anchor)
        for name in base:
            np.testing.assert_allclose(base[name][i], historical_base[name][j])
            np.testing.assert_allclose(peak_base[name][i], historical_peak[name][j])
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "day.joblib"
            joblib.dump({"model": model, "chosen": "extra_8", "features": features,
                         "train_end": "2021-01-31 23:00", "peak_score_threshold": 182.}, artifact)
            reference = forecast_trajectory(DATA, artifact, origin)
            raw = pd.read_csv(DATA)
            raw.loc[raw["날짜"] >= 20210205, ["평균", "생산량", "15분", "30분", "45분", "60분"]] = 999999
            path = Path(directory) / "changed.csv"
            raw.to_csv(path, index=False)
            changed = forecast_trajectory(path, artifact, origin)
            np.testing.assert_allclose(reference.predicted_mean, changed.predicted_mean)
            np.testing.assert_allclose(reference.predicted_quarter_max, changed.predicted_quarter_max)
            expected, expected_peak = model.predict(frame.loc[[anchor]])
            np.testing.assert_allclose(reference.predicted_mean, expected.ravel())
            self.assertEqual(len(reference), 24)
            self.assertTrue((reference.predicted_quarter_max >= reference.predicted_mean).all())
            self.assertEqual(reference.timestamp.iloc[-1], pd.Timestamp("2021-02-05 23:00"))
            with self.assertRaisesRegex(ValueError, "future observations"):
                forecast_trajectory(DATA, artifact, pd.Timestamp("2021-01-30 23:00"))
            joblib.dump({"model": None, "chosen": "previous_week", "features": features,
                         "train_end": "2021-01-31 23:00", "peak_score_threshold": 182.}, artifact)
            baseline_forecast = forecast_trajectory(path, artifact, origin)
            np.testing.assert_allclose(baseline_forecast.predicted_mean, base["previous_week"][i])

    def test_july_outcomes_do_not_change_choice_or_alert_threshold(self):
        from horizon import NAMES
        dates = pd.to_datetime(["2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01"])
        oof = pd.DataFrame({"anchor_time": dates, "actual": [10., 20., 30., 40.],
                            "actual_quarter": [190., 100., 200., 300.]})
        for i, name in enumerate(NAMES):
            oof[name] = oof.actual + i
            oof[f"quarter_{name}"] = [180., 90., 190., 280.]
        decision = choose_trajectory(oof)
        oof.loc[oof.anchor_time >= "2021-07-01", ["actual", "actual_quarter"]] = 999999
        self.assertEqual(decision, choose_trajectory(oof))
