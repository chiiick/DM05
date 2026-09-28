import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from deploy import forecast_next, prediction_frame, train_bundle
from forecast import idle_mask, make_features, read_data


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "okm_augumented_2021.csv"


class DeploymentTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.artifact = Path(cls.temporary.name) / "july_model.joblib"
        train_bundle(DATA, ROOT / "outputs/stage7/summary.json",
                     ROOT / "outputs/summary.json", cls.artifact,
                     through=pd.Timestamp("2021-07-31 23:00:00"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_unknown_target_features_match_historical_batch(self):
        clean, _ = read_data(DATA)
        batch, names = make_features(clean)
        target = pd.Timestamp("2021-08-09 08:00:00")
        history = clean.loc[clean.index < target]
        extended, row, deployed_names = prediction_frame(history)
        self.assertTrue(pd.isna(row.loc[target, "target"]))
        self.assertEqual(names, deployed_names)
        np.testing.assert_allclose(row.loc[target, names], batch.loc[target, names])
        np.testing.assert_array_equal(idle_mask(extended, row, 6, 25),
                                      idle_mask(clean, batch.loc[[target]], 6, 25))

    def test_replay_matches_saved_out_of_time_predictions(self):
        mean = pd.read_csv(ROOT / "outputs/stage7/predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
        peak = pd.read_csv(ROOT / "outputs/test_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
        for target in ("2021-08-01 00:00:00", "2021-08-09 08:00:00"):
            forecast = forecast_next(DATA, self.artifact, pd.Timestamp(target) - pd.Timedelta(hours=1))
            self.assertAlmostEqual(forecast["predicted_mean"], mean.loc[target, "prediction"], places=8)
            self.assertAlmostEqual(forecast["peak_risk_score"], peak.loc[target, "peak_risk_score"], places=8)

    def test_removing_or_changing_future_rows_does_not_change_forecast(self):
        raw = pd.read_csv(DATA)
        origin = pd.Timestamp("2021-08-09 07:00:00")
        observed = (raw["날짜"] < 20210809) | ((raw["날짜"] == 20210809) & (raw["시간"] <= 7))
        history_path = Path(self.temporary.name) / "history.csv"
        raw.loc[observed].to_csv(history_path, index=False)
        altered_path = Path(self.temporary.name) / "altered.csv"
        raw.loc[~observed, ["평균", "생산량", "15분", "30분", "45분", "60분"]] = 999999
        raw.to_csv(altered_path, index=False)
        results = [forecast_next(path, self.artifact, origin) for path in (DATA, history_path, altered_path)]
        for key in ("predicted_mean", "peak_risk_score", "peak_alert", "empirical_interval_90"):
            self.assertEqual(results[0][key], results[1][key])
            self.assertEqual(results[0][key], results[2][key])

    def test_model_cannot_predict_from_before_its_training_end(self):
        with self.assertRaisesRegex(ValueError, "later than this forecast origin"):
            forecast_next(DATA, self.artifact, pd.Timestamp("2021-07-30 23:00:00"))

    def test_missing_required_lag_is_rejected(self):
        clean, _ = read_data(DATA)
        target = pd.Timestamp("2021-08-09 08:00:00")
        history = clean.loc[clean.index < target].drop(target - pd.Timedelta(hours=24))
        with self.assertRaisesRegex(ValueError, "Insufficient exact hourly history"):
            prediction_frame(history)
