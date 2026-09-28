import tempfile
import unittest
from pathlib import Path

import pandas as pd

from release import forecast_release, train_release


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "okm_augumented_2021.csv"


class FinalReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.artifact = Path(cls.temporary.name) / "final_july.joblib"
        cls.manifest = train_release(DATA, ROOT / "outputs", cls.artifact,
                                     through=pd.Timestamp("2021-07-31 23:00:00"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_replay_matches_mean_intervals_and_retained_peak_policy(self):
        mean = pd.read_csv(ROOT / "outputs/stage14/interval_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
        peak = pd.read_csv(ROOT / "outputs/test_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
        for target in ("2021-08-01 00:00:00", "2021-08-09 08:00:00"):
            origin = pd.Timestamp(target) - pd.Timedelta(hours=1)
            prediction = forecast_release(DATA, self.artifact, origin)
            self.assertEqual(prediction["forecast_hour"], target)
            self.assertLessEqual(pd.Timestamp(prediction["calibration_end"]), origin)
            self.assertAlmostEqual(prediction["predicted_mean"], mean.loc[target, "prediction"], places=8)
            self.assertAlmostEqual(prediction["peak_risk_score"], peak.loc[target, "peak_risk_score"], places=8)
            self.assertEqual(prediction["peak_alert"], peak.loc[target, "peak_classifier_alert"])
            lower, upper = prediction["conditional_interval_90"]
            self.assertAlmostEqual(lower, mean.loc[target, "conditional_lower"], places=8)
            self.assertAlmostEqual(upper, mean.loc[target, "conditional_upper"], places=8)
            self.assertLessEqual(lower, prediction["predicted_mean"])
            self.assertGreaterEqual(upper, prediction["predicted_mean"])

    def test_future_outcomes_cannot_change_inference(self):
        raw = pd.read_csv(DATA)
        origin = pd.Timestamp("2021-08-09 07:00:00")
        observed = (raw["날짜"] < 20210809) | ((raw["날짜"] == 20210809) & (raw["시간"] <= 7))
        history = Path(self.temporary.name) / "history.csv"
        altered = Path(self.temporary.name) / "altered.csv"
        raw.loc[observed].to_csv(history, index=False)
        raw.loc[~observed, ["평균", "생산량", "15분", "30분", "45분", "60분"]] = 999999
        raw.to_csv(altered, index=False)
        results = [forecast_release(path, self.artifact, origin) for path in (DATA, history, altered)]
        for result in results[1:]:
            self.assertAlmostEqual(results[0]["predicted_mean"], result["predicted_mean"], places=8)
            self.assertAlmostEqual(results[0]["peak_risk_score"], result["peak_risk_score"], places=8)
            self.assertEqual(results[0]["peak_alert"], result["peak_alert"])
            for reference, actual in zip(results[0]["conditional_interval_90"], result["conditional_interval_90"]):
                self.assertAlmostEqual(reference, actual, places=8)

    def test_future_trained_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "later than the requested forecast origin"):
            forecast_release(DATA, self.artifact, pd.Timestamp("2021-07-30 23:00:00"))

    def test_missing_exact_lags_are_rejected(self):
        raw = pd.read_csv(DATA)
        missing_day = Path(self.temporary.name) / "missing_day.csv"
        raw.loc[raw["날짜"] != 20210808].to_csv(missing_day, index=False)
        with self.assertRaisesRegex(ValueError, "Required exact historical lags are missing"):
            forecast_release(missing_day, self.artifact, pd.Timestamp("2021-08-09 07:00:00"))
