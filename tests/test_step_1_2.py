import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import POWER_COLUMNS
from step_1_2 import forecast, recent_error_table, state_weight


ROOT = Path(__file__).resolve().parents[1]


class Step12Test(unittest.TestCase):
    def test_router_uses_only_targets_matured_before_origin(self):
        first = pd.Timestamp("2021-05-01 00:00")
        second = first + pd.Timedelta(hours=6)
        rows = pd.DataFrame({"origin": [first, first, second],
                             "timestamp": [first + pd.Timedelta(hours=1),
                                           first + pd.Timedelta(hours=7),
                                           second + pd.Timedelta(hours=1)],
                             "actual": [10., 10., 10.],
                             "residual_hgb": [20., 20., 20.],
                             "state_projection": [10., 10., 10.]})
        before = recent_error_table(rows, 5).set_index("origin").loc[second]
        changed = rows.copy()
        changed.loc[1, "actual"] = 9999  # the second target is after issue time
        after = recent_error_table(changed, 5).set_index("origin").loc[second]
        pd.testing.assert_series_equal(before, after)
        self.assertEqual(int(before.matured_request_hours), 1)

    def test_insufficient_matured_history_uses_ai(self):
        weight = state_weight(np.array([100.]), np.array([1.]), np.array([23]),
                              margin=1, temperature=1)
        self.assertEqual(float(weight[0]), 0.0)

    def test_api_matches_backtest_and_ignores_future_measurement(self):
        origin = "2021-09-13 18:00"
        start = end = "2021-09-14 09:00"
        policy = ROOT / "outputs/step_1_2/policy.json"
        original, _ = forecast(ROOT / "okm_augumented_2021.csv", policy, origin, start, end)
        archived = pd.read_csv(ROOT / "outputs/step_1_2/evaluation_requests.csv")
        expected = archived.loc[(archived.origin == "2021-09-13 18:00:00") &
                                (archived.timestamp == "2021-09-14 09:00:00")].iloc[0]
        self.assertAlmostEqual(float(original.predicted_hourly_mean.iloc[0]), float(expected.ensemble))
        self.assertAlmostEqual(float(original.state_weight.iloc[0]), float(expected.state_weight))
        raw = pd.read_csv(ROOT / "okm_augumented_2021.csv")
        future = raw["날짜"].eq(20210913) & raw["시간"].eq(19)
        raw.loc[future, ["평균", "생산량", *POWER_COLUMNS]] = 999999
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "future_changed.csv"
            raw.to_csv(altered, index=False)
            changed, _ = forecast(altered, policy, origin, start, end)
        pd.testing.assert_frame_equal(original, changed)

    def test_month_boundary_uses_only_matured_training_targets(self):
        origin = "2021-09-01 00:00"
        target = "2021-09-01 09:00"
        rows, _ = forecast(ROOT / "okm_augumented_2021.csv",
                           ROOT / "outputs/step_1_2/policy.json", origin, target, target)
        archived = pd.read_csv(ROOT / "outputs/step_1_2/evaluation_requests.csv")
        expected = archived.loc[(archived.origin == "2021-09-01 00:00:00") &
                                (archived.timestamp == "2021-09-01 09:00:00")].iloc[0]
        self.assertAlmostEqual(float(rows.predicted_hourly_mean.iloc[0]), float(expected.ensemble))
        self.assertAlmostEqual(float(rows.residual_hgb.iloc[0]), float(expected.residual_hgb))


if __name__ == "__main__":
    unittest.main()
