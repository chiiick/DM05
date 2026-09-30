import unittest
from pathlib import Path

import pandas as pd

from forecast import POWER_COLUMNS, read_data
from specified_interval import candidates
from step_1_1 import forecast, project


ROOT = Path(__file__).resolve().parents[1]


class Step11Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean, _ = read_data(ROOT / "okm_augumented_2021.csv")

    def test_future_observations_cannot_change_projection(self):
        origin = pd.Timestamp("2021-08-09 07:00")
        targets = pd.date_range("2021-08-09 09:00", "2021-08-09 12:00", freq="h")
        rows = candidates(self.clean, origin, targets)
        original, _ = project(self.clean, rows, pd.Timestamp("2021-07-31 23:00"))
        altered = self.clean.copy()
        altered.loc[altered.index > origin, ["평균", "생산량", *POWER_COLUMNS]] = 999999
        changed, _ = project(altered, candidates(altered, origin, targets),
                             pd.Timestamp("2021-07-31 23:00"))
        pd.testing.assert_series_equal(original.state_projection, changed.state_projection)

    def test_cutoff_after_origin_is_rejected(self):
        origin = pd.Timestamp("2021-08-09 07:00")
        rows = candidates(self.clean, origin, pd.DatetimeIndex([origin + pd.Timedelta(hours=1)]))
        with self.assertRaisesRegex(ValueError, "after a forecast origin"):
            project(self.clean, rows, origin + pd.Timedelta(hours=1))

    def test_forecast_returns_only_requested_interval(self):
        rows, report = forecast(ROOT / "okm_augumented_2021.csv", "2021-09-14 23:00",
                                "2021-09-15 08:00", "2021-09-15 14:00")
        self.assertEqual(rows.lead_hours.tolist(), list(range(9, 16)))
        self.assertEqual(report["hours"], 7)
        self.assertTrue(rows.predicted_hourly_mean.notna().all())


if __name__ == "__main__":
    unittest.main()
