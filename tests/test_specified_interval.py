import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecast import POWER_COLUMNS, read_data
from specified_interval import candidates, forecast_interval, historical_requests, validate_request

ROOT = Path(__file__).resolve().parents[1]


class SpecifiedIntervalTest(unittest.TestCase):
    def test_future_values_cannot_change_requested_forecast(self):
        clean, _ = read_data(ROOT / "okm_augumented_2021.csv")
        origin = pd.Timestamp("2021-08-09 07:00")
        target = pd.date_range("2021-08-09 09:00", "2021-08-09 12:00", freq="h")
        original = candidates(clean, origin, target)
        altered = clean.copy()
        altered.loc[altered.index > origin, ["평균", "생산량", *POWER_COLUMNS]] = 999999
        pd.testing.assert_frame_equal(original, candidates(altered, origin, target))

    def test_api_selects_exact_interval_and_rejects_unvalidated_lead(self):
        origin = pd.Timestamp("2021-09-14 23:00")
        rows, report = forecast_interval(ROOT / "okm_augumented_2021.csv",
            ROOT / "outputs/stage22/selection.json", str(origin), "2021-09-15 08:00", "2021-09-15 14:00")
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows.lead_hours.tolist(), list(range(9, 16)))
        self.assertEqual(report["hours"], 7)
        with self.assertRaisesRegex(ValueError, "within 24 hours"):
            validate_request(origin, origin + pd.Timedelta(hours=25), origin + pd.Timedelta(hours=25))

    def test_forecast_refuses_selection_from_the_future(self):
        with self.assertRaisesRegex(ValueError, "predates method selection"):
            forecast_interval(ROOT / "okm_augumented_2021.csv",
                ROOT / "outputs/stage22/selection.json", "2021-06-01 18:00", "2021-06-02 08:00", "2021-06-02 08:00")

    def test_full_ai_forecast_ignores_observations_after_origin(self):
        data = ROOT / "okm_augumented_2021.csv"
        selection = ROOT / "outputs/stage22/selection.json"
        original, _ = forecast_interval(data, selection, "2021-09-14 22:00", "2021-09-15 08:00", "2021-09-15 09:00")
        raw = pd.read_csv(data)
        future = (raw["날짜"] == 20210914) & (raw["시간"] == 23)
        raw.loc[future, ["평균", "생산량", *POWER_COLUMNS]] = 999999
        with tempfile.TemporaryDirectory() as folder:
            altered = Path(folder) / "changed.csv"
            raw.to_csv(altered, index=False)
            changed, _ = forecast_interval(altered, selection, "2021-09-14 22:00", "2021-09-15 08:00", "2021-09-15 09:00")
        pd.testing.assert_frame_equal(original, changed)

    def test_monthly_selection_excludes_july_targets(self):
        clean, _ = read_data(ROOT / "okm_augumented_2021.csv")
        rows = historical_requests(clean, "2021-06-29", "2021-06-30 23:00", target_before="2021-07-01")
        self.assertLess(rows.timestamp.max(), pd.Timestamp("2021-07-01"))
