import unittest
from pathlib import Path

import pandas as pd

from decision_guard import decision, origin_signals
from forecast import read_data

ROOT = Path(__file__).resolve().parents[1]


class DecisionGuardTest(unittest.TestCase):
    def test_forecast_day_observations_cannot_change_decision(self):
        clean, _ = read_data(ROOT / "okm_augumented_2021.csv")
        anchor = pd.Timestamp("2021-08-09 00:00:00")
        first = decision(origin_signals(clean, anchor), 80.73958333333333)
        changed = clean.copy()
        changed.loc[changed.index >= anchor, ["평균", "생산량"]] = 999999
        self.assertEqual(first, decision(origin_signals(changed, anchor), 80.73958333333333))

    def test_missing_history_refuses_decision(self):
        clean, _ = read_data(ROOT / "okm_augumented_2021.csv")
        with self.assertRaisesRegex(ValueError, "Incomplete past"):
            origin_signals(clean.drop(pd.Timestamp("2021-08-08 09:00:00")), pd.Timestamp("2021-08-09"))

    def test_idle_day_always_requires_review(self):
        result = decision({"past_profile_disagreement": 0., "previous_24h_no_production": True}, 80.)
        self.assertTrue(result["review_required"])
        self.assertEqual(result["reason"], "prior_day_no_production")
