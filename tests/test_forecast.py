"""Checks that forecast features only use information available at prediction time."""

import unittest
from pathlib import Path

import numpy as np

from forecast import alert_scores, idle_mask, make_features, read_data, threshold_for_recall


DATA = Path(__file__).resolve().parents[1] / "okm_augumented_2021.csv"


class FeatureTimingTest(unittest.TestCase):
    def test_same_hour_observations_do_not_change_features(self):
        clean, _ = read_data(DATA)
        stamp = "2021-06-15 10:00:00"
        original, names = make_features(clean)
        changed = clean.copy()
        for name in ("평균", "생산량", "15분", "30분", "45분", "60분"):
            changed.loc[stamp, name] = 99999
        altered, _ = make_features(changed)
        self.assertEqual(original.loc[stamp, names].to_dict(),
                         altered.loc[stamp, names].to_dict())

    def test_missing_days_do_not_become_adjacent_lags(self):
        clean, _ = read_data(DATA)
        frame, _ = make_features(clean)
        self.assertNotIn("2021-07-14 00:00:00", frame.index)
        self.assertNotIn("2021-07-16 00:00:00", frame.index)

    def test_peak_alert_uses_separate_event_and_alarm_cutoffs(self):
        actual = np.array([180, 190, 200, 170])
        predicted = np.array([174, 171, 190, 180])
        report = alert_scores(actual, predicted, peak_cutoff=182,
                              alert_threshold=172)
        self.assertEqual((report["tp"], report["fp"], report["fn"]), (1, 2, 1))

    def test_idle_signal_excludes_current_hour_production(self):
        clean, _ = read_data(DATA)
        frame, _ = make_features(clean)
        stamp = "2021-06-15 10:00:00"
        row = frame.loc[[stamp]]
        original = idle_mask(clean, row, 6, 25)
        changed = clean.copy()
        changed.loc[stamp, "생산량"] = 99999
        altered = idle_mask(changed, row, 6, 25)
        np.testing.assert_array_equal(original, altered)

    def test_recall_threshold_uses_highest_eligible_score(self):
        score = np.array([0.1, 0.4, 0.6, 0.9])
        observed = np.array([False, True, True, False])
        self.assertEqual(threshold_for_recall(score, observed, 1.0), 0.4)
        self.assertEqual(threshold_for_recall(score, observed, 0.5), 0.6)


if __name__ == "__main__":
    unittest.main()
