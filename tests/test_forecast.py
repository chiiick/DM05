"""Checks that forecast features only use information available at prediction time."""

import unittest
from pathlib import Path

from forecast import make_features, read_data


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


if __name__ == "__main__":
    unittest.main()
