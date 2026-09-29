import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from condition_analysis import known_conditions
from forecast import POWER_COLUMNS, read_data

ROOT = Path(__file__).resolve().parents[1]


class KnownConditionTest(unittest.TestCase):
    def test_target_and_future_observations_cannot_change_flags(self):
        clean, _ = read_data(ROOT / "okm_augumented_2021.csv")
        target = pd.Timestamp("2021-08-02 08:00")
        index = pd.DatetimeIndex([target])
        original = known_conditions(clean, index, 176)
        changed = clean.copy()
        changed.loc[changed.index >= target, [*POWER_COLUMNS, "생산량", "평균"]] = 999999
        pd.testing.assert_frame_equal(original, known_conditions(changed, index, 176))

    def test_prior_quarter_threshold_is_pre_evaluation(self):
        clean, _ = read_data(ROOT / "okm_augumented_2021.csv")
        before_july = clean.loc[clean.index < "2021-07-01", POWER_COLUMNS].max(axis=1).quantile(.9)
        altered = clean.copy()
        altered.loc[altered.index >= "2021-07-01", POWER_COLUMNS] = 999999
        after = altered.loc[altered.index < "2021-07-01", POWER_COLUMNS].max(axis=1).quantile(.9)
        self.assertEqual(before_july, after)
        self.assertEqual(float(after), 176.0)
