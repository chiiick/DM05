import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import read_data
from research import richer_features
from transitions import CANDIDATES, TransitionModel, select_candidate

ROOT = Path(__file__).resolve().parents[1]


class TransitionTest(unittest.TestCase):
    def test_overall_guard_and_july_acceptance_are_enforced(self):
        dates = pd.to_datetime([f"2021-{m:02d}-01 {h:02d}:00" for m in (4, 5, 6, 7) for h in (7, 8)])
        oof = pd.DataFrame({name: 2. for name in CANDIDATES}, index=dates)
        oof["actual"] = 0.
        oof["observed_state"] = ["startup", "idle"] * 4
        oof["ungated"] = [1., 8.] * 4
        oof["protect_025"] = [1., 2.] * 3 + [2., 2.]
        decision = select_candidate(oof)
        self.assertFalse(decision["selection_scores"]["ungated"]["eligible"])
        self.assertEqual(decision["selected_candidate"], "protect_025")
        self.assertEqual(decision["adopted"], "baseline")
        future = oof.iloc[:2].copy()
        future.index = pd.to_datetime(["2021-08-01 07:00", "2021-08-01 08:00"])
        future["protect_025"] = 99999.
        self.assertEqual(decision, select_candidate(pd.concat([oof, future])))

    def test_state_prediction_never_reads_target_hour_production(self):
        clean, _ = read_data(ROOT / "okm_augumented_2021.csv")
        frame, names = richer_features(clean)
        model = TransitionModel(names).fit(clean, frame.loc[frame.index < "2021-02-01"])
        target = pd.Timestamp("2021-02-05 07:00")
        row = frame.loc[[target]].copy()
        original = model.predict(clean, row)
        altered = clean.copy()
        altered.loc[altered.index >= target, ["생산량", "평균", "15분", "30분", "45분", "60분"]] = 999999
        row["target"] = 999999
        changed = model.predict(altered, row)
        np.testing.assert_allclose(original, changed)
