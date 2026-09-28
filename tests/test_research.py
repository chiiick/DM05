import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import read_data
from research import richer_features, selection_scores, training_window


class ResearchTimingTest(unittest.TestCase):
    def test_rich_features_ignore_current_and_future_outcomes(self):
        data = Path(__file__).resolve().parents[1] / "okm_augumented_2021.csv"
        clean, _ = read_data(data)
        target = pd.Timestamp("2021-08-09 08:00:00")
        original, names = richer_features(clean)
        changed = clean.copy()
        changed.loc[changed.index >= target, ["평균", "생산량", "15분", "30분", "45분", "60분"]] = 999999
        alternate, _ = richer_features(changed)
        np.testing.assert_allclose(original.loc[target, names], alternate.loc[target, names])

    def test_july_does_not_change_candidate_choice(self):
        index = pd.to_datetime(["2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01"])
        oof = pd.DataFrame({"actual": [0., 0., 0., 0.], "a": [1., 1., 1., 9999.],
                            "b": [2., 2., 2., 0.]}, index=index)
        score = selection_scores(oof, ["a", "b"])
        self.assertEqual(min(score, key=score.get), "a")

    def test_training_window_excludes_boundary_and_future_rows(self):
        frame = pd.DataFrame({"target": range(97)}, index=pd.date_range("2021-01-01", periods=97, freq="h"))
        selected = training_window(frame, "2021-01-04", window_days=2)
        self.assertEqual(selected.index.min(), pd.Timestamp("2021-01-02"))
        self.assertEqual(selected.index.max(), pd.Timestamp("2021-01-03 23:00"))
        self.assertEqual(len(selected), 48)
