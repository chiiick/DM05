import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import read_data
from reliability import attach_intervals, interval_calibration
from research import richer_features


class CalibrationTimingTest(unittest.TestCase):
    def test_current_month_outcomes_do_not_change_its_bounds(self):
        root = Path(__file__).resolve().parents[1]
        clean, _ = read_data(root / "okm_augumented_2021.csv")
        report = json.loads((root / "outputs/stage13/summary.json").read_text())
        frame, _ = richer_features(clean, report["config"]["features"])
        evaluation = pd.read_csv(root / "outputs/stage13/predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
        oof = pd.read_csv(root / "outputs/stage13/validation_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
        july = oof.loc["2021-07-01":, ["actual", report["chosen"]]].rename(columns={report["chosen"]: "prediction"})
        original, _ = attach_intervals(clean, frame, evaluation, july)
        changed = evaluation.copy()
        changed.loc["2021-09-01":, "actual"] += 9999
        alternate, _ = attach_intervals(clean, frame, changed, july)
        columns = ["global_lower", "global_upper", "conditional_lower", "conditional_upper"]
        np.testing.assert_allclose(original[columns], alternate[columns])

    def test_small_regime_samples_use_the_global_radius(self):
        fitted = interval_calibration(np.arange(10), np.zeros(10), np.arange(10) < 5)
        for value in fitted["regimes"].values():
            self.assertTrue(value["fallback_to_global"])
            self.assertEqual(value["radius"], fitted["global_radius"])
