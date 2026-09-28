import unittest

import numpy as np
import pandas as pd

from operations import evaluate_scenarios, plan_load


class ScenarioConstraintsTest(unittest.TestCase):
    def test_peak_reduction_preserves_load_and_all_constraints(self):
        forecast = np.full(24, 100.)
        forecast[12] = 150.
        plan = plan_load(forecast, .20)
        self.assertAlmostEqual(plan["adjusted"].sum(), forecast.sum(), places=6)
        self.assertAlmostEqual(plan["adjusted"].max(), 120., places=5)
        self.assertTrue((plan["outgoing"] <= .20 * forecast + 1e-6).all())
        self.assertTrue((plan["incoming"] <= .10 * forecast.max() + 1e-6).all())
        for flow in plan["flows"]:
            self.assertTrue(7 <= flow["from_hour"] <= 19)
            self.assertTrue(7 <= flow["to_hour"] <= 19)
            self.assertLessEqual(abs(flow["from_hour"] - flow["to_hour"]), 2)

    def test_zero_flexibility_and_unmovable_peak_do_not_create_moves(self):
        forecast = np.full(24, 100.)
        forecast[2] = 150.
        for fraction in (0., .20):
            plan = plan_load(forecast, fraction)
            np.testing.assert_allclose(plan["adjusted"], forecast)
            self.assertEqual(plan["flows"], [])

    def test_actual_outcomes_never_choose_plan_and_violations_are_reported(self):
        forecast = np.full(24, 100.)
        forecast[12] = 150.
        frame = pd.DataFrame({"anchor_time": pd.Timestamp("2021-08-01"),
                              "timestamp": pd.date_range("2021-08-01", periods=24, freq="h"),
                              "horizon": np.arange(1, 25), "prediction": forecast, "actual": forecast})
        _, original, _, _ = evaluate_scenarios(frame)
        frame.loc[12, "actual"] = 1.
        daily, altered, _, _ = evaluate_scenarios(frame)
        np.testing.assert_allclose(original[["outgoing", "incoming", "planned_forecast"]],
                                   altered[["outgoing", "incoming", "planned_forecast"]])
        moved = daily.loc[daily.fraction > 0]
        self.assertTrue((~moved.feasible_under_assumed_share).all())
        self.assertTrue((moved.share_violation_sum > 0).all())
        self.assertTrue((moved.simulated_peak_change > 0).all())
