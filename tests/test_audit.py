import unittest

import pandas as pd

from audit import forward_gate_selection


class ForwardSelectionTest(unittest.TestCase):
    def test_later_outcomes_cannot_change_earlier_selection(self):
        index = pd.to_datetime(["2021-04-01", "2021-05-01", "2021-06-01"])
        candidates = pd.DataFrame({"no_gate": [0., 0., 0.], "idle": [1., 1., 1.]}, index=index)
        actual = pd.Series([1., 0., 1.], index=index)
        original, records = forward_gate_selection(candidates, actual)
        altered_actual = actual.copy()
        altered_actual.iloc[-1] = 10000
        changed, _ = forward_gate_selection(candidates, altered_actual)
        self.assertEqual(original.iloc[1], changed.iloc[1])
        self.assertEqual(original.iloc[1], 1.)
        for row in records:
            self.assertLess(pd.Timestamp(row["selection_end"]), pd.Timestamp(row["evaluation_month"]))
