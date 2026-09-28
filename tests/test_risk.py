import unittest

import pandas as pd

from risk import choose_policy


class PeakSelectionTimingTest(unittest.TestCase):
    def test_july_outcomes_do_not_change_selection_or_threshold(self):
        index = pd.to_datetime(["2021-04-01", "2021-04-02", "2021-05-01", "2021-05-02",
                                "2021-06-01", "2021-06-02", "2021-07-01", "2021-07-02"])
        frame = pd.DataFrame({"actual": [100, 190] * 4,
                              "a": [.1, .9] * 4, "b": [.8, .2] * 4}, index=index)
        before = choose_policy(frame, ["a", "b"], 182.)
        frame.loc[frame.index >= "2021-07-01", ["actual", "a", "b"]] = 999
        after = choose_policy(frame, ["a", "b"], 182.)
        self.assertEqual(before, after)
