import unittest

import pandas as pd

from advanced_forecast import candidate_scores


class MonthlySelectionTest(unittest.TestCase):
    def test_months_have_equal_weight_despite_different_row_counts(self):
        index = pd.date_range("2021-04-01", periods=100, freq="h").append(
            pd.DatetimeIndex(["2021-05-01"]))
        actual = pd.Series(0., index=index)
        predictions = pd.DataFrame({"many_rows_winner": [0.] * 100 + [10.],
                                    "month_winner": [3.] * 101}, index=index)
        result = candidate_scores(predictions, actual)
        self.assertEqual(result["many_rows_winner"]["mean_month_mae"], 5.)
        self.assertEqual(result["month_winner"]["mean_month_mae"], 3.)
