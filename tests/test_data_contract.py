import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_contract import run

ROOT = Path(__file__).resolve().parents[1]


class DataContractTest(unittest.TestCase):
    def test_changed_source_is_rejected_before_interpretation(self):
        original = ROOT / "okm_augumented_2021.csv"
        with tempfile.TemporaryDirectory() as folder:
            tampered = Path(folder) / "changed.csv"
            raw = pd.read_csv(original)
            raw.loc[0, "평균"] += 1
            raw.to_csv(tampered, index=False)
            with self.assertRaisesRegex(ValueError, "different source CSV"):
                run(tampered, ROOT / "outputs")

    def test_recorded_contract_exposes_same_hour_derived_workforce(self):
        contract = json.loads((ROOT / "outputs/stage19/contract.json").read_text())
        dictionary = pd.read_csv(ROOT / "outputs/stage19/source_dictionary.csv").set_index("column")
        self.assertEqual(dictionary.loc["공장인원", "role_at_forecast_origin"], "excluded_same_hour_derivation")
        self.assertLess(contract["workforce_derivation"]["max_absolute_difference"], 1e-7)
        self.assertEqual(contract["valid_rows_by_period"]["august_september_repeatedly_inspected"]["valid_model_rows"], 1080)
