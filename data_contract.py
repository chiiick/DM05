"""Stage 19: audit original columns, time availability, and evaluation lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deploy import digest
from forecast import POWER_COLUMNS, read_data
from research import richer_features


COLUMN_ROLES = {
    "날짜": ("calendar", "Forecast date is known in advance."),
    "시간": ("calendar", "Target hour is known in advance; damaged dates are removed."),
    **{name: ("observed_after_hour", "Only timestamps earlier than the forecast target can be used.")
       for name in (*POWER_COLUMNS, "평균", "생산량")},
    **{name: ("excluded_unavailable_future", "No forecast issued at the same future timestamp was supplied.")
       for name in ("기온", "풍속", "습도", "강수량")},
    "전기요금(계절)": ("excluded_undocumented", "Currency/unit and schedule availability are unconfirmed."),
    **{name: ("redundant_calendar", "Equivalent to a parsed component of 날짜.") for name in ("day", "d", "m")},
    "공장인원": ("excluded_same_hour_derivation", "Near-identical to production / same-hour quarter sum; not independent staffing."),
    "인건비": ("excluded_undocumented", "Meaning and availability are unconfirmed."),
}


def run(data: Path, root: Path) -> dict:
    source_hash = digest(data)
    mean_selection = json.loads((root / "stage13/summary.json").read_text())
    day_selection = json.loads((root / "stage17/selection.json").read_text())
    if mean_selection["input_sha256"] != source_hash or day_selection["input_sha256"] != source_hash:
        raise ValueError("Recorded model evaluations use a different source CSV")
    raw = pd.read_csv(data)
    if set(raw.columns) != set(COLUMN_ROLES):
        raise ValueError("CSV columns changed; revise the source dictionary before interpreting results")
    clean, diagnostics = read_data(data)
    frame, features = richer_features(clean)
    if features != mean_selection["features"]:
        raise ValueError("Saved mean feature contract differs from current code")
    date = pd.to_datetime(raw["날짜"].astype(str), format="%Y%m%d")
    calendar = {"day_matches_iso_weekday": bool(raw.day.eq(date.dt.isocalendar().day.astype(int)).all()),
                "d_matches_day_of_month": bool(raw.d.eq(date.dt.day).all()),
                "m_matches_month": bool(raw.m.eq(date.dt.month).all())}
    ratio = (clean["생산량"] / clean[POWER_COLUMNS].sum(axis=1).replace(0, np.nan) - clean["공장인원"]).abs().dropna()
    quarter_average_difference = (clean["평균"] - clean[POWER_COLUMNS].mean(axis=1)).abs()
    profiles = clean["평균"].groupby(clean.index.normalize()).apply(tuple)
    train = profiles.loc[profiles.index < "2021-08-01"]
    evaluation = profiles.loc[profiles.index >= "2021-08-01"]
    train_vectors = np.array(train.tolist(), dtype=float)
    eval_vectors = np.array(evaluation.tolist(), dtype=float)
    nearest = np.mean(np.abs(eval_vectors[:, None, :] - train_vectors[None, :, :]), axis=2).min(axis=1)
    profile_record = pd.DataFrame({"date": evaluation.index, "nearest_pre_august_daily_mae": nearest,
                                   "exact_profile_seen_before_august": evaluation.isin(set(train)).to_numpy()})
    dictionary = pd.DataFrame([{"column": name, "dtype": str(raw[name].dtype),
                               "missing_raw": int(raw[name].isna().sum()),
                               "n_unique_raw": int(raw[name].nunique(dropna=True)),
                               "role_at_forecast_origin": COLUMN_ROLES[name][0], "interpretation": COLUMN_ROLES[name][1]}
                              for name in raw.columns])
    by_period = {label: {"valid_model_rows": int(len(group)), "start": str(group.index.min()),
                          "end": str(group.index.max())} for label, group in {
        "training_before_july": frame.loc[frame.index < "2021-07-01"],
        "july_diagnostic": frame.loc[(frame.index >= "2021-07-01") & (frame.index < "2021-08-01")],
        "august_september_repeatedly_inspected": frame.loc[frame.index >= "2021-08-01"],
    }.items()}
    report = {"source_sha256": source_hash, "source_rows": len(raw), "source_columns": len(raw.columns),
              "raw_missing_by_column": {name: int(raw[name].isna().sum()) for name in raw.columns},
              "cleaning": diagnostics, "mean_feature_count": len(features), "valid_mean_rows": len(frame),
              "valid_rows_by_period": by_period, "calendar_redundancy": calendar,
              "average_vs_quarters": {"n_nonzero_difference": int(quarter_average_difference.gt(0).sum()),
                                      "max_absolute_difference": float(quarter_average_difference.max()),
                                      "target_averages_are_integers": bool(clean["평균"].mod(1).eq(0).all())},
              "workforce_derivation": {"comparable_rows": len(ratio), "max_absolute_difference": float(ratio.max()),
                                       "formula": "production / sum of same-hour four quarter values"},
              "daily_profiles": {"retained_days": len(profiles), "unique_hourly_mean_profiles": int(profiles.nunique()),
                                 "evaluation_days": len(evaluation),
                                 "evaluation_exact_profile_seen_before_august": int(profile_record.exact_profile_seen_before_august.sum()),
                                 "nearest_pre_august_profile_mae_median": float(np.median(nearest)),
                                 "nearest_pre_august_profile_mae_p10": float(np.quantile(nearest, .1)),
                                 "nearest_pre_august_profile_mae_p90": float(np.quantile(nearest, .9))},
              "evaluation_lineage": {"mean_selection_months": mean_selection["selection_months"],
                                     "mean_candidate": mean_selection["chosen"],
                                     "day_ahead_selection_months": day_selection["selection_months"],
                                     "day_ahead_candidate": day_selection["chosen"],
                                     "model_comparison_warning": "One-hour updating and midnight-only 24-hour forecasting use different information sets."},
              "unknown_from_supplied_material": ["official forecast origin", "official target horizon",
                                                "physical unit of 평균 and quarter columns", "contract demand interval and threshold",
                                                "availability time of production/equipment plans", "real equipment shift constraints"],
              "source_note": "This file is a provenance and availability audit, not an independent model test."}
    output = root / "stage19"
    output.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(output / "source_dictionary.csv", index=False)
    profile_record.to_csv(output / "nearest_daily_profiles.csv", index=False)
    (output / "contract.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: report[name] for name in ("source_rows", "valid_mean_rows", "valid_rows_by_period", "daily_profiles")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.data, args.output_root)
