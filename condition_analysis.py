"""Stage 20: prespecified, forecast-time error and peak condition audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deploy import digest
from forecast import POWER_COLUMNS, read_data


def known_conditions(clean: pd.DataFrame, index: pd.DatetimeIndex, high_cutoff: float) -> pd.DataFrame:
    """All signals end before the target hour. Missing history is kept explicit."""
    hourly = clean.asfreq("h")
    production = hourly["생산량"].shift(1)
    previous_quarters = hourly[POWER_COLUMNS].shift(1)
    previous_max = previous_quarters.max(axis=1, skipna=False).reindex(index)
    previous_ramp = (previous_quarters["60분"] - previous_quarters["15분"]).reindex(index)
    last_six = production.rolling(6, min_periods=6).sum().reindex(index)
    previous_active = production.reindex(index).gt(0)
    workhour = pd.Series(index.hour >= 8, index=index) & pd.Series(index.hour <= 14, index=index)
    result = pd.DataFrame(index=index)
    result["prior_idle6"] = last_six.eq(0)
    result["weekday_work_start"] = (index.dayofweek < 5) & (index.hour >= 7) & (index.hour <= 9)
    result["prior_quarter_ramp_positive"] = previous_ramp.gt(0)
    result["prior_quarter_high"] = previous_max.ge(high_cutoff)
    result["workhour_prior_active"] = np.where(workhour, "workhour", "other") + "_" + np.where(previous_active, "active", "inactive")
    if result.isna().any().any() or previous_max.isna().any() or last_six.isna().any():
        raise ValueError("Missing prior history for condition analysis")
    return result


def metrics(rows: pd.DataFrame, large_cutoff: float) -> dict:
    error = (rows.prediction - rows.actual).abs()
    actual = rows.peak_actual.astype(bool)
    alert = rows.peak_alert.astype(bool)
    return {"n": int(len(rows)), "mae": float(error.mean()),
            "rmse": float(np.sqrt(np.mean((rows.prediction - rows.actual)**2))),
            "p90_absolute_error": float(error.quantile(.9)),
            "large_error_count": int(error.ge(large_cutoff).sum()),
            "large_error_rate": float(error.ge(large_cutoff).mean()),
            "peak_hours": int(actual.sum()), "peak_rate": float(actual.mean()),
            "tp": int((actual & alert).sum()), "fp": int((~actual & alert).sum()),
            "fn": int((actual & ~alert).sum())}


def day_bootstrap_difference(rows: pd.DataFrame, flag: str, measure: str, repeats=2000) -> dict:
    """Day-cluster CI for a descriptive true-minus-false difference."""
    value = (rows.prediction - rows.actual).abs() if measure == "absolute_error" else rows.peak_actual.astype(float)
    temp = pd.DataFrame({"flag": rows[flag].astype(bool), "value": value, "day": rows.index.normalize()})
    days = [group for _, group in temp.groupby("day")]
    rng = np.random.default_rng(20260929)
    estimates = []
    for _ in range(repeats):
        sampled = pd.concat([days[k] for k in rng.integers(0, len(days), len(days))])
        means = sampled.groupby("flag").value.mean()
        if True in means.index and False in means.index:
            estimates.append(float(means[True] - means[False]))
    observed = temp.groupby("flag").value.mean()
    if len(observed) != 2:
        return {"difference": None, "ci95": None}
    return {"difference": float(observed[True] - observed[False]),
            "ci95": [float(x) for x in np.quantile(estimates, [.025, .975])],
            "resampled_days": len(days)}


def run(data: Path, root: Path) -> dict:
    contract = json.loads((root / "stage19/contract.json").read_text())
    if digest(data) != contract["source_sha256"]:
        raise ValueError("Source differs from audited CSV")
    clean, _ = read_data(data)
    prediction = pd.read_csv(root / "stage15/diagnostic_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
    quarter_train = clean.loc[clean.index < "2021-07-01", POWER_COLUMNS].max(axis=1)
    high_cutoff = float(quarter_train.quantile(.9))
    validation = pd.read_csv(root / "stage13/validation_predictions.csv", parse_dates=["timestamp"])
    validation = validation.loc[validation.timestamp < "2021-07-01"]
    large_cutoff = float((validation["expanding"] - validation.actual).abs().quantile(.9))
    factors = known_conditions(clean, prediction.index, high_cutoff)
    rows = prediction.join(factors)
    definitions = ["prior_idle6", "weekday_work_start", "prior_quarter_ramp_positive", "prior_quarter_high", "workhour_prior_active"]
    summary = {"input_sha256": contract["source_sha256"], "high_quarter_cutoff_from_jan_june": high_cutoff,
               "large_error_cutoff_from_apr_june_oof": large_cutoff,
               "overall": metrics(rows, large_cutoff),
               "conditions": {name: {str(key): metrics(group, large_cutoff) for key, group in rows.groupby(name)} for name in definitions},
               "day_cluster_95pct_difference": {
                   "prior_idle6_mae": day_bootstrap_difference(rows, "prior_idle6", "absolute_error"),
                   "weekday_work_start_mae": day_bootstrap_difference(rows, "weekday_work_start", "absolute_error"),
                   "prior_quarter_high_peak_rate": day_bootstrap_difference(rows, "prior_quarter_high", "peak")},
               "limitations": ["August-September were repeatedly inspected; descriptive associations, not independent proof.",
                               "Condition flags use observations through t-1; their association is not a causal effect.",
                               "The peak threshold 182 is a prior operational convention, not an official tariff threshold.",
                               "Day-cluster intervals condition on these 45 observed days and omit model/threshold selection uncertainty."]}
    output = root / "stage20"
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "condition_rows.csv", index_label="timestamp")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.data, args.output_root)
