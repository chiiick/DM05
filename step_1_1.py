"""Step 1-1: causal load-state projection for a requested future interval.

The rule was reconstructed from the user-supplied alternative ZIP. It is a
mathematical forecast; the existing Stage 22 AI remains a separate comparator.
"""
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from forecast import read_data
from specified_interval import (MAX_LEAD_HOURS, historical_requests,
                                validate_request, whole_hour)


def load_levels(power: pd.Series, cutoff: pd.Timestamp) -> dict:
    """Locate the largest observed idle/active gap using fitting history only."""
    train = power.loc[:cutoff].dropna()
    if train.empty:
        raise ValueError("No power observations before fitting cutoff")
    values = np.sort(train.unique())
    gaps = np.diff(values)
    eligible = (values[:-1] >= train.quantile(.10)) & (values[:-1] <= train.quantile(.60))
    threshold = (float((values[np.flatnonzero(eligible)[np.argmax(gaps[eligible])]] +
                        values[np.flatnonzero(eligible)[np.argmax(gaps[eligible])]+1]) / 2)
                 if eligible.any() and gaps[eligible].max() > 0 else float(train.quantile(.25)))
    return {"threshold": threshold, "idle": float(train.loc[train.le(threshold)].median()),
            "fit_cutoff": str(cutoff)}


def _weekly_normal(power: pd.Series, stamps: pd.DatetimeIndex) -> np.ndarray:
    refs = np.column_stack([power.reindex(stamps - pd.Timedelta(weeks=week)).to_numpy(dtype=float)
                            for week in range(2, 9)])
    # pandas median skips missing references and returns NaN for empty rows.
    return pd.DataFrame(refs).median(axis=1).to_numpy(dtype=float)


def project(clean: pd.DataFrame, rows: pd.DataFrame, fit_cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Append a forecast to requests; no target or post-origin measurement is read."""
    if rows.empty:
        raise ValueError("No requested hours")
    if fit_cutoff > rows.origin.min():
        raise ValueError("Load-level fitting cutoff is after a forecast origin")
    if not rows.lead_hours.between(1, MAX_LEAD_HOURS).all():
        raise ValueError("Forecast lead is outside the validated range")
    power = clean["평균"].asfreq("h")
    levels = load_levels(power, fit_cutoff)
    threshold, idle = levels["threshold"], levels["idle"]
    grid = power.index
    normal_origin = pd.Series(_weekly_normal(power, grid), index=grid).fillna(power.shift(168))
    informative = power.notna() & normal_origin.gt(threshold)
    ratio_raw = ((power - idle).clip(lower=0) /
                 (normal_origin - idle).clip(lower=0).where(informative)).clip(0, 3)
    ratio_now = ratio_raw.ffill()
    ratio_ewm = ratio_raw.ewm(halflife=1, adjust=False, ignore_na=True,
                              min_periods=1).mean().ffill()
    origins = pd.DatetimeIndex(rows.origin)
    targets = pd.DatetimeIndex(rows.timestamp)
    if any((targets - pd.Timedelta(weeks=week) > origins).any() for week in range(2, 9)):
        raise AssertionError("Weekly reference after origin")
    target_normal = _weekly_normal(power, targets)
    target_normal = pd.Series(target_normal).fillna(rows.previous_week.reset_index(drop=True)).fillna(
        rows.previous_day.reset_index(drop=True)).to_numpy(dtype=float)
    origin_power = power.reindex(origins).to_numpy(dtype=float)
    origin_normal = normal_origin.reindex(origins).to_numpy(dtype=float)
    current_ratio = ratio_now.reindex(origins).to_numpy(dtype=float)
    smooth_ratio = ratio_ewm.reindex(origins).fillna(1).to_numpy(dtype=float)
    low_state = (origin_power <= threshold) & (current_ratio < .3)
    projected_low = idle + np.maximum(target_normal - idle, 0) * smooth_ratio
    projected_other = np.maximum(target_normal + np.nan_to_num(origin_power-origin_normal, nan=0) *
                                 np.exp(-rows.lead_hours.to_numpy(dtype=float)/12), 0)
    out = rows.copy()
    out["state_projection"] = np.where(low_state, projected_low, projected_other)
    out["low_state"] = low_state
    out["target_normal"] = target_normal
    out["origin_ratio"] = current_ratio
    out["state_when_low_else_week"] = np.where(low_state, out.state_projection, out.previous_week)
    if not np.isfinite(out.state_projection).all():
        raise ValueError("Projection has missing past references")
    return out, levels


def assess(data: Path, output_dir: Path) -> dict:
    clean, _ = read_data(data)
    # Identical four issuance hours and complete 24-hour requests to Stage 22.
    all_rows = historical_requests(clean, "2021-01-08", "2021-09-13 18:00",
                                   target_before="2021-09-15")
    periods = (("2021-04", "2021-03-31 23:00", "2021-04-01", "2021-05-01"),
               ("2021-05", "2021-04-30 23:00", "2021-05-01", "2021-06-01"),
               ("2021-06", "2021-05-31 23:00", "2021-06-01", "2021-07-01"),
               ("2021-07", "2021-06-30 23:00", "2021-07-01", "2021-08-01"),
               ("2021-08", "2021-07-31 23:00", "2021-08-01", "2021-09-01"),
               ("2021-09", "2021-08-31 23:00", "2021-09-01", "2021-09-15"))
    outputs, fitted = [], {}
    for name, cutoff, start, end in periods:
        # Stage 22 keeps cross-month targets except when the June selection
        # window would read a July label. July similarly stays diagnostic-only.
        within_month = all_rows.timestamp.lt(end) if name in ("2021-06", "2021-07") else True
        rows = all_rows.loc[(all_rows.origin >= start) & (all_rows.origin < end) &
                            within_month].copy()
        rows = rows.groupby("origin").filter(lambda group: len(group) == MAX_LEAD_HOURS)
        result, levels = project(clean, rows, pd.Timestamp(cutoff))
        fitted[name] = levels
        outputs.append(result)
    combined = pd.concat(outputs, ignore_index=True)
    def score(rows: pd.DataFrame, column: str) -> float:
        return float((rows[column] - rows.actual).abs().mean())
    by_month = {name: {"requests": int(len(group.origin.unique())), "request_hours": len(group),
                       "state_projection_mae": score(group, "state_projection"),
                       "previous_week_mae": score(group, "previous_week"),
                       "state_when_low_else_week_mae": score(group, "state_when_low_else_week"),
                       "low_state_share": float(group.low_state.mean()),
                       "by_observed_low_state": {str(state).lower(): {
                           "request_hours": len(subset),
                           "state_projection_mae": score(subset, "state_projection"),
                           "previous_week_mae": score(subset, "previous_week")}
                           for state, subset in group.groupby("low_state")}}
                for name, group in combined.groupby(combined.origin.dt.strftime("%Y-%m"))}
    held = combined.loc[combined.origin.ge("2021-08-01")].copy()
    previous = pd.read_csv(output_dir.parent / "stage22/evaluation_requests.csv",
                           parse_dates=["origin", "timestamp"])
    common = held.merge(previous[["origin", "timestamp", "lead_hours", "actual", "residual_hgb"]],
                        on=["origin", "timestamp", "lead_hours"], suffixes=("", "_stage22"),
                        validate="one_to_one")
    if len(common) != len(held) or len(common) != len(previous) or not np.allclose(
            common.actual, common.actual_stage22):
        raise AssertionError("Stage 22 and 1-1 requests differ")
    report = {"approach": "origin-observed load state times causal weekly template",
              "selection_apr_jun_monthly_equal_mae": {method: float(np.mean([
                  by_month[name][method] for name in ("2021-04", "2021-05", "2021-06")]))
                  for method in ("state_projection_mae", "previous_week_mae",
                                 "state_when_low_else_week_mae")},
              "by_month": by_month, "fit_levels": fitted,
              "same_4224_request_hours": {"state_projection_mae": score(common, "state_projection"),
                                          "state_when_low_else_week_mae": score(common, "state_when_low_else_week"),
                                          "stage22_ai_mae": score(common, "residual_hgb"),
                                          "previous_week_mae": score(common, "previous_week")},
              "limitations": ["August-September and the alternative ZIP were inspected before this 1-1 implementation; these are retrospective figures, not a held-out test.",
                              "The state rule is a mathematical baseline. Stage 22 supplies the separate AI comparator; an AI candidate on this new baseline is future work.",
                              "Only hourly mean is forecast. Peak score and operational effect are not validated by this step."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    common[["origin", "timestamp", "lead_hours", "actual", "state_projection", "residual_hgb",
            "previous_week", "state_when_low_else_week", "low_state", "target_normal", "origin_ratio"]].to_csv(
                output_dir / "evaluation_requests.csv", index=False)
    combined[["origin", "timestamp", "lead_hours", "actual", "state_projection",
              "previous_week", "state_when_low_else_week", "low_state"]].to_csv(
                  output_dir / "development_requests.csv", index=False)
    return report


def compare_archive(archive: Path, evaluation_csv: Path) -> dict:
    """Read archived CSV predictions only; serialized models are never loaded."""
    with ZipFile(archive) as bundle:
        old = pd.read_csv(BytesIO(bundle.read("predictions/current/development.csv")),
                          usecols=["origin_timestamp", "target_timestamp", "horizon_hours",
                                   "actual", "projected_power_state", "huber_five_e11_rec30"])
    old = old.rename(columns={"origin_timestamp": "origin", "target_timestamp": "timestamp",
                              "horizon_hours": "lead_hours", "actual": "archive_actual"})
    current = pd.read_csv(evaluation_csv)
    common = current.merge(old, on=["origin", "timestamp", "lead_hours"], validate="one_to_one")
    if len(common) != len(current) or not np.allclose(common.actual, common.archive_actual):
        raise AssertionError("Archive comparison requests or labels do not match")
    result = {"common_request_hours": len(common),
              "actual_max_abs_difference": float(np.abs(common.actual-common.archive_actual).max()),
              "reproduction_mean_abs_difference": float(np.abs(
                  common.state_projection-common.projected_power_state).mean()),
              "reproduction_max_abs_difference": float(np.abs(
                  common.state_projection-common.projected_power_state).max()),
              "mae": {name: float(np.abs(common[name]-common.actual).mean())
                      for name in ("state_projection", "residual_hgb", "previous_week",
                                   "projected_power_state", "huber_five_e11_rec30")}}
    (evaluation_csv.parent / "zip_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def forecast(data: Path, origin: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    from specified_interval import candidates
    origin, start, end = (whole_hour(value, label) for value, label in
                          ((origin, "origin"), (start, "start"), (end, "end")))
    targets = validate_request(origin, start, end)
    clean, _ = read_data(data, allow_partial_last_day=True, observed_through=origin)
    if clean.index.max() != origin:
        raise ValueError("Origin is absent from available observations")
    rows, levels = project(clean, candidates(clean, origin, targets), origin)
    out = rows[["origin", "timestamp", "lead_hours", "state_projection", "low_state"]].rename(
        columns={"state_projection": "predicted_hourly_mean"})
    return out, {"origin": str(origin), "start": str(start), "end": str(end),
                 "hours": len(out), "interval_mean_of_hourly_means": float(out.predicted_hourly_mean.mean()),
                 "interval_max_of_hourly_means": float(out.predicted_hourly_mean.max()),
                 "load_levels": levels,
                 "unit_note": "Hourly mean units and conversion to energy are unconfirmed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step_1_1"))
    parser.add_argument("--compare-zip", type=Path, help="Optional user-supplied alternative project ZIP")
    parser.add_argument("--origin")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    if any((args.origin, args.start, args.end)):
        if not all((args.origin, args.start, args.end)):
            parser.error("--origin, --start and --end must be supplied together")
        rows, report = forecast(args.data, args.origin, args.start, args.end)
        print(rows.to_csv(index=False), end="")
    else:
        report = assess(args.data, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.compare_zip:
        if args.origin:
            parser.error("--compare-zip requires evaluation mode")
        print(json.dumps(compare_archive(args.compare_zip,
                                         args.output_dir / "evaluation_requests.csv"),
                         ensure_ascii=False, indent=2))
