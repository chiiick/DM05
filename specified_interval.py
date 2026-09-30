"""Stage 22: issue one forecast for a requested future hourly interval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from deploy import digest
from audit import block_bootstrap_improvement
from forecast import POWER_COLUMNS, read_data, threshold_for_recall

MAX_LEAD_HOURS = 24
ISSUE_HOURS = (0, 6, 12, 18)
METHODS = ("persistence", "previous_day", "previous_week", "residual_hgb")
MODEL_FEATURES = ("lead_hours", "persistence", "previous_day", "previous_week",
                  "quarter_previous_day", "quarter_previous_week", "origin_production",
                  "past24_production_sum", "past24_power_mean", "target_hour_sin",
                  "target_hour_cos", "target_weekday_sin", "target_weekday_cos",
                  "target_month_sin", "target_month_cos")


def whole_hour(value: str | pd.Timestamp, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None or stamp != stamp.floor("h"):
        raise ValueError(f"{label} must be a timezone-naive whole hour")
    return stamp


def validate_request(origin: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    if start <= origin or end < start or end > origin + pd.Timedelta(hours=MAX_LEAD_HOURS):
        raise ValueError("Interval must be future hourly timestamps within 24 hours of the last observation")
    return pd.date_range(start, end, freq="h")


def candidates(clean: pd.DataFrame, origin: pd.Timestamp, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """Every source timestamp is no later than the forecast origin."""
    hourly = clean.asfreq("h")
    if origin not in hourly.index or not np.isfinite(hourly.loc[origin, "평균"]):
        raise ValueError("Forecast origin is not an observed hour")
    out = pd.DataFrame({"origin": origin, "timestamp": targets, "lead_hours": ((targets - origin) / pd.Timedelta(hours=1)).astype(int)})
    last_mean = float(hourly.loc[origin, "평균"])
    last_peak = float(hourly.loc[origin, POWER_COLUMNS].max())
    out["persistence"] = last_mean
    out["quarter_persistence"] = last_peak
    recent = hourly.loc[origin - pd.Timedelta(hours=23):origin]
    if len(recent) != 24 or recent[["평균", "생산량"]].isna().any().any():
        raise ValueError("Incomplete 24-hour history at forecast origin")
    out["origin_production"] = float(hourly.loc[origin, "생산량"])
    out["past24_production_sum"] = float(recent["생산량"].sum())
    out["past24_power_mean"] = float(recent["평균"].mean())
    out["target_hour_sin"] = np.sin(2 * np.pi * targets.hour / 24)
    out["target_hour_cos"] = np.cos(2 * np.pi * targets.hour / 24)
    out["target_weekday_sin"] = np.sin(2 * np.pi * targets.dayofweek / 7)
    out["target_weekday_cos"] = np.cos(2 * np.pi * targets.dayofweek / 7)
    out["target_month_sin"] = np.sin(2 * np.pi * targets.month / 12)
    out["target_month_cos"] = np.cos(2 * np.pi * targets.month / 12)
    for label, lag in (("previous_day", 24), ("previous_week", 168)):
        source = targets - pd.Timedelta(hours=lag)
        if source.max() > origin:
            raise ValueError("Reference would read a future observation")
        mean = hourly["평균"].reindex(source).to_numpy(dtype=float)
        peak = hourly[POWER_COLUMNS].max(axis=1, skipna=False).reindex(source).to_numpy(dtype=float)
        if not (np.isfinite(mean).all() and np.isfinite(peak).all()):
            raise ValueError("Missing exact prior day/week reference")
        out[label] = mean
        out[f"quarter_{label}"] = peak
    return out


def historical_requests(clean: pd.DataFrame, first: str, last: str, *, target_before: str | None = None) -> pd.DataFrame:
    hourly = clean.asfreq("h")
    origins = pd.date_range(first, last, freq="h")
    records = []
    for origin in origins:
        if origin.hour not in ISSUE_HOURS:
            continue
        targets = pd.date_range(origin + pd.Timedelta(hours=1), periods=MAX_LEAD_HOURS, freq="h")
        if target_before is not None and targets[-1] >= pd.Timestamp(target_before):
            continue
        try:
            frame = candidates(clean, origin, targets)
        except ValueError:
            continue
        actual = hourly["평균"].reindex(targets).to_numpy(dtype=float)
        peak = hourly[POWER_COLUMNS].max(axis=1, skipna=False).reindex(targets).to_numpy(dtype=float)
        if not (np.isfinite(actual).all() and np.isfinite(peak).all()):
            continue
        frame["actual"] = actual
        frame["actual_quarter"] = peak
        records.append(frame)
    if not records:
        raise ValueError("No complete historical requests")
    return pd.concat(records, ignore_index=True)


def fit_residual_model(train: pd.DataFrame, fit_before: pd.Timestamp) -> HistGradientBoostingRegressor:
    if train.empty or train["timestamp"].max() >= fit_before:
        raise ValueError("Training labels cross the forecast fit boundary")
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=120, max_leaf_nodes=15,
        min_samples_leaf=50, learning_rate=.05, l2_regularization=10., random_state=42,
        early_stopping=False)
    return model.fit(train[list(MODEL_FEATURES)], train.actual - train.previous_week)


def predict_residual_model(model: HistGradientBoostingRegressor, rows: pd.DataFrame) -> np.ndarray:
    return np.maximum(0, rows.previous_week.to_numpy() + model.predict(rows[list(MODEL_FEATURES)]))


def regression(frame: pd.DataFrame, candidate: str) -> dict:
    error = frame[candidate] - frame.actual
    return {"request_hours": int(len(frame)), "requests": int(frame.origin.nunique()),
            "mae": float(error.abs().mean()), "rmse": float(np.sqrt((error**2).mean()))}


def peak_counts(frame: pd.DataFrame, candidate: str, threshold: float) -> dict:
    event = frame.actual_quarter.ge(182)
    alert = frame[f"quarter_{candidate}"].ge(threshold)
    return {"tp": int((event & alert).sum()), "fp": int((~event & alert).sum()),
            "fn": int((event & ~alert).sum()), "peak_hours_in_requests": int(event.sum())}


def diagnostic_groups(frame: pd.DataFrame, candidate: str, threshold: float) -> dict:
    work_start = (frame.timestamp.dt.dayofweek < 5) & frame.timestamp.dt.hour.between(7, 9)
    no_recent_production = frame.past24_production_sum.eq(0)
    return {"weekday_7_to_9": {str(flag): {**regression(group, candidate),
                                           **peak_counts(group, candidate, threshold)}
                                  for flag, group in frame.groupby(work_start)},
            "no_production_before_origin": {str(flag): {**regression(group, candidate),
                                                        **peak_counts(group, candidate, threshold)}
                                           for flag, group in frame.groupby(no_recent_production)}}


def july_feature_sensitivity(model: HistGradientBoostingRegressor, rows: pd.DataFrame) -> dict:
    """Joint permutation sensitivity on a separate diagnostic month; not causality."""
    groups = {"past_profile": ["persistence", "previous_day", "previous_week",
                                "quarter_previous_day", "quarter_previous_week"],
              "origin_operation": ["origin_production", "past24_production_sum", "past24_power_mean"],
              "target_calendar_and_lead": ["lead_hours", "target_hour_sin", "target_hour_cos",
                                           "target_weekday_sin", "target_weekday_cos",
                                           "target_month_sin", "target_month_cos"]}
    base = float(np.abs(predict_residual_model(model, rows) - rows.actual).mean())
    rng = np.random.default_rng(42)
    result = {}
    for name, features in groups.items():
        changes = []
        for _ in range(3):
            shuffled = rows.copy()
            shuffled.loc[:, features] = rows[features].iloc[rng.permutation(len(rows))].to_numpy()
            changes.append(float(np.abs(predict_residual_model(model, shuffled) - rows.actual).mean()) - base)
        result[name] = {"features": features, "mae_change_repeats": changes,
                        "mean_mae_change": float(np.mean(changes))}
    return {"base_july_mae": base, "groups": result,
            "limitation": "Joint row permutation breaks temporal and cross-group dependence; predictive sensitivity only."}


def run(data: Path, root: Path) -> dict:
    clean, _ = read_data(data)
    # Every comparison uses the same four daily issuance times and 24 forecast hours.
    all_rows = historical_requests(clean, "2021-01-08", "2021-09-13 18:00", target_before="2021-09-15")
    selection_rows = all_rows.loc[(all_rows.origin >= "2021-04-01") & (all_rows.timestamp < "2021-07-01")].copy()
    selection_rows = selection_rows.groupby("origin").filter(lambda group: len(group) == MAX_LEAD_HOURS)
    # Expanding-window fits: every training target ends before its validation month.
    for start, end in (("2021-04-01", "2021-05-01"), ("2021-05-01", "2021-06-01"),
                       ("2021-06-01", "2021-07-01")):
        before = pd.Timestamp(start)
        train = all_rows.loc[all_rows.timestamp < before].copy()
        valid = selection_rows.origin.between(before, pd.Timestamp(end), inclusive="left")
        model = fit_residual_model(train, before)
        selection_rows.loc[valid, "residual_hgb"] = predict_residual_model(model, selection_rows.loc[valid])
    if selection_rows.residual_hgb.isna().any():
        raise AssertionError("Every selection request needs an out-of-fold AI prediction")
    selection_rows["quarter_residual_hgb"] = np.maximum(selection_rows.quarter_previous_week, selection_rows.residual_hgb)
    month = selection_rows.origin.dt.strftime("%Y-%m")
    scores = {name: float((selection_rows[name] - selection_rows.actual).abs().groupby(month).mean().mean()) for name in METHODS}
    chosen = min(scores, key=scores.get)
    # An hourly quarter maximum cannot be less than its predicted hourly mean.
    # For the AI candidate the prior-week quarter shape is a floor, not a separate learned peak model.
    peak_method = chosen
    peak_threshold = threshold_for_recall(selection_rows[f"quarter_{peak_method}"].to_numpy(),
                                           selection_rows.actual_quarter.ge(182).to_numpy(), .85)
    diagnostic = all_rows.loc[(all_rows.origin >= "2021-07-01") & (all_rows.timestamp < "2021-08-01")].copy()
    diagnostic = diagnostic.groupby("origin").filter(lambda group: len(group) == MAX_LEAD_HOURS)
    evaluation = all_rows.loc[all_rows.origin >= "2021-08-01"].copy()
    evaluation = evaluation.groupby("origin").filter(lambda group: len(group) == MAX_LEAD_HOURS)
    # The comparison candidate is refit at each month boundary using past labels only.
    july_model = None
    for start, end, frame in (("2021-07-01", "2021-08-01", diagnostic),
                              ("2021-08-01", "2021-09-01", evaluation),
                              ("2021-09-01", "2021-09-15", evaluation)):
        before = pd.Timestamp(start)
        model = fit_residual_model(all_rows.loc[all_rows.timestamp < before], before)
        if start == "2021-07-01":
            july_model = model
        valid = frame.origin.between(before, pd.Timestamp(end), inclusive="left")
        frame.loc[valid, "residual_hgb"] = predict_residual_model(model, frame.loc[valid])
    if diagnostic.residual_hgb.isna().any() or evaluation.residual_hgb.isna().any():
        raise AssertionError("Every diagnostic request needs an AI prediction")
    for frame in (diagnostic, evaluation):
        frame["quarter_residual_hgb"] = np.maximum(frame.quarter_previous_week, frame.residual_hgb)
    hourly_production = clean["생산량"].asfreq("h")
    current_prod = hourly_production.reindex(evaluation.timestamp).to_numpy() > 0
    prior_prod = hourly_production.shift(1).reindex(evaluation.timestamp).to_numpy() > 0
    evaluation["observed_state_posthoc"] = np.select(
        [current_prod & ~prior_prod, ~current_prod & prior_prod, current_prod & prior_prod],
        ["startup", "shutdown", "running"], default="idle")
    output = root / "stage22"
    output.mkdir(parents=True, exist_ok=True)
    selection = {"source_sha256": digest(data), "selection_end": "2021-06-30 23:00:00",
                 "candidate_monthly_equal_mae": scores, "chosen": chosen, "peak_method": peak_method,
                 "peak_score_threshold": float(peak_threshold),
                 "peak_event_threshold": 182, "issue_hours": list(ISSUE_HOURS), "max_lead_hours": MAX_LEAD_HOURS,
                 "note": "Only Apr-Jun requests choose the method and peak threshold; July and Aug-Sep do not."}
    report = {"selection": selection,
              "july_diagnostic": regression(diagnostic, chosen),
              "july_candidate_comparison": {name: regression(diagnostic, name) for name in METHODS},
              "august_september_repeatedly_inspected": regression(evaluation, chosen),
              "by_month": {str(month): regression(group, chosen) for month, group in
                           evaluation.groupby(evaluation.origin.dt.strftime("%Y-%m"))},
              "by_issued_hour": {str(hour): regression(group, chosen) for hour, group in evaluation.groupby(evaluation.origin.dt.hour)},
              "by_lead_bucket": {label: regression(group, chosen) for label, group in {
                  "1_to_6": evaluation.loc[evaluation.lead_hours.between(1, 6)],
                  "7_to_12": evaluation.loc[evaluation.lead_hours.between(7, 12)],
                  "13_to_24": evaluation.loc[evaluation.lead_hours.between(13, 24)]}.items()},
              "candidate_comparison": {name: regression(evaluation, name) for name in METHODS},
              "mae_reduction_vs_previous_week": block_bootstrap_improvement(
                  evaluation.set_index("origin"), "previous_week", chosen),
              "peak_alert": peak_counts(evaluation, peak_method, peak_threshold),
              "peak_alert_00_issue_nonoverlapping": peak_counts(
                  evaluation.loc[evaluation.origin.dt.hour.eq(0)], peak_method, peak_threshold),
              "conditions": diagnostic_groups(evaluation, chosen, peak_threshold),
              "posthoc_production_state": {str(state): {**regression(group, chosen),
                                                       **peak_counts(group, peak_method, peak_threshold)}
                                           for state, group in evaluation.groupby("observed_state_posthoc")},
              "july_feature_sensitivity": july_feature_sensitivity(july_model, diagnostic),
              "limitations": ["Specified future intervals within 24 hours are a provisional interpretation; official horizon is unknown.",
                              "Four issuance hours test availability at different times; API also accepts other whole-hour origins.",
                              "Overlapping requests repeat target hours, so request-hour rows are not independent observations.",
                              "August-September were repeatedly inspected and are retrospective, not a new test.",
                              "A pooled residual HGB AI candidate is selected only if it beats baselines on Apr-Jun; no claimed gain otherwise.",
                              "AI quarter-peak score is max(previous-week quarter profile, AI hourly mean); this enforces a physical lower bound but is not a separately learned peak model."]}
    selection_path = output / "selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation[["origin", "timestamp", "lead_hours", "actual", "actual_quarter", *METHODS,
                *(f"quarter_{name}" for name in METHODS), "observed_state_posthoc"]].to_csv(
                    output / "evaluation_requests.csv", index=False)
    if chosen == "residual_hgb":
        fit_before = clean.index.max() + pd.Timedelta(hours=1)
        final_train = all_rows.loc[all_rows.timestamp < fit_before]
        artifact = root.parent / "artifacts/specified_interval.joblib"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": fit_residual_model(final_train, fit_before), "train_end": str(final_train.timestamp.max()),
                    "features": list(MODEL_FEATURES)}, artifact)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def forecast_interval(data: Path, selection_path: Path, origin: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    origin, start, end = (whole_hour(value, name) for value, name in ((origin, "origin"), (start, "start"), (end, "end")))
    targets = validate_request(origin, start, end)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if origin <= pd.Timestamp(selection["selection_end"]):
        raise ValueError("This forecast origin predates method selection")
    clean, _ = read_data(data, allow_partial_last_day=True, observed_through=origin)
    if clean.index.max() != origin:
        raise ValueError("Origin is absent from the available observations")
    refs = candidates(clean, origin, targets)
    method = selection["chosen"]
    if method == "residual_hgb":
        artifact = selection_path.parents[2] / "artifacts/specified_interval.joblib"
        bundle = joblib.load(artifact)
        if pd.Timestamp(bundle["train_end"]) > origin or bundle["features"] != list(MODEL_FEATURES):
            raise ValueError("Interval model was trained beyond forecast origin or features changed")
        refs[method] = predict_residual_model(bundle["model"], refs)
        refs["quarter_residual_hgb"] = np.maximum(refs.quarter_previous_week, refs.residual_hgb)
    result = refs[["origin", "timestamp", "lead_hours"]].copy()
    result["predicted_mean"] = refs[method]
    result["predicted_quarter_max"] = refs[f"quarter_{selection['peak_method']}"]
    result["peak_alert"] = result.predicted_quarter_max.ge(selection["peak_score_threshold"])
    report = {"origin": str(origin), "start": str(start), "end": str(end), "method": method,
              "hours": len(result), "interval_mean_of_hourly_means": float(result.predicted_mean.mean()),
              "interval_max_of_hourly_means": float(result.predicted_mean.max()),
              "interval_max_of_predicted_quarter_values": float(result.predicted_quarter_max.max()),
              "alerted_hours": int(result.peak_alert.sum()),
              "unit_note": "Physical units and energy conversion are unconfirmed; hourly means are not asserted kWh.",
              "input_file_sha256": digest(data), "selection_sha256": digest(selection_path)}
    return result, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--origin", help="Last observed hour, e.g. 2021-09-14 23:00")
    parser.add_argument("--start", help="First requested future hour")
    parser.add_argument("--end", help="Last requested future hour, inclusive")
    args = parser.parse_args()
    if any(value is not None for value in (args.origin, args.start, args.end)):
        if not all(value is not None for value in (args.origin, args.start, args.end)):
            parser.error("--origin, --start, and --end must be supplied together")
        rows, summary = forecast_interval(args.data, args.output_root / "stage22/selection.json", args.origin, args.start, args.end)
        print(rows.to_csv(index=False), end="")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        run(args.data, args.output_root)
