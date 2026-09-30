"""Step 1-2: point-in-time adaptive ensemble for requested power intervals.

Combines Stage 22's monthly refit AI with Step 1-1's state projection. Expert
weights use errors whose target measurements were already observed at issue time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import block_bootstrap_improvement
from forecast import POWER_COLUMNS, read_data, threshold_for_recall
from specified_interval import (ISSUE_HOURS, MAX_LEAD_HOURS, candidates,
                                fit_residual_model, historical_requests,
                                peak_counts, predict_residual_model,
                                validate_request, whole_hour)
from step_1_1 import project


EXPERT_COLUMNS = ("origin", "timestamp", "lead_hours", "actual", "actual_quarter",
                  "previous_week", "quarter_previous_week", "residual_hgb",
                  "state_projection", "low_state", "origin_ratio", "target_normal")
SELECTION_END = pd.Timestamp("2021-06-30 23:00")
POLICY_GRID = tuple((days, margin, temperature)
                    for days in (3, 5, 7, 14)
                    for margin in (0.0, 1.0)
                    for temperature in (1.0, 3.0))


def _complete_requests(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby("origin").filter(lambda group: len(group) == MAX_LEAD_HOURS).copy()


def build_experts(clean: pd.DataFrame) -> pd.DataFrame:
    """Monthly expert predictions; each fit ends before its forecast month."""
    last_observed = clean.index.max()
    last_origin = last_observed - pd.Timedelta(hours=MAX_LEAD_HOURS)
    all_rows = historical_requests(clean, "2021-01-08", str(last_origin),
                                   target_before=str(last_observed + pd.Timedelta(hours=1)))
    parts = []
    for month in pd.period_range("2021-03", last_origin.to_period("M"), freq="M"):
        start = month.to_timestamp()
        end = start + pd.offsets.MonthBegin(1)
        valid = all_rows.loc[all_rows.origin.ge(start) & all_rows.origin.lt(end)].copy()
        if valid.empty:
            continue
        model = fit_residual_model(all_rows.loc[all_rows.timestamp.lt(start)], start)
        valid["residual_hgb"] = predict_residual_model(model, valid)
        valid, _ = project(clean, valid, start - pd.Timedelta(hours=1))
        parts.append(valid[list(EXPERT_COLUMNS)])
    if not parts:
        raise ValueError("Insufficient history for expert evaluation")
    return pd.concat(parts, ignore_index=True)


def recent_error_table(experts: pd.DataFrame, days: int) -> pd.DataFrame:
    """At each origin, score only predictions made earlier with target < origin."""
    history = experts.sort_values(["origin", "timestamp"])
    records = []
    for origin in history.origin.drop_duplicates().sort_values():
        past = history.loc[history.origin.lt(origin) &
                           history.origin.ge(origin - pd.Timedelta(days=days)) &
                           history.timestamp.lt(origin)]
        records.append({"origin": origin, "matured_request_hours": int(len(past)),
                        "recent_hgb_mae": float((past.residual_hgb-past.actual).abs().mean()),
                        "recent_state_mae": float((past.state_projection-past.actual).abs().mean())})
    return pd.DataFrame(records)


def state_weight(hgb_mae: np.ndarray, state_mae: np.ndarray, count: np.ndarray,
                 *, margin: float, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    advantage = (hgb_mae - state_mae - margin) / temperature
    weights = 1 / (1 + np.exp(-np.clip(advantage, -50, 50)))
    return np.where((count >= MAX_LEAD_HOURS) & np.isfinite(advantage), weights, 0.0)


def apply_policy(experts: pd.DataFrame, policy: dict,
                 recent: pd.DataFrame | None = None) -> pd.DataFrame:
    out = experts.copy()
    method = policy["method"]
    if method == "hgb":
        out["state_weight"] = 0.0
    elif method == "state":
        out["state_weight"] = 1.0
    elif method == "equal":
        out["state_weight"] = .5
    elif method == "adaptive":
        if recent is None:
            recent = recent_error_table(experts, policy["lookback_days"])
        out = out.merge(recent, on="origin", validate="many_to_one")
        out["state_weight"] = state_weight(
            out.recent_hgb_mae.to_numpy(), out.recent_state_mae.to_numpy(),
            out.matured_request_hours.to_numpy(), margin=policy["margin"],
            temperature=policy["temperature"])
    else:
        raise ValueError(f"Unknown policy method: {method}")
    if method != "adaptive":
        out["recent_hgb_mae"] = np.nan
        out["recent_state_mae"] = np.nan
        out["matured_request_hours"] = 0
    out["ensemble"] = np.maximum(0, (1-out.state_weight)*out.residual_hgb +
                                  out.state_weight*out.state_projection)
    out["quarter_ensemble"] = np.maximum(out.quarter_previous_week, out.ensemble)
    return out


def _selection_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return _complete_requests(rows.loc[rows.origin.between("2021-04-01", SELECTION_END) &
                                       rows.timestamp.le(SELECTION_END)])


def _july_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return _complete_requests(rows.loc[rows.origin.between("2021-07-01", "2021-07-31 23:00") &
                                       rows.timestamp.lt("2021-08-01")])


def _mae(rows: pd.DataFrame, column: str) -> float:
    return float((rows[column]-rows.actual).abs().mean())


def _monthly_equal_mae(rows: pd.DataFrame, column: str) -> float:
    return float((rows[column]-rows.actual).abs().groupby(
        rows.origin.dt.strftime("%Y-%m")).mean().mean())


def choose_policy(experts: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Select using April-June only; July and August-September are excluded."""
    scored = []
    for method in ("hgb", "state", "equal"):
        policy = {"method": method}
        scored.append({**policy, "selection_mae": _monthly_equal_mae(
            _selection_rows(apply_policy(experts, policy)), "ensemble")})
    for days in sorted({item[0] for item in POLICY_GRID}):
        recent = recent_error_table(experts, days)
        for lookback, margin, temperature in (item for item in POLICY_GRID if item[0] == days):
            policy = {"method": "adaptive", "lookback_days": lookback,
                      "margin": margin, "temperature": temperature}
            scored.append({**policy, "selection_mae": _monthly_equal_mae(
                _selection_rows(apply_policy(experts, policy, recent)), "ensemble")})
    winner = min(scored, key=lambda item: item["selection_mae"])
    return winner, sorted(scored, key=lambda item: item["selection_mae"])


def _scores(rows: pd.DataFrame) -> dict:
    return {name: _mae(rows, name) for name in
            ("ensemble", "residual_hgb", "state_projection", "previous_week")}


def _interval_scores(rows: pd.DataFrame) -> dict:
    grouped = rows.groupby("origin")
    actual_mean, pred_mean = grouped.actual.mean(), grouped.ensemble.mean()
    actual_max, pred_max = grouped.actual.max(), grouped.ensemble.max()
    return {"requests": int(grouped.ngroups),
            "mean_of_hourly_means_mae": float((pred_mean-actual_mean).abs().mean()),
            "max_of_hourly_means_mae": float((pred_max-actual_max).abs().mean())}


def evaluate(data: Path, output_dir: Path) -> dict:
    clean, _ = read_data(data)
    experts = build_experts(clean)
    policy, candidate_scores = choose_policy(experts)
    final = apply_policy(experts, policy)
    selection = _selection_rows(final)
    july = _july_rows(final)
    development = _complete_requests(final.loc[final.origin.ge("2021-08-01")])
    peak_threshold = threshold_for_recall(selection.quarter_ensemble.to_numpy(),
                                           selection.actual_quarter.ge(182).to_numpy(), .85)
    policy = {**policy, "selection_end": str(SELECTION_END),
              "peak_score_threshold": float(peak_threshold), "peak_event_threshold": 182,
              "issue_hours": list(ISSUE_HOURS), "max_lead_hours": MAX_LEAD_HOURS}
    report = {"policy": policy, "candidate_selection_scores": candidate_scores,
              "selection_apr_jun": _scores(selection),
              "selection_monthly_equal_mae": _monthly_equal_mae(selection, "ensemble"),
              "july_diagnostic": _scores(july), "august_september_repeatedly_inspected": _scores(development),
              "by_month": {name: {"request_hours": len(group), **_scores(group),
                                  "mean_state_weight": float(group.state_weight.mean())}
                           for name, group in development.groupby(
                               development.origin.dt.strftime("%Y-%m"))},
              "by_lead_bucket": {name: _scores(group) for name, group in {
                  "1_to_6": development.loc[development.lead_hours.between(1, 6)],
                  "7_to_12": development.loc[development.lead_hours.between(7, 12)],
                  "13_to_24": development.loc[development.lead_hours.between(13, 24)]}.items()},
              "by_observed_low_state": {str(state).lower(): _scores(group) for state, group in
                                        development.groupby("low_state")},
              "interval_24h": _interval_scores(development),
              "selection_gain_vs_hgb_block_bootstrap": block_bootstrap_improvement(
                  selection.set_index("origin"), "residual_hgb", "ensemble"),
              "peak_alert": peak_counts(development, "ensemble", float(peak_threshold)),
              "peak_alert_00_issue_nonoverlapping": peak_counts(
                  development.loc[development.origin.dt.hour.eq(0)], "ensemble", float(peak_threshold)),
              "mae_reduction_vs_stage22_block_bootstrap": block_bootstrap_improvement(
                  development.set_index("origin"), "residual_hgb", "ensemble"),
              "mae_reduction_vs_state_block_bootstrap": block_bootstrap_improvement(
                  development.set_index("origin"), "state_projection", "ensemble"),
              "limitations": ["April-June choose the policy; July is diagnostic and August-September were repeatedly inspected, so there is no untouched test.",
                              "The 19 candidate policies are compared on only three selection months; the winner may overfit them.",
                              "The ensemble uses repeated forecast requests for recent errors; these are correlated, not independent observations.",
                              "Peak score uses the previous-week quarter maximum as a floor; this is not a separate learned peak model or verified savings estimate."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    final[["origin", "timestamp", "lead_hours", "actual", "residual_hgb", "state_projection",
           "previous_week", "low_state", "state_weight", "ensemble"]].to_csv(
               output_dir / "expert_requests.csv", index=False)
    development[["origin", "timestamp", "lead_hours", "actual", "actual_quarter",
                 "residual_hgb", "state_projection", "previous_week", "low_state",
                 "recent_hgb_mae", "recent_state_mae", "matured_request_hours",
                 "state_weight", "ensemble", "quarter_ensemble"]].to_csv(
                     output_dir / "evaluation_requests.csv", index=False)
    return report


def _history_model(history: pd.DataFrame, month_start: pd.Timestamp,
                   models: dict) -> object:
    if month_start not in models:
        train = history.loc[history.timestamp.lt(month_start)]
        models[month_start] = fit_residual_model(train, month_start)
    return models[month_start]


def _add_matured_boundary_rows(clean: pd.DataFrame, history: pd.DataFrame,
                               month_start: pd.Timestamp) -> pd.DataFrame:
    """Include known targets of previous-day requests whose 24h horizon is incomplete.

    Retrospective monthly training includes their labels before month_start.
    At the boundary, later targets have not arrived and are never constructed.
    """
    hourly = clean.asfreq("h")
    existing = set(history.origin.unique())
    additions = []
    for old_origin in pd.date_range(month_start - pd.Timedelta(hours=24),
                                    month_start - pd.Timedelta(hours=1), freq="h"):
        if old_origin.hour not in ISSUE_HOURS or old_origin in existing:
            continue
        final_target = min(old_origin + pd.Timedelta(hours=MAX_LEAD_HOURS),
                           month_start - pd.Timedelta(hours=1))
        targets = pd.date_range(old_origin + pd.Timedelta(hours=1), final_target, freq="h")
        if targets.empty:
            continue
        try:
            rows = candidates(clean, old_origin, targets)
        except ValueError:
            continue
        actual = hourly["평균"].reindex(targets).to_numpy(dtype=float)
        peak = hourly[POWER_COLUMNS].max(axis=1, skipna=False).reindex(targets).to_numpy(dtype=float)
        if not (np.isfinite(actual).all() and np.isfinite(peak).all()):
            continue
        rows["actual"] = actual
        rows["actual_quarter"] = peak
        additions.append(rows)
    return pd.concat([history, *additions], ignore_index=True) if additions else history


def forecast(data: Path, policy_path: Path, origin: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    origin, start, end = (whole_hour(value, label) for value, label in
                          ((origin, "origin"), (start, "start"), (end, "end")))
    targets = validate_request(origin, start, end)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if origin <= pd.Timestamp(policy["selection_end"]):
        raise ValueError("Forecast origin predates policy selection")
    clean, _ = read_data(data, allow_partial_last_day=True, observed_through=origin)
    if clean.index.max() != origin:
        raise ValueError("Origin is absent from available observations")
    history = historical_requests(clean, "2021-01-08",
                                  str(origin - pd.Timedelta(hours=MAX_LEAD_HOURS)),
                                  target_before=str(origin + pd.Timedelta(hours=1)))
    models = {}
    month_start = origin.to_period("M").to_timestamp()
    history = _add_matured_boundary_rows(clean, history, month_start)
    model = _history_model(history, month_start, models)
    rows = candidates(clean, origin, targets)
    rows["residual_hgb"] = predict_residual_model(model, rows)
    rows, _ = project(clean, rows, month_start - pd.Timedelta(hours=1))
    if policy["method"] == "adaptive":
        lookback = pd.Timedelta(days=policy["lookback_days"])
        past = []
        for previous_origin in pd.date_range(origin - lookback, origin - pd.Timedelta(hours=1), freq="h"):
            if previous_origin.hour not in ISSUE_HOURS or previous_origin not in clean.index:
                continue
            previous_month = previous_origin.to_period("M").to_timestamp()
            old_targets = pd.date_range(previous_origin + pd.Timedelta(hours=1),
                                        periods=MAX_LEAD_HOURS, freq="h")
            try:
                old_rows = candidates(clean, previous_origin, old_targets)
            except ValueError:
                continue
            old_rows["residual_hgb"] = predict_residual_model(
                _history_model(history, previous_month, models), old_rows)
            old_rows, _ = project(clean, old_rows,
                                  previous_month - pd.Timedelta(hours=1))
            observed = old_rows.timestamp.lt(origin)
            old_rows = old_rows.loc[observed].copy()
            old_rows["actual"] = clean["평균"].reindex(old_rows.timestamp).to_numpy(dtype=float)
            past.append(old_rows.loc[old_rows.actual.notna(),
                                     ["actual", "residual_hgb", "state_projection"]])
        matured = pd.concat(past, ignore_index=True) if past else pd.DataFrame(
            columns=["actual", "residual_hgb", "state_projection"])
        hgb_error = float((matured.residual_hgb-matured.actual).abs().mean())
        state_error = float((matured.state_projection-matured.actual).abs().mean())
        weight = float(state_weight(np.array([hgb_error]), np.array([state_error]),
                                    np.array([len(matured)]), margin=policy["margin"],
                                    temperature=policy["temperature"])[0])
    else:
        weight = {"hgb": 0.0, "state": 1.0, "equal": .5}[policy["method"]]
        matured, hgb_error, state_error = pd.DataFrame(), None, None
    rows["state_weight"] = weight
    rows["predicted_hourly_mean"] = np.maximum(0, (1-weight)*rows.residual_hgb +
                                    weight*rows.state_projection)
    rows["predicted_quarter_score"] = np.maximum(rows.quarter_previous_week,
                                                  rows.predicted_hourly_mean)
    rows["peak_alert"] = rows.predicted_quarter_score.ge(policy["peak_score_threshold"])
    output = rows[["origin", "timestamp", "lead_hours", "predicted_hourly_mean",
                   "residual_hgb", "state_projection", "state_weight", "low_state",
                   "predicted_quarter_score", "peak_alert"]]
    report = {"origin": str(origin), "start": str(start), "end": str(end),
              "hours": len(output), "method": policy["method"],
              "recent_matured_request_hours": len(matured),
              "recent_hgb_mae": hgb_error, "recent_state_mae": state_error,
              "state_weight": weight,
              "interval_mean_of_hourly_means": float(output.predicted_hourly_mean.mean()),
              "interval_max_of_hourly_means": float(output.predicted_hourly_mean.max()),
              "alerted_hours": int(output.peak_alert.sum()),
              "unit_note": "Hourly mean units and energy conversion are unconfirmed."}
    return output, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step_1_2"))
    parser.add_argument("--origin")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    if any((args.origin, args.start, args.end)):
        if not all((args.origin, args.start, args.end)):
            parser.error("--origin, --start, and --end must be supplied together")
        rows, report = forecast(args.data, args.output_dir / "policy.json",
                                args.origin, args.start, args.end)
        print(rows.to_csv(index=False), end="")
    else:
        report = evaluate(args.data, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
