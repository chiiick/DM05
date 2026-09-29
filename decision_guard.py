"""Stage 21: forecast-origin-only advisory gate for daily load-shift proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deploy import digest
from forecast import read_data


def origin_signals(clean: pd.DataFrame, anchor: pd.Timestamp) -> dict:
    """Read only the two complete past daily profiles and prior 24-hour production."""
    if anchor.hour != 0:
        raise ValueError("Daily decision must be made at midnight")
    hourly = clean.asfreq("h")
    before = pd.date_range(anchor - pd.Timedelta(hours=24), periods=24, freq="h")
    week = pd.date_range(anchor - pd.Timedelta(hours=168), periods=24, freq="h")
    yesterday = hourly["평균"].reindex(before).to_numpy(dtype=float)
    last_week = hourly["평균"].reindex(week).to_numpy(dtype=float)
    production = hourly["생산량"].reindex(before).to_numpy(dtype=float)
    if not (np.isfinite(yesterday).all() and np.isfinite(last_week).all() and np.isfinite(production).all()):
        raise ValueError("Incomplete past daily profile or production history")
    return {"past_profile_disagreement": float(np.abs(yesterday - last_week).mean()),
            "previous_24h_no_production": bool(np.all(production == 0))}


def decision(signal: dict, cutoff: float) -> dict:
    high_disagreement = signal["past_profile_disagreement"] > cutoff
    idle_day = signal["previous_24h_no_production"]
    return {**signal, "above_calibration_disagreement": bool(high_disagreement),
            "review_required": bool(high_disagreement or idle_day),
            "reason": "+".join(reason for check, reason in ((high_disagreement, "past_profile_disagreement"),
                                                                  (idle_day, "prior_day_no_production")) if check) or "standard_review"}


def summarize_group(days: pd.DataFrame) -> dict:
    if days.empty:
        return {"n_days": 0}
    moved = days.loc[days.has_movement]
    return {"n_days": int(len(days)), "hourly_forecast_mae": float(days.daily_mae.mean()),
            "planned_move_days": int(days.has_movement.sum()),
            "assumed_flex_violation_days": int((~days.feasible_under_assumed_share & days.has_movement).sum()),
            "arithmetic_rebound_days": int((days.simulated_peak_change.gt(1e-6) & days.has_movement).sum()),
            "mean_simulated_peak_change_all_days": float(days.simulated_peak_change.mean()),
            "mean_simulated_peak_change_moved_days": float(moved.simulated_peak_change.mean()) if len(moved) else None}


def run(data: Path, root: Path) -> dict:
    stage17 = json.loads((root / "stage17/selection.json").read_text())
    stage19 = json.loads((root / "stage19/contract.json").read_text())
    if digest(data) != stage17["input_sha256"] or digest(data) != stage19["source_sha256"]:
        raise ValueError("Source differs from recorded model and data audit")
    clean, _ = read_data(data)
    calibrate = pd.date_range("2021-01-08", "2021-06-30", freq="D")
    calibration = pd.DataFrame([origin_signals(clean, anchor) for anchor in calibrate], index=calibrate)
    cutoff = float(calibration.past_profile_disagreement.quantile(.75))
    predictions = pd.read_csv(root / "stage17/predictions.csv", parse_dates=["anchor_time"])
    scenarios = pd.read_csv(root / "stage18/daily_scenarios.csv", parse_dates=["date"])
    scenarios = scenarios.loc[np.isclose(scenarios.fraction, .10)].copy().set_index("date")
    daily_error = predictions.assign(error=lambda x: (x.prediction - x.actual).abs()).groupby("anchor_time").error.mean()
    if not scenarios.index.equals(daily_error.index):
        raise ValueError("Daily scenario and forecast dates do not match")
    signals = pd.DataFrame([decision(origin_signals(clean, anchor), cutoff) for anchor in scenarios.index], index=scenarios.index)
    days = scenarios.join(daily_error.rename("daily_mae")).join(signals)
    days.index.name = "date"
    held = days.loc[days.review_required]
    standard = days.loc[~days.review_required]
    # A review gate is assessed against hypothetical arithmetic outcomes, not real interventions.
    summary = {"input_sha256": digest(data), "calibration": {"first_origin": str(calibrate[0].date()),
                   "last_origin": str(calibrate[-1].date()), "n_days": len(calibration),
                   "disagreement_p75": cutoff,
                   "rule": "review if yesterday-vs-last-week past profile MAE > p75, or previous 24h production was zero"},
               "evaluation": {"all": summarize_group(days), "review_required": summarize_group(held),
                              "standard_review": summarize_group(standard),
                              "held_fraction": float(days.review_required.mean()),
                              "high_disagreement_days": int(days.above_calibration_disagreement.sum()),
                              "previous_day_no_production_days": int(days.previous_24h_no_production.sum())},
               "note": "Standard review still requires a human decision. No real schedule was changed. Retrospective August-September outcomes were already inspected; do not claim independent validation or savings."}
    next_forecast = pd.read_csv(root / "stage17/next_day_prediction.csv", parse_dates=["timestamp"])
    if len(next_forecast) != 24 or next_forecast.timestamp.dt.normalize().nunique() != 1:
        raise ValueError("Expected one complete next-day forecast")
    next_anchor = next_forecast.timestamp.iloc[0].normalize()
    if next_anchor <= clean.index.max():
        raise ValueError("Next-day card must follow all observations")
    next_signal = decision(origin_signals(clean, next_anchor), cutoff)
    next_day = {"date": str(next_anchor.date()), **next_signal,
                "forecast_mean_daily_max": float(next_forecast.predicted_mean.max()),
                "forecast_quarter_max": float(next_forecast.predicted_quarter_max.max()),
                "forecast_peak_alert_hours": int(next_forecast.peak_alert.astype(bool).sum()),
                "status": "hold_for_human_review" if next_signal["review_required"] else "candidate_for_human_review",
                "can_execute_automatically": False,
                "missing_operational_inputs": ["equipment-level flexible load", "fixed jobs and due times", "contract peak interval and units", "operator approval"]}
    output = root / "stage21"
    output.mkdir(parents=True, exist_ok=True)
    days.to_csv(output / "decision_cards.csv")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "next_day_decision.json").write_text(json.dumps(next_day, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluation": summary["evaluation"], "next_day": next_day}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.data, args.output_root)
