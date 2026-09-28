"""Stage 14: chronological interval calibration and peak-policy cost diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import POWER_COLUMNS, alert_scores, conformal_radius, idle_mask, read_data, threshold_for_recall
from research import richer_features, training_window
from risk import PeakModel


def interval_calibration(actual: np.ndarray, prediction: np.ndarray, idle: np.ndarray) -> dict:
    errors = np.abs(np.asarray(actual) - np.asarray(prediction))
    global_radius = conformal_radius(errors)
    regimes = {}
    for name, mask in (("idle", idle), ("other", ~idle)):
        regimes[name] = {"n": int(mask.sum()),
                         "radius": conformal_radius(errors[mask]) if mask.sum() >= 30 else global_radius,
                         "fallback_to_global": bool(mask.sum() < 30)}
    return {"n": len(errors), "global_radius": global_radius, "regimes": regimes}


def interval_metrics(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    actual, lower, upper = np.asarray(actual), np.asarray(lower), np.asarray(upper)
    width = upper - lower
    score = width + 20 * np.maximum(lower - actual, 0) + 20 * np.maximum(actual - upper, 0)
    return {"n": len(actual), "coverage": float(np.mean((lower <= actual) & (actual <= upper))),
            "mean_width": float(width.mean()), "mean_interval_score_90": float(score.mean())}


def attach_intervals(clean: pd.DataFrame, frame: pd.DataFrame, evaluation: pd.DataFrame,
                     july_predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    output = evaluation.copy()
    output["known_idle"] = idle_mask(clean, frame.loc[output.index], 6, 25)
    records = {}
    for month, start, end in (("2021-08", "2021-08-01", "2021-09-01"),
                              ("2021-09", "2021-09-01", "2021-09-15")):
        calibration = july_predictions if month == "2021-08" else output.loc[output.index < start]
        cal_idle = idle_mask(clean, frame.loc[calibration.index], 6, 25)
        fitted = interval_calibration(calibration.actual.to_numpy(), calibration.prediction.to_numpy(), cal_idle)
        mask = (output.index >= start) & (output.index < end)
        target = output.loc[mask]
        if calibration.index.max() >= target.index.min():
            raise ValueError("Calibration observations must precede every target hour")
        radii = np.where(target.known_idle, fitted["regimes"]["idle"]["radius"], fitted["regimes"]["other"]["radius"])
        for label, radius in (("global", fitted["global_radius"]), ("conditional", radii)):
            output.loc[mask, f"{label}_lower"] = np.maximum(0, target.prediction.to_numpy() - radius)
            output.loc[mask, f"{label}_upper"] = target.prediction.to_numpy() + radius
        records[month] = {**fitted, "calibration_start": str(calibration.index.min()),
                          "calibration_end": str(calibration.index.max()),
                          "evaluation_start": str(target.index.min())}
    return output, records


def chronological_peak_policy(clean: pd.DataFrame, frame: pd.DataFrame, features: list[str],
                               algorithm: str, cutoff: float) -> tuple[pd.DataFrame, dict]:
    parts, records = [], {}
    for month, train_before, cal_end, eval_end in [
        ("2021-08", "2021-07-01", "2021-08-01", "2021-09-01"),
        ("2021-09", "2021-08-01", "2021-09-01", "2021-09-15")]:
        train = training_window(frame, train_before)
        calibration = frame.loc[(frame.index >= train_before) & (frame.index < cal_end)]
        test = frame.loc[(frame.index >= cal_end) & (frame.index < eval_end)]
        y_train = clean.loc[train.index, POWER_COLUMNS].max(axis=1).ge(cutoff).to_numpy()
        model = PeakModel(algorithm, features).fit(train, y_train)
        cal_score = model.predict(calibration)
        observed = clean.loc[calibration.index, POWER_COLUMNS].max(axis=1).ge(cutoff).to_numpy()
        threshold = threshold_for_recall(cal_score, observed, .85)
        score = model.predict(test)
        parts.append(pd.DataFrame({"recent_score": score, "recent_threshold": threshold,
                                   "recent_alert": score >= threshold}, index=test.index))
        records[month] = {"train_end": str(train.index.max()),
                          "calibration_start": str(calibration.index.min()),
                          "calibration_end": str(calibration.index.max()), "score_threshold": threshold}
    return pd.concat(parts), records


def run(data: Path, root: Path) -> dict:
    clean, _ = read_data(data)
    mean_report = json.loads((root / "stage13/summary.json").read_text())
    peak_report = json.loads((root / "stage12/summary.json").read_text())
    frame, _ = richer_features(clean, mean_report["config"]["features"])
    evaluation = pd.read_csv(root / "stage13/predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
    oof = pd.read_csv(root / "stage13/validation_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
    july = oof.loc[oof.index >= "2021-07-01", ["actual", mean_report["chosen"]]].rename(columns={mean_report["chosen"]: "prediction"})
    intervals, calibration = attach_intervals(clean, frame, evaluation, july)
    interval_results = {name: interval_metrics(intervals.actual, intervals[f"{name}_lower"], intervals[f"{name}_upper"])
                        for name in ("global", "conditional")}
    by_regime = {str(regime): {name: interval_metrics(group.actual, group[f"{name}_lower"], group[f"{name}_upper"])
                              for name in ("global", "conditional")}
                 for regime, group in intervals.groupby("known_idle")}

    risk_frame, risk_features = richer_features(clean, peak_report["config"]["features"])
    cutoff = peak_report["event_cutoff"]
    risk, risk_calibration = chronological_peak_policy(clean, risk_frame, risk_features,
                                                       peak_report["config"]["model"], cutoff)
    legacy = pd.read_csv(root / "test_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
    frozen = pd.read_csv(root / "stage12/predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
    risk["actual"] = frozen.actual
    risk["legacy_alert"] = legacy.peak_classifier_alert
    risk["stage12_alert"] = frozen.alert
    policy_results = {name: alert_scores(risk.actual.to_numpy(), risk[name].astype(float).to_numpy(), cutoff, .5)
                      for name in ("legacy_alert", "stage12_alert", "recent_alert")}
    cost_table = {str(ratio): {name: metric["fp"] + ratio * metric["fn"] for name, metric in policy_results.items()}
                  for ratio in (1, 2, 5, 10)}
    report = {"interval_calibration": calibration, "intervals": interval_results,
              "intervals_by_prior_state": by_regime, "risk_calibration": risk_calibration,
              "risk_policies": policy_results, "miss_cost_relative_to_false_alarm": cost_table,
              "note": "Cost units are hypothetical; report tradeoffs, do not choose a policy from these test costs."}
    output = root / "stage14"
    output.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output / "interval_predictions.csv", index_label="timestamp")
    risk.to_csv(output / "risk_policies.csv", index_label="timestamp")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.data, args.output_root)
