"""Stage 15: selected demand model, state intervals, and retained peak policy."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from deploy import calibration_errors, digest
from forecast import POWER_COLUMNS, idle_mask, make_features, make_peak_classifier, read_data
from reliability import interval_calibration
from research import DemandModel, apply_gate, richer_features, training_window


def train_release(data: Path, root: Path, artifact: Path,
                    through: pd.Timestamp | None = None) -> dict:
    mean_report = json.loads((root / "stage13/summary.json").read_text())
    original = json.loads((root / "summary.json").read_text())
    source_hash = digest(data)
    if mean_report["input_sha256"] != source_hash or original["reproducibility"]["input_sha256"] != source_hash:
        raise ValueError("Data changed since model selection; regenerate evaluation first")
    clean, _ = read_data(data, allow_partial_last_day=True, observed_through=through)
    through = clean.index.max() if through is None else pd.Timestamp(through)
    if through != clean.index.max() or through < pd.Timestamp("2021-07-31 23:00:00"):
        raise ValueError("Training must end at an observed hour after peak-policy selection completed")
    config = mean_report["config"]
    full_frame, features = richer_features(clean, config["features"])
    if features != mean_report["features"]:
        raise ValueError("Mean feature contract changed")
    train = training_window(full_frame, through + pd.Timedelta(hours=1), config.get("window_days"))
    mean_model = DemandModel(config["model"], features).fit(train)
    base, peak_features = make_features(clean)
    peak_train = base.loc[base.index < "2021-08-01"]
    cutoff = original["observed_test_conditions"]["quarter_peak_threshold"]
    labels = clean.loc[peak_train.index, POWER_COLUMNS].max(axis=1).ge(cutoff)
    peak_models = [make_peak_classifier(leaf).fit(peak_train[peak_features], labels) for leaf in (20, 50)]
    errors = calibration_errors(root / "stage13", mean_report["chosen"], through)
    prior_idle = idle_mask(clean, full_frame.loc[errors.index], 6, 25)
    calibration = interval_calibration(errors.to_numpy(), np.zeros(len(errors)), prior_idle)
    metadata = {"format_version": 2, "stage": 15, "python": platform.python_version(),
                "scikit_learn": sklearn.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "joblib": joblib.__version__,
                "input_sha256": source_hash,
                "selection_report_sha256": digest(root / "stage13/summary.json"),
                "peak_report_sha256": digest(root / "summary.json"),
                "calibration_source_sha256": {name: digest(root / "stage13" / name)
                    for name in ("validation_predictions.csv", "predictions.csv")},
                "mean_config": config, "mean_features": features, "peak_features": peak_features,
                "mean_train_start": str(train.index.min()), "mean_train_end": str(train.index.max()),
                "mean_train_rows": len(train), "peak_train_end": str(peak_train.index.max()),
                "peak_policy": "retained_stage4_recall_policy",
                "peak_event_cutoff": cutoff, "peak_score_cutoff": original["stage_four_validation"]["alert_threshold"],
                "interval_calibration": {**calibration, "start": str(errors.index.min()), "end": str(errors.index.max())}}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"metadata": metadata, "mean_model": mean_model, "peak_models": peak_models}, artifact)
    manifest = {**metadata, "artifact_sha256": digest(artifact)}
    artifact.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def forecast_release(data: Path, artifact: Path, last_observed_hour: pd.Timestamp | None = None) -> dict:
    bundle = joblib.load(artifact)
    metadata = bundle["metadata"]
    if metadata["format_version"] != 2 or metadata["scikit_learn"] != sklearn.__version__:
        raise ValueError("Model format/environment mismatch")
    clean, _ = read_data(data, allow_partial_last_day=True, observed_through=last_observed_hour)
    observed = clean.index.max()
    if last_observed_hour is not None and observed != pd.Timestamp(last_observed_hour):
        raise ValueError("Last observed hour is absent or invalid")
    if any(observed < pd.Timestamp(end) for end in
           (metadata["mean_train_end"], metadata["peak_train_end"], metadata["interval_calibration"]["end"])):
        raise ValueError("Model contains observations later than the requested forecast origin")
    target = observed + pd.Timedelta(hours=1)
    extended = clean.reindex(clean.index.union(pd.DatetimeIndex([target])))
    config = metadata["mean_config"]
    mean_frame, mean_names = richer_features(extended, config["features"], require_target=False)
    base, peak_names = make_features(extended, require_target=False)
    if target not in mean_frame.index or target not in base.index:
        raise ValueError("Required exact historical lags are missing")
    if mean_names != metadata["mean_features"] or peak_names != metadata["peak_features"]:
        raise ValueError("Feature contract changed")
    row = mean_frame.loc[[target]]
    if row.target.notna().any():
        raise AssertionError("Future target must be unknown")
    prediction = apply_gate(extended, row, bundle["mean_model"].predict(row), config["gate"])[0]
    known_idle = bool(idle_mask(extended, row, 6, 25)[0])
    state = "idle" if known_idle else "other"
    calibration = metadata["interval_calibration"]
    radius = calibration["regimes"][state]["radius"]
    global_radius = calibration["global_radius"]
    risk = float(np.mean([model.predict_proba(base.loc[[target], peak_names])[0, 1] for model in bundle["peak_models"]]))
    return {"last_observed_hour": str(observed), "forecast_hour": str(target),
            "predicted_mean": float(prediction), "prior_idle_state": known_idle,
            "conditional_interval_90": [float(max(0, prediction - radius)), float(prediction + radius)],
            "global_interval_reference_90": [float(max(0, prediction - global_radius)), float(prediction + global_radius)],
            "peak_risk_score": risk, "peak_alert": bool(risk >= metadata["peak_score_cutoff"]),
            "peak_policy": metadata["peak_policy"], "peak_score_cutoff": metadata["peak_score_cutoff"],
            "mean_train_end": metadata["mean_train_end"], "peak_train_end": metadata["peak_train_end"],
            "calibration_end": calibration["end"], "artifact_sha256": digest(artifact),
            "history_sha256": digest(data),
            "notes": ["Source power units unconfirmed; score is not a calibrated probability.",
                      "Intervals are empirical and may fail during abrupt startup or regime changes."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/final.joblib"))
    parser.add_argument("--through", type=pd.Timestamp)
    parser.add_argument("--last-observed-hour", type=pd.Timestamp)
    args = parser.parse_args()
    if args.command == "train":
        result = train_release(args.data, args.output_root, args.artifact, args.through)
    else:
        result = forecast_release(args.data, args.artifact, args.last_observed_hour)
        destination = args.output_root / "stage15/next_prediction.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
