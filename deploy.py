"""Stage 8: save a selected model and forecast one hour from observed history."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from advanced_forecast import fit_pair, parse_candidate, predict_candidate
from forecast import (POWER_COLUMNS, conformal_radius, make_features,
                      make_peak_classifier, read_data)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_frame(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Append an unknown target hour; every observed feature is shifted into it."""
    if history.empty:
        raise ValueError("Observed history is empty")
    target_time = history.index.max() + pd.Timedelta(hours=1)
    extended = history.reindex(history.index.union(pd.DatetimeIndex([target_time])))
    features, names = make_features(extended, require_target=False)
    if target_time not in features.index:
        raise ValueError("Insufficient exact hourly history: required lags up to 169 hours are missing")
    row = features.loc[[target_time]]
    if row.target.notna().any():
        raise AssertionError("The forecast hour must have an unknown outcome")
    return extended, row, names


def calibration_errors(output_root: Path, candidate: str, through: pd.Timestamp) -> pd.Series:
    validation = pd.read_csv(output_root / "validation_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
    evaluation = pd.read_csv(output_root / "predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
    error = pd.concat([(validation[candidate] - validation.actual).abs(),
                       (evaluation.prediction - evaluation.actual).abs()])
    error = error.loc[error.index <= through]
    # Use the latest observed calendar month with at least two days of errors.
    for month in sorted(error.index.to_period("M").unique(), reverse=True):
        recent = error.loc[error.index.to_period("M") == month]
        if len(recent) >= 48:
            return recent
    raise ValueError("No sufficiently large past out-of-time calibration period")


def train_bundle(data: Path, selection_path: Path, alert_path: Path,
                 artifact: Path, through: pd.Timestamp | None = None) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    alert = json.loads(alert_path.read_text(encoding="utf-8"))
    input_hash = digest(data)
    if selection["input_sha256"] != input_hash or alert["reproducibility"]["input_sha256"] != input_hash:
        raise ValueError("Input differs from the recorded selection data; rerun evaluation before training")
    clean, _ = read_data(data, allow_partial_last_day=True, observed_through=through)
    through = clean.index.max() if through is None else pd.Timestamp(through)
    if clean.index.max() != through:
        raise ValueError("The last observed hour is absent or was removed as invalid")
    if through < pd.Timestamp(selection["selection_end"]):
        raise ValueError("Cannot train a release for a date before candidate selection completed")
    frame, names = make_features(clean)
    if names != selection["features"] or names != alert["reproducibility"]["feature_names"]:
        raise ValueError("Feature definitions differ from recorded evaluation")
    candidate = selection["chosen_candidate"]
    kind, _ = parse_candidate(candidate)
    models = fit_pair(frame, names, kind)
    # Retain the validated classifier's training date and threshold. Updating
    # the mean forecast must not silently recalibrate its separate peak policy.
    risk_train = frame.loc[frame.index < "2021-08-01"]
    peak_cutoff = alert["observed_test_conditions"]["quarter_peak_threshold"]
    peak_labels = clean.loc[risk_train.index, POWER_COLUMNS].max(axis=1).ge(peak_cutoff)
    peak_models = [make_peak_classifier(leaf).fit(risk_train[names], peak_labels) for leaf in (20, 50)]
    residuals = calibration_errors(selection_path.parent, candidate, through)
    radius = conformal_radius(residuals.to_numpy())
    metadata = {
        "format_version": 1, "input_sha256": input_hash,
        "selection_sha256": digest(selection_path), "alert_report_sha256": digest(alert_path),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
        "candidate": candidate, "feature_names": names, "train_rows": len(frame),
        "mean_model_observed_through": str(through),
        "peak_model_observed_through": str(risk_train.index.max()),
        "peak_event_cutoff": peak_cutoff,
        "peak_score_cutoff": alert["stage_four_validation"]["alert_threshold"],
        "interval_radius": radius, "calibration_n": len(residuals),
        "calibration_start": str(residuals.index.min()), "calibration_end": str(residuals.index.max()),
        "interval_note": "Empirical residual range; time-series coverage is not guaranteed after refitting.",
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"metadata": metadata, "models": models, "peak_models": peak_models}, artifact)
    manifest = {**metadata, "artifact_sha256": digest(artifact)}
    artifact.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def forecast_next(data: Path, artifact: Path, last_observed_hour: pd.Timestamp | None = None) -> dict:
    bundle = joblib.load(artifact)
    metadata = bundle["metadata"]
    if metadata["format_version"] != 1 or metadata["scikit_learn"] != sklearn.__version__:
        raise ValueError("Model format or scikit-learn version differs; use the pinned environment")
    clean, _ = read_data(data, allow_partial_last_day=True, observed_through=last_observed_hour)
    observed = clean.index.max()
    if last_observed_hour is not None and observed != pd.Timestamp(last_observed_hour):
        raise ValueError("Requested last observed hour is missing or invalid")
    if observed < pd.Timestamp(metadata["mean_model_observed_through"]):
        raise ValueError("The model was trained on observations later than this forecast origin")
    extended, row, names = prediction_frame(clean)
    if names != metadata["feature_names"]:
        raise ValueError("Feature definition mismatch")
    prediction = float(predict_candidate(bundle["models"], extended, row, names, metadata["candidate"])[0])
    risk = float(np.mean([model.predict_proba(row[names])[0, 1] for model in bundle["peak_models"]]))
    radius = metadata["interval_radius"]
    return {"last_observed_hour": str(observed), "forecast_hour": str(row.index[0]),
            "predicted_mean": prediction,
            "empirical_interval_90": [max(0., prediction - radius), prediction + radius],
            "peak_risk_score": risk, "peak_alert": bool(risk >= metadata["peak_score_cutoff"]),
            "peak_score_cutoff": metadata["peak_score_cutoff"],
            "peak_event_cutoff": metadata["peak_event_cutoff"],
            "mean_model_observed_through": metadata["mean_model_observed_through"],
            "peak_model_observed_through": metadata["peak_model_observed_through"],
            "history_file_sha256": digest(data), "model_sha256": digest(artifact),
            "notes": ["The forecast hour's observations are not used.",
                      "Power units remain unconfirmed; risk score is not a calibrated probability.",
                      metadata["interval_note"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    train.add_argument("--selection", type=Path, default=Path("outputs/stage7/summary.json"))
    train.add_argument("--alerts", type=Path, default=Path("outputs/summary.json"))
    train.add_argument("--artifact", type=Path, default=Path("artifacts/latest.joblib"))
    train.add_argument("--through", type=pd.Timestamp)
    predict = commands.add_parser("predict")
    predict.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    predict.add_argument("--artifact", type=Path, default=Path("artifacts/latest.joblib"))
    predict.add_argument("--last-observed-hour", type=pd.Timestamp)
    predict.add_argument("--output", type=Path, default=Path("outputs/stage8/next_prediction.json"))
    args = parser.parse_args()
    if args.command == "train":
        result = train_bundle(args.data, args.selection, args.alerts, args.artifact, args.through)
    else:
        result = forecast_next(args.data, args.artifact, args.last_observed_hour)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
