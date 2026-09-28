"""Stage 7: compare loss functions and idle overrides on chronological folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecast import idle_mask, make_features, read_data


FOLDS = [("2021-04-01", "2021-05-01"), ("2021-05-01", "2021-06-01"),
         ("2021-06-01", "2021-07-01"), ("2021-07-01", "2021-08-01")]
BASE_MODELS = ("squared", "absolute", "residual_absolute")


def fit_pair(train: pd.DataFrame, features: list[str], kind: str) -> list:
    if kind not in BASE_MODELS:
        raise ValueError(f"Unknown model: {kind}")
    target = train.target.to_numpy()
    if kind == "residual_absolute":
        target = target - train.power_lag_1.to_numpy()
    loss = "squared_error" if kind == "squared" else "absolute_error"
    return [HistGradientBoostingRegressor(
        loss=loss, max_iter=150, max_leaf_nodes=15, min_samples_leaf=leaf,
        learning_rate=0.05, l2_regularization=10, random_state=42,
        early_stopping=False).fit(train[features], target) for leaf in (20, 50)]


def predict_pair(models: list, frame: pd.DataFrame, features: list[str], kind: str) -> np.ndarray:
    estimates = np.array([model.predict(frame[features]) for model in models])
    if kind == "residual_absolute":
        estimates += frame.power_lag_1.to_numpy()
    return np.maximum(0, estimates).mean(axis=0)


def parse_candidate(candidate: str) -> tuple[str, bool]:
    gated = candidate.endswith("_idle")
    kind = candidate[:-5] if gated else candidate
    if kind not in BASE_MODELS:
        raise ValueError(f"Unknown candidate: {candidate}")
    return kind, gated


def predict_candidate(models: list, clean: pd.DataFrame, frame: pd.DataFrame,
                       features: list[str], candidate: str) -> np.ndarray:
    kind, gated = parse_candidate(candidate)
    estimate = predict_pair(models, frame, features, kind)
    return np.where(idle_mask(clean, frame, 6, 25), frame.power_lag_1, estimate) if gated else estimate


def candidate_scores(predictions: pd.DataFrame, actual: pd.Series) -> dict:
    errors = predictions.sub(actual, axis=0).abs()
    folds = errors.groupby(errors.index.strftime("%Y-%m")).mean()
    return {name: {"mean_month_mae": float(folds[name].mean()),
                   "fold_mae": {str(k): float(v) for k, v in folds[name].items()}}
            for name in predictions.columns}


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.asarray(prediction) - np.asarray(actual)
    return {"n": len(error), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "wape": float(np.abs(error).sum() / np.abs(actual).sum())}


def run(data: Path, output: Path) -> None:
    clean, _ = read_data(data)
    frame, features = make_features(clean)
    fold_results = []
    for start, end in FOLDS:
        train = frame.loc[frame.index < start]
        valid = frame.loc[(frame.index >= start) & (frame.index < end)]
        result = pd.DataFrame({"actual": valid.target}, index=valid.index)
        for kind in BASE_MODELS:
            estimate = predict_pair(fit_pair(train, features, kind), valid, features, kind)
            result[kind] = estimate
            result[f"{kind}_idle"] = np.where(idle_mask(clean, valid, 6, 25), valid.power_lag_1, estimate)
        fold_results.append(result)
        print(f"Validated {start[:7]}", flush=True)
    oof = pd.concat(fold_results)
    candidates = [c for c in oof if c != "actual"]
    comparison = candidate_scores(oof[candidates], oof.actual)
    chosen = min(comparison, key=lambda name: comparison[name]["mean_month_mae"])
    forward = []
    for start, end in FOLDS[1:]:
        past = oof.loc[oof.index < start]
        current = oof.loc[(oof.index >= start) & (oof.index < end)]
        past_scores = candidate_scores(past[candidates], past.actual)
        name = min(past_scores, key=lambda c: past_scores[c]["mean_month_mae"])
        forward.append({"month": start[:7], "selected_candidate": name,
                        "selection_end": str(past.index.max()),
                        "mae": float((current[name] - current.actual).abs().mean()),
                        "stage5_style_mae": float((current.squared_idle - current.actual).abs().mean())})
    print(f"Selected from validation: {chosen}", flush=True)
    kind, _ = parse_candidate(chosen)
    monthly = []
    for start, end in [("2021-08-01", "2021-09-01"), ("2021-09-01", "2021-09-15")]:
        train = frame.loc[frame.index < start]
        test = frame.loc[(frame.index >= start) & (frame.index < end)]
        model = fit_pair(train, features, kind)
        prediction = predict_candidate(model, clean, test, features, chosen)
        monthly.append(pd.DataFrame({"actual": test.target, "prediction": prediction,
                                     "model_fit_before": start}, index=test.index))
    evaluation = pd.concat(monthly)
    report = {"input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
              "features": features, "selection_end": "2021-07-31 23:00:00",
              "candidate_validation": comparison, "chosen_candidate": chosen,
              "forward_month_selection": forward,
              "evaluation": regression_metrics(evaluation.actual.to_numpy(), evaluation.prediction.to_numpy()),
              "evaluation_by_month": {str(m): regression_metrics(g.actual.to_numpy(), g.prediction.to_numpy())
                                      for m, g in evaluation.groupby(evaluation.index.strftime("%Y-%m"))},
              "after_first_august_week": regression_metrics(
                  evaluation.loc["2021-08-08":, "actual"].to_numpy(),
                  evaluation.loc["2021-08-08":, "prediction"].to_numpy()),
              "limitation": "Repeated retrospective evaluation; candidate choice uses only April-July."}
    output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output / "validation_predictions.csv", index_label="timestamp")
    evaluation.to_csv(output / "predictions.csv", index_label="timestamp")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage7"))
    args = parser.parse_args()
    run(args.data, args.output)
