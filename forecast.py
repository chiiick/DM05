"""One-hour-ahead electricity forecast with strictly historical inputs.

Run from this directory:
    python3 forecast.py --data /path/to/okm_augumented_2021.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POWER_COLUMNS = ["15분", "30분", "45분", "60분"]
REQUIRED = ["날짜", "시간", *POWER_COLUMNS, "평균", "생산량", "기온", "풍속", "습도", "강수량"]


def read_data(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    raw["date"] = pd.to_datetime(raw["날짜"].astype(str), format="%Y%m%d", errors="raise")
    bad_hour = ~raw["시간"].between(0, 23)
    bad_dates = sorted(raw.loc[bad_hour, "date"].dt.strftime("%Y-%m-%d").unique().tolist())
    # On these days the hour column has been overwritten. Row order is not
    # sufficient evidence for reconstructing true hours, so drop whole days.
    clean = raw.loc[~raw["date"].dt.strftime("%Y-%m-%d").isin(bad_dates)].copy()
    clean["timestamp"] = clean["date"] + pd.to_timedelta(clean["시간"], unit="h")
    if clean["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps remain after removing invalid days")
    counts = clean.groupby("date")["시간"].nunique()
    if not counts.eq(24).all():
        raise ValueError("A retained date does not contain exactly 24 distinct hours")
    clean = clean.sort_values("timestamp").set_index("timestamp")
    for name in ["평균", "생산량", "기온", "풍속", "습도", "강수량"]:
        clean[name] = pd.to_numeric(clean[name], errors="raise")
    if clean[["평균", "생산량"]].isna().any().any():
        raise ValueError("Missing values in the target or historical production input")
    same_hour_mean = clean[POWER_COLUMNS].mean(axis=1)
    diagnostics = {
        "raw_rows": int(len(raw)),
        "bad_hour_rows": int(bad_hour.sum()),
        "removed_dates": bad_dates,
        "retained_rows": int(len(clean)),
        "missing_values_after_day_removal": {name: int(value) for name, value in
                                            clean[REQUIRED].isna().sum().items() if value},
        "start": str(clean.index.min()),
        "end": str(clean.index.max()),
        "max_difference_between_average_and_quarter_mean": float((clean["평균"] - same_hour_mean).abs().max()),
    }
    return clean, diagnostics


def make_features(clean: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # asfreq inserts missing hours where the invalid days were removed. Exact
    # timestamp shifts therefore cannot accidentally cross those gaps.
    hourly = clean.asfreq("h")
    frame = pd.DataFrame(index=hourly.index)
    frame["target"] = hourly["평균"]
    for lag in (1, 2, 24, 168):
        frame[f"power_lag_{lag}"] = hourly["평균"].shift(lag)
    for lag in (1, 24):
        frame[f"production_lag_{lag}"] = hourly["생산량"].shift(lag)
    frame["power_past_6h_mean"] = hourly["평균"].shift(1).rolling(6, min_periods=6).mean()
    frame["power_past_24h_mean"] = hourly["평균"].shift(1).rolling(24, min_periods=24).mean()
    hour = frame.index.hour
    day = frame.index.dayofweek
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * day / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * day / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame.index.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame.index.month / 12)
    frame["weekend"] = (day >= 5).astype(int)
    frame = frame.dropna().copy()
    return frame, [name for name in frame if name != "target"]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - center) / scale
    target_center = y.mean()
    coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - target_center))
    return {"center": center, "scale": scale, "target_center": target_center,
            "coefficients": coefficients, "alpha": alpha}


def ridge_predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - model["center"]) / model["scale"]
    return np.maximum(0, model["target_center"] + z @ model["coefficients"])


def scores(y: np.ndarray, pred: np.ndarray, peak_cutoff: float) -> dict:
    error = pred - y
    observed_peak = y >= peak_cutoff
    predicted_peak = pred >= peak_cutoff
    tp = int(np.sum(observed_peak & predicted_peak))
    fp = int(np.sum(~observed_peak & predicted_peak))
    fn = int(np.sum(observed_peak & ~predicted_peak))
    return {
        "n": len(y),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.sum(np.abs(error)) / np.sum(np.abs(y))) if np.sum(np.abs(y)) else None,
        "peak_tp": tp, "peak_fp": fp, "peak_fn": fn,
        "peak_precision": tp / (tp + fp) if tp + fp else 0.0,
        "peak_recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def error_conditions(frame: pd.DataFrame, prediction: np.ndarray, peak_cutoff: float) -> dict:
    detail = frame[["target", "production_lag_1"]].copy()
    detail["absolute_error"] = np.abs(prediction - detail["target"].to_numpy())
    detail["hour"] = detail.index.hour
    detail["is_weekend"] = detail.index.dayofweek >= 5
    detail["is_peak"] = detail["target"] >= peak_cutoff
    return {
        "mae_by_hour": {str(k): float(v) for k, v in detail.groupby("hour")["absolute_error"].mean().items()},
        "mae_by_weekend": {str(k): float(v) for k, v in detail.groupby("is_weekend")["absolute_error"].mean().items()},
        "mae_by_peak": {str(k): float(v) for k, v in detail.groupby("is_peak")["absolute_error"].mean().items()},
        "mae_by_previous_production": {str(k): float(v) for k, v in
             detail.groupby(detail["production_lag_1"].gt(0))["absolute_error"].mean().items()},
    }


def save_plot(predictions: pd.DataFrame, output: Path) -> None:
    first_week = predictions.iloc[:168]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(first_week.index, first_week["actual"], label="Actual", linewidth=1)
    ax.plot(first_week.index, first_week["ridge"], label="Ridge 1h ahead", linewidth=1)
    ax.set(title="First test week: one-hour-ahead electricity demand", ylabel="Hourly mean demand")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "test_first_week.png", dpi=160)
    plt.close(fig)


def run(data_path: Path, output: Path) -> None:
    clean, diagnostics = read_data(data_path)
    frame, feature_names = make_features(clean)
    train = frame.loc[frame.index < "2021-07-01"]
    validation = frame.loc[(frame.index >= "2021-07-01") & (frame.index < "2021-08-01")]
    test = frame.loc[frame.index >= "2021-08-01"]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One of the chronological splits is empty")
    x_train, y_train = train[feature_names].to_numpy(), train.target.to_numpy()
    x_val, y_val = validation[feature_names].to_numpy(), validation.target.to_numpy()
    x_test, y_test = test[feature_names].to_numpy(), test.target.to_numpy()
    peak_cutoff = float(np.quantile(y_train, 0.95))

    trials = []
    for alpha in (0.1, 10.0, 100.0, 1000.0):
        candidate = fit_ridge(x_train, y_train, alpha)
        mae = float(np.mean(np.abs(ridge_predict(candidate, x_val) - y_val)))
        trials.append((mae, alpha))
    _, chosen_alpha = min(trials)
    model = fit_ridge(x_train, y_train, chosen_alpha)
    prediction = ridge_predict(model, x_test)
    validation_prediction = ridge_predict(model, x_val)

    predictions = pd.DataFrame({"actual": y_test, "persistence": test["power_lag_1"].to_numpy(),
                                "weekly": test["power_lag_168"].to_numpy(), "ridge": prediction},
                               index=test.index)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "test_predictions.csv", index_label="timestamp")
    save_plot(predictions, output)
    report = {
        "data": diagnostics,
        "task": "At the end of hour t-1, predict the mean electricity demand during hour t.",
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "train_peak_95th_percentile": peak_cutoff,
        "validation_ridge_mae_by_alpha": {str(a): m for m, a in trials},
        "chosen_ridge_alpha": chosen_alpha,
        "validation_metrics": {
            "persistence": scores(y_val, validation["power_lag_1"].to_numpy(), peak_cutoff),
            "weekly": scores(y_val, validation["power_lag_168"].to_numpy(), peak_cutoff),
            "ridge": scores(y_val, validation_prediction, peak_cutoff),
        },
        "test_metrics": {name: scores(y_test, predictions[name].to_numpy(), peak_cutoff)
                         for name in ("persistence", "weekly", "ridge")},
        "ridge_error_conditions": error_conditions(test, prediction, peak_cutoff),
        "ridge_coefficients_standardized": {name: float(value) for name, value in
                                             zip(feature_names, model["coefficients"])},
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"data": diagnostics, "split_rows": report["split_rows"],
                      "chosen_ridge_alpha": chosen_alpha,
                      "validation_metrics": report["validation_metrics"],
                      "test_metrics": report["test_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    arguments = parser.parse_args()
    run(arguments.data, arguments.output)
