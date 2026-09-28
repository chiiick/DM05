"""One-hour-ahead electricity forecast with strictly historical inputs.

Select a ridge and histogram gradient boosting model with expanding-window
validation, then fit through July and evaluate August-September once.

Run from this directory:
    python3 forecast.py --data okm_augumented_2021.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor


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
    for name in [*POWER_COLUMNS, "평균", "생산량", "기온", "풍속", "습도", "강수량"]:
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
    for lag in (1, 2, 3, 23, 24, 25, 48, 167, 168, 169):
        frame[f"power_lag_{lag}"] = hourly["평균"].shift(lag)
    for lag in (1, 2, 3, 24, 25):
        frame[f"production_lag_{lag}"] = hourly["생산량"].shift(lag)
    # The preceding hour's final quarter is available at the forecast origin.
    frame["previous_hour_final_quarter"] = hourly["60분"].shift(1)
    frame["power_past_6h_mean"] = hourly["평균"].shift(1).rolling(6, min_periods=6).mean()
    frame["power_past_24h_mean"] = hourly["평균"].shift(1).rolling(24, min_periods=24).mean()
    frame["power_past_6h_std"] = hourly["평균"].shift(1).rolling(6, min_periods=6).std()
    frame["power_recent_change"] = hourly["평균"].shift(1) - hourly["평균"].shift(2)
    frame["production_recent_change"] = hourly["생산량"].shift(1) - hourly["생산량"].shift(2)
    frame["previous_hour_active"] = hourly["생산량"].shift(1).gt(0).astype(int)
    frame["past_6h_active_share"] = hourly["생산량"].shift(1).gt(0).rolling(6, min_periods=6).mean()
    hour = frame.index.hour
    day = frame.index.dayofweek
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * day / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * day / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame.index.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame.index.month / 12)
    frame["weekend"] = (day >= 5).astype(int)
    frame["hour"] = hour
    frame["weekday"] = day
    frame["day_of_year"] = frame.index.dayofyear
    frame["weekday_work_start"] = ((day < 5) & np.isin(hour, [7, 8, 9])).astype(int)
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


def rolling_validation(frame: pd.DataFrame, feature_names: list[str]) -> tuple[dict, float, int]:
    """Select hyperparameters using expanding-window, future-month folds."""
    folds = [("2021-04-01", "2021-05-01"), ("2021-05-01", "2021-06-01"),
             ("2021-06-01", "2021-07-01"), ("2021-07-01", "2021-08-01")]
    ridge_alphas = (0.1, 10.0, 100.0, 1000.0)
    hgb_leaves = (20, 50)
    errors = {name: [] for name in ("persistence", "weekly", *[f"ridge_{a}" for a in ridge_alphas],
                                    *[f"hgb_leaf_{leaf}" for leaf in hgb_leaves])}
    fold_rows = {}
    for start, end in folds:
        train = frame.loc[frame.index < start]
        valid = frame.loc[(frame.index >= start) & (frame.index < end)]
        if train.empty or valid.empty:
            raise ValueError(f"Empty rolling validation fold: {start}")
        x_train, y_train = train[feature_names].to_numpy(), train.target.to_numpy()
        x_valid, y_valid = valid[feature_names].to_numpy(), valid.target.to_numpy()
        predictions = {"persistence": valid.power_lag_1.to_numpy(),
                       "weekly": valid.power_lag_168.to_numpy()}
        for alpha in ridge_alphas:
            predictions[f"ridge_{alpha}"] = ridge_predict(fit_ridge(x_train, y_train, alpha), x_valid)
        for leaf in hgb_leaves:
            model = make_hgb(leaf)
            model.fit(x_train, y_train)
            predictions[f"hgb_leaf_{leaf}"] = np.maximum(0, model.predict(x_valid))
        for name, pred in predictions.items():
            errors[name].append(float(np.mean(np.abs(pred - y_valid))))
        fold_rows[start[:7]] = len(valid)
    mean_mae = {name: float(np.mean(values)) for name, values in errors.items()}
    best_alpha = min(ridge_alphas, key=lambda a: mean_mae[f"ridge_{a}"])
    best_leaf = min(hgb_leaves, key=lambda leaf: mean_mae[f"hgb_leaf_{leaf}"])
    return {"fold_rows": fold_rows, "fold_mae": errors, "mean_mae": mean_mae}, best_alpha, best_leaf


def make_hgb(min_samples_leaf: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                                         min_samples_leaf=min_samples_leaf,
                                         learning_rate=0.05, l2_regularization=10,
                                         random_state=42)


def condition_analysis(clean: pd.DataFrame, test: pd.DataFrame,
                       prediction: np.ndarray, mean_peak_cutoff: float,
                       quarter_peak_cutoff: float) -> dict:
    """Post-hoc descriptions; same-hour production and quarter power are never features."""
    detail = pd.DataFrame(index=test.index)
    detail["actual"] = test.target
    detail["absolute_error"] = np.abs(prediction - test.target.to_numpy())
    detail["observed_production"] = clean.loc[test.index, "생산량"]
    detail["previous_production"] = test.production_lag_1
    detail["quarter_max"] = clean.loc[test.index, POWER_COLUMNS].max(axis=1)
    detail["mean_peak"] = detail.actual >= mean_peak_cutoff
    detail["quarter_peak"] = detail.quarter_max >= quarter_peak_cutoff
    active = detail.observed_production.gt(0)
    previous_active = detail.previous_production.gt(0)
    detail["production_state"] = np.select(
        [~previous_active & ~active, ~previous_active & active,
         previous_active & ~active, previous_active & active],
        ["idle", "start", "stop", "running"], default="unknown")
    detail["hour"] = detail.index.hour
    detail["day_type"] = np.where(detail.index.dayofweek >= 5, "weekend", "weekday")

    def aggregate(column: str) -> dict:
        return {str(key): {"n": int(len(group)),
                           "actual_mean": float(group.actual.mean()),
                           "mae": float(group.absolute_error.mean()),
                           "mean_peak_rate": float(group.mean_peak.mean()),
                           "quarter_peak_rate": float(group.quarter_peak.mean())}
                for key, group in detail.groupby(column)}

    return {"note": "Same-hour production and 15-minute power are observed outcomes for diagnosis only.",
            "by_production_state": aggregate("production_state"),
            "by_hour": aggregate("hour"),
            "by_day_type": aggregate("day_type"),
            "quarter_peak_threshold": quarter_peak_cutoff,
            "quarter_peak_hours": int(detail.quarter_peak.sum()),
            "mean_peak_hours": int(detail.mean_peak.sum()),
            "mean_peak_misses_quarter_peak": int((detail.quarter_peak & ~detail.mean_peak).sum())}


def save_plot(predictions: pd.DataFrame, output: Path) -> None:
    first_week = predictions.iloc[:168]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(first_week.index, first_week["actual"], label="Actual", linewidth=1)
    ax.plot(first_week.index, first_week["hgb"], label="HGB 1h ahead", linewidth=1)
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
    selection, chosen_alpha, chosen_leaf = rolling_validation(frame.loc[frame.index < "2021-08-01"],
                                                                feature_names)
    development = frame.loc[frame.index < "2021-08-01"]
    x_train, y_train = development[feature_names].to_numpy(), development.target.to_numpy()
    x_test, y_test = test[feature_names].to_numpy(), test.target.to_numpy()
    # Freeze the peak definition at the start of validation for comparability.
    peak_cutoff = float(np.quantile(train.target.to_numpy(), 0.95))
    quarter_train_max = clean.loc[train.index, POWER_COLUMNS].max(axis=1)
    quarter_peak_cutoff = float(np.quantile(quarter_train_max.to_numpy(), 0.95))
    model = fit_ridge(x_train, y_train, chosen_alpha)
    ridge_prediction = ridge_predict(model, x_test)
    hgb = make_hgb(chosen_leaf)
    hgb.fit(x_train, y_train)
    hgb_prediction = np.maximum(0, hgb.predict(x_test))

    predictions = pd.DataFrame({"actual": y_test, "persistence": test["power_lag_1"].to_numpy(),
                                "weekly": test["power_lag_168"].to_numpy(),
                                "ridge": ridge_prediction, "hgb": hgb_prediction},
                               index=test.index)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "test_predictions.csv", index_label="timestamp")
    save_plot(predictions, output)
    report = {
        "reproducibility": {
            "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__, "scikit_learn": sklearn.__version__,
            "feature_names": feature_names,
            "validation_months": ["2021-04", "2021-05", "2021-06", "2021-07"],
            "final_test_start": "2021-08-01",
        },
        "data": diagnostics,
        "task": "At the end of hour t-1, predict the mean electricity demand during hour t.",
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "train_peak_95th_percentile": peak_cutoff,
        "model_selection": selection,
        "chosen_ridge_alpha": chosen_alpha,
        "chosen_hgb_min_samples_leaf": chosen_leaf,
        "final_fit_rows": len(development),
        "test_metrics": {name: scores(y_test, predictions[name].to_numpy(), peak_cutoff)
                         for name in ("persistence", "weekly", "ridge", "hgb")},
        "hgb_error_conditions": error_conditions(test, hgb_prediction, peak_cutoff),
        "observed_test_conditions": condition_analysis(clean, test, hgb_prediction,
                                                         peak_cutoff, quarter_peak_cutoff),
        "ridge_coefficients_standardized": {name: float(value) for name, value in
                                             zip(feature_names, model["coefficients"])},
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"data": diagnostics, "split_rows": report["split_rows"],
                      "chosen_ridge_alpha": chosen_alpha,
                      "chosen_hgb_min_samples_leaf": chosen_leaf,
                      "model_selection_mean_mae": selection["mean_mae"],
                      "test_metrics": report["test_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    arguments = parser.parse_args()
    run(arguments.data, arguments.output)
