"""Stage 17: forecast all next 24 hours from one observed origin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from advanced_forecast import FOLDS, regression_metrics
from deploy import digest
from forecast import POWER_COLUMNS, alert_scores, read_data, threshold_for_recall
from research import richer_features

HOURS = 24
NAMES = ("persistence", "previous_day", "previous_week", "extra_2", "extra_8",
         "profile_2", "profile_8", "residual_8")


def shifted_matrix(series, anchors, offset):
    return np.column_stack([series.reindex(anchors + pd.Timedelta(hours=offset + h)).to_numpy()
                            for h in range(HOURS)])


def trajectory_data(clean, require_targets=True):
    frame, features = richer_features(clean, require_target=False)
    power = clean["평균"]
    peak = clean[POWER_COLUMNS].max(axis=1, skipna=False)
    means = shifted_matrix(power, frame.index, 0)
    peaks = shifted_matrix(peak, frame.index, 0)
    base = {"persistence": np.repeat(frame.power_lag_1.to_numpy()[:, None], HOURS, axis=1),
            "previous_day": shifted_matrix(power, frame.index, -24),
            "previous_week": shifted_matrix(power, frame.index, -168)}
    peak_base = {"persistence": np.repeat(frame.previous_quarter_max.to_numpy()[:, None], HOURS, axis=1),
                 "previous_day": shifted_matrix(peak, frame.index, -24),
                 "previous_week": shifted_matrix(peak, frame.index, -168)}
    valid = np.ones(len(frame), dtype=bool)
    for value in [*base.values(), *peak_base.values()]:
        valid &= np.isfinite(value).all(axis=1)
    if require_targets:
        valid &= np.isfinite(means).all(axis=1) & np.isfinite(peaks).all(axis=1)
    # Give learned candidates the same complete past trajectories as baselines.
    added = {}
    for label, source in (("mean", base), ("peak", peak_base)):
        for period, key in (("day", "previous_day"), ("week", "previous_week")):
            for h in range(HOURS):
                added[f"profile_{label}_{period}_{h}"] = source[key][:, h]
    frame = pd.concat([frame, pd.DataFrame(added, index=frame.index)], axis=1)
    features = features + list(added)
    return (frame.loc[valid], features, means[valid], peaks[valid],
            {k: v[valid] for k, v in base.items()}, {k: v[valid] for k, v in peak_base.items()})


def training_mask(anchors, before):
    # Purge origins whose final training label would be on/after the cutoff.
    return anchors + pd.Timedelta(hours=HOURS - 1) < pd.Timestamp(before)


class TrajectoryModel:
    def __init__(self, leaf, features, residual=False):
        self.leaf, self.features, self.residual = leaf, features, residual

    def fit(self, frame, means, peaks):
        if self.residual:
            means = means - frame[[f"profile_mean_week_{h}" for h in range(HOURS)]].to_numpy()
            peaks = peaks - frame[[f"profile_peak_week_{h}" for h in range(HOURS)]].to_numpy()
        self.models = [ExtraTreesRegressor(n_estimators=240, min_samples_leaf=self.leaf,
                         max_features=1., random_state=42, n_jobs=2).fit(frame[self.features], target)
                       for target in (means, peaks)]
        return self

    def predict(self, frame):
        mean, peak = (model.predict(frame[self.features]) for model in self.models)
        if self.residual:
            mean += frame[[f"profile_mean_week_{h}" for h in range(HOURS)]].to_numpy()
            peak += frame[[f"profile_peak_week_{h}" for h in range(HOURS)]].to_numpy()
        mean = np.maximum(0, mean)
        return mean, np.maximum(mean, peak)


def evaluate_fold(dataset, start, end, names=NAMES):
    frame, features, means, peaks, base, peak_base = dataset
    train = training_mask(frame.index, start)
    valid = ((frame.index >= pd.Timestamp(start)) & (frame.index + pd.Timedelta(hours=23) < pd.Timestamp(end))
             & (frame.index.hour == 0))
    anchors = frame.index[valid]
    out = pd.DataFrame({"anchor_time": np.repeat(anchors, HOURS),
                        "last_observed_hour": np.repeat(anchors - pd.Timedelta(hours=1), HOURS),
                        "horizon": np.tile(np.arange(1, HOURS + 1), len(anchors)),
                        "actual": means[valid].ravel(), "actual_quarter": peaks[valid].ravel()})
    out["timestamp"] = out.anchor_time + pd.to_timedelta(out.horizon - 1, unit="h")
    for name in names:
        if name in base:
            mean_prediction, peak_prediction = base[name][valid], peak_base[name][valid]
        else:
            chosen_features = [n for n in features if not n.startswith("profile_")] if name.startswith("extra_") else features
            model = TrajectoryModel(int(name.split("_")[1]), chosen_features, name.startswith("residual_")).fit(frame.loc[train], means[train], peaks[train])
            mean_prediction, peak_prediction = model.predict(frame.loc[valid])
        out[name] = mean_prediction.ravel()
        out[f"quarter_{name}"] = peak_prediction.ravel()
    out["fit_before"] = start
    provenance = {"train_origins": int(train.sum()), "train_last_origin": str(frame.index[train].max()),
                  "train_last_target": str(frame.index[train].max() + pd.Timedelta(hours=23)),
                  "first_evaluation_origin": str(anchors.min()), "evaluation_days": len(anchors)}
    return out, provenance


def choose_trajectory(oof):
    selection = oof.loc[oof.anchor_time < pd.Timestamp("2021-07-01")]
    errors = selection[list(NAMES)].sub(selection.actual, axis=0).abs()
    monthly = errors.groupby(selection.anchor_time.dt.strftime("%Y-%m")).mean()
    scores = {name: float(monthly[name].mean()) for name in NAMES}
    chosen = min(scores, key=scores.get)
    threshold = threshold_for_recall(selection[f"quarter_{chosen}"].to_numpy(),
                                     selection.actual_quarter.to_numpy() >= 182, .85)
    return {"chosen": chosen, "candidate_scores": scores,
            "peak_score_threshold": threshold, "peak_event_threshold": 182,
            "selection_months": ["2021-04", "2021-05", "2021-06"]}


def metrics(frame, name, threshold):
    result = regression_metrics(frame.actual.to_numpy(), frame[name].to_numpy())
    result["by_horizon"] = {str(h): regression_metrics(g.actual.to_numpy(), g[name].to_numpy())
                            for h, g in frame.groupby("horizon")}
    daily = frame.groupby("anchor_time")
    result["daily_mean_max_mae"] = float((daily.actual.max() - daily[name].max()).abs().mean())
    result["daily_quarter_max_mae"] = float((daily.actual_quarter.max() - daily[f"quarter_{name}"].max()).abs().mean())
    actual_hours = frame.loc[daily.actual.idxmax().to_numpy(), "horizon"].to_numpy()
    predicted_hours = frame.loc[daily[name].idxmax().to_numpy(), "horizon"].to_numpy()
    result["mean_peak_time_abs_hours"] = float(np.abs(actual_hours - predicted_hours).mean())
    result["quarter_alerts"] = alert_scores(frame.actual_quarter.to_numpy(), frame[f"quarter_{name}"].to_numpy(), 182, threshold)
    return result


def plot_evaluation(evaluated, output, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    for name in dict.fromkeys(["persistence", "previous_day", "previous_week", chosen]):
        error = (evaluated[name] - evaluated.actual).abs().groupby(evaluated.horizon).mean()
        axes[0].plot(error.index, error, label=name)
    axes[0].set(title="24-hour forecasts issued once at midnight", xlabel="Lead hour (also target clock hour + 1)", ylabel="MAE (source units)")
    axes[0].legend()
    daily = evaluated.groupby("anchor_time")[["actual", chosen]].max()
    axes[1].plot(daily.index, daily.actual, label="Observed daily maximum of hourly mean")
    axes[1].plot(daily.index, daily[chosen], label="Forecast daily maximum of hourly mean")
    axes[1].set(title="Longer planning horizon exposes shutdown and restart failures", ylabel="Source units")
    axes[1].legend()
    fig.savefig(output / "horizon_diagnostics.png", dpi=160)
    plt.close(fig)


def forecast_trajectory(data, artifact, observed_through=None):
    bundle = joblib.load(artifact)
    clean, _ = read_data(data, allow_partial_last_day=True, observed_through=observed_through)
    observed = clean.index.max()
    if observed_through is not None and observed != pd.Timestamp(observed_through):
        raise ValueError("Requested last observed hour is absent")
    if observed < pd.Timestamp(bundle["train_end"]):
        raise ValueError("Model training contains future observations")
    anchor = observed + pd.Timedelta(hours=1)
    if anchor.hour != 0:
        raise ValueError("Validated planning forecasts must start at midnight")
    extended = clean.reindex(clean.index.union(pd.DatetimeIndex([anchor])))
    frame, features, _, _, base, peak_base = trajectory_data(extended, require_targets=False)
    if features != bundle["features"] or anchor not in frame.index:
        raise ValueError("Feature contract mismatch or missing exact history")
    index = frame.index.get_loc(anchor)
    if bundle["chosen"] in base:
        mean, peak = base[bundle["chosen"]][[index]], peak_base[bundle["chosen"]][[index]]
    else:
        mean, peak = bundle["model"].predict(frame.loc[[anchor]])
    result = pd.DataFrame({"timestamp": pd.date_range(anchor, periods=HOURS, freq="h"),
                           "horizon": np.arange(1, HOURS+1), "predicted_mean": mean.ravel(),
                           "predicted_quarter_max": peak.ravel()})
    result["peak_alert"] = result.predicted_quarter_max >= bundle["peak_score_threshold"]
    result["last_observed_hour"] = observed
    result["model_train_end"] = bundle["train_end"]
    return result


def run(data, root):
    clean, _ = read_data(data)
    dataset = trajectory_data(clean)
    output = root / "stage17"
    output.mkdir(parents=True, exist_ok=True)
    parts, provenance = [], {}
    for start, end in FOLDS:
        result, info = evaluate_fold(dataset, start, end)
        parts.append(result)
        provenance[start[:7]] = info
        print(f"Stage 17: {start[:7]} completed ({info['evaluation_days']} daily origins)", flush=True)
    oof = pd.concat(parts, ignore_index=True)
    selection = {**choose_trajectory(oof), "input_sha256": digest(data), "horizon_hours": HOURS,
                 "features": dataset[1], "origin_hour": 0,
                 "note": "24 outputs share one origin; no within-day observations used."}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Stage 17: locked {selection['chosen']}", flush=True)
    parts = []
    names = tuple(dict.fromkeys(["persistence", "previous_day", "previous_week", selection["chosen"]]))
    for start, end in [("2021-08-01", "2021-09-01"), ("2021-09-01", "2021-09-15")]:
        result, info = evaluate_fold(dataset, start, end, names)
        parts.append(result)
        provenance[start[:7]] = info
    evaluated = pd.concat(parts, ignore_index=True)
    chosen, threshold = selection["chosen"], selection["peak_score_threshold"]
    evaluated["prediction"] = evaluated[chosen]
    evaluated["quarter_prediction"] = evaluated[f"quarter_{chosen}"]
    evaluated["peak_alert"] = evaluated.quarter_prediction >= threshold
    july = oof.loc[oof.anchor_time >= "2021-07-01"]
    report = {**selection, "training_provenance": provenance,
              "july": metrics(july, chosen, threshold),
              "evaluation": {name: metrics(evaluated, name, threshold) for name in names},
              "by_month": {str(m): metrics(g, chosen, threshold)
                           for m, g in evaluated.groupby(evaluated.anchor_time.dt.strftime("%Y-%m"))},
              "limitation": "Exploratory 24-hour planning horizon, not confirmed official target. Previously inspected data. Baseline peak metrics use the selected policy's threshold for reference only."}
    oof.to_csv(output / "validation_predictions.csv", index=False)
    evaluated.to_csv(output / "predictions.csv", index=False)
    plot_evaluation(evaluated, output, chosen)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    frame, features, means, peaks, _, _ = dataset
    chosen_features = [n for n in features if not n.startswith("profile_")] if chosen.startswith("extra_") else features
    model = None if chosen in ("persistence", "previous_day", "previous_week") else TrajectoryModel(int(chosen.split("_")[1]), chosen_features, chosen.startswith("residual_")).fit(frame, means, peaks)
    metadata = {**selection, "train_end": str(clean.index.max()), "last_train_origin": str(frame.index.max()),
                "last_train_target": str(frame.index.max() + pd.Timedelta(hours=23))}
    artifact = Path("artifacts/day_ahead.joblib")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**metadata, "model": model}, artifact)
    artifact.with_suffix(".json").write_text(json.dumps({**metadata, "artifact_sha256": digest(artifact)}, indent=2), encoding="utf-8")
    forecast_trajectory(data, artifact).to_csv(output / "next_day_prediction.csv", index=False)
    print(json.dumps({"selection": selection["candidate_scores"], "chosen": chosen,
                      "evaluation": {name: report["evaluation"][name]["mae"] for name in names}}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/day_ahead.joblib"))
    parser.add_argument("--last-observed-hour", type=pd.Timestamp)
    args = parser.parse_args()
    from horizon import run, forecast_trajectory
    if args.predict_only:
        print(forecast_trajectory(args.data, args.artifact, args.last_observed_hour).to_csv(index=False))
    else:
        run(args.data, args.output_root)
