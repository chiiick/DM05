"""Stage 15: error diagnosis and sensitivity; does not select another model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_forecast import regression_metrics
from audit import block_bootstrap_improvement
from deploy import digest
from forecast import idle_mask, read_data
from research import DemandModel, apply_gate, richer_features


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")


def grouped_metrics(frame: pd.DataFrame, column: str) -> dict:
    result = {}
    for key, rows in frame.groupby(column):
        result[str(key)] = {
            **regression_metrics(rows.actual.to_numpy(), rows.prediction.to_numpy()),
            "absolute_error_p95": float(rows.absolute_error.quantile(.95)),
            "interval_coverage": float(rows.covered.mean()),
            "peak_tp": int((rows.peak_actual & rows.peak_alert).sum()),
            "peak_fp": int((~rows.peak_actual & rows.peak_alert).sum()),
            "peak_fn": int((rows.peak_actual & ~rows.peak_alert).sum()),
        }
    return result


def july_sensitivity(clean: pd.DataFrame, config: dict) -> dict:
    frame, features = richer_features(clean, config["features"])
    july = frame.loc[(frame.index >= "2021-07-01") & (frame.index < "2021-08-01")]
    model = DemandModel(config["model"], features).fit(frame.loc[frame.index < "2021-07-01"])
    baseline = apply_gate(clean, july, model.predict(july), config["gate"])
    mae = float(np.abs(baseline - july.target).mean())
    calendar = {"hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
                "weekend", "hour", "weekday", "day_of_year", "weekday_work_start"}
    groups = {"calendar": [], "quarter_shape": [], "production_history": [], "power_history": []}
    for name in features:
        if name in calendar:
            group = "calendar"
        elif "quarter" in name or name.startswith("previous_") and "분" in name:
            group = "quarter_shape"
        elif any(word in name for word in ("production", "active", "idle")):
            group = "production_history"
        else:
            group = "power_history"
        groups[group].append(name)
    rng = np.random.default_rng(42)
    results = {}
    for group, names in groups.items():
        differences = []
        for _ in range(3):
            altered = july.copy()
            altered.loc[:, names] = july[names].iloc[rng.permutation(len(july))].to_numpy()
            # Hold the observed operational gate and its fallback prediction fixed.
            prediction = apply_gate(clean, july, model.predict(altered), config["gate"])
            differences.append(float(np.abs(prediction - july.target).mean()) - mae)
        results[group] = {"features": names, "mae_increase_repeats": differences,
                          "mean_mae_increase": float(np.mean(differences))}
    return {"train_end": "2021-06-30 23:00:00", "diagnostic_month": "2021-07",
            "n": len(july), "baseline_mae": mae, "groups": results,
            "limitation": "Joint row permutation within each feature group breaks temporal and cross-group dependence. Predictive sensitivity, not causal importance; idle gate held fixed. No model selection from this diagnostic."}


def run(data: Path, root: Path) -> dict:
    selection = json.loads((root / "stage13/summary.json").read_text())
    if digest(data) != selection["input_sha256"]:
        raise ValueError("Data differs from recorded evaluation")
    clean, _ = read_data(data)
    result = read_predictions(root / "stage14/interval_predictions.csv")
    legacy = read_predictions(root / "test_predictions.csv")
    if not result.index.equals(legacy.index):
        raise ValueError("Evaluation indices differ")
    np.testing.assert_allclose(result.actual, legacy.actual)
    hourly = clean.asfreq("h")
    current = hourly["생산량"].reindex(result.index).gt(0)
    previous = hourly["생산량"].shift(1).reindex(result.index).gt(0)
    result["observed_state"] = np.select([current & ~previous, ~current & previous, current & previous],
                                         ["startup", "shutdown", "running"], default="idle")
    result["production_actual"] = hourly["생산량"].reindex(result.index)
    result["previous_power"] = hourly["평균"].shift(1).reindex(result.index)
    result["peak_actual"] = legacy.quarter_actual.ge(182)
    result["peak_alert"] = legacy.peak_classifier_alert
    result["legacy_prediction"] = legacy.operational
    result["persistence"] = legacy.persistence
    result["absolute_error"] = (result.prediction - result.actual).abs()
    result["covered"] = result.actual.between(result.conditional_lower, result.conditional_upper)
    result["hour"] = result.index.hour
    result["month"] = result.index.strftime("%Y-%m")
    report = {"input_sha256": digest(data), "evaluation": selection["evaluation"],
              "by_observed_state": grouped_metrics(result, "observed_state"),
              "by_prior_idle_state": grouped_metrics(result, "known_idle"),
              "by_hour": grouped_metrics(result, "hour"), "by_month": grouped_metrics(result, "month"),
              "bootstrap": [block_bootstrap_improvement(result, name, "prediction")
                            for name in ("legacy_prediction", "persistence")],
              "july_feature_sensitivity": july_sensitivity(clean, selection["config"]),
              "limitations": ["August-September were inspected repeatedly; results are retrospective.",
                              "Observed startup/shutdown labels use target-hour production for diagnosis only, never for prediction.",
                              "Bootstrap is conditional on these 45 days and excludes model-selection uncertainty."]}
    output = root / "stage15"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "diagnostic_predictions.csv", index_label="timestamp")
    result.nlargest(20, "absolute_error").to_csv(output / "largest_errors.csv", index_label="timestamp")
    (output / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    errors = result[["legacy_prediction", "prediction"]].sub(result.actual, axis=0).abs().resample("D").mean()
    axes[0].plot(errors.index, errors.legacy_prediction, label="Stage 5 HGB", alpha=.8)
    axes[0].plot(errors.index, errors.prediction, label="Final ExtraTrees", alpha=.8)
    axes[0].set(title="Daily MAE: retrospective evaluation (45 days)", ylabel="MAE (source units)")
    axes[0].legend(loc="upper left")
    week = result.loc["2021-08-09":"2021-08-15"]
    axes[1].fill_between(week.index, week.conditional_lower, week.conditional_upper, alpha=.2, label="Empirical 90% range")
    axes[1].plot(week.index, week.actual, color="black", lw=1, label="Observed")
    axes[1].plot(week.index, week.prediction, lw=1, label="Forecast")
    axes[1].set(title="Working-week detail, including transitions", ylabel="Hourly mean (source units)")
    axes[1].legend(loc="upper left", ncol=3)
    states = pd.DataFrame(report["by_observed_state"]).T.reindex(["idle", "startup", "running", "shutdown"])
    axes[2].bar(states.index, states.mae, color=["#72b7b2", "#e45756", "#4c78a8", "#f2cf5b"])
    for i, (_, row) in enumerate(states.iterrows()):
        axes[2].text(i, row.mae + .3, f"n={int(row.n)}; coverage={row.interval_coverage:.0%}", ha="center")
    axes[2].set(title="Post-hoc production states: transitions remain difficult", ylabel="MAE (source units)")
    axes[2].set_ylim(0, states.mae.max() * 1.25)
    fig.savefig(output / "final_diagnostics.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("evaluation", "by_observed_state", "bootstrap")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.data, args.output_root)
