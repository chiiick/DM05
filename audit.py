"""Stage 6: data redundancy and time-ordered forecast improvement audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import POWER_COLUMNS, idle_mask, make_features, read_data, stage_two_validation


def forward_gate_selection(candidates: pd.DataFrame, actual: pd.Series) -> tuple[pd.Series, list]:
    """Evaluate each month using a rule selected from earlier OOF months only."""
    if not candidates.index.equals(actual.index):
        raise ValueError("Prediction and outcome indices must match")
    month = candidates.index.to_period("M")
    prediction = pd.Series(np.nan, index=actual.index, name="forward_gate")
    records = []
    for current in sorted(month.unique())[1:]:
        earlier, current_rows = month < current, month == current
        errors = candidates.loc[earlier].sub(actual.loc[earlier], axis=0).abs()
        mean_month_errors = errors.groupby(errors.index.to_period("M")).mean().mean()
        chosen = mean_month_errors.idxmin()
        prediction.loc[current_rows] = candidates.loc[current_rows, chosen]
        records.append({"evaluation_month": str(current), "selected_rule": chosen,
                        "selection_end": str(actual.index[earlier].max()),
                        "n": int(current_rows.sum()),
                        "mae": float((prediction.loc[current_rows] - actual.loc[current_rows]).abs().mean()),
                        "no_gate_mae": float((candidates.loc[current_rows, "no_gate"] - actual.loc[current_rows]).abs().mean())})
    return prediction, records


def block_bootstrap_improvement(predictions: pd.DataFrame, baseline: str,
                                improved: str, seed: int = 42) -> dict:
    """Resample circular blocks of seven days, preserving all hours per day."""
    difference = ((predictions[baseline] - predictions.actual).abs()
                  - (predictions[improved] - predictions.actual).abs())
    daily = difference.groupby(difference.index.normalize()).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    n, block = len(daily), min(7, len(daily))
    means = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        sample = daily.iloc[indices]
        means.append(float(sample["sum"].sum() / sample["count"].sum()))
    return {"baseline": baseline, "improved": improved, "n_days": n,
            "mean_mae_reduction": float(difference.mean()),
            "percentile_95_interval": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "method": "2000 circular moving-block resamples, 7 days per block, seed 42",
            "limitation": "Conditional on this already inspected period; does not remove model-selection bias."}


def run(data: Path, predictions_path: Path, output: Path) -> None:
    clean, diagnostics = read_data(data)
    frame, features = make_features(clean)
    development = frame.loc[frame.index < "2021-08-01"]
    quarter = clean.loc[frame.index, POWER_COLUMNS].max(axis=1)
    _, oof = stage_two_validation(development, quarter, features, 182.0)
    candidates = pd.DataFrame({"no_gate": oof.predicted_mean}, index=oof.index)
    signal = frame.loc[oof.index]
    for hours in (6, 12, 24):
        for cap in (25, 30, 35, 40, 50):
            mask = idle_mask(clean, signal, hours, cap)
            candidates[f"idle_{hours}h_cap_{cap}"] = np.where(
                mask, signal.power_lag_1, oof.predicted_mean)
    forward, selection = forward_gate_selection(candidates, oof.actual_mean)
    oof["forward_gate"] = forward

    profiles = clean["평균"].groupby(clean.index.normalize()).apply(tuple)
    train_profiles = set(profiles.loc[profiles.index < "2021-08-01"])
    later_profiles = profiles.loc[profiles.index >= "2021-08-01"]
    ratio = clean["생산량"] / clean[POWER_COLUMNS].sum(axis=1).replace(0, np.nan)
    difference = (ratio - clean["공장인원"]).abs().dropna()
    predictions = pd.read_csv(predictions_path, parse_dates=["timestamp"]).set_index("timestamp")
    periods = {"all": predictions, "first_august_week": predictions.loc[predictions.index < "2021-08-08"],
               "after_first_august_week": predictions.loc[predictions.index >= "2021-08-08"]}
    report = {
        "data": diagnostics,
        "profile_redundancy": {"days": len(profiles), "unique_hourly_mean_profiles": profiles.nunique(),
                               "repeated_quarter_vectors": int(clean.duplicated(POWER_COLUMNS).sum()),
                               "evaluation_days": len(later_profiles),
                               "evaluation_days_with_profile_seen_before_august": int(later_profiles.isin(train_profiles).sum()),
                               "interpretation": "Repeated profiles may reflect augmentation or recurring operation; provenance is unverified."},
        "derived_workforce": {"n_comparable": len(difference), "max_absolute_difference": float(difference.max()),
                              "formula": "production / sum of same-hour quarter power",
                              "interpretation": "Do not treat this field as an independent measured staffing count."},
        "period_mae": {name: {"n": len(part), **{model: float((part[model] - part.actual).abs().mean())
                                                for model in ("persistence", "ensemble", "idle_gated", "operational")}}
                       for name, part in periods.items()},
        "gate_forward_selection": selection,
        "block_bootstrap": [block_bootstrap_improvement(predictions, baseline, "operational")
                            for baseline in ("persistence", "ensemble", "idle_gated")],
        "status": "Retrospective audit, not a new untouched test set.",
    }
    output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output / "validation_predictions.csv", index_label="timestamp")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/test_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage6"))
    args = parser.parse_args()
    run(args.data, args.predictions, args.output)
