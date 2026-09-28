"""Stage 12: classify quarter-hour peak events using past observations only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

from advanced_forecast import FOLDS
from forecast import POWER_COLUMNS, alert_scores, make_peak_classifier, read_data, threshold_for_recall
from research import richer_features


class PeakModel:
    def __init__(self, algorithm: str, features: list[str]):
        self.algorithm, self.features = algorithm, features

    def fit(self, frame: pd.DataFrame, labels: np.ndarray):
        if self.algorithm == "hgb":
            self.estimators = [make_peak_classifier(leaf) for leaf in (20, 50)]
        elif self.algorithm.startswith("extra_"):
            leaf = int(self.algorithm.split("_")[1])
            self.estimators = [ExtraTreesClassifier(n_estimators=240, min_samples_leaf=leaf,
                                                    class_weight="balanced", random_state=42, n_jobs=2)]
        elif self.algorithm == "forest":
            self.estimators = [RandomForestClassifier(n_estimators=240, min_samples_leaf=3,
                                                      max_features=.8, class_weight="balanced",
                                                      random_state=42, n_jobs=2)]
        else:
            raise ValueError(f"Unknown peak algorithm: {self.algorithm}")
        if len(np.unique(labels)) != 2:
            raise ValueError("Peak training requires both classes")
        for estimator in self.estimators:
            estimator.fit(frame[self.features], labels)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.mean([e.predict_proba(frame[self.features])[:, 1] for e in self.estimators], axis=0)


def choose_policy(oof: pd.DataFrame, names: list[str], cutoff: float) -> dict:
    selection = oof.loc[oof.index < "2021-07-01"]
    monthly_ap = {name: {str(m): float(average_precision_score(g.actual >= cutoff, g[name]))
                         for m, g in selection.groupby(selection.index.strftime("%Y-%m"))} for name in names}
    mean_ap = {name: float(np.mean(list(v.values()))) for name, v in monthly_ap.items()}
    chosen = max(mean_ap, key=mean_ap.get)
    observed = selection.actual.to_numpy() >= cutoff
    threshold = threshold_for_recall(selection[chosen].to_numpy(), observed, .85)
    return {"chosen": chosen, "monthly_average_precision": monthly_ap,
            "mean_month_average_precision": mean_ap, "score_cutoff": threshold,
            "selection_event_recall_floor": .85,
            "selection_counts": alert_scores(selection.actual.to_numpy(), selection[chosen].to_numpy(), cutoff, threshold)}


def run(data: Path, output: Path) -> dict:
    clean, _ = read_data(data)
    frames = {group: richer_features(clean, group) for group in ("base", "combined")}
    configs = {"legacy_hgb": {"features": "base", "model": "hgb"},
               **{name: {"features": "combined", "model": name} for name in ("hgb", "extra_2", "extra_8", "forest")}}
    # Preserve the previously defined event for comparison across stages.
    cutoff = 182.
    parts = []
    for start, end in FOLDS:
        predictions = {}
        for name, config in configs.items():
            frame, features = frames[config["features"]]
            train = frame.loc[frame.index < start]
            valid = frame.loc[(frame.index >= start) & (frame.index < end)]
            target = clean.loc[train.index, POWER_COLUMNS].max(axis=1).ge(cutoff).to_numpy()
            predictions[name] = PeakModel(config["model"], features).fit(train, target).predict(valid)
        part = pd.DataFrame(predictions, index=valid.index)
        part.insert(0, "actual", clean.loc[valid.index, POWER_COLUMNS].max(axis=1))
        parts.append(part)
        print(f"Peak validation {start[:7]}", flush=True)
    oof = pd.concat(parts)
    policy = choose_policy(oof, list(configs), cutoff)
    config = configs[policy["chosen"]]
    output.mkdir(parents=True, exist_ok=True)
    lock = {**policy, "config": config, "event_cutoff": cutoff,
            "selection_end": "2021-06-30 23:00:00",
            "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
    (output / "selection.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    frame, features = frames[config["features"]]
    train = frame.loc[frame.index < "2021-08-01"]
    test = frame.loc[frame.index >= "2021-08-01"]
    target = clean.loc[train.index, POWER_COLUMNS].max(axis=1).ge(cutoff).to_numpy()
    prediction = PeakModel(config["model"], features).fit(train, target).predict(test)
    evaluation = pd.DataFrame({"actual": clean.loc[test.index, POWER_COLUMNS].max(axis=1),
                               "risk_score": prediction, "alert": prediction >= policy["score_cutoff"]}, index=test.index)
    july = oof.loc[oof.index >= "2021-07-01"]
    report = {**lock, "features": features, "configs": configs,
              "july_audit": alert_scores(july.actual.to_numpy(), july[policy["chosen"]].to_numpy(), cutoff, policy["score_cutoff"]),
              "evaluation": alert_scores(evaluation.actual.to_numpy(), prediction, cutoff, policy["score_cutoff"]),
              "evaluation_average_precision": float(average_precision_score(evaluation.actual >= cutoff, prediction)),
              "by_month": {str(m): alert_scores(g.actual.to_numpy(), g.risk_score.to_numpy(), cutoff, policy["score_cutoff"])
                           for m, g in evaluation.groupby(evaluation.index.strftime("%Y-%m"))},
              "note": "Scores are not calibrated probabilities. Event cutoff is the inherited retrospective 182 definition."}
    oof.to_csv(output / "validation_predictions.csv", index_label="timestamp")
    evaluation.to_csv(output / "predictions.csv", index_label="timestamp")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage12"))
    args = parser.parse_args()
    run(args.data, args.output)
