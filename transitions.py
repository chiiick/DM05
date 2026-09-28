"""Stage 16: protect operating transitions using strictly historical inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from advanced_forecast import FOLDS, regression_metrics
from deploy import digest
from forecast import idle_mask, read_data
from research import DemandModel, richer_features

CANDIDATES = ("baseline", "ungated", "protect_025", "protect_050", "soft_gate", "state_mixture")


def observed_states(clean, frame):
    active = clean["생산량"].reindex(frame.index).gt(0)
    previous = frame.production_lag_1.gt(0)
    return pd.Series(np.select([active & ~previous, ~active & previous, active & previous],
                              ["startup", "shutdown", "running"], default="idle"), index=frame.index)


class TransitionModel:
    def __init__(self, features):
        self.features = features

    def fit(self, clean, frame):
        active = clean.loc[frame.index, "생산량"].gt(0)
        self.base = DemandModel("extra_2", self.features).fit(frame)
        self.state = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
            min_samples_leaf=20, learning_rate=.05, l2_regularization=10,
            early_stopping=False, random_state=42).fit(frame[self.features], active)
        self.experts = [DemandModel("extra_2", self.features).fit(frame.loc[active == state])
                        for state in (False, True)]
        return self

    def predict(self, clean, frame):
        raw = self.base.predict(frame)
        score = self.state.predict_proba(frame[self.features])[:, 1]
        idle = idle_mask(clean, frame, 6, 25)
        old = frame.power_lag_1.to_numpy()
        mixture = (1 - score) * self.experts[0].predict(frame) + score * self.experts[1].predict(frame)
        return pd.DataFrame({"baseline": np.where(idle, old, raw), "ungated": raw,
            "protect_025": np.where(idle & (score < .25), old, raw),
            "protect_050": np.where(idle & (score < .50), old, raw),
            "soft_gate": np.where(idle, (1-score) * old + score * raw, raw),
            "state_mixture": mixture, "active_score": score}, index=frame.index)


def select_candidate(oof):
    selection = oof.loc[oof.index < "2021-07-01"]
    err = selection[list(CANDIDATES)].sub(selection.actual, axis=0).abs()
    monthly = err.groupby(err.index.strftime("%Y-%m")).mean()
    transition = err.loc[selection.observed_state.isin(["startup", "shutdown"])]
    transition_monthly = transition.groupby(transition.index.strftime("%Y-%m")).mean()
    scores = {}
    for name in CANDIDATES:
        eligible = bool(monthly[name].mean() <= 1.01 * monthly.baseline.mean()
                        and (monthly[name] <= 1.05 * monthly.baseline).all())
        scores[name] = {"mean_month_mae": float(monthly[name].mean()),
                       "transition_month_mae": float(transition_monthly[name].mean()),
                       "eligible": eligible}
    chosen = min((name for name in CANDIDATES if scores[name]["eligible"]),
                 key=lambda name: scores[name]["transition_month_mae"])
    july = oof.loc[(oof.index >= "2021-07-01") & (oof.index < "2021-08-01")]
    transition = july.observed_state.isin(["startup", "shutdown"])
    errors = july[list(CANDIDATES)].sub(july.actual, axis=0).abs()
    accepted = bool(errors[chosen].mean() <= 1.01 * errors.baseline.mean()
                    and errors.loc[transition, chosen].mean() <= .95 * errors.loc[transition, "baseline"].mean())
    return {"selection_scores": scores, "selected_candidate": chosen,
            "july_acceptance": accepted, "adopted": chosen if accepted else "baseline",
            "july": {name: {"mae": float(errors[name].mean()),
                            "transition_mae": float(errors.loc[transition, name].mean())} for name in CANDIDATES}}


def evaluate_part(part, candidate):
    return {"all": regression_metrics(part.actual.to_numpy(), part[candidate].to_numpy()),
            "by_state": {state: regression_metrics(g.actual.to_numpy(), g[candidate].to_numpy())
                         for state, g in part.groupby("observed_state")}}


def run(data, root):
    clean, _ = read_data(data)
    frame, features = richer_features(clean)
    output = root / "stage16"
    output.mkdir(parents=True, exist_ok=True)
    parts = []
    for start, end in FOLDS:
        train = frame.loc[frame.index < start]
        valid = frame.loc[(frame.index >= start) & (frame.index < end)]
        result = TransitionModel(features).fit(clean, train).predict(clean, valid)
        result["actual"] = valid.target
        result["observed_state"] = observed_states(clean, valid)
        result["fit_before"] = start
        parts.append(result)
        print(f"Stage 16: {start[:7]} validation complete", flush=True)
    oof = pd.concat(parts)
    decision = select_candidate(oof)
    lock = {**decision, "input_sha256": digest(data), "features": features,
            "selection_months": ["2021-04", "2021-05", "2021-06"],
            "acceptance_month": "2021-07", "candidate_names": CANDIDATES}
    (output / "selection.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Stage 16: locked {decision['adopted']} before August evaluation", flush=True)
    parts = []
    for start, end in [("2021-08-01", "2021-09-01"), ("2021-09-01", "2021-09-15")]:
        valid = frame.loc[(frame.index >= start) & (frame.index < end)]
        result = TransitionModel(features).fit(clean, frame.loc[frame.index < start]).predict(clean, valid)
        result["actual"] = valid.target
        result["observed_state"] = observed_states(clean, valid)
        result["fit_before"] = start
        parts.append(result)
    evaluated = pd.concat(parts)
    evaluated["prediction"] = evaluated[decision["adopted"]]
    report = {**lock, "evaluation": {name: evaluate_part(evaluated, name)
                                    for name in dict.fromkeys(["baseline", decision["selected_candidate"], decision["adopted"]])},
              "by_month": {str(m): evaluate_part(g, decision["adopted"])
                           for m, g in evaluated.groupby(evaluated.index.strftime("%Y-%m"))},
              "note": "Retrospective data; July is an acceptance gate, not an untouched test. State labels are diagnostic only; score is not a calibrated probability."}
    oof.to_csv(output / "validation_predictions.csv", index_label="timestamp")
    evaluated.to_csv(output / "predictions.csv", index_label="timestamp")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Store a mean-only candidate bundle. Existing release remains reproducible.
    model = TransitionModel(features).fit(clean, frame)
    artifact = Path("artifacts/stage16.joblib")
    joblib.dump({"model": model, "candidate": decision["adopted"], "train_end": str(frame.index.max()),
                 "features": features, "input_sha256": digest(data)}, artifact)
    target = clean.index.max() + pd.Timedelta(hours=1)
    extended = clean.reindex(clean.index.union(pd.DatetimeIndex([target])))
    next_frame, _ = richer_features(extended, require_target=False)
    next_values = model.predict(extended, next_frame.loc[[target]])
    next_report = {"forecast_hour": str(target), "train_end": str(frame.index.max()),
                   "candidate": decision["adopted"], "predicted_mean": float(next_values.loc[target, decision["adopted"]]),
                   "active_score": float(next_values.loc[target, "active_score"]),
                   "artifact_sha256": digest(artifact), "note": "Mean-only extension; stage 15 peak policy is unchanged."}
    (output / "next_prediction.json").write_text(json.dumps(next_report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("selected_candidate", "july_acceptance", "adopted", "july", "evaluation")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    # Stable module name for loading the saved model in a separate process.
    from transitions import run
    run(args.data, args.output_root)
