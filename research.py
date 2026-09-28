"""Controlled stage 9+ experiments; select on April-June, audit July separately."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from advanced_forecast import FOLDS, regression_metrics
from forecast import idle_mask, make_features, read_data


def richer_features(clean: pd.DataFrame, group: str = "combined",
                    require_target: bool = True) -> tuple[pd.DataFrame, list[str]]:
    base, names = make_features(clean, require_target=require_target)
    if group == "base":
        return base, names
    hourly = clean.asfreq("h")
    quarter = pd.DataFrame(index=hourly.index)
    for column in ("15분", "30분", "45분"):
        quarter[f"previous_{column}"] = hourly[column].shift(1)
    quarter["previous_quarter_ramp"] = (hourly["60분"] - hourly["15분"]).shift(1)
    quarter["previous_last_quarter_ramp"] = (hourly["60분"] - hourly["45분"]).shift(1)
    values = hourly[["15분", "30분", "45분", "60분"]]
    quarter["previous_quarter_range"] = (values.max(axis=1) - values.min(axis=1)).shift(1)
    quarter["previous_quarter_max"] = values.max(axis=1).shift(1)
    quarter["day_ago_quarter_max"] = values.max(axis=1).shift(24)
    regime = pd.DataFrame(index=hourly.index)
    past_power = hourly["평균"].shift(1)
    past_production = hourly["생산량"].shift(1)
    active = past_production.gt(0).astype(float).where(past_production.notna())
    for window in (6, 12, 24, 48):
        minimum = min(window, 24)
        regime[f"past_{window}_active_share"] = active.rolling(window, min_periods=minimum).mean()
        regime[f"past_{window}_production_logmean"] = np.log1p(
            past_production.rolling(window, min_periods=minimum).mean())
    regime["past_24_power_min"] = past_power.rolling(24, min_periods=24).min()
    regime["past_24_power_max"] = past_power.rolling(24, min_periods=24).max()
    regime["past_48_power_mean"] = past_power.rolling(48, min_periods=24).mean()
    regime["power_relative_to_day_mean"] = past_power / past_power.rolling(24, min_periods=24).mean().clip(lower=1)
    zero = hourly["생산량"].eq(0) & hourly["생산량"].notna()
    regime["preceding_idle_hours"] = zero.groupby((~zero).cumsum()).cumsum().shift(1)
    groups = {"quarter": quarter, "regime": regime, "combined": pd.concat([quarter, regime], axis=1)}
    if group not in groups:
        raise ValueError(f"Unknown feature group: {group}")
    extra = groups[group].reindex(base.index)
    frame = pd.concat([base, extra], axis=1)
    features = names + list(extra.columns)
    if frame[features].isna().any().any():
        raise ValueError("New features unexpectedly changed the common evaluation rows")
    return frame, features


class DemandModel:
    def __init__(self, algorithm: str, features: list[str]):
        self.algorithm = algorithm
        self.features = features
        self.estimators = []

    def fit(self, frame: pd.DataFrame, target: np.ndarray | None = None):
        y = frame.target.to_numpy() if target is None else np.asarray(target)
        if self.algorithm.startswith("blend:"):
            _, tree, fraction = self.algorithm.split(":")
            self.tree_weight = float(fraction)
            self.children = [DemandModel("hgb", self.features).fit(frame, y),
                             DemandModel(tree, self.features).fit(frame, y)]
            return self
        if self.algorithm == "hgb":
            self.estimators = [HistGradientBoostingRegressor(
                max_iter=150, max_leaf_nodes=15, min_samples_leaf=leaf,
                learning_rate=.05, l2_regularization=10, random_state=42,
                early_stopping=False) for leaf in (20, 50)]
        elif self.algorithm.startswith("extra_"):
            leaf = int(self.algorithm.split("_")[1])
            self.estimators = [ExtraTreesRegressor(n_estimators=240, min_samples_leaf=leaf,
                                                   max_features=1.0, random_state=42, n_jobs=2)]
        elif self.algorithm == "forest":
            self.estimators = [RandomForestRegressor(n_estimators=240, min_samples_leaf=3,
                                                     max_features=.8, random_state=42, n_jobs=2)]
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        for estimator in self.estimators:
            estimator.fit(frame[self.features], y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.algorithm.startswith("blend:"):
            return ((1 - self.tree_weight) * self.children[0].predict(frame)
                    + self.tree_weight * self.children[1].predict(frame))
        return np.maximum(0, np.array([e.predict(frame[self.features]) for e in self.estimators])).mean(axis=0)


def apply_gate(clean: pd.DataFrame, frame: pd.DataFrame, prediction: np.ndarray, enabled: bool) -> np.ndarray:
    return np.where(idle_mask(clean, frame, 6, 25), frame.power_lag_1, prediction) if enabled else prediction


def selection_scores(oof: pd.DataFrame, names: list[str]) -> dict:
    selection = oof.loc[oof.index < "2021-07-01"]
    errors = selection[names].sub(selection.actual, axis=0).abs()
    monthly = errors.groupby(errors.index.strftime("%Y-%m")).mean()
    return {name: float(monthly[name].mean()) for name in names}


def evaluate_configs(data: Path, output: Path, configs: dict, stage: int) -> dict:
    clean, _ = read_data(data)
    frames = {group: richer_features(clean, group) for group in {c["features"] for c in configs.values()}}
    oof_parts = []
    for start, end in FOLDS:
        predictions = {}
        cache = {}
        for name, config in configs.items():
            frame, features = frames[config["features"]]
            train = frame.loc[frame.index < start]
            valid = frame.loc[(frame.index >= start) & (frame.index < end)]
            key = (config["features"], config["model"])
            if key not in cache:
                cache[key] = DemandModel(config["model"], features).fit(train).predict(valid)
            predictions[name] = apply_gate(clean, valid, cache[key], config["gate"])
        result = pd.DataFrame(predictions, index=valid.index)
        result.insert(0, "actual", valid.target)
        oof_parts.append(result)
        print(f"Stage {stage}: evaluated {start[:7]}", flush=True)
    oof = pd.concat(oof_parts)
    return finish_experiment(data, output, clean, frames, oof, configs, stage)


def finish_experiment(data: Path, output: Path, clean: pd.DataFrame, frames: dict,
                       oof: pd.DataFrame, configs: dict, stage: int) -> dict:
    scores = selection_scores(oof, list(configs))
    chosen = min(scores, key=scores.get)
    config = configs[chosen]
    output.mkdir(parents=True, exist_ok=True)
    lock = {"stage": stage, "selection_months": ["2021-04", "2021-05", "2021-06"],
            "selection_metric": "equal-weighted monthly MAE", "candidate_scores": scores,
            "chosen": chosen, "config": config, "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
    (output / "selection.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Stage {stage}: locked {chosen} before August-September evaluation", flush=True)
    frame, features = frames[config["features"]]
    parts = []
    for start, end in [("2021-08-01", "2021-09-01"), ("2021-09-01", "2021-09-15")]:
        train = frame.loc[frame.index < start]
        test = frame.loc[(frame.index >= start) & (frame.index < end)]
        prediction = DemandModel(config["model"], features).fit(train).predict(test)
        prediction = apply_gate(clean, test, prediction, config["gate"])
        parts.append(pd.DataFrame({"actual": test.target, "prediction": prediction,
                                   "model_fit_before": start}, index=test.index))
    evaluated = pd.concat(parts)
    july = oof.loc[oof.index >= "2021-07-01"]
    report = {**lock, "features": features, "configs": configs,
              "july_audit": {n: regression_metrics(july.actual.to_numpy(), july[n].to_numpy()) for n in configs},
              "evaluation": regression_metrics(evaluated.actual.to_numpy(), evaluated.prediction.to_numpy()),
              "by_month": {str(m): regression_metrics(g.actual.to_numpy(), g.prediction.to_numpy())
                           for m, g in evaluated.groupby(evaluated.index.strftime("%Y-%m"))},
              "after_august_first_week": regression_metrics(
                  evaluated.loc["2021-08-08":, "actual"].to_numpy(), evaluated.loc["2021-08-08":, "prediction"].to_numpy()),
              "note": "Retrospective study on previously inspected data. July is a separate diagnostic, not pristine external validation."}
    oof.to_csv(output / "validation_predictions.csv", index_label="timestamp")
    evaluated.to_csv(output / "predictions.csv", index_label="timestamp")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def evaluate_blends(data: Path, root: Path) -> dict:
    previous = json.loads((root / "stage10/summary.json").read_text())
    if previous["input_sha256"] != hashlib.sha256(data.read_bytes()).hexdigest():
        raise ValueError("Stage 10 used a different input dataset")
    candidates = [n for n, c in previous["configs"].items() if c["model"] != "hgb"]
    tree_name = min(candidates, key=lambda name: previous["candidate_scores"][name])
    config = previous["configs"][tree_name]
    hgb_name = "hgb_idle" if config["gate"] else "hgb"
    source = pd.read_csv(root / "stage10/validation_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
    oof = source[["actual"]].copy()
    configs = {}
    for weight in (0., .25, .5, .75, 1.):
        name = f"tree_weight_{int(100 * weight)}"
        oof[name] = (1 - weight) * source[hgb_name] + weight * source[tree_name]
        algorithm = "hgb" if weight == 0 else config["model"] if weight == 1 else f"blend:{config['model']}:{weight}"
        configs[name] = {**config, "model": algorithm}
    clean, _ = read_data(data)
    frames = {config["features"]: richer_features(clean, config["features"])}
    return finish_experiment(data, root / "stage11", clean, frames, oof, configs, 11)


def stage_configs(stage: int, root: Path) -> dict:
    if stage == 9:
        return {group: {"features": group, "model": "hgb", "gate": True}
                for group in ("base", "quarter", "regime", "combined")}
    if stage == 10:
        previous = json.loads((root / "stage9/summary.json").read_text())
        group = previous["config"]["features"]
        return {f"{model}{'_idle' if gate else ''}": {"features": group, "model": model, "gate": gate}
                for model in ("hgb", "extra_2", "extra_8", "forest") for gate in (False, True)}
    raise ValueError("Unknown experiment stage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[9, 10, 11], required=True)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.stage == 11:
        evaluate_blends(args.data, args.output_root)
    else:
        configs = stage_configs(args.stage, args.output_root)
        evaluate_configs(args.data, args.output_root / f"stage{args.stage}", configs, args.stage)
