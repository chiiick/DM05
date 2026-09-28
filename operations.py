"""Stage 18: constrained, explicitly hypothetical load-shifting scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from deploy import digest
from forecast import POWER_COLUMNS, read_data

FRACTIONS = (0., .05, .10, .20)


def plan_load(predicted, fraction, max_shift=2, allowed_hours=tuple(range(7, 20)), arrival_fraction=.10):
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != (24,) or not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Expected 24 finite nonnegative hourly mean forecasts")
    if not 0 <= fraction <= 1 or not 0 <= arrival_fraction <= 1 or max_shift < 0:
        raise ValueError("Invalid flexibility constraints")
    allowed = sorted(set(allowed_hours))
    if any(h < 0 or h > 23 for h in allowed):
        raise ValueError("Allowed hours must be within the same day")
    edges = [(i, j) for i in allowed for j in allowed if 0 < abs(i-j) <= max_shift]
    zero = {"outgoing": np.zeros(24), "incoming": np.zeros(24), "adjusted": predicted.copy(), "flows": []}
    if fraction == 0 or not edges or predicted.max() == 0:
        return zero
    n = len(edges)
    net = np.zeros((24, n))
    outgoing, incoming = np.zeros((24, n)), np.zeros((24, n))
    for k, (source, target) in enumerate(edges):
        net[source, k], net[target, k] = -1., 1.
        outgoing[source, k], incoming[target, k] = 1., 1.
    constraints = np.vstack([np.column_stack([net, -np.ones(24)]),
                             np.column_stack([outgoing, np.zeros(24)]),
                             np.column_stack([incoming, np.zeros(24)])])
    limits = np.r_[-predicted, fraction * predicted, np.full(24, arrival_fraction * predicted.max())]
    objective = np.r_[np.zeros(n), 1.]
    optimum = linprog(objective, A_ub=constraints, b_ub=limits, bounds=(0, None), method="highs")
    if not optimum.success:
        raise RuntimeError(f"Planning optimization failed: {optimum.message}")
    # Among equal peak solutions choose minimum movement; avoid needless cycles.
    bounds = [(0, None)] * n + [(0, optimum.x[-1] + 1e-7)]
    parsimonious = linprog(np.r_[np.ones(n), 0.], A_ub=constraints, b_ub=limits,
                           bounds=bounds, method="highs")
    if not parsimonious.success:
        raise RuntimeError(f"Minimum-movement optimization failed: {parsimonious.message}")
    values = parsimonious.x[:n]
    return {"outgoing": outgoing @ values, "incoming": incoming @ values,
            "adjusted": predicted + net @ values,
            "flows": [{"from_hour": i, "to_hour": j, "amount": float(v)}
                      for (i, j), v in zip(edges, values) if v > 1e-7]}


def condition_metrics(frame, column):
    return {str(key): {"n_hours": len(g), "peak_hours": int(g.peak.sum()),
                       "peak_rate": float(g.peak.mean()), "quarter_max": float(g.quarter_actual.max())}
            for key, g in frame.groupby(column)}


def peak_conditions(clean, root):
    legacy = pd.read_csv(root / "test_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
    frame = legacy[["quarter_actual", "peak_classifier_alert"]].copy()
    frame["peak"] = frame.quarter_actual >= 182
    frame["hour"] = frame.index.hour
    frame["weekday"] = frame.index.dayofweek
    production = clean["생산량"].asfreq("h")
    previous = production.shift(1).reindex(frame.index).gt(0)
    current = production.reindex(frame.index).gt(0)
    frame["prior_active"] = previous
    frame["observed_state"] = np.select([current & ~previous, ~current & previous, current & previous],
                                         ["startup", "shutdown", "running"], default="idle")
    starts = frame.peak & ~frame.peak.shift(1, fill_value=False)
    episodes = []
    for _, g in frame.loc[frame.peak].groupby(starts.cumsum().loc[frame.peak]):
        start = g.index.min()
        episodes.append({"start": str(start), "end": str(g.index.max()), "duration_hours": len(g),
                         "quarter_max": float(g.quarter_actual.max()),
                         "one_hour_policy_alert_at_onset": bool(frame.loc[start, "peak_classifier_alert"])})
    return {"by_known_target_hour": condition_metrics(frame, "hour"),
            "by_known_weekday": condition_metrics(frame, "weekday"),
            "by_previous_hour_production": condition_metrics(frame, "prior_active"),
            "by_observed_production_state": condition_metrics(frame, "observed_state"),
            "episodes": episodes, "episode_count": len(episodes),
            "alerted_at_onset": sum(e["one_hour_policy_alert_at_onset"] for e in episodes),
            "note": "Hour/weekday known at planning time; previous-hour production known only one hour ahead; same-hour state is post-hoc. Associations, not causal explanations. Episode onset alerts are issued after the preceding hour is observed."}


def evaluate_scenarios(predictions):
    rows, plans, flow_rows = [], [], []
    for day, group in predictions.groupby("anchor_time"):
        group = group.sort_values("horizon")
        if len(group) != 24 or not np.array_equal(group.horizon, np.arange(1, 25)):
            raise ValueError("Scenario requires a complete forecast from one daily origin")
        forecast, actual = group.prediction.to_numpy(), group.actual.to_numpy()
        for fraction in FRACTIONS:
            plan = plan_load(forecast, fraction)
            simulated = actual - plan["outgoing"] + plan["incoming"]
            # Check the claimed flexible share against actual load after decisions.
            violation = np.maximum(plan["outgoing"] - fraction * actual, 0.)
            feasible = bool(violation.max() <= 1e-6 and simulated.min() >= -1e-6)
            rows.append({"date": str(day.date()), "fraction": fraction,
                         "forecast_peak_before": float(forecast.max()), "forecast_peak_after": float(plan["adjusted"].max()),
                         "observed_peak_before": float(actual.max()), "simulated_peak_after": float(simulated.max()),
                         "simulated_peak_change": float(simulated.max() - actual.max()),
                         "feasible_under_assumed_share": feasible,
                         "has_movement": bool(plan["outgoing"].sum() > 1e-6),
                         "share_violation_sum": float(violation.sum()),
                         "moved_load_sum": float(plan["outgoing"].sum())})
            detail = pd.DataFrame({"timestamp": group.timestamp.to_numpy(), "fraction": fraction,
                                   "forecast": forecast, "planned_forecast": plan["adjusted"], "actual": actual,
                                   "outgoing": plan["outgoing"], "incoming": plan["incoming"],
                                   "simulated_actual": simulated, "share_violation": violation})
            plans.append(detail)
            flow_rows.extend([{"date": str(day.date()), "fraction": fraction, **flow} for flow in plan["flows"]])
    daily = pd.DataFrame(rows)
    summaries = {}
    for fraction, group in daily.groupby("fraction"):
        feasible = group.loc[group.feasible_under_assumed_share]
        feasible_moved = feasible.loc[feasible.has_movement]
        summaries[str(fraction)] = {"n_days": len(group), "feasible_days": len(feasible),
            "planned_move_days": int(group.has_movement.sum()), "feasible_moved_days": len(feasible_moved),
            "forecast_mean_peak_reduction": float((group.forecast_peak_before - group.forecast_peak_after).mean()),
            "arithmetic_simulated_mean_peak_reduction_all_proposals": float((-group.simulated_peak_change).mean()),
            "simulated_mean_peak_reduction_feasible_only": float((-feasible.simulated_peak_change).mean()) if len(feasible) else None,
            "simulated_mean_peak_reduction_feasible_moved_only": float((-feasible_moved.simulated_peak_change).mean()) if len(feasible_moved) else None,
            "simulated_rebound_days_all_proposals": int((group.simulated_peak_change > 1e-6).sum()),
            "simulated_rebound_days_feasible_only": int((feasible.simulated_peak_change > 1e-6).sum()),
            "assumed_share_violation_days": int((~group.feasible_under_assumed_share).sum()),
            "mean_moved_load_sum": float(group.moved_load_sum.mean())}
    return daily, pd.concat(plans, ignore_index=True), pd.DataFrame(flow_rows, columns=["date", "fraction", "from_hour", "to_hour", "amount"]), summaries


def run(data, root):
    selection = json.loads((root / "stage17/selection.json").read_text())
    if selection["input_sha256"] != digest(data):
        raise ValueError("Data does not match the recorded daily forecast experiment")
    clean, _ = read_data(data)
    predictions = pd.read_csv(root / "stage17/predictions.csv", parse_dates=["anchor_time", "timestamp"])
    output = root / "stage18"
    output.mkdir(parents=True, exist_ok=True)
    daily, plans, flows, summaries = evaluate_scenarios(predictions)
    conditions = peak_conditions(clean, root)
    assumptions = {"flexible_fractions": FRACTIONS, "max_shift_hours": 2, "allowed_hours": list(range(7, 20)),
                   "arrival_limit_fraction_of_forecast_daily_peak": .10,
                   "objective": "Minimize forecast hourly-mean maximum, then minimize movement",
                   "units": "Unconfirmed source units. Sum of hourly mean values is a load proxy, not asserted kWh.",
                   "limitation": "Assumed divisible flexible load, perfect execution and additive response. No real job schedule, equipment constraints, production feasibility or tariff. Not causal savings or 15-minute demand reduction."}
    report = {"input_sha256": digest(data), "forecast_source_sha256": digest(root / "stage17/predictions.csv"),
              "daily_forecast_method": selection["chosen"],
              "assumptions": assumptions, "scenarios": summaries,
              "peak_conditions": conditions, "deployment_status": "Scenario decision support only; no automatic operational changes"}
    daily.to_csv(output / "daily_scenarios.csv", index=False)
    plans.to_csv(output / "hourly_scenarios.csv", index=False)
    flows.to_csv(output / "load_transfers.csv", index=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    future = pd.read_csv(root / "stage17/next_day_prediction.csv", parse_dates=["timestamp"])
    future_plans, future_flows = [], []
    for fraction in FRACTIONS:
        plan = plan_load(future.predicted_mean.to_numpy(), fraction)
        detail = future.copy()
        detail = detail.rename(columns={"peak_alert": "unmodified_plan_peak_alert",
                                        "predicted_quarter_max": "unmodified_plan_predicted_quarter_max"})
        detail["scenario_status"] = "assumptions_only"
        detail["assumed_flexible_fraction"] = fraction
        detail["planned_mean"] = plan["adjusted"]
        detail["outgoing"] = plan["outgoing"]
        detail["incoming"] = plan["incoming"]
        future_plans.append(detail)
        future_flows.extend([{"date": str(future.timestamp.iloc[0].date()),
                              "assumed_flexible_fraction": fraction, "scenario_status": "assumptions_only", **flow}
                             for flow in plan["flows"]])
    pd.concat(future_plans).to_csv(output / "next_day_scenarios.csv", index=False)
    pd.DataFrame(future_flows, columns=["date", "assumed_flexible_fraction", "scenario_status", "from_hour", "to_hour", "amount"]).to_csv(output / "next_day_transfers.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    for fraction in FRACTIONS[1:]:
        part = daily.loc[daily.fraction == fraction]
        axes[0].plot(pd.to_datetime(part.date), -part.simulated_peak_change, label=f"{fraction:.0%} assumed flexibility")
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set(title="Hypothetical hourly-mean peak reduction (all proposals, including infeasible)", ylabel="Source units; below 0 = rebound")
    axes[0].legend()
    future_plan = future_plans[2]
    axes[1].plot(future.timestamp, future.predicted_mean, label="Forecast")
    axes[1].plot(future.timestamp, future_plan.planned_mean, label="10% flexibility scenario")
    axes[1].set(title="Next-day scenario: availability of flexible load must be confirmed", ylabel="Hourly mean (source units)")
    axes[1].legend()
    fig.savefig(output / "operating_scenarios.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"scenarios": summaries, "episodes": conditions["episode_count"],
                      "alerted_at_onset": conditions["alerted_at_onset"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("okm_augumented_2021.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.data, args.output_root)
