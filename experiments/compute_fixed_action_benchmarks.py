"""Compute full-GIC monopsony and zero-profit benchmarks at fixed actions."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .prototype import (
    ce_wage,
    classify,
    expected_revenue,
    find_best_deviation,
    make_problem,
    reservation_utility,
)


def _solve(
    mhp: Any,
    utility_cfg: dict[str, Any],
    case: dict[str, Any],
    numerics: dict[str, Any],
    wage: float,
) -> dict[str, Any]:
    action = float(case["fixed_action"])
    action_lb, action_ub = map(float, case["action_bounds"])
    ru = reservation_utility(utility_cfg, wage)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mhp.solve_cost_minimization_problem(
            intended_action=action,
            reservation_utility=ru,
            a_ic_lb=action_lb,
            a_ic_ub=action_ub,
            n_a_iterations=int(numerics["full_ic_iterations"]),
            a_always_check_global_ic=np.array([action_lb, action_ub]),
        )
    contract = result.optimal_contract
    deviation = find_best_deviation(
        mhp,
        utility_cfg,
        contract,
        action,
        (action_lb, action_ub),
        int(numerics["deviation"]["coarse_action_points"]),
        int(numerics["deviation"].get("multistart_count", 12)),
    )
    delivered_utility = float(np.asarray(mhp.U(contract, action)).reshape(()))
    delivered_ce = float(ce_wage(utility_cfg, delivered_utility))
    expected_wage = float(result.constraints["Ewage"])
    revenue = float(expected_revenue(case, action))
    return {
        "reservation_wage": float(wage),
        "action": action,
        "revenue": revenue,
        "expected_wage": expected_wage,
        "profit": revenue - expected_wage,
        "delivered_ce_wage": delivered_ce,
        "ir_multiplier": float(np.asarray(result.multipliers.get("lam", math.nan)).reshape(())),
        "ir_slack_ce": delivered_ce - float(wage),
        "deviation_ce_gain": float(deviation.ce_gain),
        "global_ic_classification": classify(
            deviation.ce_gain,
            float(numerics["deviation"]["valid_tolerance_ce"]),
            float(numerics["deviation"]["invalid_tolerance_ce"]),
        ),
        "warnings": sorted({str(item.message) for item in caught}),
    }


def _global_ic_status(rows: list[dict[str, Any]]) -> str:
    classifications = {row["global_ic_classification"] for row in rows}
    if "invalid" in classifications:
        return "failed"
    if "unresolved" in classifications:
        return "unresolved"
    return "passed"


def fixed_action_benchmarks(
    case: dict[str, Any],
    numerics: dict[str, Any],
    *,
    initial_upper: float,
    wage_tolerance: float,
    max_expansions: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Compute slack-IR and zero-profit full-GIC benchmarks at one action."""
    mhp, utility_cfg = make_problem(case)
    cache: dict[float, dict[str, Any]] = {}

    def solve(wage: float) -> dict[str, Any]:
        key = float(wage)
        if key not in cache:
            cache[key] = _solve(mhp, utility_cfg, case, numerics, key)
        return cache[key]

    monopsony_history = []
    selected = None
    candidates = [float(x) for x in numerics["monopsony"]["candidate_reservation_wages"]]
    max_extensions = int(numerics["monopsony"].get("max_downward_extensions", 0))
    downward_step = float(numerics["monopsony"].get("downward_step", 10.0))
    minimum_wage = float(numerics["monopsony"].get("minimum_reservation_wage", -math.inf))
    extensions = 0
    index = 0
    while index < len(candidates):
        row = solve(candidates[index])
        index += 1
        monopsony_history.append(row)
        slack = (
            row["ir_multiplier"] <= float(numerics["monopsony"]["lambda_tolerance"])
            and row["ir_slack_ce"] >= float(numerics["monopsony"].get("ir_slack_ce_tolerance", 0.0))
            and row["global_ic_classification"] != "invalid"
        )
        if slack:
            previous = monopsony_history[-2] if len(monopsony_history) >= 2 else None
            previous_slack = previous is not None and (
                previous["ir_multiplier"] <= float(numerics["monopsony"]["lambda_tolerance"])
                and previous["ir_slack_ce"]
                >= float(numerics["monopsony"].get("ir_slack_ce_tolerance", 0.0))
                and previous["global_ic_classification"] != "invalid"
            )
            selected = row
            if previous_slack:
                break
        if index == len(candidates) and extensions < max_extensions:
            next_wage = max(minimum_wage, candidates[-1] - downward_step)
            if next_wage < candidates[-1]:
                candidates.append(next_wage)
                extensions += 1

    if selected is None:
        monopsony = {"status": "not_found", "history": monopsony_history}
    else:
        # If numerically equivalent slack-IR solves differ slightly, retain the
        # least-cost globally feasible candidate as the economic benchmark.
        slack_rows = [
            row for row in monopsony_history
            if row["ir_multiplier"] <= float(numerics["monopsony"]["lambda_tolerance"])
            and row["ir_slack_ce"]
            >= float(numerics["monopsony"].get("ir_slack_ce_tolerance", 0.0))
            and row["global_ic_classification"] != "invalid"
        ]
        selected = min(slack_rows, key=lambda row: row["expected_wage"])
        previous_slack = next((
            row for row in reversed(monopsony_history)
            if row is not selected
            and row["ir_multiplier"] <= float(numerics["monopsony"]["lambda_tolerance"])
            and row["ir_slack_ce"]
            >= float(numerics["monopsony"].get("ir_slack_ce_tolerance", 0.0))
            and row["global_ic_classification"] != "invalid"
        ), None)
        stable = previous_slack is not None and (
            abs(selected["expected_wage"] - previous_slack["expected_wage"])
            <= float(numerics["monopsony"].get("expected_wage_tolerance", numerics["monopsony"]["ce_tolerance"]))
            and abs(selected["delivered_ce_wage"] - previous_slack["delivered_ce_wage"])
            <= float(numerics["monopsony"]["ce_tolerance"])
        )
        if not stable:
            status = "unverified_plateau"
        else:
            gic = _global_ic_status([selected, previous_slack])
            status = "ok" if gic == "passed" else f"{gic}_global_ic"
        monopsony = {
            "status": status,
            "selected": selected,
            "history": monopsony_history,
        }

    lower_wage = float(numerics["monopsony"]["candidate_reservation_wages"][0])
    lower = solve(lower_wage)
    if lower["profit"] < 0:
        competitive = {
            "status": "no_nonnegative_profit_bracket",
            "lower": lower,
        }
    else:
        upper_wage = max(float(initial_upper), 0.0, lower_wage + 1.0)
        upper = solve(upper_wage)
        expansions = 0
        while upper["profit"] > 0 and expansions < max_expansions:
            span = upper_wage - lower_wage
            upper_wage += max(span, 10.0)
            upper = solve(upper_wage)
            expansions += 1
        if upper["profit"] > 0:
            competitive = {
                "status": "not_bracketed",
                "lower": lower,
                "upper": upper,
                "expansions": expansions,
            }
        else:
            iterations = 0
            while upper_wage - lower_wage > wage_tolerance and iterations < max_iterations:
                midpoint_wage = (lower_wage + upper_wage) / 2
                midpoint = solve(midpoint_wage)
                if midpoint["profit"] >= 0:
                    lower_wage, lower = midpoint_wage, midpoint
                else:
                    upper_wage, upper = midpoint_wage, midpoint
                iterations += 1
            profit_span = lower["profit"] - upper["profit"]
            if profit_span > 0 and math.isfinite(profit_span):
                weight = lower["profit"] / profit_span
                estimate = lower_wage + weight * (upper_wage - lower_wage)
            else:
                estimate = (lower_wage + upper_wage) / 2
            competitive = {
                "status": "ok",
                "global_ic_status": _global_ic_status([lower, upper]),
                "competitive_ce_wage": float(estimate),
                "bracket": [float(lower_wage), float(upper_wage)],
                "profit_bracket": [float(lower["profit"]), float(upper["profit"])],
                "lower": lower,
                "upper": upper,
                "expansions": expansions,
                "iterations": iterations,
                "wage_tolerance": wage_tolerance,
            }

    return {
        "status": "ok" if monopsony.get("selected") is not None else "incomplete",
        "intended_action": float(case["fixed_action"]),
        "monopsony": monopsony,
        "competitive": competitive,
        "history": sorted(cache.values(), key=lambda row: row["reservation_wage"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/foa-internal-atlas-final-v2")
    parser.add_argument("--output")
    parser.add_argument("--wage-tolerance", type=float, default=0.025)
    parser.add_argument("--max-expansions", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=16)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = (
        Path(args.output) if args.output
        else input_dir / "fixed_action_benchmarks.json"
    )
    records = []
    for atomic_path in sorted((input_dir / "atomic").glob("*.json")):
        atomic = json.loads(atomic_path.read_text())
        result = atomic["result"]
        exercise = result.get("exercises", {}).get("fixed_action")
        if exercise is None:
            continue
        record: dict[str, Any]
        if exercise.get("status") == "infeasible_local_incentives":
            record = {
                "status": "infeasible_local_incentives",
                "intended_action": float(result["effective_configuration"]["fixed_action"]),
            }
        else:
            initial_wages = [
                float(point["reservation_wage"])
                for point in exercise.get("points", [])
            ]
            record = fixed_action_benchmarks(
                result["effective_configuration"],
                atomic["numerical_configuration"],
                initial_upper=max(initial_wages, default=100.0),
                wage_tolerance=args.wage_tolerance,
                max_expansions=args.max_expansions,
                max_iterations=args.max_iterations,
            )
        record.update({"case_id": atomic["case_id"], "task_hash": atomic["task_hash"]})
        records.append(record)
        competitive = record.get("competitive", {})
        print(
            f"{atomic['case_id']}: {record['status']}"
            + (
                f"; monopsony={record['monopsony']['selected']['delivered_ce_wage']:.4f}"
                if record.get("monopsony", {}).get("selected") else ""
            )
            + (
                f"; competitive={competitive['competitive_ce_wage']:.4f}"
                if competitive.get("status") == "ok" else ""
            )
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_atlas": str(input_dir),
        "definition": {
            "monopsony": "Delivered CE wage in the minimum-cost full-GIC contract at the fixed action when participation is slack.",
            "competitive": "Reservation CE wage where full-GIC compensation cost at the fixed action equals declared expected revenue.",
        },
        "units": "USD_1000",
        "numerics": {
            "wage_tolerance": args.wage_tolerance,
            "max_expansions": args.max_expansions,
            "max_iterations": args.max_iterations,
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"Saved {len(records)} benchmark(s) to {output_path}")


if __name__ == "__main__":
    main()
