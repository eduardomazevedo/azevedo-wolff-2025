"""Compute and save full-GIC zero-profit competitive-wage benchmarks.

This is an experiment command, not reporting code: it solves models and writes
results that the reporting layer can subsequently consume without rerunning.
"""

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


def _solve(case: dict[str, Any], numerics: dict[str, Any], wage: float) -> dict[str, Any]:
    mhp, utility_cfg = make_problem(case)
    action_lb, action_ub = map(float, case["action_bounds"])
    principal_lb, principal_ub = map(
        float, case.get("principal_intended_action_bounds", case["action_bounds"])
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solution = mhp.solve_principal_problem(
            revenue_function=lambda action: expected_revenue(case, action),
            reservation_utility=reservation_utility(utility_cfg, wage),
            a_min=principal_lb,
            a_max=principal_ub,
            a_ic_lb=action_lb,
            a_ic_ub=action_ub,
            n_a_iterations=int(numerics["full_ic_iterations"]),
            a_always_check_global_ic=np.array([action_lb, action_ub]),
        )
    contract = solution.cmp_result.optimal_contract
    action = float(solution.optimal_action)
    deviation = find_best_deviation(
        mhp,
        utility_cfg,
        contract,
        action,
        (action_lb, action_ub),
        int(numerics["deviation"]["coarse_action_points"]),
        int(numerics["deviation"].get("multistart_count", 12)),
    )
    delivered_utility = float(np.asarray(mhp.U(contract, action)).reshape(-1)[0])
    return {
        "reservation_wage": float(wage),
        "profit": float(solution.profit),
        "action": action,
        "expected_wage": float(solution.cmp_result.constraints["Ewage"]),
        "delivered_ce_wage": float(ce_wage(utility_cfg, delivered_utility)),
        "deviation_ce_gain": float(deviation.ce_gain),
        "global_ic_classification": classify(
            deviation.ce_gain,
            float(numerics["deviation"]["valid_tolerance_ce"]),
            float(numerics["deviation"]["invalid_tolerance_ce"]),
        ),
        "warnings": sorted({str(item.message) for item in caught}),
    }


def competitive_benchmark(
    case: dict[str, Any],
    numerics: dict[str, Any],
    *,
    initial_upper: float,
    wage_tolerance: float,
    profit_tolerance: float,
    max_expansions: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Bracket and bisect the reservation CE wage at which profit is zero."""
    cache: dict[float, dict[str, Any]] = {}

    def solve(wage: float) -> dict[str, Any]:
        key = float(wage)
        if key not in cache:
            cache[key] = _solve(case, numerics, key)
        return cache[key]

    # A sufficiently low requirement is on the monopsony plateau and must have
    # nonnegative maximal profit whenever employment is economically viable.
    lower_wage = float(numerics["monopsony"]["candidate_reservation_wages"][0])
    lower = solve(lower_wage)
    if lower["profit"] < 0:
        return {"status": "no_nonnegative_profit_bracket", "history": list(cache.values())}

    upper_wage = max(float(initial_upper), lower_wage + 1.0)
    upper = solve(upper_wage)
    expansions = 0
    while upper["profit"] > 0 and expansions < max_expansions:
        span = upper_wage - lower_wage
        upper_wage = upper_wage + max(span, 10.0)
        upper = solve(upper_wage)
        expansions += 1
    if upper["profit"] > 0:
        return {
            "status": "not_bracketed",
            "lower": lower,
            "upper": upper,
            "expansions": expansions,
            "history": sorted(cache.values(), key=lambda row: row["reservation_wage"]),
        }

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
    endpoint_classifications = {lower["global_ic_classification"], upper["global_ic_classification"]}
    if "invalid" in endpoint_classifications:
        global_ic_status = "failed"
    elif "unresolved" in endpoint_classifications:
        global_ic_status = "unresolved"
    else:
        global_ic_status = "passed"
    return {
        "status": "ok",
        "global_ic_status": global_ic_status,
        "competitive_ce_wage": float(estimate),
        "bracket": [float(lower_wage), float(upper_wage)],
        "profit_bracket": [float(lower["profit"]), float(upper["profit"])],
        "lower": lower,
        "upper": upper,
        "expansions": expansions,
        "iterations": iterations,
        "wage_tolerance": wage_tolerance,
        "profit_tolerance": profit_tolerance,
        "history": sorted(cache.values(), key=lambda row: row["reservation_wage"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/foa-internal-atlas-final-v2")
    parser.add_argument("--output")
    parser.add_argument("--wage-tolerance", type=float, default=0.025)
    parser.add_argument("--profit-tolerance", type=float, default=0.025)
    parser.add_argument("--max-expansions", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=16)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output) if args.output else input_dir / "competitive_benchmarks.json"
    records: list[dict[str, Any]] = []
    for atomic_path in sorted((input_dir / "atomic").glob("*.json")):
        atomic = json.loads(atomic_path.read_text())
        result = atomic["result"]
        if "principal" not in result.get("exercises", {}):
            continue
        principal = result["exercises"]["principal"]
        initial_wages = [
            float(point["reservation_wage"])
            for point in principal.get("points", [])
            if point.get("profit") is not None
        ]
        if not initial_wages:
            continue
        benchmark = competitive_benchmark(
            result["effective_configuration"],
            atomic["numerical_configuration"],
            initial_upper=max(initial_wages),
            wage_tolerance=args.wage_tolerance,
            profit_tolerance=args.profit_tolerance,
            max_expansions=args.max_expansions,
            max_iterations=args.max_iterations,
        )
        benchmark.update({"case_id": atomic["case_id"], "task_hash": atomic["task_hash"]})
        records.append(benchmark)
        print(
            f"{atomic['case_id']}: {benchmark['status']}"
            + (f"; CE={benchmark['competitive_ce_wage']:.4f}" if benchmark["status"] == "ok" else "")
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_atlas": str(input_dir),
        "definition": "Reservation certainty-equivalent wage at which the optimized full-GIC principal profit is zero.",
        "units": "USD_1000",
        "numerics": {
            "wage_tolerance": args.wage_tolerance,
            "profit_tolerance": args.profit_tolerance,
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
