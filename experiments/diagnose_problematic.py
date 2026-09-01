"""Generate internal figures for predeclared numerically suspicious solves.

This is an explicit diagnostic runner, not a reporting module. It intentionally
re-solves only the cells listed under ``problem_diagnostics`` in the manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.prototype import (
    certify_outcome_support,
    classify,
    find_best_deviation,
    make_problem,
    reservation_utility,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _effective_case(case: dict[str, Any], numerics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    support = certify_outcome_support(case, numerics)
    effective = copy.deepcopy(case)
    effective["outcome_grid"] = copy.deepcopy(support["effective_outcome_grid"])
    return effective, support


def _select_action(mhp, utility_cfg, case, numerics, diagnostic):
    wage = float(diagnostic["reservation_wage"])
    ru = reservation_utility(utility_cfg, wage)
    lb, ub = map(float, case["action_bounds"])
    mode = diagnostic["selection"]
    if mode == "fixed":
        return float(diagnostic.get("intended_action", case["fixed_action"])), None, []
    iterations = 0 if mode == "principal_relaxed" else int(numerics["full_ic_iterations"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solution = mhp.solve_principal_problem(
            revenue_function=lambda action: action,
            reservation_utility=ru,
            a_min=lb,
            a_max=ub,
            a_ic_lb=lb,
            a_ic_ub=ub,
            n_a_iterations=iterations,
        )
    return float(solution.optimal_action), solution, sorted({str(item.message) for item in caught})


def _solve_contracts(mhp, utility_cfg, case, numerics, action, wage):
    ru = reservation_utility(utility_cfg, wage)
    lb, ub = map(float, case["action_bounds"])
    a_hat = np.unique(np.append(
        np.linspace(lb, ub, int(numerics["validation"]["cvxpy_action_points"])), action
    ))
    specs = {
        "active_relaxed": lambda: mhp.solve_cost_minimization_problem(
            intended_action=action, reservation_utility=ru, a_ic_lb=lb, a_ic_ub=ub,
            n_a_iterations=0,
        ),
        "active_full": lambda: mhp.solve_cost_minimization_problem(
            intended_action=action, reservation_utility=ru, a_ic_lb=lb, a_ic_ub=ub,
            n_a_iterations=int(numerics["full_ic_iterations"]),
        ),
        "cvxpy_relaxed": lambda: mhp.solve_cost_minimization_problem_cvxpy(
            intended_action=action, reservation_utility=ru, a_hat=np.array([]),
        ),
        "cvxpy_full": lambda: mhp.solve_cost_minimization_problem_cvxpy(
            intended_action=action, reservation_utility=ru, a_hat=a_hat,
        ),
    }
    contracts: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for name, solve in specs.items():
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = solve()
            warning_messages = sorted({str(item.message) for item in caught})
            if name.startswith("active"):
                contract = np.asarray(result.optimal_contract, dtype=float)
                expected_wage = float(result.constraints["Ewage"])
                solver_status = str(result.solver_state.get("status", "unknown"))
            else:
                contract_value = result.get("optimal_contract")
                contract = None if contract_value is None else np.asarray(contract_value, dtype=float)
                expected_wage = result.get("expected_wage")
                solver_status = str(result["status"])
            record: dict[str, Any] = {
                "solver_status": solver_status,
                "expected_wage": expected_wage,
                "warnings": warning_messages,
            }
            if contract is not None and np.all(np.isfinite(contract)):
                contracts[name] = contract
                deviation = find_best_deviation(
                    mhp, utility_cfg, contract, action, (lb, ub),
                    int(numerics["deviation"]["coarse_action_points"]),
                    int(numerics["deviation"].get("multistart_count", 12)),
                )
                record.update({
                    "classification": classify(
                        deviation.ce_gain,
                        float(numerics["deviation"]["valid_tolerance_ce"]),
                        float(numerics["deviation"]["invalid_tolerance_ce"]),
                    ),
                    "deviation": asdict(deviation),
                })
            else:
                record["error"] = "No finite contract returned"
            metadata[name] = record
        except Exception as error:
            metadata[name] = {
                "error_type": type(error).__name__,
                "error": str(error),
            }
    return contracts, metadata


def _plot_contract_diagnostic(path, mhp, utility_cfg, case, action, contracts, title):
    lb, ub = map(float, case["action_bounds"])
    action_grid = np.linspace(lb, ub, 721)
    figure, axes = plt.subplots(3, 1, figsize=(9, 11), constrained_layout=True)

    for name, contract in contracts.items():
        utility_values = np.asarray(mhp.U(contract, action_grid), dtype=float)
        intended_utility = float(np.asarray(mhp.U(contract, action)).reshape(()))
        axes[0].plot(action_grid, utility_values - intended_utility, label=name)
        axes[1].plot(mhp.y_grid, contract, label=name)
        with np.errstate(over="ignore", invalid="ignore"):
            wages = np.asarray(mhp.k(contract), dtype=float)
        finite = np.isfinite(wages)
        axes[2].plot(mhp.y_grid[finite], wages[finite], label=name)

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axvline(action, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="Action", ylabel="Utility relative to intended action")
    axes[1].set(xlabel="Outcome", ylabel="Contract in utility units")
    axes[2].set(xlabel="Outcome", ylabel="Wage", yscale="symlog")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_support_diagnostic(path, case, support, title):
    history = support["history"]
    expansions = [row["expansion"] for row in history]
    omitted = [row["diagnostics"]["omitted_mass"] for row in history]
    score_means = [abs(row["diagnostics"]["score_mean"]) for row in history]
    bounds = [row["outcome_grid"]["y_max"] - row["outcome_grid"]["y_min"] for row in history]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
    axes[0].semilogy(expansions, omitted, marker="o", label="omitted mass")
    axes[0].semilogy(expansions, score_means, marker="s", label="absolute score mean")
    axes[0].set(xlabel="Support expansion", ylabel="Diagnostic magnitude")
    axes[0].legend()
    axes[1].plot(expansions, bounds, marker="o")
    axes[1].set(xlabel="Support expansion", ylabel="Outcome support width")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_diagnostic(manifest, diagnostic, output_root: Path) -> dict[str, Any]:
    cases = {case["id"]: case for case in manifest["cases"]}
    case = cases[diagnostic["case_id"]]
    numerics = manifest["numerics"]
    effective, support = _effective_case(case, numerics)
    output = output_root / diagnostic["id"]
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "diagnostic": diagnostic,
        "original_configuration": case,
        "effective_configuration": effective,
        "support_validation": support,
    }

    figures: list[str] = []
    if diagnostic["kind"] == "support":
        figure_name = "support_convergence.png"
        _plot_support_diagnostic(output / figure_name, case, support, diagnostic["id"])
        figures.append(figure_name)
    else:
        mhp, utility_cfg = make_problem(effective)
        action, selected_solution, selection_warnings = _select_action(
            mhp, utility_cfg, effective, numerics, diagnostic
        )
        wage = float(diagnostic["reservation_wage"])
        contracts, solves = _solve_contracts(
            mhp, utility_cfg, effective, numerics, action, wage
        )
        metadata.update({
            "selected_action": action,
            "selection_profit": None if selected_solution is None else float(selected_solution.profit),
            "selection_warnings": selection_warnings,
            "solves": solves,
        })
        figure_name = "contracts_and_agent_objective.png"
        _plot_contract_diagnostic(
            output / figure_name, mhp, utility_cfg, effective,
            action, contracts, diagnostic["id"],
        )
        figures.append(figure_name)
        np.savez_compressed(
            output / "diagnostic_arrays.npz",
            outcome_grid=mhp.y_grid,
            action_grid=np.linspace(*map(float, effective["action_bounds"]), 721),
            **{f"contract_{name}": contract for name, contract in contracts.items()},
        )

    (output / "metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, allow_nan=False)
    )
    return {
        "id": diagnostic["id"],
        "case_id": diagnostic["case_id"],
        "kind": diagnostic["kind"],
        "output": str(output),
        "figures": figures,
        "support_status": support["status"],
        "status": "completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments/foa_experiments.yaml")
    parser.add_argument("--output", default="output/foa-problem-diagnostics")
    parser.add_argument("--id", action="append", dest="ids", help="Run only this diagnostic ID; repeatable")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    diagnostics = manifest.get("problem_diagnostics", [])
    if args.ids:
        requested = set(args.ids)
        diagnostics = [item for item in diagnostics if item["id"] in requested]
        missing = requested - {item["id"] for item in diagnostics}
        if missing:
            raise ValueError(f"Unknown diagnostic IDs: {sorted(missing)}")
    output_root = Path(args.output)
    results = [run_diagnostic(manifest, item, output_root) for item in diagnostics]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.json").write_text(json.dumps(results, indent=2))
    readme_lines = [
        "# FOA Problem Diagnostics",
        "",
        "Internal numerical diagnostics only; these are not paper figures.",
        "",
    ]
    for result in results:
        readme_lines.extend([
            f"## {result['id']}",
            "",
            f"- Case: `{result['case_id']}`",
            f"- Support status: `{result['support_status']}`",
            f"- Metadata: [`metadata.json`]({result['id']}/metadata.json)",
            *[
                f"- Figure: [`{figure}`]({result['id']}/{figure})"
                for figure in result["figures"]
            ],
            "",
        ])
    (output_root / "README.md").write_text("\n".join(readme_lines))
    print(f"Completed {len(results)} diagnostic(s) under {args.output}")
    for result in results:
        print(f"  {result['id']}: {result['output']}")


if __name__ == "__main__":
    main()
