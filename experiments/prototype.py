"""Small-scale prototype for the FOA experiment plan.

This module deliberately keeps the result schema explicit and independent of
plotting code. It is a prototype: the convex cross-check and safe-region
metrics described in the full plan are not implemented yet.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.optimize import brentq, minimize_scalar

from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_distribution_cfg, make_utility_cfg


@dataclass
class DeviationResult:
    intended_action: float
    intended_utility: float
    best_action: float
    best_utility: float
    utility_gain: float
    ce_gain: float
    local_maxima: list[dict[str, float]]
    diagnostics: dict[str, Any] | None = None


@dataclass
class PointResult:
    exercise: str
    reservation_wage: float
    reservation_utility: float
    intended_action: float
    expected_wage: float
    profit: float | None
    delivered_utility: float
    delivered_ce_wage: float
    ir_multiplier: float
    solver_foa_flag: bool | None
    classification: str
    deviation: DeviationResult
    warnings: list[str]


def _float(value: Any) -> float:
    """Convert scalar NumPy values to JSON-safe Python floats."""
    return float(np.asarray(value).reshape(()))


def make_problem(case: dict[str, Any]) -> tuple[MoralHazardProblem, dict[str, Any]]:
    """Construct one moral-hazard problem from a manifest case."""
    w0 = float(case["initial_wealth"])
    utility = case["utility"]
    if utility["kind"] == "risk_neutral":
        def linear_u(x: Any) -> Any:
            return np.asarray(x) + w0

        def linear_k(uval: Any, xp: Any = np) -> Any:
            return uval - w0

        def linear_link(z: Any) -> Any:
            return np.maximum(np.asarray(z), w0)

        utility_cfg = {"u": linear_u, "k": linear_k, "link_function": linear_link}
    else:
        utility_cfg = make_utility_cfg(
            utility["kind"],
            w0=w0,
            gamma=utility.get("gamma"),
            alpha=utility.get("alpha"),
        )
    distribution = case["distribution"]
    dist_cfg = make_distribution_cfg(distribution["kind"], **distribution.get("params", {}))

    target_action = float(case["target_action"])
    base_theta = 1.0 / target_action / (target_action + w0)
    h = 1e-4
    uprime0 = _float((utility_cfg["u"](h) - utility_cfg["u"](-h)) / (2 * h))
    normalization = case.get("cost_normalization", "paper_log")
    if normalization == "paper_log":
        theta = base_theta
    elif normalization == "local_consumption_equivalent":
        # Preserve c'(target)/u'(wage=0) relative to the paper's log case.
        # Estimate u'(0) accurately without depending on utility internals.
        log_uprime0 = 1.0 / w0
        theta = base_theta * uprime0 / log_uprime0
    else:
        raise ValueError(f"Unknown cost normalization: {normalization}")
    # Gamma and binomial parameterize action as scale/probability, so one unit
    # of action can produce several units of expected output. This explicit
    # factor keeps marginal effort cost comparable to marginal revenue.
    output_slope = float(case.get("cost_output_slope", 1.0))
    cost_scale = float(case.get("cost_scale", 1.0))
    theta *= output_slope * cost_scale
    cost_zero_at_action = float(case.get("cost_zero_at_action", 0.0))
    lower_action_bound = float(case.get("action_bounds", [0.0])[0])
    if not (0.0 <= cost_zero_at_action <= lower_action_bound):
        raise ValueError("cost_zero_at_action must lie between zero and the lower action bound")
    cost_metadata = {
        "primitive_theta": theta,
        "base_paper_log_theta": base_theta,
        "normalization": normalization,
        "output_slope": output_slope,
        "cost_scale": cost_scale,
        "cost_zero_at_action": cost_zero_at_action,
        "target_effort_cost": theta * (target_action**2 - cost_zero_at_action**2) / 2,
        "target_marginal_utility_cost": theta * target_action,
        "target_marginal_ce_cost_at_zero_wage": theta * target_action / uprime0,
    }

    def cost(a: Any) -> Any:
        # A constant shift preserves marginal incentives and the economic scale
        # while allowing positive-domain families to assign zero cost to their
        # lowest feasible action.
        return theta * (np.asarray(a) ** 2 - cost_zero_at_action**2) / 2

    def cost_prime(a: Any) -> Any:
        return theta * np.asarray(a)

    cfg = {
        "problem_params": {
            **utility_cfg,
            **dist_cfg,
            "C": cost,
            "Cprime": cost_prime,
        },
        "computational_params": case["outcome_grid"],
    }
    if utility["kind"] == "cara" or (
        utility["kind"] == "crra" and float(utility.get("gamma", 1.0)) > 1.0
    ):
        utility_upper_bound = 0.0
    else:
        utility_upper_bound = math.inf
    utility_result = {
        **utility_cfg,
        "cost_metadata": cost_metadata,
        "utility_metadata": {
            "kind": utility["kind"],
            "upper_bound": utility_upper_bound,
        },
    }
    return MoralHazardProblem(cfg), utility_result


def reservation_utility(utility_cfg: dict[str, Any], wage: float) -> float:
    return _float(utility_cfg["u"](wage))


def ce_wage(utility_cfg: dict[str, Any], utility: float) -> float:
    return _float(utility_cfg["k"](utility))


def _distribution_moments(
    distribution: dict[str, Any] | None, action: float
) -> tuple[float | None, float | None]:
    """Return analytic raw moments when they exist for a supported family."""
    if not distribution:
        return None, None
    kind = distribution["kind"]
    params = distribution.get("params", {})
    if kind == "gaussian":
        variance = float(params.get("sigma", 1.0)) ** 2
        return action, action * action + variance
    if kind == "poisson":
        return action, action * action + action
    if kind == "exponential":
        return action, 2 * action * action
    if kind == "bernoulli":
        return action, action
    if kind == "geometric":
        return action, 2 * action * action - action
    if kind == "binomial":
        n = float(params.get("n", 1))
        mean = n * action
        return mean, n * action * (1 - action) + mean * mean
    if kind == "gamma":
        n = float(params.get("n", 1))
        return n * action, n * (n + 1) * action * action
    if kind == "student_t":
        nu = float(params.get("nu", 5))
        mean = action if nu > 1 else None
        if nu <= 2:
            return mean, None
        variance = float(params.get("sigma", 1.0)) ** 2 * nu / (nu - 2)
        return mean, action * action + variance
    return None, None


def distribution_diagnostics(
    mhp: MoralHazardProblem,
    action: float,
    *,
    derivative_step: float | None = None,
    distribution: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """Check support truncation, score identities, and action derivatives.

    These are grid diagnostics, not analytic tail certificates.  In particular,
    ``omitted_mass`` includes both genuine omitted tails and quadrature error.
    """
    y = np.asarray(mhp.y_grid, dtype=float)
    weights = np.asarray(mhp.w, dtype=float)
    f0 = np.asarray(mhp.f(y, action), dtype=float)
    score = np.asarray(mhp.score(y, action), dtype=float)
    h = derivative_step or max(1e-5, 1e-4 * max(1.0, abs(action)))
    f_plus = np.asarray(mhp.f(y, action + h), dtype=float)
    f_minus = np.asarray(mhp.f(y, action - h), dtype=float)
    score_plus = np.asarray(mhp.score(y, action + h), dtype=float)
    score_minus = np.asarray(mhp.score(y, action - h), dtype=float)

    fa_identity = f0 * score
    fa_fd = (f_plus - f_minus) / (2 * h)
    score_a_fd = (score_plus - score_minus) / (2 * h)
    faa_identity = f0 * (score * score + score_a_fd)
    faa_fd = (f_plus - 2 * f0 + f_minus) / (h * h)

    def integral(values: np.ndarray) -> float:
        return _float(np.sum(weights * values))

    def relative_error(left: np.ndarray, right: np.ndarray) -> float:
        scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-14)
        return float(np.max(np.abs(left - right)) / scale)

    mass = integral(f0)
    grid_first = integral(y * f0)
    grid_second = integral(y * y * f0)
    expected_first, expected_second = _distribution_moments(distribution, action)
    return {
        "grid_mass": mass,
        "omitted_mass": max(0.0, 1.0 - mass),
        "mass_error": abs(1.0 - mass),
        "grid_first_moment": grid_first,
        "expected_first_moment": expected_first,
        "omitted_first_moment_error": None if expected_first is None else abs(expected_first - grid_first),
        "grid_second_moment": grid_second,
        "expected_second_moment": expected_second,
        "omitted_second_moment_error": None if expected_second is None else abs(expected_second - grid_second),
        "score_mean": integral(f0 * score),
        "fa_relative_error": relative_error(fa_identity, fa_fd),
        "faa_relative_error": relative_error(faa_identity, faa_fd),
        "derivative_step": float(h),
        "left_boundary_mass_or_density": float(f0[0]),
        "right_boundary_mass_or_density": float(f0[-1]),
    }


def local_incentive_capacity(
    mhp: MoralHazardProblem,
    utility_cfg: dict[str, Any],
    action: float,
) -> dict[str, float | bool]:
    """Fast necessary local-implementability check under limited liability."""
    upper = float(utility_cfg["utility_metadata"]["upper_bound"])
    required = _float(mhp.Cprime(action))
    if math.isinf(upper):
        return {
            "action": float(action), "bounded_utility": False,
            "capacity": math.inf, "required_incentive": required,
            "slack": math.inf, "feasible": True,
        }
    utility_at_zero_wage = reservation_utility(utility_cfg, 0.0)
    density = np.asarray(mhp.f(mhp.y_grid, action), dtype=float)
    score = np.asarray(mhp.score(mhp.y_grid, action), dtype=float)
    positive_score_moment = _float(np.sum(mhp.w * density * np.maximum(score, 0.0)))
    capacity = (upper - utility_at_zero_wage) * positive_score_moment
    slack = capacity - required
    return {
        "action": float(action), "bounded_utility": True,
        "utility_upper_bound": upper,
        "utility_at_zero_wage": utility_at_zero_wage,
        "positive_score_moment": positive_score_moment,
        "capacity": capacity,
        "required_incentive": required,
        "slack": slack,
        "feasible": bool(slack > 0.0),
    }


def incentive_capacity_precheck(
    mhp: MoralHazardProblem,
    utility_cfg: dict[str, Any],
    action_bounds: tuple[float, float],
    *,
    grid_points: int = 257,
    safety_fraction: float = 1e-3,
) -> dict[str, Any]:
    """Find the highest locally feasible intended action for bounded utility."""
    lb, ub = map(float, action_bounds)
    upper_check = local_incentive_capacity(mhp, utility_cfg, ub)
    if not upper_check["bounded_utility"] or upper_check["feasible"]:
        return {
            "status": "all_actions_locally_feasible",
            "configured_action_bounds": [lb, ub],
            "highest_feasible_action": ub,
            "computational_action_upper_bound": ub,
            "upper_action_check": upper_check,
        }

    grid = np.linspace(lb, ub, max(3, int(grid_points)))
    slacks = np.array([
        float(local_incentive_capacity(mhp, utility_cfg, action)["slack"])
        for action in grid
    ])
    feasible = slacks > 0.0
    feasible_indices = np.flatnonzero(feasible)
    if not len(feasible_indices):
        return {
            "status": "no_locally_feasible_action",
            "configured_action_bounds": [lb, ub],
            "highest_feasible_action": None,
            "computational_action_upper_bound": None,
            "upper_action_check": upper_check,
        }
    last = int(feasible_indices[-1])
    if last == len(grid) - 1:
        root = ub
    else:
        left, right = float(grid[last]), float(grid[last + 1])
        root = float(brentq(
            lambda action: float(local_incentive_capacity(mhp, utility_cfg, action)["slack"]),
            left, right,
        ))
    computational_upper = max(lb, root - float(safety_fraction) * (ub - lb))
    transitions = int(np.sum(feasible[1:] != feasible[:-1]))
    return {
        "status": "upper_action_infeasible",
        "configured_action_bounds": [lb, ub],
        "highest_feasible_action": root,
        "computational_action_upper_bound": computational_upper,
        "safety_fraction": float(safety_fraction),
        "feasibility_transitions_on_grid": transitions,
        "upper_action_check": upper_check,
        "computational_upper_check": local_incentive_capacity(mhp, utility_cfg, computational_upper),
    }


def certify_outcome_support(case: dict[str, Any], numerics: dict[str, Any]) -> dict[str, Any]:
    """Expand the outcome support until mass and score diagnostics pass."""
    cfg = numerics.get("support", {})
    mass_tolerance = float(cfg.get("mass_tolerance", 1e-6))
    score_tolerance = float(cfg.get("score_mean_tolerance", 1e-6))
    max_expansions = int(cfg.get("max_expansions", 4))
    expansion_factor = float(cfg.get("expansion_factor", 1.5))
    working_case = copy.deepcopy(case)
    history: list[dict[str, Any]] = []

    certification_actions = sorted({
        float(case["target_action"]),
        *([float(case["cost_zero_at_action"])] if "cost_zero_at_action" in case else []),
    })
    for expansion in range(max_expansions + 1):
        mhp, _ = make_problem(working_case)
        action_diagnostics = {
            str(action): distribution_diagnostics(
                mhp, action, distribution=case.get("distribution")
            ) for action in certification_actions
        }
        diagnostics = action_diagnostics[str(float(case["target_action"]))]
        history.append({
            "expansion": expansion,
            "outcome_grid": copy.deepcopy(working_case["outcome_grid"]),
            "diagnostics": diagnostics,
            "action_diagnostics": action_diagnostics,
        })
        if all(
            row["mass_error"] <= mass_tolerance
            and abs(row["score_mean"]) <= score_tolerance
            for row in action_diagnostics.values()
        ):
            return {
                "status": "passed",
                "expansions": expansion,
                "effective_outcome_grid": copy.deepcopy(working_case["outcome_grid"]),
                "history": history,
            }
        if expansion == max_expansions:
            break

        grid = working_case["outcome_grid"]
        if grid["distribution_type"] == "continuous":
            center = float(case["target_action"])
            nonnegative_support = case["distribution"]["kind"] in {"exponential", "gamma"}
            if not nonnegative_support:
                grid["y_min"] = center - expansion_factor * (center - float(grid["y_min"]))
            # Preserve known nonnegative support boundaries. Scale one-sided
            # supports from zero; location-family supports expand around a0.
            grid["y_max"] = (
                expansion_factor * float(grid["y_max"])
                if nonnegative_support
                else center + expansion_factor * (float(grid["y_max"]) - center)
            )
            old_n = int(grid["n"])
            new_n = int(math.ceil((old_n - 1) * expansion_factor)) + 1
            grid["n"] = new_n if new_n % 2 == 1 else new_n + 1
        else:
            old_max = float(grid["y_max"])
            lower = float(grid["y_min"])
            grid["y_max"] = lower + expansion_factor * (old_max - lower)
            step = float(grid.get("step_size", 1.0))
            grid["y_max"] = lower + math.ceil((grid["y_max"] - lower) / step) * step

    return {
        "status": "not_converged",
        "expansions": max_expansions,
        "effective_outcome_grid": copy.deepcopy(working_case["outcome_grid"]),
        "history": history,
    }


def safe_region_metrics(
    mhp: MoralHazardProblem,
    case: dict[str, Any],
    numerics: dict[str, Any],
    *,
    support_status: str,
    intended_action: float | None = None,
) -> dict[str, Any]:
    """Numerically certify theorem-relevant safe-region quantities.

    Certification is only over the declared outcome and action grids. It is a
    diagnostic, never a proof of the uniform full-support condition.
    """
    cfg = numerics.get("safe_region", {})
    a_lb, a_ub = map(float, case["action_bounds"])
    action_points = int(cfg.get("action_points", 121))
    actions = np.linspace(a_lb, a_ub, action_points)
    y = np.asarray(mhp.y_grid, dtype=float)
    weights = np.asarray(mhp.w, dtype=float)
    a0 = float(
        intended_action if intended_action is not None
        else case.get("fixed_action", case["target_action"])
    )
    h = float(cfg.get("derivative_step", max(1e-5, 1e-4 * max(1.0, a_ub - a_lb))))

    action_matrix = actions[:, None]
    outcome_matrix = y[None, :]
    density = np.asarray(mhp.f(outcome_matrix, action_matrix), dtype=float)
    score_plus = np.asarray(mhp.score(outcome_matrix, action_matrix + h), dtype=float)
    score_minus = np.asarray(mhp.score(outcome_matrix, action_matrix - h), dtype=float)
    score_a = (score_plus - score_minus) / (2 * h)
    score_all = np.asarray(mhp.score(outcome_matrix, action_matrix), dtype=float)
    faa = density * (score_all * score_all + score_a)

    score0 = np.asarray(mhp.score(y, a0), dtype=float)
    density0 = np.asarray(mhp.f(y, a0), dtype=float)
    faa_tolerance = float(cfg.get("faa_tolerance", 1e-12))
    point_safe = np.min(faa, axis=0) >= -faa_tolerance

    nonpositive_scores = np.unique(score0[score0 <= 0.0])
    first_unsafe_score = None
    for score_value in nonpositive_scores:
        same_score = np.isclose(score0, score_value, rtol=0.0, atol=1e-12)
        if not np.all(point_safe[same_score]):
            first_unsafe_score = float(score_value)
            break
    cutoff_attained = first_unsafe_score is None
    safe_cutoff = 0.0 if cutoff_attained else first_unsafe_score
    safe_mask = score0 < safe_cutoff if not cutoff_attained else score0 <= safe_cutoff

    safe_mass = _float(np.sum(weights * density0 * safe_mask))
    safe_capacity = _float(-np.sum(weights * score0 * density0 * safe_mask))
    if cutoff_attained:
        curvature_mask = score0 > safe_cutoff
    else:
        # q approaches the unsafe boundary from below, so its score group
        # remains in {s > q} in the infimum defining safe curvature.
        curvature_mask = score0 >= safe_cutoff
    curvature_by_action = np.sum(
        weights[None, :] * np.maximum(score0[None, :] * faa, 0.0) * curvature_mask[None, :],
        axis=1,
    )
    safe_curvature = float(np.max(curvature_by_action))

    score_inf_grid = float(np.min(score0))
    grid_width = max(0.0, safe_cutoff - score_inf_grid)
    distribution_kind = case["distribution"]["kind"].lower()
    analytic_infinite_width = distribution_kind == "gaussian"
    safe_width = math.inf if analytic_infinite_width else grid_width
    score_mean = _float(np.sum(weights * density0 * score0))
    score_variance = _float(np.sum(weights * density0 * (score0 - score_mean) ** 2))
    score_sd = math.sqrt(max(0.0, score_variance))
    normalized_width = math.inf if math.isinf(safe_width) else (
        safe_width / score_sd if score_sd > 0 else math.nan
    )

    cost_h = max(1e-6, 1e-4 * max(1.0, a_ub - a_lb))
    cost_curvature = np.asarray(
        (mhp.Cprime(actions + cost_h) - mhp.Cprime(actions - cost_h)) / (2 * cost_h),
        dtype=float,
    )
    cost_curvature_floor = float(np.min(cost_curvature))
    curvature_width_ratio = 0.0 if math.isinf(safe_width) and math.isfinite(safe_curvature) else (
        safe_curvature / safe_width if safe_width > 0 else math.inf
    )
    safe_indices = np.flatnonzero(safe_mask)
    outcome_region = None if len(safe_indices) == 0 else {
        "grid_min": float(np.min(y[safe_indices])),
        "grid_max": float(np.max(y[safe_indices])),
        "grid_point_count": int(len(safe_indices)),
    }
    grid_certified = bool(np.all(point_safe[safe_mask]))
    status = "passed" if grid_certified and support_status == "passed" else "unresolved"
    return {
        "status": status,
        "numerical_diagnostic_not_proof": True,
        "support_status": support_status,
        "intended_action": a0,
        "action_grid": {"lower": a_lb, "upper": a_ub, "points": action_points},
        "derivative_step": h,
        "faa_tolerance": faa_tolerance,
        "safe_cutoff_supremum": safe_cutoff,
        "cutoff_attained_on_grid": cutoff_attained,
        "safe_mass": safe_mass,
        "safe_incentive_capacity": safe_capacity,
        "safe_curvature": safe_curvature,
        "safe_width": None if math.isinf(safe_width) else safe_width,
        "safe_width_infinite": math.isinf(safe_width),
        "grid_safe_width": grid_width,
        "normalized_safe_width_score_sd": None if math.isinf(normalized_width) or math.isnan(normalized_width) else normalized_width,
        "score_infimum_on_grid": score_inf_grid,
        "safe_outcome_region_on_grid": outcome_region,
        "minimum_faa_on_safe_grid": float(np.min(faa[:, safe_mask])) if np.any(safe_mask) else None,
        "cost_curvature_floor": cost_curvature_floor,
        "curvature_width_ratio": None if math.isinf(curvature_width_ratio) else curvature_width_ratio,
        "curvature_width_ratio_infinite": math.isinf(curvature_width_ratio),
        "log_condition_margin": None if math.isinf(curvature_width_ratio) else cost_curvature_floor - curvature_width_ratio,
        "log_curvature_width_condition_on_grid": curvature_width_ratio < cost_curvature_floor,
    }


def safe_region_convergence(
    mhp: MoralHazardProblem,
    case: dict[str, Any],
    numerics: dict[str, Any],
    *,
    support_status: str,
) -> dict[str, Any]:
    """Check safe metrics over action-grid and derivative-step refinements."""
    cfg = numerics.get("safe_region", {})
    action_points = [int(value) for value in cfg.get("convergence_action_points", [91, 181, 361])]
    derivative_steps = [float(value) for value in cfg.get("convergence_derivative_steps", [1e-2, 1e-3, 1e-4])]
    baseline_points = int(cfg.get("action_points", action_points[-1]))
    finest_step = derivative_steps[-1]
    records: list[dict[str, Any]] = []

    def evaluate(label: str, points: int, step: float) -> None:
        local_numerics = copy.deepcopy(numerics)
        local_numerics.setdefault("safe_region", {})["action_points"] = points
        local_numerics["safe_region"]["derivative_step"] = step
        metrics = safe_region_metrics(
            mhp, case, local_numerics, support_status=support_status
        )
        records.append({
            "dimension": label,
            "action_points": points,
            "derivative_step": step,
            "safe_cutoff_supremum": metrics["safe_cutoff_supremum"],
            "safe_mass": metrics["safe_mass"],
            "safe_incentive_capacity": metrics["safe_incentive_capacity"],
            "safe_curvature": metrics["safe_curvature"],
            "grid_safe_width": metrics["grid_safe_width"],
            "log_condition": metrics["log_curvature_width_condition_on_grid"],
        })

    for points in action_points:
        evaluate("action_grid", points, finest_step)
    for step in derivative_steps:
        evaluate("derivative_step", baseline_points, step)

    def relative_change(left: float, right: float) -> float:
        return abs(right - left) / max(abs(left), abs(right), 1e-14)

    action_records = [row for row in records if row["dimension"] == "action_grid"]
    derivative_records = [row for row in records if row["dimension"] == "derivative_step"]
    comparisons: dict[str, Any] = {}
    stable = support_status == "passed"
    for label, rows in (("action_grid", action_records), ("derivative_step", derivative_records)):
        if len(rows) < 2:
            comparisons[label] = {"stable": False, "reason": "fewer_than_two_levels"}
            stable = False
            continue
        left, right = rows[-2], rows[-1]
        cutoff_change = abs(right["safe_cutoff_supremum"] - left["safe_cutoff_supremum"])
        capacity_change = relative_change(
            left["safe_incentive_capacity"], right["safe_incentive_capacity"]
        )
        curvature_change = relative_change(left["safe_curvature"], right["safe_curvature"])
        dimension_stable = (
            cutoff_change <= float(cfg.get("convergence_cutoff_tolerance", 1e-6))
            and capacity_change <= float(cfg.get("convergence_relative_tolerance", 0.01))
            and curvature_change <= float(cfg.get("convergence_relative_tolerance", 0.01))
            and left["log_condition"] == right["log_condition"]
        )
        comparisons[label] = {
            "stable": dimension_stable,
            "cutoff_absolute_change": cutoff_change,
            "capacity_relative_change": capacity_change,
            "curvature_relative_change": curvature_change,
            "log_condition_agrees": left["log_condition"] == right["log_condition"],
        }
        stable = stable and dimension_stable
    return {
        "status": "passed" if stable else "unresolved",
        "support_status": support_status,
        "comparisons": comparisons,
        "records": records,
    }


def find_best_deviation(
    mhp: MoralHazardProblem,
    utility_cfg: dict[str, Any],
    contract: np.ndarray,
    intended_action: float,
    action_bounds: tuple[float, float],
    coarse_points: int,
    multistart_count: int = 12,
) -> DeviationResult:
    """Globally search for the agent's best action using a grid and local refinement."""
    lb, ub = action_bounds
    if coarse_points < 3:
        raise ValueError("coarse_points must be at least 3")
    grid = np.linspace(lb, ub, coarse_points)
    values = np.asarray(mhp.U(contract, grid), dtype=float)
    if values.shape != grid.shape or not np.all(np.isfinite(values)):
        raise ValueError("Agent utility scan returned nonfinite values or the wrong shape")

    local_indices = [
        i for i in range(1, len(grid) - 1)
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]
    ]
    candidate_indices = [0, len(grid) - 1, *local_indices]
    # Also refine neighborhoods around the highest coarse values. This guards
    # against nearly tied/flat peaks whose sampled center is not a strict local
    # maximum. Uniform starts make the search less dependent on grid alignment.
    top_count = min(max(1, multistart_count), len(grid))
    candidate_indices.extend(np.argsort(values)[-top_count:].tolist())
    if multistart_count > 0:
        candidate_indices.extend(
            np.linspace(1, len(grid) - 2, min(multistart_count, len(grid) - 2), dtype=int).tolist()
        )
    candidate_indices = sorted(set(candidate_indices))

    candidates: list[tuple[float, float]] = [(float(grid[i]), float(values[i])) for i in candidate_indices]
    for i in candidate_indices:
        if i in (0, len(grid) - 1):
            continue
        result = minimize_scalar(
            lambda a: -_float(mhp.U(contract, a)),
            bounds=(float(grid[i - 1]), float(grid[i + 1])),
            method="bounded",
            options={"xatol": 1e-9},
        )
        if result.success and np.isfinite(result.fun):
            candidates.append((float(result.x), float(-result.fun)))

    intended_u = _float(mhp.U(contract, intended_action))
    candidates.append((float(intended_action), intended_u))
    candidates.sort(key=lambda item: item[1], reverse=True)
    max_utility = candidates[0][1]
    utility_tie_tolerance = 1e-12 * max(1.0, abs(max_utility))
    # Numerical integration can make a flat objective differ at machine
    # precision. Resolve such ties deterministically toward the lower action.
    best_a, best_u = min(
        (item for item in candidates if max_utility - item[1] <= utility_tie_tolerance),
        key=lambda item: item[0],
    )

    unique_maxima: list[dict[str, float]] = []
    action_tolerance = max(1e-7, (ub - lb) * 1e-8)
    neighborhood = max(action_tolerance * 10, (ub - lb) / (coarse_points - 1) * 1e-4)
    for action, utility in candidates:
        if any(abs(action - old["action"]) <= action_tolerance for old in unique_maxima):
            continue
        if action <= lb + action_tolerance or action >= ub - action_tolerance:
            is_local_maximum = True  # endpoints are one-sided candidates
        else:
            neighbor_values = np.asarray(mhp.U(contract, [action - neighborhood, action + neighborhood]))
            is_local_maximum = bool(np.all(utility >= neighbor_values - utility_tie_tolerance))
        if is_local_maximum:
            unique_maxima.append({"action": action, "utility": utility})

    return DeviationResult(
        intended_action=float(intended_action),
        intended_utility=intended_u,
        best_action=best_a,
        best_utility=best_u,
        utility_gain=best_u - intended_u,
        ce_gain=ce_wage(utility_cfg, best_u) - ce_wage(utility_cfg, intended_u),
        local_maxima=unique_maxima,
        diagnostics={
            "coarse_points": int(coarse_points),
            "multistart_count": int(multistart_count),
            "coarse_best_action": float(grid[int(np.argmax(values))]),
            "coarse_best_utility": float(np.max(values)),
            "candidate_local_maxima": len(local_indices),
            "endpoint_utilities": [float(values[0]), float(values[-1])],
            "endpoint_checked": True,
        },
    )


def classify(ce_gain: float, valid_tolerance: float, invalid_tolerance: float) -> str:
    if ce_gain <= valid_tolerance:
        return "valid"
    if ce_gain >= invalid_tolerance:
        return "invalid"
    return "unresolved"


def _constraint_diagnostics(
    mhp: MoralHazardProblem,
    contract: np.ndarray,
    intended_action: float,
    reservation_utility_value: float,
) -> dict[str, float]:
    density = np.asarray(mhp.f(mhp.y_grid, intended_action), dtype=float)
    score = np.asarray(mhp.score(mhp.y_grid, intended_action), dtype=float)
    foc = _float(np.sum(mhp.w * density * score * contract) - mhp.Cprime(intended_action))
    return {
        "ir_slack_utility": _float(mhp.U(contract, intended_action) - reservation_utility_value),
        "foc_residual": foc,
        "limited_liability_slack": _float(np.min(contract) - reservation_utility(
            {"u": mhp._primitives["u"]}, 0.0
        )),
    }


def crosscheck_fixed_action(
    *,
    mhp: MoralHazardProblem,
    utility_cfg: dict[str, Any],
    case: dict[str, Any],
    numerics: dict[str, Any],
    intended_action: float,
    wage: float,
) -> dict[str, Any]:
    """Compare active-set and discretized CVXPY solutions at one action.

    Principal points use this same fixed-action check at their endogenously
    selected action; this validates the inner contract problem, not the outer
    principal action search.
    """
    ru = reservation_utility(utility_cfg, wage)
    a_lb, a_ub = map(float, case["action_bounds"])
    validation_cfg = numerics["validation"]
    a_hat = np.linspace(a_lb, a_ub, int(validation_cfg["cvxpy_action_points"]))
    a_hat = np.unique(np.append(a_hat, intended_action))
    records: dict[str, Any] = {}
    caught_messages: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        active_relaxed = mhp.solve_cost_minimization_problem(
            intended_action=intended_action, reservation_utility=ru,
            a_ic_lb=a_lb, a_ic_ub=a_ub, n_a_iterations=0,
        )
        active_full = mhp.solve_cost_minimization_problem(
            intended_action=intended_action, reservation_utility=ru,
            a_ic_lb=a_lb, a_ic_ub=a_ub, n_a_iterations=int(numerics["full_ic_iterations"]),
        )
        cvx_relaxed = mhp.solve_cost_minimization_problem_cvxpy(
            intended_action=intended_action, reservation_utility=ru, a_hat=np.array([]),
        )
        cvx_full = mhp.solve_cost_minimization_problem_cvxpy(
            intended_action=intended_action, reservation_utility=ru, a_hat=a_hat,
        )
        caught_messages.extend(str(item.message) for item in caught)

    cvx_results = {"cvxpy_relaxed": cvx_relaxed, "cvxpy_full": cvx_full}
    for name, cmp in (("active_relaxed", active_relaxed), ("active_full", active_full)):
        deviation = find_best_deviation(
            mhp, utility_cfg, cmp.optimal_contract, intended_action, (a_lb, a_ub),
            int(numerics["deviation"]["coarse_action_points"]),
            int(numerics["deviation"].get("multistart_count", 12)),
        )
        records[name] = {
            "status": str(cmp.solver_state.get("status", "unknown")),
            "expected_wage": _float(cmp.constraints["Ewage"]),
            "classification": classify(
                deviation.ce_gain,
                float(numerics["deviation"]["valid_tolerance_ce"]),
                float(numerics["deviation"]["invalid_tolerance_ce"]),
            ),
            "deviation_ce_gain": deviation.ce_gain,
            "best_deviation": deviation.best_action,
            "constraints": _constraint_diagnostics(mhp, cmp.optimal_contract, intended_action, ru),
        }
    for name, cvx_result in cvx_results.items():
        contract = cvx_result.get("optimal_contract")
        record: dict[str, Any] = {"status": str(cvx_result["status"])}
        if contract is not None:
            deviation = find_best_deviation(
                mhp, utility_cfg, np.asarray(contract), intended_action, (a_lb, a_ub),
                int(numerics["deviation"]["coarse_action_points"]),
                int(numerics["deviation"].get("multistart_count", 12)),
            )
            record.update({
                "expected_wage": _float(cvx_result["expected_wage"]),
                "classification": classify(
                    deviation.ce_gain,
                    float(numerics["deviation"]["valid_tolerance_ce"]),
                    float(numerics["deviation"]["invalid_tolerance_ce"]),
                ),
                "deviation_ce_gain": deviation.ce_gain,
                "best_deviation": deviation.best_action,
                "constraints": _constraint_diagnostics(mhp, np.asarray(contract), intended_action, ru),
                "minimum_discrete_ic_slack": (
                    _float(np.min(cvx_result["ic_slack"])) if len(cvx_result["ic_slack"]) else None
                ),
            })
        records[name] = record

    relaxed_gap = records["cvxpy_relaxed"].get("expected_wage", math.nan) - records["active_relaxed"]["expected_wage"]
    full_gap = records["cvxpy_full"].get("expected_wage", math.nan) - records["active_full"]["expected_wage"]
    objective_tolerance = float(validation_cfg["objective_tolerance"])
    foc_tolerance = float(validation_cfg["foc_tolerance"])
    cvx_ok = all(records[name]["status"] in {"optimal", "optimal_inaccurate"} for name in cvx_results)
    residuals_ok = all(
        abs(record.get("constraints", {}).get("foc_residual", math.inf)) <= foc_tolerance
        and record.get("constraints", {}).get("ir_slack_utility", -math.inf) >= -foc_tolerance
        for record in records.values()
    )
    objectives_ok = abs(relaxed_gap) <= objective_tolerance and abs(full_gap) <= objective_tolerance
    relaxed_classifications_agree = (
        records["active_relaxed"]["classification"] == records["cvxpy_relaxed"].get("classification")
    )
    full_ic_ok = records["active_full"]["classification"] == "valid" and records["cvxpy_full"].get("classification") == "valid"
    if cvx_ok and residuals_ok and objectives_ok and relaxed_classifications_agree and full_ic_ok:
        status = "passed"
    elif not cvx_ok or not np.isfinite(relaxed_gap) or not np.isfinite(full_gap):
        status = "failed"
    else:
        status = "unresolved"
    return {
        "status": status,
        "intended_action": float(intended_action),
        "reservation_wage": float(wage),
        "cvxpy_action_points": len(a_hat),
        "objective_gaps_cvxpy_minus_active": {"relaxed": relaxed_gap, "full": full_gap},
        "contract_comparisons": {
            "relaxed_max_abs_utility_difference": _float(np.max(np.abs(
                np.asarray(cvx_relaxed["optimal_contract"]) - active_relaxed.optimal_contract
            ))) if cvx_relaxed.get("optimal_contract") is not None else None,
            "full_max_abs_utility_difference": _float(np.max(np.abs(
                np.asarray(cvx_full["optimal_contract"]) - active_full.optimal_contract
            ))) if cvx_full.get("optimal_contract") is not None else None,
            "relaxed_classifications_agree": relaxed_classifications_agree,
        },
        "solutions": records,
        "warnings": sorted(set(caught_messages)),
    }


def support_grid_convergence_at_point(
    *,
    case: dict[str, Any],
    numerics: dict[str, Any],
    intended_action: float,
    wage: float,
) -> dict[str, Any]:
    """Re-solve a relaxed inner problem on predeclared support/grid variants."""
    support_cfg = numerics.get("support", {})
    variants = support_cfg.get("contract_variants", ["baseline", "dense", "expanded_dense"])
    records: list[dict[str, Any]] = []
    for variant in variants:
        variant_case = copy.deepcopy(case)
        grid = variant_case["outcome_grid"]
        if variant == "dense" and grid["distribution_type"] == "discrete":
            continue
        if variant in {"dense", "expanded_dense"} and grid["distribution_type"] == "continuous":
            old_n = int(grid["n"])
            grid["n"] = 2 * (old_n - 1) + 1
        if variant == "expanded_dense":
            factor = float(support_cfg.get("contract_expansion_factor", 1.5))
            if grid["distribution_type"] == "continuous":
                center = float(case["target_action"])
                nonnegative_support = case["distribution"]["kind"] in {"exponential", "gamma"}
                if not nonnegative_support:
                    grid["y_min"] = center - factor * (center - float(grid["y_min"]))
                grid["y_max"] = (
                    factor * float(grid["y_max"])
                    if nonnegative_support
                    else center + factor * (float(grid["y_max"]) - center)
                )
            else:
                lower = float(grid["y_min"])
                step = float(grid.get("step_size", 1.0))
                expanded = lower + factor * (float(grid["y_max"]) - lower)
                grid["y_max"] = lower + math.ceil((expanded - lower) / step) * step
        mhp, utility_cfg = make_problem(variant_case)
        ru = reservation_utility(utility_cfg, wage)
        a_lb, a_ub = map(float, case["action_bounds"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cmp = mhp.solve_cost_minimization_problem(
                intended_action=intended_action,
                reservation_utility=ru,
                a_ic_lb=a_lb,
                a_ic_ub=a_ub,
                n_a_iterations=0,
            )
        deviation = find_best_deviation(
            mhp, utility_cfg, cmp.optimal_contract, intended_action, (a_lb, a_ub),
            int(numerics["deviation"]["coarse_action_points"]),
            int(numerics["deviation"].get("multistart_count", 12)),
        )
        records.append({
            "variant": variant,
            "outcome_grid": copy.deepcopy(grid),
            "expected_wage": _float(cmp.constraints["Ewage"]),
            "classification": classify(
                deviation.ce_gain,
                float(numerics["deviation"]["valid_tolerance_ce"]),
                float(numerics["deviation"]["invalid_tolerance_ce"]),
            ),
            "deviation_ce_gain": deviation.ce_gain,
            "best_deviation": deviation.best_action,
            "distribution_diagnostics": distribution_diagnostics(
                mhp, intended_action, distribution=variant_case.get("distribution")
            ),
            "warnings": sorted({str(item.message) for item in caught}),
        })
    wage_range = max(row["expected_wage"] for row in records) - min(row["expected_wage"] for row in records)
    classifications = {row["classification"] for row in records}
    serious_warnings = any(row["warnings"] for row in records)
    stable = (
        len(classifications) == 1
        and "unresolved" not in classifications
        and wage_range <= float(support_cfg.get("contract_wage_tolerance", 0.02))
        and not serious_warnings
    )
    return {
        "status": "passed" if stable else "unresolved",
        "intended_action": float(intended_action),
        "reservation_wage": float(wage),
        "expected_wage_range": wage_range,
        "classifications": sorted(classifications),
        "records": records,
    }


def _solve_point(
    *,
    exercise: str,
    mhp: MoralHazardProblem,
    utility_cfg: dict[str, Any],
    case: dict[str, Any],
    wage: float,
    numerics: dict[str, Any],
) -> PointResult:
    ru = reservation_utility(utility_cfg, wage)
    a_lb, a_ub = map(float, case["action_bounds"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if exercise == "principal":
            principal_lb, principal_ub = map(
                float, case.get("principal_intended_action_bounds", case["action_bounds"])
            )
            solution = mhp.solve_principal_problem(
                revenue_function=lambda a: a,
                reservation_utility=ru,
                a_min=principal_lb,
                a_max=principal_ub,
                a_ic_lb=a_lb,
                a_ic_ub=a_ub,
                n_a_iterations=0,
            )
            intended_action = float(solution.optimal_action)
            cmp = solution.cmp_result
            profit = float(solution.profit)
        elif exercise == "fixed_action":
            intended_action = float(case["fixed_action"])
            cmp = mhp.solve_cost_minimization_problem(
                intended_action=intended_action,
                reservation_utility=ru,
                a_ic_lb=a_lb,
                a_ic_ub=a_ub,
                n_a_iterations=0,
            )
            profit = None
        else:
            raise ValueError(exercise)

    deviation = find_best_deviation(
        mhp,
        utility_cfg,
        cmp.optimal_contract,
        intended_action,
        (a_lb, a_ub),
        int(numerics["deviation"]["coarse_action_points"]),
        int(numerics["deviation"].get("multistart_count", 12)),
    )
    classification = classify(
        deviation.ce_gain,
        float(numerics["deviation"]["valid_tolerance_ce"]),
        float(numerics["deviation"]["invalid_tolerance_ce"]),
    )
    lam = _float(cmp.multipliers.get("lam", math.nan))
    expected_wage = _float(cmp.constraints["Ewage"])
    delivered_u = deviation.intended_utility

    return PointResult(
        exercise=exercise,
        reservation_wage=float(wage),
        reservation_utility=ru,
        intended_action=intended_action,
        expected_wage=expected_wage,
        profit=profit,
        delivered_utility=delivered_u,
        delivered_ce_wage=ce_wage(utility_cfg, delivered_u),
        ir_multiplier=lam,
        solver_foa_flag=cmp.first_order_approach_holds,
        classification=classification,
        deviation=deviation,
        warnings=sorted({str(item.message) for item in caught}),
    )


def _solve_monopsony(
    *,
    relaxed: bool,
    mhp: MoralHazardProblem,
    utility_cfg: dict[str, Any],
    case: dict[str, Any],
    numerics: dict[str, Any],
    candidate_wages: list[float],
) -> dict[str, Any]:
    """Find a slack-IR principal solution by scanning low reservation wages."""
    a_lb, a_ub = map(float, case["action_bounds"])
    principal_lb, principal_ub = map(
        float, case.get("principal_intended_action_bounds", case["action_bounds"])
    )
    history: list[dict[str, Any]] = []
    selected = None
    wages_to_try = list(candidate_wages)
    max_extensions = int(numerics["monopsony"].get("max_downward_extensions", 0))
    downward_step = float(numerics["monopsony"].get("downward_step", 10.0))
    minimum_wage = float(numerics["monopsony"].get("minimum_reservation_wage", -math.inf))
    extensions = 0
    index = 0
    while index < len(wages_to_try):
        wage = wages_to_try[index]
        index += 1
        ru = reservation_utility(utility_cfg, wage)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solution = mhp.solve_principal_problem(
                revenue_function=lambda a: a,
                reservation_utility=ru,
                a_min=principal_lb,
                a_max=principal_ub,
                a_ic_lb=a_lb,
                a_ic_ub=a_ub,
                n_a_iterations=0 if relaxed else int(numerics["full_ic_iterations"]),
            )
        cmp = solution.cmp_result
        delivered_u = _float(mhp.U(cmp.optimal_contract, solution.optimal_action))
        deviation = find_best_deviation(
            mhp,
            utility_cfg,
            cmp.optimal_contract,
            float(solution.optimal_action),
            (a_lb, a_ub),
            int(numerics["deviation"]["coarse_action_points"]),
            int(numerics["deviation"].get("multistart_count", 12)),
        )
        global_ic_classification = classify(
            deviation.ce_gain,
            float(numerics["deviation"]["valid_tolerance_ce"]),
            float(numerics["deviation"]["invalid_tolerance_ce"]),
        )
        row = {
            "reservation_wage": float(wage),
            "ir_multiplier": _float(cmp.multipliers.get("lam", math.nan)),
            "action": float(solution.optimal_action),
            "profit": float(solution.profit),
            "expected_wage": _float(cmp.constraints["Ewage"]),
            "delivered_utility": delivered_u,
            "delivered_ce_wage": ce_wage(utility_cfg, delivered_u),
            "ir_slack_utility": delivered_u - ru,
            "ir_slack_ce": ce_wage(utility_cfg, delivered_u) - float(wage),
            "deviation_ce_gain": deviation.ce_gain,
            "global_ic_classification": global_ic_classification,
            "globally_ic": global_ic_classification == "valid",
            "warnings": sorted({str(item.message) for item in caught}),
        }
        history.append(row)
        slack = (
            row["ir_multiplier"] <= float(numerics["monopsony"]["lambda_tolerance"])
            and row["ir_slack_ce"] >= float(numerics["monopsony"].get("ir_slack_ce_tolerance", 0.0))
        )
        if slack:
            selected = row
            # Two slack points allow a basic plateau check.
            previous = history[-2] if len(history) >= 2 else None
            previous_slack = previous is not None and (
                previous["ir_multiplier"] <= float(numerics["monopsony"]["lambda_tolerance"])
                and previous["ir_slack_ce"] >= float(numerics["monopsony"].get("ir_slack_ce_tolerance", 0.0))
            )
            if previous_slack:
                break
        if index == len(wages_to_try) and extensions < max_extensions:
            next_wage = max(minimum_wage, wages_to_try[-1] - downward_step)
            if next_wage < wages_to_try[-1]:
                wages_to_try.append(next_wage)
                extensions += 1

    if selected is None:
        return {"status": "not_found", "history": history}

    previous_slack = next(
        (
            row for row in reversed(history[:-1])
            if row["ir_multiplier"] <= float(numerics["monopsony"]["lambda_tolerance"])
            and row["ir_slack_ce"] >= float(numerics["monopsony"].get("ir_slack_ce_tolerance", 0.0))
        ),
        None,
    )
    stable = previous_slack is not None and (
        abs(selected["action"] - previous_slack["action"]) <= float(numerics["monopsony"]["action_tolerance"])
        and abs(selected["profit"] - previous_slack["profit"]) <= float(numerics["monopsony"]["profit_tolerance"])
        and abs(selected["expected_wage"] - previous_slack["expected_wage"])
        <= float(numerics["monopsony"].get("expected_wage_tolerance", numerics["monopsony"]["ce_tolerance"]))
        and abs(selected["delivered_ce_wage"] - previous_slack["delivered_ce_wage"])
        <= float(numerics["monopsony"]["ce_tolerance"])
    )
    if not stable:
        status = "unverified_plateau"
    elif selected["global_ic_classification"] == "invalid" or previous_slack["global_ic_classification"] == "invalid":
        status = "failed_global_ic_check"
    elif selected["global_ic_classification"] == "unresolved" or previous_slack["global_ic_classification"] == "unresolved":
        status = "unresolved_global_ic"
    else:
        status = "ok"
    return {"status": status, "selected": selected, "history": history}


def summarize_transitions(points: list[PointResult]) -> dict[str, Any]:
    ordered = sorted(points, key=lambda point: point.reservation_wage)
    transitions = []
    reversals = []
    for left, right in zip(ordered, ordered[1:]):
        if left.classification != right.classification:
            transition = {
                "lower_wage": left.reservation_wage,
                "upper_wage": right.reservation_wage,
                "from": left.classification,
                "to": right.classification,
            }
            transitions.append(transition)
            if left.classification == "valid" and right.classification == "invalid":
                reversals.append(transition)

    persistent_threshold = None
    for i, point in enumerate(ordered):
        if point.classification == "valid" and all(p.classification == "valid" for p in ordered[i:]):
            persistent_threshold = point.reservation_wage
            break

    return {
        "persistent_threshold_on_grid": persistent_threshold,
        "transitions": transitions,
        "reversals": reversals,
        "monotone_validity_on_grid": not reversals,
    }


def refine_transitions(
    points: list[PointResult],
    solve: Callable[[float], PointResult],
    wage_tolerance: float,
    max_iterations: int,
) -> list[dict[str, Any]]:
    """Bisect every valid/invalid transition without assuming monotonicity."""
    ordered = sorted(points, key=lambda point: point.reservation_wage)
    refined: list[dict[str, Any]] = []
    for initial_left, initial_right in zip(ordered, ordered[1:]):
        if {initial_left.classification, initial_right.classification} != {"valid", "invalid"}:
            continue
        left, right = initial_left, initial_right
        unresolved = None
        iterations = 0
        while right.reservation_wage - left.reservation_wage > wage_tolerance and iterations < max_iterations:
            midpoint = (left.reservation_wage + right.reservation_wage) / 2
            middle = solve(midpoint)
            iterations += 1
            if middle.classification == "unresolved":
                unresolved = asdict(middle)
                break
            if middle.classification == left.classification:
                left = middle
            else:
                right = middle
        refined.append({
            "lower_wage": left.reservation_wage,
            "upper_wage": right.reservation_wage,
            "lower_classification": left.classification,
            "upper_classification": right.classification,
            "direction": f"{initial_left.classification}_to_{initial_right.classification}",
            "iterations": iterations,
            "unresolved_midpoint": unresolved,
        })
    return refined


def run_case(case: dict[str, Any], numerics: dict[str, Any]) -> dict[str, Any]:
    support_validation = certify_outcome_support(case, numerics)
    effective_case = copy.deepcopy(case)
    effective_case["outcome_grid"] = copy.deepcopy(support_validation["effective_outcome_grid"])
    mhp, utility_cfg = make_problem(effective_case)
    capacity_cfg = numerics.get("incentive_capacity", {})
    capacity_precheck = incentive_capacity_precheck(
        mhp,
        utility_cfg,
        tuple(map(float, effective_case["action_bounds"])),
        grid_points=int(capacity_cfg.get("grid_points", 257)),
        safety_fraction=float(capacity_cfg.get("safety_fraction", 1e-3)),
    )
    if capacity_precheck["computational_action_upper_bound"] is not None:
        effective_case["principal_intended_action_bounds"] = [
            float(effective_case["action_bounds"][0]),
            float(capacity_precheck["computational_action_upper_bound"]),
        ]
    candidate_wages = [float(x) for x in numerics["monopsony"]["candidate_reservation_wages"]]
    diagnostic_actions = sorted({
        float(effective_case["target_action"]),
        *([float(effective_case["fixed_action"])] if "fixed_action" in effective_case else []),
        *([float(effective_case["cost_zero_at_action"])] if "cost_zero_at_action" in effective_case else []),
    })
    result: dict[str, Any] = {
        "case_id": case["id"],
        "configuration": case,
        "effective_configuration": effective_case,
        "distribution_diagnostics": {
            str(action): distribution_diagnostics(
                mhp, action, distribution=effective_case.get("distribution")
            ) for action in diagnostic_actions
        },
        "support_validation": support_validation,
        "cost_metadata": copy.deepcopy(utility_cfg["cost_metadata"]),
        "incentive_capacity_precheck": capacity_precheck,
        "safe_region_metrics": safe_region_metrics(
            mhp, effective_case, numerics, support_status=support_validation["status"]
        ),
        "safe_region_convergence": safe_region_convergence(
            mhp, effective_case, numerics, support_status=support_validation["status"]
        ),
        "monopsony": {},
        "exercises": {},
    }
    if effective_case.get("compute_monopsony", True):
        full_gic = _solve_monopsony(
            relaxed=False,
            mhp=mhp,
            utility_cfg=utility_cfg,
            case=effective_case,
            numerics=numerics,
            candidate_wages=candidate_wages,
        )
        if full_gic.get("selected") is not None:
            full_gic["safe_region_metrics"] = safe_region_metrics(
                mhp, effective_case, numerics,
                support_status=support_validation["status"],
                intended_action=float(full_gic["selected"]["action"]),
            )
        result["monopsony"]["full_gic"] = full_gic
    if effective_case.get("compute_monopsony", True) and numerics["monopsony"].get("compute_relaxed_lambda_zero", False):
        result["monopsony"]["relaxed_lambda_zero"] = _solve_monopsony(
            relaxed=True,
            mhp=mhp,
            utility_cfg=utility_cfg,
            case=effective_case,
            numerics=numerics,
            candidate_wages=candidate_wages,
        )

    for exercise in effective_case.get("exercises", ["principal", "fixed_action"]):
        if exercise == "fixed_action":
            fixed_check = local_incentive_capacity(
                mhp, utility_cfg, float(effective_case["fixed_action"])
            )
            if not fixed_check["feasible"]:
                result["exercises"][exercise] = {
                    "status": "infeasible_local_incentives",
                    "incentive_capacity_check": fixed_check,
                    "points": [],
                    "refinement_points": [],
                    "summary": {
                        "persistent_threshold_on_grid": None,
                        "transitions": [],
                        "reversals": [],
                        "monotone_validity_on_grid": None,
                        "refined_transitions": [],
                    },
                    "validation": [],
                }
                continue
        cache: dict[float, PointResult] = {}

        def solve(wage: float) -> PointResult:
            key = float(wage)
            if key not in cache:
                cache[key] = _solve_point(
                    exercise=exercise,
                    mhp=mhp,
                    utility_cfg=utility_cfg,
                    case=effective_case,
                    wage=key,
                    numerics=numerics,
                )
            return cache[key]

        points = [solve(float(wage)) for wage in effective_case["reservation_wages"]]
        summary = summarize_transitions(points)
        summary["refined_transitions"] = refine_transitions(
            points,
            solve,
            float(numerics["threshold"]["wage_tolerance"]),
            int(numerics["threshold"]["max_iterations"]),
        )
        validation_results = []
        if numerics.get("validation", {}).get("crosscheck_transitions", False):
            validation_locations = sorted({
                float(transition[side])
                for transition in summary["refined_transitions"]
                for side in ("lower_wage", "upper_wage")
            })
            for validation_wage in validation_locations:
                point = solve(validation_wage)
                try:
                    crosscheck = crosscheck_fixed_action(
                        mhp=mhp,
                        utility_cfg=utility_cfg,
                        case=effective_case,
                        numerics=numerics,
                        intended_action=point.intended_action,
                        wage=validation_wage,
                    )
                    crosscheck["safe_region_metrics"] = safe_region_metrics(
                        mhp, effective_case, numerics,
                        support_status=support_validation["status"],
                        intended_action=point.intended_action,
                    )
                    if numerics.get("support", {}).get("check_transition_contracts", False):
                        crosscheck["support_grid_convergence"] = support_grid_convergence_at_point(
                            case=effective_case,
                            numerics=numerics,
                            intended_action=point.intended_action,
                            wage=validation_wage,
                        )
                        if crosscheck["support_grid_convergence"]["status"] != "passed":
                            crosscheck["status"] = "unresolved"
                    validation_results.append(crosscheck)
                except Exception as error:  # Preserve failed cells for internal review.
                    validation_results.append({
                        "status": "failed",
                        "reservation_wage": validation_wage,
                        "intended_action": point.intended_action,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
        result["exercises"][exercise] = {
            "status": "completed",
            "points": [asdict(point) for point in points],
            "refinement_points": [
                asdict(point) for wage, point in sorted(cache.items())
                if wage not in {float(x) for x in effective_case["reservation_wages"]}
            ],
            "summary": summary,
            "validation": validation_results,
        }

    full_result = result["monopsony"].get("full_gic")
    full = full_result.get("selected") if full_result else None
    relaxed_result = result["monopsony"].get("relaxed_lambda_zero")
    relaxed = relaxed_result.get("selected") if relaxed_result else None
    if full and relaxed:
        result["monopsony"]["ce_gap_relaxed_minus_full"] = (
            relaxed["delivered_ce_wage"] - full["delivered_ce_wage"]
        )

    action_lb, action_ub = map(float, effective_case["action_bounds"])
    boundary_tolerance = float(numerics.get("boundary_action_tolerance", numerics["monopsony"].get("action_tolerance", 0.01)))
    boundary_records = []
    if full:
        action = float(full["action"])
        if min(abs(action - action_lb), abs(action - action_ub)) <= boundary_tolerance:
            boundary_records.append({"scope": "full_gic_monopsony", "action": action})
    for exercise, exercise_result in result["exercises"].items():
        for point in exercise_result["points"]:
            action = float(point["intended_action"])
            if min(abs(action - action_lb), abs(action - action_ub)) <= boundary_tolerance:
                boundary_records.append({
                    "scope": f"{exercise}_initial_grid", "reservation_wage": point["reservation_wage"], "action": action,
                })
    result["boundary_diagnostics"] = {
        "status": "boundary_contaminated" if boundary_records else "passed",
        "action_bounds": [action_lb, action_ub],
        "tolerance": boundary_tolerance,
        "records": boundary_records,
    }
    return result


def run_manifest(path: str | Path, suite: str = "smoke") -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = yaml.safe_load(manifest_path.read_text())
    numerics = manifest["numerics"]
    cases = [case for case in manifest["cases"] if suite in case.get("suites", [])]
    results = [run_case(case, numerics) for case in cases]
    return {
        "schema_version": manifest["schema_version"],
        "experiment_id": manifest["experiment_id"],
        "suite": suite,
        "manifest": str(manifest_path),
        "results": results,
    }


def write_outputs(payload: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "prototype_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False))

    rows = []
    for case in payload["results"]:
        for exercise, exercise_result in case["exercises"].items():
            for point in exercise_result["points"]:
                rows.append({
                    "case_id": case["case_id"],
                    "exercise": exercise,
                    "reservation_wage": point["reservation_wage"],
                    "intended_action": point["intended_action"],
                    "delivered_ce_wage": point["delivered_ce_wage"],
                    "expected_wage": point["expected_wage"],
                    "profit": point["profit"],
                    "ir_multiplier": point["ir_multiplier"],
                    "ce_deviation_gain": point["deviation"]["ce_gain"],
                    "best_deviation": point["deviation"]["best_action"],
                    "classification": point["classification"],
                    "warning_count": len(point["warnings"]),
                })
    if rows:
        with (output / "prototype_points.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
