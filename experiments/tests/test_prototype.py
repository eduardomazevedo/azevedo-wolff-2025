from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.prototype import (
    PointResult,
    DeviationResult,
    ce_wage,
    _solve_monopsony,
    certify_outcome_support,
    classify,
    distribution_diagnostics,
    find_best_deviation,
    make_problem,
    reservation_utility,
    refine_transitions,
    safe_region_convergence,
    safe_region_metrics,
    summarize_transitions,
)
from moralhazard.config_maker import make_utility_cfg


class UtilityTests(unittest.TestCase):
    def test_ce_round_trip(self) -> None:
        configs = [
            make_utility_cfg("log", w0=50),
            make_utility_cfg("crra", w0=50, gamma=0.5),
            make_utility_cfg("crra", w0=50, gamma=1.5),
            make_utility_cfg("cara", w0=50, alpha=0.02),
        ]
        for cfg in configs:
            for wage in (-1.0, 0.0, 10.0, 100.0):
                self.assertAlmostEqual(ce_wage(cfg, reservation_utility(cfg, wage)), wage, places=8)

    def test_classification_gray_zone(self) -> None:
        self.assertEqual(classify(0.0, 1e-4, 1e-3), "valid")
        self.assertEqual(classify(5e-4, 1e-4, 1e-3), "unresolved")
        self.assertEqual(classify(2e-3, 1e-4, 1e-3), "invalid")

    def test_effort_cost_is_net_of_utility_before_ce_conversion(self) -> None:
        cfg = make_utility_cfg("log", w0=50)
        gross_wage = 10.0
        effort_cost = 0.02
        delivered_utility = reservation_utility(cfg, gross_wage) - effort_cost
        expected = (50 + gross_wage) * np.exp(-effort_cost) - 50
        self.assertAlmostEqual(ce_wage(cfg, delivered_utility), expected, places=10)


class DeviationTests(unittest.TestCase):
    def test_constant_wage_prefers_lowest_effort(self) -> None:
        case = {
            "initial_wealth": 50,
            "target_action": 100,
            "action_bounds": [0, 180],
            "cost_normalization": "paper_log",
            "utility": {"kind": "log"},
            "distribution": {"kind": "gaussian", "params": {"sigma": 50}},
            "outcome_grid": {
                "distribution_type": "continuous",
                "y_min": -300,
                "y_max": 480,
                "n": 201,
            },
        }
        mhp, utility_cfg = make_problem(case)
        constant_contract = np.full_like(mhp._y_grid, reservation_utility(utility_cfg, 0.0))
        result = find_best_deviation(mhp, utility_cfg, constant_contract, 100, (0, 180), 121)
        self.assertAlmostEqual(result.best_action, 0.0, places=8)
        self.assertGreater(result.ce_gain, 0.0)
        self.assertTrue(result.diagnostics["endpoint_checked"])

    def test_narrow_and_tied_synthetic_maxima(self) -> None:
        class SyntheticProblem:
            @staticmethod
            def U(_contract, action):
                a = np.asarray(action)
                left = 1.0 - ((a - 2.03) / 0.08) ** 2
                right = 1.0 - ((a - 7.91) / 0.08) ** 2
                return np.maximum(left, right)

        identity_cfg = {"k": lambda value: np.asarray(value)}
        result = find_best_deviation(
            SyntheticProblem(), identity_cfg, np.zeros(1), 5.0, (0.0, 10.0), 101, multistart_count=16
        )
        self.assertAlmostEqual(result.best_utility, 1.0, places=7)
        high = [item for item in result.local_maxima if item["utility"] > 0.999]
        self.assertTrue(any(abs(item["action"] - 2.03) < 1e-4 for item in high))
        self.assertTrue(any(abs(item["action"] - 7.91) < 1e-4 for item in high))


class DistributionTests(unittest.TestCase):
    @staticmethod
    def case(kind: str, params: dict, grid: dict, target: float) -> dict:
        return {
            "initial_wealth": 50,
            "target_action": target,
            "action_bounds": [0, 180],
            "cost_normalization": "paper_log",
            "utility": {"kind": "log"},
            "distribution": {"kind": kind, "params": params},
            "outcome_grid": grid,
        }

    def test_gaussian_normalization_score_and_derivatives(self) -> None:
        case = self.case(
            "gaussian", {"sigma": 20},
            {"distribution_type": "continuous", "y_min": -40, "y_max": 240, "n": 801}, 100,
        )
        mhp, _ = make_problem(case)
        diagnostics = distribution_diagnostics(mhp, 100)
        self.assertLess(diagnostics["mass_error"], 1e-10)
        self.assertLess(abs(diagnostics["score_mean"]), 1e-10)
        self.assertLess(diagnostics["fa_relative_error"], 1e-6)
        self.assertLess(diagnostics["faa_relative_error"], 1e-5)

    def test_poisson_support_truncation_is_visible(self) -> None:
        short, _ = make_problem(self.case(
            "poisson", {}, {"distribution_type": "discrete", "y_min": 0, "y_max": 8, "step_size": 1}, 7,
        ))
        long, _ = make_problem(self.case(
            "poisson", {}, {"distribution_type": "discrete", "y_min": 0, "y_max": 28, "step_size": 1}, 7,
        ))
        short_d = distribution_diagnostics(short, 7)
        long_d = distribution_diagnostics(long, 7)
        self.assertGreater(short_d["omitted_mass"], 0.1)
        self.assertLess(long_d["omitted_mass"], 1e-8)
        self.assertLess(abs(long_d["score_mean"]), 1e-7)

    def test_adaptive_discrete_support_expands_until_mass_passes(self) -> None:
        case = self.case(
            "poisson", {}, {"distribution_type": "discrete", "y_min": 0, "y_max": 8, "step_size": 1}, 7,
        )
        result = certify_outcome_support(case, {"support": {
            "mass_tolerance": 1e-6, "score_mean_tolerance": 1e-6,
            "max_expansions": 4, "expansion_factor": 1.5,
        }})
        self.assertEqual(result["status"], "passed")
        self.assertGreater(result["expansions"], 0)
        self.assertLessEqual(result["history"][-1]["diagnostics"]["mass_error"], 1e-6)

    def test_heavy_tail_support_failure_is_retained(self) -> None:
        case = self.case(
            "student_t", {"sigma": 20, "nu": 1.15},
            {"distribution_type": "continuous", "y_min": -200, "y_max": 380, "n": 301}, 100,
        )
        result = certify_outcome_support(case, {"support": {
            "mass_tolerance": 1e-6, "score_mean_tolerance": 1e-6,
            "max_expansions": 2, "expansion_factor": 1.5,
        }})
        self.assertEqual(result["status"], "not_converged")
        self.assertGreater(result["history"][-1]["diagnostics"]["omitted_mass"], 1e-3)

    def test_gaussian_safe_region_has_infinite_width(self) -> None:
        case = self.case(
            "gaussian", {"sigma": 20},
            {"distribution_type": "continuous", "y_min": -100, "y_max": 300, "n": 801}, 100,
        )
        case["action_bounds"] = [0, 180]
        case["fixed_action"] = 100
        mhp, _ = make_problem(case)
        result = safe_region_metrics(
            mhp, case, {"safe_region": {"action_points": 181, "derivative_step": 0.001,
                                         "faa_tolerance": 1e-12}}, support_status="passed",
        )
        self.assertEqual(result["status"], "passed")
        self.assertIsNone(result["safe_width"])
        self.assertTrue(result["safe_width_infinite"])
        self.assertGreater(result["safe_mass"], 0)
        self.assertGreater(result["safe_incentive_capacity"], 0)
        self.assertTrue(result["log_curvature_width_condition_on_grid"])

    def test_poisson_safe_metrics_converge_at_fine_derivative_steps(self) -> None:
        case = self.case(
            "poisson", {},
            {"distribution_type": "discrete", "y_min": 0, "y_max": 28, "step_size": 1}, 7,
        )
        case["action_bounds"] = [0.05, 10]
        case["fixed_action"] = 7
        mhp, _ = make_problem(case)
        numerics = {"safe_region": {
            "action_points": 181, "derivative_step": 0.001, "faa_tolerance": 1e-12,
            "convergence_action_points": [91, 181, 361],
            "convergence_derivative_steps": [0.01, 0.001, 0.0001],
            "convergence_cutoff_tolerance": 1e-6, "convergence_relative_tolerance": 0.01,
        }}
        result = safe_region_convergence(mhp, case, numerics, support_status="passed")
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["comparisons"]["action_grid"]["stable"])
        self.assertTrue(result["comparisons"]["derivative_step"]["stable"])
        derivative_curvatures = [
            row["safe_curvature"] for row in result["records"]
            if row["dimension"] == "derivative_step"
        ]
        self.assertGreater(max(derivative_curvatures) / min(derivative_curvatures), 1.2)

    def test_student_t_empty_safe_region_is_not_promoted(self) -> None:
        case = self.case(
            "student_t", {"sigma": 20, "nu": 1.15},
            {"distribution_type": "continuous", "y_min": -200, "y_max": 380, "n": 301}, 100,
        )
        case["action_bounds"] = [0, 180]
        case["fixed_action"] = 100
        mhp, _ = make_problem(case)
        result = safe_region_metrics(
            mhp, case, {"safe_region": {"action_points": 181, "derivative_step": 0.001,
                                         "faa_tolerance": 1e-12}}, support_status="not_converged",
        )
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["safe_mass"], 0)
        self.assertEqual(result["safe_incentive_capacity"], 0)
        self.assertFalse(result["log_curvature_width_condition_on_grid"])

    def test_cost_functions_do_not_capture_later_cases(self) -> None:
        first, _ = make_problem(self.case(
            "gaussian", {"sigma": 1},
            {"distribution_type": "continuous", "y_min": -5, "y_max": 15, "n": 101}, 10,
        ))
        first_cost = float(first.C(10))
        second, _ = make_problem(self.case(
            "gaussian", {"sigma": 1},
            {"distribution_type": "continuous", "y_min": -5, "y_max": 25, "n": 101}, 20,
        ))
        self.assertEqual(float(first.C(10)), first_cost)
        self.assertNotEqual(float(first.C(10)), float(second.C(10)))


class MonopsonyTests(unittest.TestCase):
    @staticmethod
    def numerics() -> dict:
        return {
            "full_ic_iterations": 5,
            "deviation": {"coarse_action_points": 21, "valid_tolerance_ce": 1e-4, "invalid_tolerance_ce": 1e-3},
            "monopsony": {"lambda_tolerance": 1e-6, "action_tolerance": 0.01,
                           "profit_tolerance": 0.01, "ce_tolerance": 0.01},
        }

    @staticmethod
    def fake_problem(rows):
        class FakeProblem:
            def __init__(self):
                self.rows = iter(rows)

            def solve_principal_problem(self, **_kwargs):
                row = next(self.rows)
                cmp = SimpleNamespace(
                    optimal_contract=np.array([row["utility"]]),
                    multipliers={"lam": row["lambda"]},
                    constraints={"Ewage": row["wage"]},
                )
                return SimpleNamespace(optimal_action=row["action"], profit=row["profit"], cmp_result=cmp)

            @staticmethod
            def U(contract, _action):
                return float(contract[0])

        return FakeProblem()

    def test_stable_valid_plateau_is_selected(self) -> None:
        rows = [
            {"lambda": 0.0, "action": 3.0, "profit": 2.0, "wage": 1.0, "utility": 0.5},
            {"lambda": 0.0, "action": 3.005, "profit": 2.005, "wage": 1.0, "utility": 0.505},
        ]
        deviation = DeviationResult(3, .5, 3, .5, 0, 0, [])
        with patch("experiments.prototype.find_best_deviation", return_value=deviation):
            result = _solve_monopsony(
                relaxed=False, mhp=self.fake_problem(rows), utility_cfg={"u": lambda x: x, "k": lambda x: x},
                case={"action_bounds": [0, 5]}, numerics=self.numerics(), candidate_wages=[-1, -2],
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["history"]), 2)
        self.assertGreater(result["selected"]["ir_slack_ce"], 0)

    def test_search_extends_downward_until_two_slack_points(self) -> None:
        rows = [
            {"lambda": 1.0, "action": 3.0, "profit": 1.0, "wage": 1.0, "utility": -1.0},
            {"lambda": 0.0, "action": 3.0, "profit": 2.0, "wage": 1.0, "utility": 0.5},
            {"lambda": 0.0, "action": 3.0, "profit": 2.0, "wage": 1.0, "utility": 0.5},
        ]
        numerics = self.numerics()
        numerics["monopsony"].update({"max_downward_extensions": 2, "downward_step": 10})
        deviation = DeviationResult(3, .5, 3, .5, 0, 0, [])
        with patch("experiments.prototype.find_best_deviation", return_value=deviation):
            result = _solve_monopsony(
                relaxed=False, mhp=self.fake_problem(rows), utility_cfg={"u": lambda x: x, "k": lambda x: x},
                case={"action_bounds": [0, 5]}, numerics=numerics, candidate_wages=[-1],
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["reservation_wage"] for row in result["history"]], [-1, -11, -21])

    def test_invalid_gic_cannot_receive_ok_status(self) -> None:
        rows = [
            {"lambda": 0.0, "action": 3.0, "profit": 2.0, "wage": 1.0, "utility": 0.5},
            {"lambda": 0.0, "action": 3.0, "profit": 2.0, "wage": 1.0, "utility": 0.5},
        ]
        deviation = DeviationResult(3, .5, 1, .6, .1, .1, [])
        with patch("experiments.prototype.find_best_deviation", return_value=deviation):
            result = _solve_monopsony(
                relaxed=False, mhp=self.fake_problem(rows), utility_cfg={"u": lambda x: x, "k": lambda x: x},
                case={"action_bounds": [0, 5]}, numerics=self.numerics(), candidate_wages=[-1, -2],
            )
        self.assertEqual(result["status"], "failed_global_ic_check")


class TransitionTests(unittest.TestCase):
    @staticmethod
    def point(wage: float, state: str) -> PointResult:
        deviation = DeviationResult(0, 0, 0, 0, 0, 0, [])
        return PointResult("test", wage, 0, 0, 0, None, 0, 0, 0, None, state, deviation, [])

    def test_reversal_detection(self) -> None:
        points = [self.point(0, "invalid"), self.point(1, "valid"), self.point(2, "invalid"), self.point(3, "valid")]
        summary = summarize_transitions(points)
        self.assertEqual(len(summary["reversals"]), 1)
        self.assertEqual(summary["persistent_threshold_on_grid"], 3)
        self.assertFalse(summary["monotone_validity_on_grid"])

    def test_transition_refinement(self) -> None:
        points = [self.point(0, "invalid"), self.point(1, "valid")]
        refined = refine_transitions(
            points,
            lambda wage: self.point(wage, "invalid" if wage < 0.4 else "valid"),
            wage_tolerance=0.01,
            max_iterations=20,
        )
        self.assertEqual(len(refined), 1)
        self.assertLessEqual(refined[0]["lower_wage"], 0.4)
        self.assertGreaterEqual(refined[0]["upper_wage"], 0.4)
        self.assertLessEqual(refined[0]["upper_wage"] - refined[0]["lower_wage"], 0.01)


if __name__ == "__main__":
    unittest.main()
