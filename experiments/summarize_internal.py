"""Build internal-only tabular summaries from saved atomic FOA results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _warning_rows(value: Any, task_hash: str, case_id: str, path: str = "result") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key == "warnings" and isinstance(item, list):
                rows.extend({
                    "task_hash": task_hash, "case_id": case_id,
                    "path": child_path, "warning": str(warning),
                } for warning in item)
            else:
                rows.extend(_warning_rows(item, task_hash, case_id, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_warning_rows(item, task_hash, case_id, f"{path}[{index}]"))
    return rows


def _threshold(summary: dict[str, Any], points: list[dict[str, Any]]) -> tuple[Any, Any, str]:
    ordered = sorted(points, key=lambda point: point["reservation_wage"])
    transitions = summary.get("refined_transitions", [])
    if ordered and all(point["classification"] == "valid" for point in ordered):
        return None, ordered[0]["reservation_wage"], "left_censored_all_valid"
    if not ordered or ordered[-1]["classification"] != "valid":
        return None, None, "not_reached"
    candidates = [item for item in transitions if item["direction"] == "invalid_to_valid"]
    if not candidates:
        return None, summary.get("persistent_threshold_on_grid"), "grid_only"
    transition = max(candidates, key=lambda item: item["upper_wage"])
    status = "gray_zone_bracket" if transition.get("unresolved_midpoint") else "refined_bracket"
    return transition["lower_wage"], transition["upper_wage"], status


def summarize(input_dir: str | Path) -> dict[str, int]:
    root = Path(input_dir)
    summary_dir = root / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    threshold_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []

    for path in sorted((root / "atomic").glob("*.json")):
        record = json.loads(path.read_text())
        task_hash = record["task_hash"]
        case_id = record["case_id"]
        if record.get("execution_status") != "completed":
            failure_rows.append({
                "task_hash": task_hash, "case_id": case_id,
                "error_type": record.get("error_type"), "error": record.get("error"),
                "runtime_seconds": record.get("runtime_seconds"),
            })
            continue
        result = record["result"]
        warning_rows.extend(_warning_rows(result, task_hash, case_id))
        monopsony = result.get("monopsony", {}).get("full_gic", {})
        selected = monopsony.get("selected") or {}
        monopsony_ce = selected.get("delivered_ce_wage")
        boundary_status = result.get("boundary_diagnostics", {}).get("status")
        if boundary_status is None and selected.get("action") is not None:
            action_lb, action_ub = record["economic_configuration"]["action_bounds"]
            tolerance = record["numerical_configuration"]["monopsony"].get("action_tolerance", 0.01)
            boundary_status = (
                "boundary_contaminated" if min(abs(selected["action"] - action_lb), abs(selected["action"] - action_ub)) <= tolerance
                else "passed"
            )
        for exercise, exercise_result in sorted(result.get("exercises", {}).items()):
            summary = exercise_result["summary"]
            points = exercise_result["points"]
            lower, upper, threshold_status = _threshold(summary, points)
            valid_wages = [point["reservation_wage"] for point in points if point["classification"] == "valid"]
            threshold_rows.append({
                "task_hash": task_hash,
                "case_id": case_id,
                "exercise": exercise,
                "strict_numerical_status": result.get("strict_numerical_status"),
                "review_status": result.get("review_status"),
                "support_status": result.get("support_validation", {}).get("status"),
                "monopsony_status": monopsony.get("status"),
                "monopsony_boundary_status": boundary_status,
                "monopsony_ce_wage": monopsony_ce,
                "persistent_threshold_lower": lower,
                "persistent_threshold_upper": upper,
                "threshold_status": threshold_status,
                "gap_from_monopsony_lower": None if lower is None or monopsony_ce is None else lower - monopsony_ce,
                "gap_from_monopsony_upper": None if upper is None or monopsony_ce is None else upper - monopsony_ce,
                "first_valid_wage": min(valid_wages) if valid_wages else None,
                "persistent_threshold_on_grid": summary.get("persistent_threshold_on_grid"),
                "transition_count": len(summary.get("transitions", [])),
                "reversal_count": len(summary.get("reversals", [])),
                "monotone_validity_on_grid": summary.get("monotone_validity_on_grid"),
                "transitions_json": json.dumps(summary.get("refined_transitions", []), separators=(",", ":")),
            })

    _write_csv(summary_dir / "internal_thresholds.csv", threshold_rows)
    _write_csv(summary_dir / "failures.csv", failure_rows)
    _write_csv(summary_dir / "warnings.csv", warning_rows)
    counts = {"threshold_rows": len(threshold_rows), "failures": len(failure_rows), "warnings": len(warning_rows)}
    (summary_dir / "summary_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/foa-internal-atlas")
    args = parser.parse_args()
    counts = summarize(args.input)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
