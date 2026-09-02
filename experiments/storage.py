"""Expand and run the exact FOA paper specifications."""

from __future__ import annotations

import copy
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.foa import run_case


@dataclass(frozen=True)
class Task:
    case_id: str
    economic_configuration: dict[str, Any]
    numerical_configuration: dict[str, Any]


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = copy.deepcopy(value)


def _label(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    text = text.replace('"', "").replace(" ", "")
    return "".join(
        character if character.isalnum() or character in ".-" else "_"
        for character in text
    )


def _manifest_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [copy.deepcopy(case) for case in manifest.get("cases", [])]
    for family in manifest.get("case_families", []):
        axes = sorted(family.get("axes", []), key=lambda axis: axis["path"])
        for values in itertools.product(*(axis["values"] for axis in axes)):
            case = copy.deepcopy(family["base"])
            suffix = []
            for axis, value in zip(axes, values):
                _set_path(case, axis["path"], value)
                suffix.append(
                    f"{axis.get('name', axis['path'].split('.')[-1])}-{_label(value)}"
                )
            case["id"] = family["id"] + (
                "__" + "__".join(suffix) if suffix else ""
            )
            cases.append(case)
    return cases


def expand_manifest(manifest: dict[str, Any]) -> list[Task]:
    """Expand every declared specification in deterministic case-ID order."""
    cases = _manifest_cases(manifest)
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest case IDs must be unique")
    numerics = manifest["numerics"]
    return [
        Task(
            case_id=case["id"],
            economic_configuration={
                key: copy.deepcopy(value) for key, value in case.items() if key != "id"
            },
            numerical_configuration=copy.deepcopy(numerics),
        )
        for case in sorted(cases, key=lambda item: item["id"])
    ]


def _replace_nonfinite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _replace_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_nonfinite(item) for item in value]
    return value


def run_paper(
    manifest_path: str | Path = "experiments/foa_paper.yaml",
    output_dir: str | Path = "output/foa-paper",
) -> list[dict[str, Any]]:
    """Run all paper cases and save one readable JSON record per case."""
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    tasks = expand_manifest(manifest)
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task in tasks:
        case = {"id": task.case_id, **copy.deepcopy(task.economic_configuration)}
        try:
            record = {
                "case_id": task.case_id,
                "execution_status": "completed",
                "numerical_configuration": task.numerical_configuration,
                "result": run_case(case, task.numerical_configuration),
            }
        except Exception as error:
            record = {
                "case_id": task.case_id,
                "execution_status": "failed",
                "numerical_configuration": task.numerical_configuration,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        record = _replace_nonfinite(record)
        (results_dir / f"{task.case_id}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        records.append(record)
    return records
