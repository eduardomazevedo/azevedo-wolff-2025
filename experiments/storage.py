"""Deterministic task expansion and durable storage for FOA experiments."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from experiments.prototype import run_case, write_outputs


@dataclass(frozen=True)
class Task:
    task_hash: str
    case_id: str
    suites: tuple[str, ...]
    economic_configuration: dict[str, Any]
    numerical_configuration: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _economic_configuration(case: dict[str, Any]) -> dict[str, Any]:
    """Remove manifest labels that do not alter the mathematical task."""
    return {key: copy.deepcopy(value) for key, value in case.items() if key not in {"id", "suites"}}


def task_hash(case: dict[str, Any], numerics: dict[str, Any], schema_version: int) -> str:
    payload = {
        "schema_version": schema_version,
        "economic_configuration": _economic_configuration(case),
        "numerical_configuration": numerics,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = copy.deepcopy(value)


def _label(value: Any) -> str:
    text = _canonical_json(value).replace('"', "").replace(" ", "")
    return "".join(character if character.isalnum() or character in ".-" else "_" for character in text)


def _manifest_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [copy.deepcopy(case) for case in manifest.get("cases", [])]
    for family in manifest.get("case_families", []):
        axes = sorted(family.get("axes", []), key=lambda axis: axis["path"])
        value_lists = [axis["values"] for axis in axes]
        for values in itertools.product(*value_lists):
            case = copy.deepcopy(family["base"])
            suffix = []
            for axis, value in zip(axes, values):
                _set_path(case, axis["path"], value)
                suffix.append(f"{axis.get('name', axis['path'].split('.')[-1])}-{_label(value)}")
            case["id"] = family["id"] + ("__" + "__".join(suffix) if suffix else "")
            case["suites"] = copy.deepcopy(family.get("suites", case.get("suites", [])))
            cases.append(case)
    return cases


def expand_manifest(manifest: dict[str, Any], suite: str | None = "smoke") -> list[Task]:
    """Expand a manifest deterministically into case-level atomic tasks.

    A case is currently the atomic unit because support certification, monopsony,
    and threshold scans share expensive state. Labels and suite membership do not
    affect its hash, so the same task is deduplicated across suites.
    """
    cases = _manifest_cases(manifest)
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest case IDs must be unique")
    selected = [case for case in cases if suite is None or suite in case.get("suites", [])]
    tasks = [
        Task(
            task_hash=task_hash(case, manifest["numerics"], int(manifest["schema_version"])),
            case_id=case["id"],
            suites=tuple(sorted(case.get("suites", []))),
            economic_configuration=_economic_configuration(case),
            numerical_configuration=copy.deepcopy(manifest["numerics"]),
        )
        for case in selected
    ]
    tasks.sort(key=lambda task: (task.case_id, task.task_hash))
    hashes = [task.task_hash for task in tasks]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Manifest contains duplicate economic/numerical tasks")
    return tasks


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_provenance(name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name}
    try:
        distribution = importlib.metadata.distribution(name)
        result["version"] = distribution.version
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            result["direct_url"] = json.loads(direct_url)
    except importlib.metadata.PackageNotFoundError:
        result["version"] = None
    return result


def _source_state() -> tuple[str | None, dict[str, dict[str, str]]]:
    tracked_diff = _command_output(["git", "diff", "--binary", "HEAD"])
    untracked_names = (_command_output(["git", "ls-files", "--others", "--exclude-standard"]) or "").splitlines()
    untracked: dict[str, dict[str, str]] = {}
    for name in sorted(untracked_names):
        data = Path(name).read_bytes()
        untracked[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "content_base64": base64.b64encode(data).decode("ascii"),
        }
    return tracked_diff, untracked


def environment_provenance(source_state: tuple[str | None, dict[str, dict[str, str]]] | None = None) -> dict[str, Any]:
    commit = _command_output(["git", "rev-parse", "HEAD"])
    source_diff, untracked = source_state or _source_state()
    state_hash = hashlib.sha256()
    state_hash.update((source_diff or "").encode("utf-8"))
    state_hash.update(_canonical_json(untracked).encode("utf-8"))
    dirty = bool(source_diff or untracked) if commit else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": commit,
        "git_dirty": dirty,
        "git_diff_sha256": hashlib.sha256(source_diff.encode("utf-8")).hexdigest() if source_diff else None,
        "git_untracked_files": sorted(untracked),
        "git_source_state_sha256": state_hash.hexdigest() if dirty else None,
        "moralhazard": _package_provenance("moralhazard"),
    }


def _reviews_for_case(manifest: dict[str, Any], case_id: str) -> list[dict[str, Any]]:
    required = {"strict_numerical_status", "review_status", "economic_materiality", "review_notes"}
    reviews = []
    for review in manifest.get("reviews", []):
        if not review.get("active", True) or review.get("case_id") != case_id:
            continue
        missing = required - review.keys()
        if missing:
            raise ValueError(f"Review {review.get('id', case_id)!r} lacks fields: {sorted(missing)}")
        reviews.append(copy.deepcopy(review))
    return sorted(reviews, key=lambda item: item.get("id", ""))


def _strict_status(result: dict[str, Any]) -> str:
    unresolved = result.get("support_validation", {}).get("status") != "passed"
    unresolved |= result.get("boundary_diagnostics", {}).get("status", "passed") != "passed"
    for benchmark in result.get("monopsony", {}).values():
        if isinstance(benchmark, dict) and "status" in benchmark and benchmark["status"] != "ok":
            unresolved = True
    for exercise in result.get("exercises", {}).values():
        for point in exercise.get("points", []) + exercise.get("refinement_points", []):
            unresolved |= point.get("classification") == "unresolved" or bool(point.get("warnings"))
        unresolved |= any(check.get("status") != "passed" for check in exercise.get("validation", []))
    return "unresolved" if unresolved else "passed"


def _replace_nonfinite(value: Any, path: str = "$") -> tuple[Any, list[str]]:
    if isinstance(value, float) and not math.isfinite(value):
        return None, [path]
    if isinstance(value, dict):
        clean, paths = {}, []
        for key, item in value.items():
            clean[key], found = _replace_nonfinite(item, f"{path}.{key}")
            paths.extend(found)
        return clean, paths
    if isinstance(value, list):
        clean, paths = [], []
        for index, item in enumerate(value):
            replaced, found = _replace_nonfinite(item, f"{path}[{index}]")
            clean.append(replaced)
            paths.extend(found)
        return clean, paths
    if isinstance(value, tuple):
        return _replace_nonfinite(list(value), path)
    return value, []


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_index(output: Path, records: list[dict[str, Any]]) -> None:
    _write_json_atomic(output / "task_index.json", records)
    if records:
        with (output / "task_index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def run_manifest_atomic(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    suite: str | None = "smoke",
    resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Run selected tasks, writing one crash-safe JSON record per task."""
    manifest_path = Path(manifest_path)
    manifest_text = manifest_path.read_text()
    manifest = yaml.safe_load(manifest_text)
    tasks = expand_manifest(manifest, suite)
    output = Path(output_dir)
    atomic_dir = output / "atomic"
    arrays_dir = output / "arrays"
    atomic_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    (output / "manifest.snapshot.yaml").write_text(manifest_text)
    source_state = _source_state()
    source_diff, untracked = source_state
    environment = environment_provenance(source_state)
    _write_json_atomic(output / "environment.json", environment)
    diff_path = output / "source.diff.patch"
    if source_diff:
        diff_path.write_text(source_diff + "\n")
    elif diff_path.exists():
        diff_path.unlink()
    untracked_path = output / "source.untracked.json"
    if untracked:
        _write_json_atomic(untracked_path, untracked)
    elif untracked_path.exists():
        untracked_path.unlink()

    index: list[dict[str, Any]] = []
    successful_results: list[dict[str, Any]] = []
    for task in tasks:
        path = atomic_dir / f"{task.task_hash}.json"
        cached = False
        record: dict[str, Any]
        if resume and path.exists():
            try:
                record = json.loads(path.read_text())
                reusable_status = record.get("execution_status") == "completed" or (
                    record.get("execution_status") == "failed" and not retry_failed
                )
                cached = reusable_status and record.get("task_hash") == task.task_hash
            except (json.JSONDecodeError, OSError):
                cached = False
        if not cached:
            started = datetime.now(timezone.utc)
            clock = time.perf_counter()
            case = {"id": task.case_id, "suites": list(task.suites), **copy.deepcopy(task.economic_configuration)}
            try:
                result = run_case(case, task.numerical_configuration)
                reviews = _reviews_for_case(manifest, task.case_id)
                result["strict_numerical_status"] = _strict_status(result)
                result["reviews"] = reviews
                result["review_status"] = "partially_reviewed" if reviews else "unreviewed"
                result["economic_materiality"] = sorted({r["economic_materiality"] for r in reviews})
                result["review_notes"] = [r["review_notes"] for r in reviews]
                result["diagnostic_paths"] = sorted({r["diagnostic_path"] for r in reviews if r.get("diagnostic_path")})
                record = {
                    "schema_version": manifest["schema_version"],
                    "experiment_id": manifest["experiment_id"],
                    "task_hash": task.task_hash,
                    "case_id": task.case_id,
                    "suites": list(task.suites),
                    "economic_configuration": task.economic_configuration,
                    "numerical_configuration": task.numerical_configuration,
                    "execution_status": "completed",
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "runtime_seconds": time.perf_counter() - clock,
                    "provenance": environment,
                    "result": result,
                }
            except Exception as error:  # Keep failed tasks indexable and resumable.
                record = {
                    "schema_version": manifest["schema_version"],
                    "experiment_id": manifest["experiment_id"],
                    "task_hash": task.task_hash,
                    "case_id": task.case_id,
                    "suites": list(task.suites),
                    "economic_configuration": task.economic_configuration,
                    "numerical_configuration": task.numerical_configuration,
                    "execution_status": "failed",
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "runtime_seconds": time.perf_counter() - clock,
                    "provenance": environment,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            record, nonfinite_paths = _replace_nonfinite(record)
            if nonfinite_paths:
                record["nonfinite_value_paths"] = nonfinite_paths
                if record.get("result"):
                    record["result"]["strict_numerical_status"] = "unresolved"
            _write_json_atomic(path, record)
        if record.get("execution_status") == "completed":
            successful_results.append(record["result"])
        index.append({
            "task_hash": task.task_hash,
            "case_id": task.case_id,
            "execution_status": record.get("execution_status", "invalid_cache"),
            "strict_numerical_status": record.get("result", {}).get("strict_numerical_status", "not_available"),
            "cached": cached,
            "runtime_seconds": record.get("runtime_seconds"),
            "atomic_path": str(path.relative_to(output)),
        })
        _write_index(output, index)

    payload = {
        "schema_version": manifest["schema_version"],
        "experiment_id": manifest["experiment_id"],
        "suite": suite if suite is not None else "all",
        "manifest": str(manifest_path),
        "results": successful_results,
    }
    # Temporary compatibility summaries; atomic records are authoritative.
    write_outputs(payload, output)
    return {"payload": payload, "task_index": index}
