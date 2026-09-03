"""Small shared helpers for the two FOA paper summary figures."""

from __future__ import annotations

from pathlib import Path

import yaml

# Paper palette and summary geometry.
TEAL = "#3b6978"
RED = "#c95f59"
BLACK = "#222222"
GRAY = "#aaaaaa"
GRID = "#dddddd"
SEPARATOR = "#eeeeee"
# US Letter with one-inch manuscript margins leaves a 6.5-inch text block and
# roughly eight inches for a full-page graphic above its caption.
SUMMARY_FIGSIZE = (6.5, 7.98)
SUMMARY_LINEWIDTH = 2.3
BENCHMARK_MARKER_SIZE = 38


def legend_marker_size() -> float:
    return BENCHMARK_MARKER_SIZE**0.5


def summary_rows(
    exercise: str,
    manifest_path: str | Path = "experiments/foa_paper.yaml",
) -> list[tuple[str, str, str | None]]:
    """Read the declared paper rows for one FOA summary."""
    if exercise not in {"principal", "fixed_action"}:
        raise ValueError(f"unknown FOA summary exercise: {exercise!r}")
    payload = yaml.safe_load(Path(manifest_path).read_text())
    section = payload.get("foa_summary", {})
    rows = list(section.get("rows", []))
    if exercise == "fixed_action":
        insertion = section.get("fixed_action_insertions", {})
        added = insertion.get("rows", [])
        after = insertion.get("after")
        positions = [index for index, row in enumerate(rows) if row.get("spec") == after]
        if len(positions) != 1 or not isinstance(added, list):
            raise ValueError("invalid fixed-action FOA row insertion")
        rows[positions[0] + 1 : positions[0] + 1] = added

    result = []
    for row in rows:
        kind, label, spec = row.get("kind"), row.get("label"), row.get("spec")
        if kind not in {"header", "subheader", "data", "data_bold"}:
            raise ValueError(f"invalid FOA summary row: {row!r}")
        if not isinstance(label, str):
            raise ValueError(f"FOA summary row lacks a label: {row!r}")
        if kind in {"data", "data_bold"} and not isinstance(spec, str):
            raise ValueError(f"FOA data row lacks a specification: {row!r}")
        if kind in {"header", "subheader"} and spec is not None:
            raise ValueError(f"FOA heading unexpectedly has a specification: {row!r}")
        result.append((kind, label, spec))
    return result
