"""Shared visual style for all paper figures."""

from __future__ import annotations

# Paper palette.
TEAL = "#3b6978"
RED = "#c95f59"
BLACK = "#222222"
GRAY = "#aaaaaa"
GRID = "#dddddd"
SEPARATOR = "#eeeeee"

# Summary-figure geometry.
SUMMARY_FIGSIZE = (8.35, 11.2)
SUMMARY_LINEWIDTH = 2.3
BENCHMARK_MARKER_SIZE = 38


def legend_marker_size(scatter_area: float = BENCHMARK_MARKER_SIZE) -> float:
    """Convert Matplotlib scatter area to the matching Line2D marker size."""
    return scatter_area**0.5
