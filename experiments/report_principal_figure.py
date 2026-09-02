"""Create the principal-problem summary figure from saved results only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from replication.config import foa_summary_rows
from replication.style import (
    BENCHMARK_MARKER_SIZE,
    BLACK,
    GRID,
    RED,
    SEPARATOR,
    SUMMARY_FIGSIZE,
    SUMMARY_LINEWIDTH,
    TEAL,
    legend_marker_size,
)

REPORT_VALID_TOLERANCE_CE = 0.001  # Atlas units are $1,000, so this is $1.

# Paper order and display labels come from the common asset manifest.
ROWS = foa_summary_rows("principal")


def _atomic_results(input_dir: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in (input_dir / "atomic").glob("*.json"):
        atomic = json.loads(path.read_text())
        if "principal" in atomic["result"].get("exercises", {}):
            results[atomic["case_id"]] = atomic
    return results


def _report_threshold(exercise: dict[str, Any]) -> tuple[float | None, str]:
    points = exercise.get("points", []) + exercise.get("refinement_points", [])
    # The paper exercise is defined only for nonnegative reservation wages.
    ordered = sorted(
        (point for point in points if float(point["reservation_wage"]) >= 0.0),
        key=lambda point: float(point["reservation_wage"]),
    )
    if not ordered:
        return None, "not_reached"
    works = [
        float(point["deviation"]["ce_gain"]) <= REPORT_VALID_TOLERANCE_CE
        for point in ordered
    ]
    for index, (point, valid) in enumerate(zip(ordered, works)):
        if valid and all(works[index:]):
            return max(0.0, float(point["reservation_wage"])), "observed"
    return None, "not_reached"


def build_rows(input_dir: Path) -> list[dict[str, Any]]:
    atomic = _atomic_results(input_dir)
    competitive_payload = json.loads((input_dir / "competitive_benchmarks.json").read_text())
    competitive = {row["case_id"]: row for row in competitive_payload["records"]}
    data = []
    for kind, label, case_id in ROWS:
        row: dict[str, Any] = {"kind": kind, "label": label, "case_id": case_id}
        if kind in {"data", "data_bold"}:
            item = atomic[case_id]
            result = item["result"]
            benchmark = competitive[case_id]
            selected = result.get("monopsony", {}).get("full_gic", {}).get("selected")
            # Student-t skips the redundant plateau scan. Its action set begins
            # at zero and C(0)=0, so limited liability guarantees CE >= 0 and
            # makes the full-GIC solve at a -1 reservation CE strictly slack.
            if selected is None and benchmark.get("history"):
                selected = min(benchmark["history"], key=lambda item: item["reservation_wage"])
            threshold, threshold_status = _report_threshold(result["exercises"]["principal"])
            row.update({
                "monopsony_ce_wage": None if selected is None else max(0.0, float(selected["delivered_ce_wage"])),
                "foa_threshold_ce_wage": threshold,
                "foa_threshold_status": threshold_status,
                "competitive_ce_wage": benchmark.get("competitive_ce_wage"),
                "competitive_status": benchmark["status"],
            })
        data.append(row)
    return data


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    records = [row for row in rows if row["kind"] in {"data", "data_bold"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_figure(rows: list[dict[str, Any]], output: Path) -> None:
    nrows = len(rows)
    fig, (label_ax, plot_ax) = plt.subplots(
        1,
        2,
        figsize=SUMMARY_FIGSIZE,
        gridspec_kw={"width_ratios": [3.65, 5.35], "wspace": 0.04},
    )
    for axis in (label_ax, plot_ax):
        axis.set_ylim(nrows - 0.35, -1.2)
        axis.set_yticks([])
    label_ax.axis("off")

    data_rows = [row for row in rows if row["kind"] in {"data", "data_bold"}]
    max_competitive = max(float(row["competitive_ce_wage"]) for row in data_rows if row["competitive_ce_wage"] is not None)
    plot_ax.set_xlim(0, max(110, 1.04 * max_competitive))
    plot_ax.xaxis.tick_top()
    plot_ax.xaxis.set_label_position("top")
    plot_ax.set_xlabel("Reservation certainty-equivalent wage ($1,000)", labelpad=9)
    plot_ax.grid(axis="x", color=GRID, linewidth=0.7)
    plot_ax.spines[["left", "right", "bottom"]].set_visible(False)
    plot_ax.tick_params(axis="y", length=0)

    label_ax.text(0, -0.65, "Specification", weight="bold", va="bottom")

    for y, row in enumerate(rows):
        kind = row["kind"]
        if kind == "header":
            label_ax.text(0, y, row["label"], weight="bold", va="center", fontsize=9.5)
            for axis in (label_ax, plot_ax):
                axis.axhline(y + 0.43, color=SEPARATOR, linewidth=0.6, zorder=0)
            continue
        if kind == "subheader":
            label_ax.text(0.06, y, row["label"], style="italic", color="#666666", va="center", fontsize=8)
            continue

        if kind == "data_bold":
            for axis in (label_ax, plot_ax):
                axis.axhline(y - 0.5, color=SEPARATOR, linewidth=0.6, zorder=0)
        label_ax.text(
            0 if kind == "data_bold" else 0.12,
            y,
            row["label"],
            va="center",
            fontsize=8.3 if kind == "data" else 9.2,
            weight="bold" if kind == "data_bold" else "normal",
        )
        monopsony = row["monopsony_ce_wage"]
        threshold = row["foa_threshold_ce_wage"]
        competitive = row["competitive_ce_wage"]
        if monopsony is None or competitive is None:
            continue
        x_left, x_right = plot_ax.get_xlim()
        if threshold is None:
            plot_ax.plot([x_left, x_right], [y, y], color=RED, linewidth=SUMMARY_LINEWIDTH, zorder=3)
        else:
            plot_ax.plot([x_left, threshold], [y, y], color=RED, linewidth=SUMMARY_LINEWIDTH, zorder=3)
            plot_ax.plot([threshold, x_right], [y, y], color=TEAL, linewidth=SUMMARY_LINEWIDTH, zorder=3)
        # Downward triangles sit clear of the validity line and point to the
        # benchmark locations, leaving short red failure segments visible.
        marker_y = y - 0.18
        plot_ax.scatter(
            monopsony, marker_y, marker="v", s=BENCHMARK_MARKER_SIZE,
            facecolor=BLACK, edgecolor=BLACK, linewidth=1.0, zorder=6,
            clip_on=False,
        )
        plot_ax.scatter(
            competitive, marker_y, marker="v", s=BENCHMARK_MARKER_SIZE,
            facecolor="white", edgecolor=BLACK, linewidth=1.0, zorder=6,
        )

    marker_size = legend_marker_size()
    legend = [
        Line2D([], [], marker="v", markersize=marker_size, linestyle="none", markerfacecolor=BLACK, markeredgecolor=BLACK, label="Monopsony wage"),
        Line2D([], [], marker="v", markersize=marker_size, linestyle="none", markerfacecolor="white", markeredgecolor=BLACK, label="Competitive wage"),
        Line2D([], [], color=RED, linewidth=SUMMARY_LINEWIDTH, label="FOA fails"),
        Line2D([], [], color=TEAL, linewidth=SUMMARY_LINEWIDTH, label="FOA holds"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.67, 0.018), frameon=False, fontsize=7.5, ncol=4)
    fig.subplots_adjust(top=0.94, bottom=0.055, left=0.045, right=0.985)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/foa-internal-atlas-final-v2")
    parser.add_argument("--output", default="figures/foa-principal-summary/mock.pdf")
    args = parser.parse_args()
    input_dir = Path(args.input)
    rows = build_rows(input_dir)
    output = Path(args.output)
    write_csv(rows, output.with_name("mock_data.csv"))
    make_figure(rows, output)
    print(f"Saved {output}, {output.with_suffix('.png')}, and {output.with_name('mock_data.csv')}")


if __name__ == "__main__":
    main()
