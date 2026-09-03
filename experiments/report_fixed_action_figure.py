"""Create the fixed-action cost-minimization FOA summary from saved results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .report_common import (
    BENCHMARK_MARKER_SIZE,
    BLACK,
    GRAY,
    GRID,
    RED,
    SEPARATOR,
    SUMMARY_FIGSIZE,
    SUMMARY_LINEWIDTH,
    TEAL,
    legend_marker_size,
    summary_rows,
)

from .report_principal_figure import _format_dollars, _report_threshold


# Principal and fixed-action results are never pooled: this script reads only
# the fixed_action exercise from each paper specification.
ROWS = summary_rows("fixed_action")


def _saved_results(input_dir: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in (input_dir / "results").glob("*.json"):
        saved = json.loads(path.read_text())
        if "fixed_action" in saved["result"].get("exercises", {}):
            results[saved["case_id"]] = saved
    return results


def build_rows(input_dir: Path) -> list[dict[str, Any]]:
    saved_results = _saved_results(input_dir)
    benchmark_payload = json.loads((input_dir / "fixed_action_benchmarks.json").read_text())
    benchmarks = {row["case_id"]: row for row in benchmark_payload["records"]}
    data: list[dict[str, Any]] = []
    for kind, label, case_id in ROWS:
        row: dict[str, Any] = {"kind": kind, "label": label, "case_id": case_id}
        if kind in {"data", "data_bold"}:
            item = saved_results[case_id]
            result = item["result"]
            exercise = result["exercises"]["fixed_action"]
            infeasible = exercise.get("status") == "infeasible_local_incentives"
            threshold, threshold_status = (
                (None, "infeasible") if infeasible else _report_threshold(exercise)
            )
            benchmark = benchmarks[case_id]
            monopsony_selected = benchmark.get("monopsony", {}).get("selected")
            competitive = benchmark.get("competitive", {})
            row.update({
                "intended_action": float(result["effective_configuration"]["fixed_action"]),
                "foa_threshold_ce_wage": threshold,
                "foa_threshold_status": threshold_status,
                "fixed_action_monopsony_ce_wage": (
                    None if monopsony_selected is None
                    else max(0.0, float(monopsony_selected["delivered_ce_wage"]))
                ),
                "fixed_action_monopsony_status": benchmark.get("monopsony", {}).get("status"),
                "fixed_action_competitive_ce_wage": competitive.get("competitive_ce_wage"),
                "fixed_action_competitive_status": competitive.get("status"),
                "infeasible": infeasible,
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
    displayed_wages = [
        float(value)
        for row in data_rows
        for value in (
            row.get("foa_threshold_ce_wage"),
            row.get("fixed_action_competitive_ce_wage"),
        )
        if value is not None
    ]
    x_right = max(110.0, max(displayed_wages, default=0.0) * 1.04)
    plot_ax.set_xlim(0, x_right)
    plot_ax.xaxis.tick_top()
    plot_ax.xaxis.set_label_position("top")
    plot_ax.set_xlabel("Reservation certainty-equivalent wage ($1,000)", labelpad=9)
    plot_ax.grid(axis="x", color=GRID, linewidth=0.7)
    plot_ax.spines[["left", "right", "bottom"]].set_visible(False)
    plot_ax.tick_params(axis="y", length=0)

    label_ax.text(0, -0.65, "Specification", weight="bold", va="bottom")
    label_ax.text(
        0.98,
        -0.65,
        "FOA valid\nstarting at",
        weight="bold",
        ha="right",
        va="bottom",
        fontsize=8.3,
        linespacing=0.9,
    )

    for y, row in enumerate(rows):
        kind = row["kind"]
        if kind == "header":
            label_ax.text(0, y, row["label"], weight="bold", va="center", fontsize=9.5)
            for axis in (label_ax, plot_ax):
                axis.axhline(y + 0.43, color=SEPARATOR, linewidth=0.6, zorder=0)
            continue
        if kind == "subheader":
            label_ax.text(
                0.06, y, row["label"], style="italic", color="#666666",
                va="center", fontsize=8,
            )
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
        threshold = row["foa_threshold_ce_wage"]
        if threshold is not None:
            label_ax.text(
                0.98,
                y,
                _format_dollars(threshold),
                ha="right",
                va="center",
                fontsize=7.2,
                weight="bold" if kind == "data_bold" else "normal",
            )
        if row["infeasible"]:
            plot_ax.plot([0, x_right], [y, y], color=GRAY, linewidth=1.5, zorder=2)
            plot_ax.text(
                2.0, y - 0.13, "Locally infeasible", color="#777777",
                fontsize=7.0, va="bottom",
            )
            continue

        if threshold is None:
            plot_ax.plot([0, x_right], [y, y], color=RED, linewidth=SUMMARY_LINEWIDTH, zorder=3)
        else:
            plot_ax.plot([0, threshold], [y, y], color=RED, linewidth=SUMMARY_LINEWIDTH, zorder=3)
            plot_ax.plot([threshold, x_right], [y, y], color=TEAL, linewidth=SUMMARY_LINEWIDTH, zorder=3)

        marker_y = y - 0.18
        monopsony = row["fixed_action_monopsony_ce_wage"]
        competitive = row["fixed_action_competitive_ce_wage"]
        benchmarks_close = (
            monopsony is not None
            and competitive is not None
            and abs(competitive - monopsony) < 0.1 * x_right
        )
        if monopsony is not None:
            plot_ax.scatter(
                monopsony, marker_y, marker="v", s=BENCHMARK_MARKER_SIZE,
                facecolor=BLACK, edgecolor=BLACK, linewidth=1.0, zorder=6,
                clip_on=False,
            )
            near_edge = monopsony < 1.0
            plot_ax.annotate(
                _format_dollars(monopsony),
                xy=(monopsony, marker_y),
                xytext=(-2 if benchmarks_close else (2 if near_edge else 0), 3),
                textcoords="offset points",
                ha="right" if benchmarks_close else ("left" if near_edge else "center"),
                va="bottom",
                fontsize=5.8,
                color=BLACK,
                zorder=7,
                clip_on=False,
            )
        if competitive is not None:
            plot_ax.scatter(
                competitive, marker_y, marker="v", s=BENCHMARK_MARKER_SIZE,
                facecolor="white", edgecolor=BLACK, linewidth=1.0, zorder=6,
            )
            near_edge = competitive < 1.0
            plot_ax.annotate(
                _format_dollars(competitive),
                xy=(competitive, marker_y),
                xytext=(2 if benchmarks_close or near_edge else 0, 3),
                textcoords="offset points",
                ha="left" if benchmarks_close or near_edge else "center",
                va="bottom",
                fontsize=5.8,
                color=BLACK,
                zorder=7,
                clip_on=False,
            )

    marker_size = legend_marker_size()
    legend = [
        Line2D([], [], marker="v", markersize=marker_size, linestyle="none", markerfacecolor=BLACK, markeredgecolor=BLACK, label="Fixed-action monopsony wage"),
        Line2D([], [], marker="v", markersize=marker_size, linestyle="none", markerfacecolor="white", markeredgecolor=BLACK, label="Fixed-action competitive wage"),
        Line2D([], [], color=RED, linewidth=SUMMARY_LINEWIDTH, label="FOA fails"),
        Line2D([], [], color=TEAL, linewidth=SUMMARY_LINEWIDTH, label="FOA holds"),
        Line2D([], [], color=GRAY, linewidth=1.5, label="Locally infeasible"),
    ]
    fig.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.69, 0.018),
        frameon=False, fontsize=7.0, ncol=3,
    )
    fig.subplots_adjust(top=0.94, bottom=0.055, left=0.045, right=0.985)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/foa-paper")
    parser.add_argument("--output", default="figures/foa-fixed-action-summary.pdf")
    args = parser.parse_args()
    input_dir = Path(args.input)
    rows = build_rows(input_dir)
    output = Path(args.output)
    data_output = output.with_suffix(".csv")
    write_csv(rows, data_output)
    make_figure(rows, output)
    print(
        f"Saved {output}, {output.with_suffix('.png')}, and {data_output}"
    )


if __name__ == "__main__":
    main()
