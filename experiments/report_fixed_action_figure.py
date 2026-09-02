"""Create the fixed-action cost-minimization FOA summary from saved results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .report_principal_figure import (
    ROWS as PRINCIPAL_ROWS,
    _empty_safe_region,
    _report_threshold,
)


# Keep the principal figure's ordering and add the two dedicated Gaussian
# intended-action exercises. Principal and fixed-action results are never
# pooled: this script reads only the fixed_action exercise from each task.
ROWS: list[tuple[str, str, str | None]] = []
for row in PRINCIPAL_ROWS:
    ROWS.append(row)
    if row[2] == "gaussian_log_paper":
        ROWS.extend([
            ("subheader", "Intended action", None),
            ("data", "Low", "gaussian_fixed_actions_log__intended_action-70"),
            ("data", "Near monopsony", "gaussian_fixed_actions_log__intended_action-130"),
        ])


def _atomic_results(input_dir: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in (input_dir / "atomic").glob("*.json"):
        atomic = json.loads(path.read_text())
        if "fixed_action" in atomic["result"].get("exercises", {}):
            results[atomic["case_id"]] = atomic
    return results


def build_rows(input_dir: Path) -> list[dict[str, Any]]:
    atomic = _atomic_results(input_dir)
    data: list[dict[str, Any]] = []
    for kind, label, case_id in ROWS:
        row: dict[str, Any] = {"kind": kind, "label": label, "case_id": case_id}
        if kind in {"data", "data_bold"}:
            item = atomic[case_id]
            result = item["result"]
            exercise = result["exercises"]["fixed_action"]
            infeasible = exercise.get("status") == "infeasible_local_incentives"
            threshold, threshold_status = (
                (None, "infeasible") if infeasible else _report_threshold(exercise)
            )
            row.update({
                "intended_action": float(result["effective_configuration"]["fixed_action"]),
                "foa_threshold_ce_wage": threshold,
                "foa_threshold_status": threshold_status,
                "empty_safe_region": _empty_safe_region(result),
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


def _format_action(action: float) -> str:
    if action >= 10 or action.is_integer():
        return f"{action:g}"
    return f"{action:.3g}"


def make_figure(rows: list[dict[str, Any]], output: Path) -> None:
    nrows = len(rows)
    fig, (label_ax, action_ax, safe_ax, plot_ax) = plt.subplots(
        1,
        4,
        figsize=(8.35, 11.2),
        gridspec_kw={"width_ratios": [2.85, 1.05, 1.05, 5.1], "wspace": 0.08},
    )
    for axis in (label_ax, action_ax, safe_ax, plot_ax):
        axis.set_ylim(nrows - 0.35, -1.2)
        axis.set_yticks([])
    label_ax.axis("off")
    action_ax.axis("off")
    safe_ax.axis("off")

    data_rows = [row for row in rows if row["kind"] in {"data", "data_bold"}]
    tested_wages = []
    atomic_thresholds = [
        float(row["foa_threshold_ce_wage"])
        for row in data_rows
        if row.get("foa_threshold_ce_wage") is not None
    ]
    tested_wages.extend(atomic_thresholds)
    x_right = max(110.0, max(tested_wages, default=0.0) * 1.1)
    plot_ax.set_xlim(0, x_right)
    plot_ax.xaxis.tick_top()
    plot_ax.xaxis.set_label_position("top")
    plot_ax.set_xlabel("Reservation certainty-equivalent wage ($1,000)", labelpad=9)
    plot_ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    plot_ax.spines[["left", "right", "bottom"]].set_visible(False)
    plot_ax.tick_params(axis="y", length=0)

    label_ax.text(0, -0.65, "Specification", weight="bold", va="bottom")
    action_ax.text(
        0.5, -0.82, "Intended\naction a₀", weight="bold", ha="center",
        va="bottom", fontsize=8.2, linespacing=0.9,
    )
    safe_ax.text(
        0.5, -0.82, "Safe region\nempty?", weight="bold", ha="center",
        va="bottom", fontsize=8.2, linespacing=0.9,
    )

    for y, row in enumerate(rows):
        kind = row["kind"]
        if kind == "header":
            label_ax.text(0, y, row["label"], weight="bold", va="center", fontsize=9.5)
            for axis in (label_ax, action_ax, safe_ax, plot_ax):
                axis.axhline(y + 0.43, color="#eeeeee", linewidth=0.6, zorder=0)
            continue
        if kind == "subheader":
            label_ax.text(
                0.06, y, row["label"], style="italic", color="#666666",
                va="center", fontsize=8,
            )
            continue

        if kind == "data_bold":
            for axis in (label_ax, action_ax, safe_ax, plot_ax):
                axis.axhline(y - 0.5, color="#eeeeee", linewidth=0.6, zorder=0)
        label_ax.text(
            0 if kind == "data_bold" else 0.12,
            y,
            row["label"],
            va="center",
            fontsize=8.3 if kind == "data" else 9.2,
            weight="bold" if kind == "data_bold" else "normal",
        )
        action_ax.text(
            0.5, y, _format_action(row["intended_action"]),
            ha="center", va="center", fontsize=8.1, color="#333333",
        )
        safe_ax.text(
            0.5, y, "Yes" if row["empty_safe_region"] else "",
            ha="center", va="center", fontsize=8.1, color="#444444",
        )

        if row["infeasible"]:
            plot_ax.plot([0, x_right], [y, y], color="#aaaaaa", linewidth=1.5, zorder=2)
            plot_ax.text(
                2.0, y - 0.13, "Locally infeasible", color="#777777",
                fontsize=7.0, va="bottom",
            )
            continue

        threshold = row["foa_threshold_ce_wage"]
        if threshold is None:
            plot_ax.plot([0, x_right], [y, y], color="#c95f59", linewidth=2.3, zorder=3)
        else:
            plot_ax.plot([0, threshold], [y, y], color="#c95f59", linewidth=2.3, zorder=3)
            plot_ax.plot([threshold, x_right], [y, y], color="#41965a", linewidth=2.3, zorder=3)

    legend = [
        Line2D([], [], color="#c95f59", linewidth=2.3, label="FOA fails"),
        Line2D([], [], color="#41965a", linewidth=2.3, label="FOA holds"),
        Line2D([], [], color="#aaaaaa", linewidth=1.5, label="Locally infeasible"),
    ]
    fig.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.69, 0.018),
        frameon=False, fontsize=7.5, ncol=3,
    )
    fig.subplots_adjust(top=0.94, bottom=0.055, left=0.045, right=0.985)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/foa-internal-atlas-final-v2")
    parser.add_argument("--output", default="figures/foa-fixed-action-summary/mock.pdf")
    args = parser.parse_args()
    input_dir = Path(args.input)
    rows = build_rows(input_dir)
    output = Path(args.output)
    write_csv(rows, output.with_name("mock_data.csv"))
    make_figure(rows, output)
    print(
        f"Saved {output}, {output.with_suffix('.png')}, "
        f"and {output.with_name('mock_data.csv')}"
    )


if __name__ == "__main__":
    main()
