"""Create the principal-problem summary figure from saved results only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPORT_VALID_TOLERANCE_CE = 0.001  # Atlas units are $1,000, so this is $1.

# Deliberate paper order and short display labels. Detailed definitions remain
# in the manifest and can be reproduced in an appendix table.
ROWS = [
    ("header", "Gaussian", None),
    ("data", "Baseline (log, σ = 50)", "gaussian_log_paper"),
    ("subheader", "Risk aversion", None),
    ("data", "CRRA γ = 0.25", "gaussian_crra__gamma-0.25"),
    ("data", "CRRA γ = 0.50", "gaussian_crra__gamma-0.5"),
    ("data", "CRRA γ = 1.50", "gaussian_crra__gamma-1.5"),
    ("data", "CRRA γ = 2", "gaussian_crra__gamma-2"),
    ("data", "CARA RRA = 0.25", "gaussian_cara__alpha-0.005"),
    ("data", "CARA RRA = 0.50", "gaussian_cara__alpha-0.01"),
    ("data", "CARA RRA = 1", "gaussian_cara__alpha-0.02"),
    ("data", "CARA RRA = 2", "gaussian_cara__alpha-0.04"),
    ("subheader", "Noise", None),
    ("data", "σ = 10", "gaussian_sigma_log__sigma-10"),
    ("data", "σ = 20", "gaussian_sigma_log__sigma-20"),
    ("subheader", "Initial wealth", None),
    ("data", "w₀ = 25", "gaussian_wealth_log__wealth-25"),
    ("data", "w₀ = 100", "gaussian_wealth_log__wealth-100"),
    ("data", "w₀ = 200", "gaussian_wealth_log__wealth-200"),
    ("subheader", "Effort cost", None),
    ("data", "Half cost", "gaussian_cost_log__cost_scale-0.5"),
    ("data", "Double cost", "gaussian_cost_log__cost_scale-2"),
    ("data_bold", "Poisson", "poisson_log_paper"),
    ("data_bold", "Exponential", "exponential_log_baseline"),
    ("data_bold", "Gamma (shape = 2)", "gamma2_log_baseline"),
    ("data_bold", "Geometric", "geometric_log_baseline"),
    ("data_bold", "Bernoulli", "bernoulli_log_baseline"),
    ("data_bold", "Binomial (n = 10)", "binomial10_log_baseline"),
    ("data_bold", "Student-t (adverse case)", "student_t_log_adverse"),
]


def _atomic_results(input_dir: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in (input_dir / "atomic").glob("*.json"):
        atomic = json.loads(path.read_text())
        if "principal" in atomic["result"].get("exercises", {}):
            results[atomic["case_id"]] = atomic
    return results


def _report_threshold(exercise: dict[str, Any]) -> tuple[float | None, str]:
    points = exercise.get("points", []) + exercise.get("refinement_points", [])
    ordered = sorted(points, key=lambda point: float(point["reservation_wage"]))
    if not ordered:
        return None, "not_reached"
    works = [
        float(point["deviation"]["ce_gain"]) <= REPORT_VALID_TOLERANCE_CE
        for point in ordered
    ]
    for index, (point, valid) in enumerate(zip(ordered, works)):
        if valid and all(works[index:]):
            status = "left_censored" if index == 0 else "observed"
            return float(point["reservation_wage"]), status
    return None, "not_reached"


def _empty_safe_region(result: dict[str, Any]) -> bool:
    distribution = result["effective_configuration"]["distribution"]["kind"].lower()
    # These families have analytically nonempty safe regions under the
    # configured interior action sets. Prefer that fact to finite-difference
    # diagnostics, which are cancellation-prone for affine Bernoulli masses.
    known_nonempty = {
        "gaussian", "poisson", "exponential", "gamma", "geometric",
        "bernoulli", "binomial",
    }
    if distribution in known_nonempty:
        return False
    full = result.get("monopsony", {}).get("full_gic", {})
    metrics = full.get("safe_region_metrics") or result.get("safe_region_metrics", {})
    return metrics.get("safe_outcome_region_on_grid") is None


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
            # The Student-t adverse control deliberately skipped the atlas
            # monopsony scan. Its competitive benchmark's lowest-requirement
            # full-GIC solve is slack and supplies the display benchmark.
            if selected is None and benchmark.get("history"):
                selected = min(benchmark["history"], key=lambda item: item["reservation_wage"])
            threshold, threshold_status = _report_threshold(result["exercises"]["principal"])
            row.update({
                "monopsony_ce_wage": None if selected is None else float(selected["delivered_ce_wage"]),
                "foa_threshold_ce_wage": threshold,
                "foa_threshold_status": threshold_status,
                "competitive_ce_wage": benchmark.get("competitive_ce_wage"),
                "competitive_status": benchmark["status"],
                "empty_safe_region": _empty_safe_region(result),
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
    fig, (label_ax, safe_ax, plot_ax) = plt.subplots(
        1,
        3,
        figsize=(8.35, 11.2),
        gridspec_kw={"width_ratios": [3.15, 0.9, 5.25], "wspace": 0.03},
    )
    for axis in (label_ax, safe_ax, plot_ax):
        axis.set_ylim(nrows - 0.35, -1.2)
        axis.set_yticks([])
    label_ax.axis("off")
    safe_ax.axis("off")

    data_rows = [row for row in rows if row["kind"] in {"data", "data_bold"}]
    max_competitive = max(float(row["competitive_ce_wage"]) for row in data_rows if row["competitive_ce_wage"] is not None)
    plot_ax.set_xlim(-5, max(110, 1.04 * max_competitive))
    plot_ax.xaxis.tick_top()
    plot_ax.xaxis.set_label_position("top")
    plot_ax.set_xlabel("Reservation certainty-equivalent wage ($1,000)", labelpad=9)
    plot_ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    plot_ax.spines[["left", "right", "bottom"]].set_visible(False)
    plot_ax.tick_params(axis="y", length=0)

    label_ax.text(0, -0.65, "Specification", weight="bold", va="bottom")
    safe_ax.text(0.5, -0.82, "Safe region\nempty?", weight="bold", ha="center", va="bottom", fontsize=8.5, linespacing=0.9)

    for y, row in enumerate(rows):
        kind = row["kind"]
        if kind == "header":
            label_ax.text(0, y, row["label"], weight="bold", va="center", fontsize=9.5)
            for axis in (label_ax, safe_ax, plot_ax):
                axis.axhline(y + 0.43, color="#eeeeee", linewidth=0.6, zorder=0)
            continue
        if kind == "subheader":
            label_ax.text(0.06, y, row["label"], style="italic", color="#666666", va="center", fontsize=8)
            continue

        if kind == "data_bold":
            for axis in (label_ax, safe_ax, plot_ax):
                axis.axhline(y - 0.5, color="#eeeeee", linewidth=0.6, zorder=0)
        label_ax.text(
            0 if kind == "data_bold" else 0.12,
            y,
            row["label"],
            va="center",
            fontsize=8.3 if kind == "data" else 9.2,
            weight="bold" if kind == "data_bold" else "normal",
        )
        safe_ax.text(
            0.5,
            y,
            "Yes" if row["empty_safe_region"] else "", 
            ha="center",
            va="center",
            fontsize=8.1,
            color="#444444",
        )
        monopsony = row["monopsony_ce_wage"]
        threshold = row["foa_threshold_ce_wage"]
        competitive = row["competitive_ce_wage"]
        if monopsony is None or competitive is None:
            continue
        x_left, x_right = plot_ax.get_xlim()
        if threshold is None:
            plot_ax.plot([x_left, x_right], [y, y], color="#c95f59", linewidth=2.3, zorder=2)
        else:
            plot_ax.plot([x_left, threshold], [y, y], color="#c95f59", linewidth=2.3, zorder=2)
            plot_ax.plot([threshold, x_right], [y, y], color="#41965a", linewidth=2.3, zorder=2)
            marker = "<" if row["foa_threshold_status"] == "left_censored" else "D"
            plot_ax.scatter(threshold, y, marker=marker, s=42, color="#41965a", edgecolor="white", linewidth=0.35, zorder=5)
        plot_ax.scatter(monopsony, y, marker="o", s=18, color="#2967a3", edgecolor="white", linewidth=0.4, zorder=6)
        plot_ax.scatter(competitive, y, marker="o", s=28, facecolor="white", edgecolor="#222222", linewidth=1.0, zorder=6)

    legend = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#2967a3", markeredgecolor="white", label="Monopsony wage"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="#41965a", markeredgecolor="white", label="Lowest FOA-valid wage"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#222222", label="Competitive wage"),
        Line2D([], [], color="#41965a", linewidth=2.3, label="FOA works"),
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
