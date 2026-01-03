"""
Timing benchmark for MoralHazardProblem solvers (dual and CVXPY).

What this script does
---------------------
1) Runs multiple wall-clock timing repetitions for each solver variant:
   - Dual cost minimization problem (CMP): relaxed (no global IC refinement) and full
   - Dual principal problem
   - CVXPY CMP: relaxed (no global IC constraints) and full (101-point global IC grid)
   - CVXPY principal problem (100-action discretized grid)

2) Reports a timing table in milliseconds using robust summary statistics:
   - median_ms: median runtime over repetitions
   - iqr_ms: interquartile range (p75 - p25) over repetitions
   - mean_ms, std_ms: included for reference

3) Writes outputs to ./output/:
   - timing_results.csv  (wide table of summary stats)
   - machine_specs.txt   (hardware + OS + Python + key package versions + thread env)

Notes for papers
----------------
- Prefer reporting median and IQR (less sensitive to noise).
- Include machine_specs.txt (or its contents) in an appendix/supplement.
- For fair comparisons, keep solver tolerances/iterations fixed (as done here).
"""

import os
import sys
import time
import warnings
import platform
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg


# -----------------------------
# User-tunable benchmark knobs
# -----------------------------
warnings.filterwarnings("ignore")

# Primitives
initial_wealth = 50
sigma_gaussian = 40.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)

a_ic_lb_default = 0.0
a_ic_ub_default = 130.0

def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a

# Timing controls
intended_action = first_best_effort
n_timing_iterations = 20
warmup_iterations = 1  # discard these for each solver/case (reduces cold-start effects)

# Global IC refinement controls for dual solver
a_always_check_global_ic = np.array([])  # as in your original script
dual_full_n_a_iterations = 10            # fixed for comparability


# -----------------------------
# Helpers
# -----------------------------
def _run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"<unavailable: {e}>"

def write_machine_specs(path: Path) -> None:
    # Basic info
    lines = []
    lines.append(f"timestamp_utc: {datetime.utcnow().isoformat()}Z")
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"python: {sys.version.replace(os.linesep, ' ')}")
    lines.append(f"executable: {sys.executable}")
    lines.append("")

    # OS / kernel
    lines.append("uname -a:")
    lines.append(_run_cmd(["uname", "-a"]))
    lines.append("")

    # CPU / memory (best-effort, Linux-oriented)
    lines.append("lscpu:")
    lines.append(_run_cmd(["bash", "-lc", "command -v lscpu >/dev/null 2>&1 && lscpu || echo '<lscpu not found>'"]))
    lines.append("")
    lines.append("meminfo (first 40 lines):")
    lines.append(_run_cmd(["bash", "-lc", "test -r /proc/meminfo && head -n 40 /proc/meminfo || echo '<no /proc/meminfo>'"]))
    lines.append("")

    # Threading / BLAS env (important for reproducibility)
    env_keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    lines.append("thread_env:")
    for k in env_keys:
        v = os.environ.get(k, "")
        lines.append(f"  {k}={v!r}")
    lines.append("")

    # Key package versions (minimal, avoids dumping full pip freeze)
    lines.append("package_versions:")
    lines.append(f"  numpy={np.__version__}")
    lines.append(f"  pandas={pd.__version__}")
    try:
        import cvxpy as cp  # noqa
        lines.append(f"  cvxpy={cp.__version__}")
    except Exception as e:
        lines.append(f"  cvxpy=<unavailable: {e}>")
    try:
        import moralhazard  # noqa
        lines.append(f"  moralhazard={getattr(moralhazard, '__version__', '<no __version__>')}")
    except Exception as e:
        lines.append(f"  moralhazard=<unavailable: {e}>")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def summarize_times(seconds: list[float]) -> dict:
    arr = np.asarray(seconds, dtype=float)
    # Convert to ms
    ms = arr * 1000.0
    q25, q50, q75 = np.percentile(ms, [25, 50, 75])
    return {
        "n_runs": int(ms.size),
        "median_ms": float(q50),
        "iqr_ms": float(q75 - q25),
        "mean_ms": float(ms.mean()),
        "std_ms": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
        "p25_ms": float(q25),
        "p75_ms": float(q75),
    }

def timeit(fn, n_runs: int, n_warmup: int) -> dict:
    # Warmup (discard)
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return summarize_times(times)


# -----------------------------
# Build problem cases
# -----------------------------
utility_cfg = make_utility_cfg("log", w0=initial_wealth)
u_fun = utility_cfg["u"]

cases = []

# Gaussian (easy / hard share same mhp but different reservation utility)
sigma = sigma_gaussian
dist_cfg_gaussian = make_distribution_cfg("gaussian", sigma=sigma)
cfg_gaussian = {
    "problem_params": {**utility_cfg, **dist_cfg_gaussian, "C": C, "Cprime": Cprime},
    "computational_params": {
        "distribution_type": "continuous",
        "y_min": a_ic_lb_default - 3 * sigma,
        "y_max": a_ic_ub_default + 3 * sigma,
        "n": 201,  # must be odd
    },
}
mhp_gaussian = MoralHazardProblem(cfg_gaussian)
cases.append(("gaussian-easy", mhp_gaussian, u_fun(10)))
cases.append(("gaussian-hard", mhp_gaussian, u_fun(-10)))

# Student-t case
sigma_t = 20.0
a_min_t = 0.0
a_max_t = 100.0
dist_cfg_t = make_distribution_cfg("student_t", nu=1.15, sigma=sigma_t)
cfg_t = {
    "problem_params": {**utility_cfg, **dist_cfg_t, "C": C, "Cprime": Cprime},
    "computational_params": {
        "distribution_type": "continuous",
        "y_min": a_min_t - 10 * sigma_t,
        "y_max": a_max_t + 10 * sigma_t,
        "n": 201,  # must be odd
    },
}
mhp_t = MoralHazardProblem(cfg_t)
cases.append(("t", mhp_t, u_fun(0)))


# -----------------------------
# Run timings
# -----------------------------
print("=== Timing benchmarks ===\n")

rows = []

for case_name, mhp, reservation_utility in cases:
    print(f"Case: {case_name} (reservation_utility = {reservation_utility:.4f})")

    # Bounds / grids per case (kept consistent with your original script)
    if case_name == "t":
        a_lb, a_ub = a_min_t, a_max_t
        a_pp_min, a_pp_max = a_min_t, a_max_t
    else:
        a_lb, a_ub = a_ic_lb_default, a_ic_ub_default
        a_pp_min, a_pp_max = 0.0, 100.0

    a_hat_cvxpy = np.linspace(a_lb, a_ub, 101)
    a_grid_principal = np.linspace(a_pp_min, a_pp_max, 100)

    # 1) Dual relaxed CMP
    print("  Timing dual relaxed CMP...")
    stats_dual_relaxed = timeit(
        fn=lambda: mhp.solve_cost_minimization_problem(
            intended_action=intended_action,
            reservation_utility=reservation_utility,
            a_ic_lb=a_lb,
            a_ic_ub=a_ub,
            n_a_iterations=0,
            a_always_check_global_ic=a_always_check_global_ic,
        ),
        n_runs=n_timing_iterations,
        n_warmup=warmup_iterations,
    )
    print(f"    median: {stats_dual_relaxed['median_ms']:.2f} ms (IQR {stats_dual_relaxed['iqr_ms']:.2f})")

    # 2) Dual full CMP
    print("  Timing dual CMP...")
    stats_dual_cmp = timeit(
        fn=lambda: mhp.solve_cost_minimization_problem(
            intended_action=intended_action,
            reservation_utility=reservation_utility,
            a_ic_lb=a_lb,
            a_ic_ub=a_ub,
            n_a_iterations=dual_full_n_a_iterations,
            a_always_check_global_ic=a_always_check_global_ic,
        ),
        n_runs=n_timing_iterations,
        n_warmup=warmup_iterations,
    )
    print(f"    median: {stats_dual_cmp['median_ms']:.2f} ms (IQR {stats_dual_cmp['iqr_ms']:.2f})")

    # 3) Dual principal problem
    print("  Timing dual principal problem...")
    stats_dual_principal = timeit(
        fn=lambda: mhp.solve_principal_problem(
            revenue_function=lambda a: a,
            reservation_utility=reservation_utility,
            a_min=a_pp_min,
            a_max=a_pp_max,
            a_ic_lb=a_lb,
            a_ic_ub=a_ub,
        ),
        n_runs=n_timing_iterations,
        n_warmup=warmup_iterations,
    )
    print(f"    median: {stats_dual_principal['median_ms']:.2f} ms (IQR {stats_dual_principal['iqr_ms']:.2f})")

    # 4) CVXPY relaxed CMP
    print("  Timing CVXPY relaxed CMP...")
    stats_cvx_relaxed = timeit(
        fn=lambda: mhp.solve_cost_minimization_problem_cvxpy(
            intended_action=intended_action,
            reservation_utility=reservation_utility,
            a_hat=np.array([]),
        ),
        n_runs=n_timing_iterations,
        n_warmup=warmup_iterations,
    )
    print(f"    median: {stats_cvx_relaxed['median_ms']:.2f} ms (IQR {stats_cvx_relaxed['iqr_ms']:.2f})")

    # 5) CVXPY full CMP
    print("  Timing CVXPY CMP (101 a_hat)...")
    stats_cvx_cmp = timeit(
        fn=lambda: mhp.solve_cost_minimization_problem_cvxpy(
            intended_action=intended_action,
            reservation_utility=reservation_utility,
            a_hat=a_hat_cvxpy,
        ),
        n_runs=n_timing_iterations,
        n_warmup=warmup_iterations,
    )
    print(f"    median: {stats_cvx_cmp['median_ms']:.2f} ms (IQR {stats_cvx_cmp['iqr_ms']:.2f})")

    # 6) CVXPY principal problem
    print("  Timing CVXPY principal problem (100 actions)...")
    stats_cvx_principal = timeit(
        fn=lambda: mhp.solve_principal_problem_cvxpy(
            revenue_function=lambda a: a,
            reservation_utility=reservation_utility,
            discretized_a_grid=a_grid_principal,
        ),
        n_runs=n_timing_iterations,
        n_warmup=warmup_iterations,
    )
    print(f"    median: {stats_cvx_principal['median_ms']:.2f} ms (IQR {stats_cvx_principal['iqr_ms']:.2f})")

    print()

    # Wide row: keep the same solver columns, but store median+IQR (and mean/std for reference)
    def pack(prefix: str, stats: dict) -> dict:
        return {
            f"{prefix}_median_ms": stats["median_ms"],
            f"{prefix}_iqr_ms": stats["iqr_ms"],
            f"{prefix}_mean_ms": stats["mean_ms"],
            f"{prefix}_std_ms": stats["std_ms"],
        }

    row = {"case": case_name}
    row |= pack("dual_relaxed", stats_dual_relaxed)
    row |= pack("dual_cmp", stats_dual_cmp)
    row |= pack("dual_principal", stats_dual_principal)
    row |= pack("cvxpy_relaxed", stats_cvx_relaxed)
    row |= pack("cvxpy_cmp", stats_cvx_cmp)
    row |= pack("cvxpy_principal", stats_cvx_principal)
    row["n_runs"] = n_timing_iterations
    row["warmup_runs_discarded"] = warmup_iterations
    rows.append(row)


# -----------------------------
# Output table + files
# -----------------------------
df = pd.DataFrame(rows)

# Human-friendly table: median (IQR)
table_cols = ["case"]
for base in ["dual_relaxed", "dual_cmp", "dual_principal", "cvxpy_relaxed", "cvxpy_cmp", "cvxpy_principal"]:
    table_cols += [f"{base}_median_ms", f"{base}_iqr_ms"]

df_table = df[table_cols].copy()

print("=== Results Table: median_ms (IQR) ===")
print(df_table.to_string(index=False, float_format="%.2f"))
print()

# Output directory: prefer script directory; fall back to CWD (works in notebooks too)
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path.cwd()

output_dir = base_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "timing_results.csv"
df.to_csv(csv_path, index=False)

specs_path = output_dir / "machine_specs.txt"
write_machine_specs(specs_path)

print(f"Saved timing CSV: {csv_path}")
print(f"Saved machine specs: {specs_path}")
