"""
Comparison of dual and CVXPY solvers for the log-gaussian example.
Solves the cost minimization problem using both methods and creates comparison figures.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# -------------------------------------------------------------
# Figure formatting (from figure_maker.py)
# -------------------------------------------------------------
PLOT_LINEWIDTH = 1.2
FONT_SIZE = 6
LABEL_FONT_SIZE = 7.2
TITLE_FONT_SIZE = 8.4
MARKER_SIZE = 3

# Colors for the two solvers
DUAL_COLOR = (59/255.0, 105/255.0, 120/255.0)  # Deep teal
CVXPY_COLOR = (255/255.0, 92/255.0, 92/255.0)  # Red

# -------------------------------------------------------------
# Log-gaussian example setup (from main_figures.py)
# -------------------------------------------------------------
initial_wealth = 50
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)

def C(a):
    return theta * a ** 2 / 2

def Cprime(a):
    return theta * a

sigma = 10.0
utility_cfg = make_utility_cfg("log", w0=initial_wealth)
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)

comp_cfg = {
    "distribution_type": "continuous",
    "y_min": 0.0 - 6 * sigma,
    "y_max": 180.0 + 6 * sigma,
    "n": 201,  # must be odd
}

cfg = {
    "problem_params": {**utility_cfg, **dist_cfg, "C": C, "Cprime": Cprime},
    "computational_params": comp_cfg
}

mhp = MoralHazardProblem(cfg)

# -------------------------------------------------------------
# Problem parameters
# -------------------------------------------------------------
intended_action = first_best_effort
reservation_utility = utility_cfg["u"](0.0) - 1.0  # Reservation utility not binding
n_a_iterations = 100
a_min, a_max = 0.0, 180.0
action_grid_plot = np.linspace(a_min, a_max, 200)

# Output directory
output_dir = "figures/solver_comparison"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# Solve using dual solver
# -------------------------------------------------------------
print(f"Comparing solvers for intended action a = {intended_action}")
print("=" * 60)

print("\nRunning dual solver...")
result_dual = mhp.solve_cost_minimization_problem(
    intended_action=intended_action,
    reservation_utility=reservation_utility,
    a_ic_lb=a_min,
    a_ic_ub=a_max,
    n_a_iterations=n_a_iterations,
)
v_dual = result_dual.optimal_contract

print("\n" + "-" * 60)
print("DUAL SOLVER RESULT:")
print("-" * 60)
print(f"  Expected wage: {result_dual.expected_wage:.4f}")
print(f"  FOA holds: {result_dual.first_order_approach_holds}")

# -------------------------------------------------------------
# Solve using CVXPY solver
# -------------------------------------------------------------
print("\nRunning CVXPY solver...")
a_hat = np.linspace(a_min, a_max, 200)  # Grid of alternative actions for IC constraints
result_cvxpy = mhp.solve_cost_minimization_problem_cvxpy(
    intended_action=intended_action,
    reservation_utility=reservation_utility,
    a_hat=a_hat,
)
v_cvxpy = result_cvxpy['optimal_contract']

print("\n" + "-" * 60)
print("CVXPY SOLVER RESULT:")
print("-" * 60)
print(f"  Expected wage: {result_cvxpy['expected_wage']:.4f}")
print(f"  Optimal contract: array(shape={v_cvxpy.shape}, min={v_cvxpy.min():.4f}, max={v_cvxpy.max():.4f})")

# -------------------------------------------------------------
# Compute derived quantities
# -------------------------------------------------------------
y_grid = mhp.y_grid
w_dual = mhp.k(v_dual)
w_cvxpy = mhp.k(v_cvxpy)

# Agent utility vs action
U_dual = mhp.U(v_dual, action_grid_plot)
U_cvxpy = mhp.U(v_cvxpy, action_grid_plot)

# Find best action for each contract
best_a_dual = action_grid_plot[np.argmax(U_dual)]
best_a_cvxpy = action_grid_plot[np.argmax(U_cvxpy)]

print("\n" + "-" * 60)
print("AGENT'S BEST ACTION:")
print("-" * 60)
print(f"  Dual solver - Best action for agent: {best_a_dual:.2f} (intended: {intended_action})")
print(f"  CVXPY solver - Best action for agent: {best_a_cvxpy:.2f} (intended: {intended_action})")

# -------------------------------------------------------------
# Figure 1: Optimal contract w(y) vs y
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.375))

ax.plot(y_grid, w_dual, color=DUAL_COLOR, linestyle='-', linewidth=PLOT_LINEWIDTH * 1.5,
        label=f'Algorithm 1 (E[w]={result_dual.expected_wage:.2f})')
ax.plot(y_grid, w_cvxpy, color=CVXPY_COLOR, linestyle='--', linewidth=PLOT_LINEWIDTH * 1.5,
        label=f'Convex program (E[w]={result_cvxpy["expected_wage"]:.2f})')

# Mark the intended action (no legend entry)
ax.axvline(intended_action, color='gray', linestyle=':', alpha=0.7, linewidth=PLOT_LINEWIDTH)

ax.set_xlabel(r"Output $y$ (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax.set_ylabel(r"Wage $w(y)$ (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax.set_title("Optimal Contract Comparison", fontsize=TITLE_FONT_SIZE)
ax.tick_params(labelsize=FONT_SIZE)
ax.legend(fontsize=FONT_SIZE, loc='best')
ax.set_xlim(a_min, a_max)

fig.tight_layout()
fig.savefig(f"{output_dir}/wage_comparison.png", dpi=300)
fig.savefig(f"{output_dir}/wage_comparison.pdf")
plt.close(fig)

print(f"\nSaved: {output_dir}/wage_comparison.png")

# -------------------------------------------------------------
# Figure 2: Agent utility U vs a
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.375))

ax.plot(action_grid_plot, U_dual, color=DUAL_COLOR, linestyle='-', linewidth=PLOT_LINEWIDTH * 1.5,
        label='Algorithm 1')
ax.plot(action_grid_plot, U_cvxpy, color=CVXPY_COLOR, linestyle='--', linewidth=PLOT_LINEWIDTH * 1.5,
        label='Convex program')

# Mark intended action (no legend entry)
ax.axvline(intended_action, color='green', linestyle='-', alpha=0.7, linewidth=PLOT_LINEWIDTH)

# Mark reservation utility
ax.axhline(reservation_utility, color='gray', linestyle='--', alpha=0.5, linewidth=PLOT_LINEWIDTH,
           label='Reservation utility')

# Mark the utility at intended action for each solver
U_dual_at_intended = mhp.U(v_dual, intended_action)
U_cvxpy_at_intended = mhp.U(v_cvxpy, intended_action)
ax.scatter([intended_action], [U_dual_at_intended], color=DUAL_COLOR, s=MARKER_SIZE * 10, zorder=5)
ax.scatter([intended_action], [U_cvxpy_at_intended], color=CVXPY_COLOR, s=MARKER_SIZE * 10, zorder=5, marker='s')

ax.set_xlabel("Action (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax.set_ylabel("Agent expected utility (certain equivalent, USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax.set_title("Agent Utility vs Action", fontsize=TITLE_FONT_SIZE)
ax.tick_params(labelsize=FONT_SIZE)
ax.legend(fontsize=FONT_SIZE, loc='best')
ax.set_xlim(a_min, a_max)

# Set y-axis ticks at natural certain equivalent values (from figure_maker.py)
ylim = ax.get_ylim()
u_min, u_max = ylim

# Convert to certain equivalent range
ce_min = mhp.k(u_min)
ce_max = mhp.k(u_max)

# Find all multiples of 10 in the certain equivalent range
start_tick = np.ceil(ce_min / 10) * 10
end_tick = np.floor(ce_max / 10) * 10

if start_tick <= end_tick:
    ce_ticks = np.arange(start_tick, end_tick + 10, 10)
else:
    ce_ticks = np.array([])

# Convert certain equivalent ticks back to utility values for positioning
if len(ce_ticks) > 0:
    u_ticks = utility_cfg["u"](ce_ticks)
    ax.set_yticks(u_ticks)
    ax.set_yticklabels([f'{ce:.0f}' for ce in ce_ticks], fontsize=FONT_SIZE)

fig.tight_layout()
fig.savefig(f"{output_dir}/utility_comparison.png", dpi=300)
fig.savefig(f"{output_dir}/utility_comparison.pdf")
plt.close(fig)

print(f"Saved: {output_dir}/utility_comparison.png")

# -------------------------------------------------------------
# Figure 3: Stacked figure (wage on top, utility on bottom)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6.75))

# TOP PLOT: Optimal Contract w(y) vs y
line1, = ax1.plot(y_grid, w_dual, color=DUAL_COLOR, linestyle='-', linewidth=PLOT_LINEWIDTH * 1.5,
                  label='Algorithm 1')
line2, = ax1.plot(y_grid, w_cvxpy, color=CVXPY_COLOR, linestyle='--', linewidth=PLOT_LINEWIDTH * 1.5,
                  label='Convex program')
ax1.axvline(intended_action, color='gray', linestyle=':', alpha=0.7, linewidth=PLOT_LINEWIDTH)

ax1.set_xlabel(r"Output $y$ (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax1.set_ylabel(r"Wage $w(y)$ (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax1.set_title("Optimal Contract Comparison", fontsize=TITLE_FONT_SIZE)
ax1.tick_params(labelsize=FONT_SIZE)
ax1.set_xlim(a_min, a_max)

# BOTTOM PLOT: Agent utility U vs a
ax2.plot(action_grid_plot, U_dual, color=DUAL_COLOR, linestyle='-', linewidth=PLOT_LINEWIDTH * 1.5)
ax2.plot(action_grid_plot, U_cvxpy, color=CVXPY_COLOR, linestyle='--', linewidth=PLOT_LINEWIDTH * 1.5)

ax2.axvline(intended_action, color='green', linestyle='-', alpha=0.7, linewidth=PLOT_LINEWIDTH)

ax2.axhline(reservation_utility, color='gray', linestyle='--', alpha=0.5, linewidth=PLOT_LINEWIDTH)

ax2.scatter([intended_action], [U_dual_at_intended], color=DUAL_COLOR, s=MARKER_SIZE * 10, zorder=5)
ax2.scatter([intended_action], [U_cvxpy_at_intended], color=CVXPY_COLOR, s=MARKER_SIZE * 10, zorder=5, marker='s')

ax2.set_xlabel("Action (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax2.set_ylabel("Agent expected utility (certain equivalent, USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax2.set_title("Agent Utility vs Action", fontsize=TITLE_FONT_SIZE)
ax2.tick_params(labelsize=FONT_SIZE)
ax2.set_xlim(a_min, a_max)

# Set y-axis ticks at natural certain equivalent values
ylim = ax2.get_ylim()
u_min_plot, u_max_plot = ylim
ce_min_plot = mhp.k(u_min_plot)
ce_max_plot = mhp.k(u_max_plot)

start_tick = np.ceil(ce_min_plot / 10) * 10
end_tick = np.floor(ce_max_plot / 10) * 10

if start_tick <= end_tick:
    ce_ticks_stacked = np.arange(start_tick, end_tick + 10, 10)
else:
    ce_ticks_stacked = np.array([])

if len(ce_ticks_stacked) > 0:
    u_ticks_stacked = utility_cfg["u"](ce_ticks_stacked)
    ax2.set_yticks(u_ticks_stacked)
    ax2.set_yticklabels([f'{ce:.0f}' for ce in ce_ticks_stacked], fontsize=FONT_SIZE)

# Shared legend at the bottom
fig.subplots_adjust(bottom=0.12, top=0.95, hspace=0.3)
fig.legend([line1, line2], ['Algorithm 1', 'Convex program'], 
           loc='lower center', ncol=2, fontsize=FONT_SIZE, frameon=False)

fig.savefig(f"{output_dir}/stacked_comparison.png", dpi=300)
fig.savefig(f"{output_dir}/stacked_comparison.pdf")
plt.close(fig)

print(f"Saved: {output_dir}/stacked_comparison.png")

# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Dual solver:  E[w] = {result_dual.expected_wage:.4f}, FOA = {result_dual.first_order_approach_holds}")
print(f"CVXPY solver: E[w] = {result_cvxpy['expected_wage']:.4f}")
print(f"Difference in E[w]: {abs(result_dual.expected_wage - result_cvxpy['expected_wage']):.6f}")
