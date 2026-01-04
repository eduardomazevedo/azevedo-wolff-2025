"""
Generate LaTeX assets for benchmarking section in the supplementary material.
Reads timing_results.csv and machine_specs.txt, produces:
  - output/benchmarks_table.tex: LaTeX table with timing results
  - output/specs_description.tex: Sentence describing machine specs
"""
import pandas as pd
import re

# -------------------------------------------------------------
# Load data
# -------------------------------------------------------------
df = pd.read_csv("output/timing_results.csv")

with open("output/machine_specs.txt", "r") as f:
    specs_text = f.read()

# -------------------------------------------------------------
# Parse machine specs
# -------------------------------------------------------------
# Extract platform
platform_match = re.search(r"platform: (.+)", specs_text)
platform = platform_match.group(1) if platform_match else "unknown"

# Extract Python version
python_match = re.search(r"python: (\d+\.\d+\.\d+)", specs_text)
python_version = python_match.group(1) if python_match else "unknown"

# Extract chip from uname (Darwin ... arm64)
if "arm64" in specs_text:
    chip = "Apple Silicon (ARM64)"
elif "x86_64" in specs_text:
    chip = "Intel x86-64"
else:
    chip = "unknown architecture"

# Extract key package versions
numpy_match = re.search(r"numpy=(\S+)", specs_text)
cvxpy_match = re.search(r"cvxpy=(\S+)", specs_text)
numpy_version = numpy_match.group(1) if numpy_match else "unknown"
cvxpy_version = cvxpy_match.group(1) if cvxpy_match else "unknown"

# Create specs description
specs_description = (
    f"Benchmarks were run on {chip} ({platform.split('-')[0]}) "
    f"using Python {python_version}, NumPy {numpy_version}, and CVXPY {cvxpy_version}."
)

# Save specs description
with open("output/specs_description.tex", "w") as f:
    f.write(specs_description)

print(f"Saved: output/specs_description.tex")

# -------------------------------------------------------------
# Create benchmarks table
# -------------------------------------------------------------
# Map case names to nice labels
case_labels = {
    "gaussian-easy": "Gaussian (easy)",
    "gaussian-hard": "Gaussian (hard)",
    "t": "Student-$t$",
}

# We want to show: Case, Algorithm 1 CMP (ms), Convex CMP (ms), Algorithm 1 Principal (ms), Convex Principal (ms)
# Using median values

rows = []
for _, row in df.iterrows():
    case = row["case"]
    label = case_labels.get(case, case)
    
    dual_cmp = row["dual_cmp_median_ms"]
    cvxpy_cmp = row["cvxpy_cmp_median_ms"]
    dual_pp = row["dual_principal_median_ms"]
    cvxpy_pp = row["cvxpy_principal_median_ms"]
    
    rows.append({
        "Case": label,
        "Alg. 1 CMP": dual_cmp,
        "Convex CMP": cvxpy_cmp,
        "Alg. 1 PP": dual_pp,
        "Convex PP": cvxpy_pp,
    })

results_df = pd.DataFrame(rows)

# Format numbers nicely
def format_ms(val):
    if val < 10:
        return f"{val:.1f}"
    elif val < 1000:
        return f"{val:.0f}"
    else:
        return f"{val/1000:.1f}k"

# Build LaTeX table
table_lines = []
table_lines.append(r"\begin{tabular}{lrrrr}")
table_lines.append(r"\toprule")
table_lines.append(r"& \multicolumn{2}{c}{Cost Minimization (ms)} & \multicolumn{2}{c}{Principal Problem (ms)} \\")
table_lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
table_lines.append(r"Case & Algorithm 1 & Convex & Algorithm 1 & Convex \\")
table_lines.append(r"\midrule")

for _, row in results_df.iterrows():
    case = row["Case"]
    alg1_cmp = format_ms(row["Alg. 1 CMP"])
    cvx_cmp = format_ms(row["Convex CMP"])
    alg1_pp = format_ms(row["Alg. 1 PP"])
    cvx_pp = format_ms(row["Convex PP"])
    table_lines.append(f"{case} & {alg1_cmp} & {cvx_cmp} & {alg1_pp} & {cvx_pp} \\\\")

table_lines.append(r"\bottomrule")
table_lines.append(r"\end{tabular}")

table_tex = "\n".join(table_lines)

# Save table
with open("output/benchmarks_table.tex", "w") as f:
    f.write(table_tex)

print(f"Saved: output/benchmarks_table.tex")

# Print preview
print("\n--- Specs Description ---")
print(specs_description)
print("\n--- Benchmarks Table ---")
print(table_tex)
