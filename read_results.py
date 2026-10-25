
from pathlib import Path
import pandas as pd

RESULTS = Path("benchmark_results/results.csv")

if not RESULTS.exists():
    raise FileNotFoundError(
        f"No saved results found at {RESULTS.resolve()}"
    )

df = pd.read_csv(RESULTS)

# Show individual results
print("\nIndividual results:")
print(df.to_string(index=False))

# Convert numeric columns safely
for col in [
    "objective", "best_bound", "gap", "runtime_seconds",
    "num_vars", "num_constraints", "lp_bound"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Treat only OPTIMAL status as proven optimal.
df["is_optimal"] = df["status"].eq("OPTIMAL")
df["has_solution"] = df["objective"].notna()

# One summary row per model
summary = df.groupby("model").agg(
    instances=("instance", "nunique"),
    optimal=("is_optimal", "sum"),
    feasible=("has_solution", "sum"),
    mean_runtime_s=("runtime_seconds", "mean"),
    median_runtime_s=("runtime_seconds", "median"),
    mean_variables=("num_vars", "mean"),
    mean_constraints=("num_constraints", "mean"),
    mean_lp_bound=("lp_bound", "mean"),
    mean_gap=("gap", "mean"),
).round(3)

print("\nSummary by model:")
print(summary.to_string())

# Save the summary for your report
summary.to_csv("benchmark_results/summary.csv")

# Optional: compare the three models on the same instances
objectives = df.pivot(
    index="instance", columns="model", values="objective"
)
statuses = df.pivot(
    index="instance", columns="model", values="status"
)

print("\nInstances where all three models proved optimal:")
models = ["kantorovich", "gilmore-gomory", "arcflow"]

if all(m in statuses.columns for m in models):
    all_optimal = statuses[models].eq("OPTIMAL").all(axis=1)
    common = objectives.loc[all_optimal, models]
    if not common.empty:
        print(common.to_string())
        print("\nDo their objective values agree?")
        print(common.nunique(axis=1).eq(1).all())
    else:
        print("No instance was solved optimally by all three models.")
