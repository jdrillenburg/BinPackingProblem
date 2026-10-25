
from pathlib import Path
import csv
import json
import time
from datetime import datetime, timezone
from solve_bpp import MODELS, from_item_list, verify_solution


# ---------------------------- Configuration ----------------------------

INSTANCE_DIR = Path("testing_instances")
RESULTS_DIR = Path("benchmark_results")

TIME_LIMIT = 60.0       # seconds per model per instance
THREADS = 1             # fixed for reproducibility
COMPUTE_LP_BOUND = True  # LP relaxation is computed after the MIP solve
VERBOSE = False

RESULTS_CSV = RESULTS_DIR / "results.csv"
ERRORS_CSV = RESULTS_DIR / "errors.csv"
SETTINGS_JSON = RESULTS_DIR / "settings.json"

RESULT_FIELDS = [
    "instance", "model", "status", "objective", "best_bound", "gap",
    "runtime_seconds", "wall_elapsed_seconds", "lp_bound",
    "num_vars", "num_constraints", "num_nonzeros",
    "solution_feasible", "info"
]

ERROR_FIELDS = ["instance", "stage", "error"]


# ----------------------------- File reader -----------------------------

def read_instance(path):
    """
    Read a single bin-packing instance in the format:

        number_of_items
        bin_capacity
        item_size_1
        item_size_2
        ...

    Blank lines are ignored. All other content must be integers.
    """
    tokens = path.read_text(encoding="utf-8").split()

    try:
        values = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("File contains non-integer data") from exc

    if len(values) < 2:
        raise ValueError("Expected item count and bin capacity")

    n, capacity = values[0], values[1]
    sizes = values[2:]

    if n <= 0:
        raise ValueError(f"Invalid number of items: {n}")

    if capacity <= 0:
        raise ValueError(f"Invalid bin capacity: {capacity}")

    if len(sizes) != n:
        raise ValueError(
            f"Header says {n} items, but found {len(sizes)} sizes"
        )

    if any(size <= 0 or size > capacity for size in sizes):
        raise ValueError(
            "Every item size must satisfy 0 < size <= capacity"
        )

    # Convert the individual items to the format expected by your models.
    widths, demands = from_item_list(sizes)
    return widths, demands, capacity


# ----------------------------- CSV helpers -----------------------------

def append_csv(path, fields, row):
    """Append a row, writing the header if the file is new or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def discover_instances(root):
    """Find files recursively, excluding common repository metadata."""
    ignored_names = {
        "readme.md", "license", "manifest.csv",
        "results.csv", "errors.csv", "settings.json",
    }
    ignored_suffixes = {".py", ".csv", ".json", ".png", ".pdf"}

    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
        and p.name.lower() not in ignored_names
        and p.suffix.lower() not in ignored_suffixes
    )


# ---------------------------- Benchmark run ----------------------------

def run_benchmarks():
    if not INSTANCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Instance directory not found: {INSTANCE_DIR.resolve()}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    instances = discover_instances(INSTANCE_DIR)

    if not instances:
        raise RuntimeError(f"No candidate instance files found in {INSTANCE_DIR}")

    settings = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "instance_directory": str(INSTANCE_DIR.resolve()),
        "time_limit_seconds": TIME_LIMIT,
        "threads": THREADS,
        "compute_lp_bound": COMPUTE_LP_BOUND,
        "verbose": VERBOSE,
        "models": list(MODELS.keys()),
        "instance_count": len(instances),
    }

    SETTINGS_JSON.write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    print(f"Found {len(instances)} candidate files.")
    print(f"Results will be saved in: {RESULTS_DIR.resolve()}")

    for index, path in enumerate(instances, start=1):
        instance_name = path.relative_to(INSTANCE_DIR).as_posix()
        print(f"\n[{index}/{len(instances)}] {instance_name}")

        try:
            widths, demands, capacity = read_instance(path)
        except Exception as exc:
            print(f"  SKIPPED: could not read instance: {exc}")
            append_csv(ERRORS_CSV, ERROR_FIELDS, {
                "instance": instance_name,
                "stage": "read_instance",
                "error": repr(exc),
            })
            continue

        print(
            f"  Items: {sum(demands)}, distinct sizes: {len(widths)}, "
            f"capacity: {capacity}"
        )

        for model_name, solve in MODELS.items():
            start = time.perf_counter()

            try:
                result = solve(
                    widths,
                    demands,
                    capacity,
                    time_limit=TIME_LIMIT,
                    verbose=VERBOSE,
                    threads=THREADS,
                    compute_lp_bound=COMPUTE_LP_BOUND,
                )

                wall_elapsed = time.perf_counter() - start

                feasible = None
                if result.solution is not None:
                    feasible = verify_solution(
                        result.solution, widths, demands, capacity
                    )
                    if not feasible:
                        raise RuntimeError(
                            "Returned solution failed feasibility verification"
                        )

                row = {
                    "instance": instance_name,
                    "model": model_name,
                    "status": result.status,
                    "objective": result.objective,
                    "best_bound": result.best_bound,
                    "gap": result.gap,
                    "runtime_seconds": result.runtime,
                    "wall_elapsed_seconds": round(wall_elapsed, 6),
                    "lp_bound": result.lp_bound,
                    "num_vars": result.num_vars,
                    "num_constraints": result.num_constrs,
                    "num_nonzeros": result.num_nonzeros,
                    "solution_feasible": feasible,
                    "info": json.dumps(result.info, sort_keys=True),
                }

                append_csv(RESULTS_CSV, RESULT_FIELDS, row)

                print(
                    f"  {model_name:16} status={result.status:18} "
                    f"obj={str(result.objective):>5} "
                    f"time={result.runtime:.2f}s "
                    f"wall={wall_elapsed:.2f}s"
                )

            except Exception as exc:
                print(f"  ERROR in {model_name}: {exc}")
                append_csv(ERRORS_CSV, ERROR_FIELDS, {
                    "instance": instance_name,
                    "stage": model_name,
                    "error": repr(exc),
                })

    settings["finished_utc"] = datetime.now(timezone.utc).isoformat()
    SETTINGS_JSON.write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    print("\nBenchmark run finished.")
    print(f"Results:  {RESULTS_CSV.resolve()}")
    print(f"Errors:   {ERRORS_CSV.resolve()}")
    print(f"Settings: {SETTINGS_JSON.resolve()}")


if __name__ == "__main__":
    run_benchmarks()
