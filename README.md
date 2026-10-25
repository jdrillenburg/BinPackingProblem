# Comparing Integer Programming Formulations for One-Dimensional Bin Packing

## Overview

This project implements and experimentally compares three integer linear programming (ILP) formulations for the one-dimensional bin packing problem:

- **Kantorovich formulation**
- **Gilmore–Gomory formulation**
- **Arc-flow formulation**

The goal is to compare their computational performance on benchmark instances, considering solution quality, computation time, model size, and solver bounds.

This project was developed as part of an academic assignment.

## Problem Description

The one-dimensional bin packing problem consists of a set of items with given sizes and a collection of identical bins with a fixed capacity. The objective is to assign every item to a bin without exceeding its capacity while minimizing the number of bins used.

## Implemented Formulations

- **Kantorovich:** Uses binary assignment variables to assign items to bins.
- **Gilmore–Gomory:** Uses feasible packing patterns as decision variables.
- **Arc-flow:** Represents feasible packings using a network-flow model.

The formulations are implemented in Python and solved using Gurobi.

## Benchmark Instances

The experiments use one-dimensional bin packing benchmark instances from the Scholl benchmark sets:

- Scholl 1
- Scholl 2
- Scholl 3

Please see the benchmark source and original publication for attribution and the terms governing use or redistribution of the instances.

## Requirements

- Python 3
- Gurobi Optimizer
- A valid Gurobi license

## Running the Experiments

1. Clone or download this repository.
2. Install the required Python packages.
3. Ensure Gurobi is installed and licensed.
4. Ensure that the folder `testing_instances` is in the same folder as benchmark_runner.py and solve_bpp.py
5. Configure the benchmark settings in `benchmark_runner.py`.
6. Run:

   ```bash
   python benchmark_runner.py
   ```

Check the runner's configuration and the code for the expected instance file format and directory structure before running.

## Experimental Settings

The benchmark runner is configured to use a time limit of 60 seconds per instance and formulation, with one solver thread. Other settings, including LP-bound computation, are specified in the runner.

The 60-second limit applies to model construction and the integer optimization. LP relaxation bounds are calculated afterward and may increase total wall-clock time. The reported solver runtime and total elapsed time are recorded separately.

## Results

The benchmark runner generates CSV files containing the results for individual instances and summary statistics. These can be used to compare:

- Number of instances solved to optimality
- Best feasible objective values
- LP relaxation bounds and MIP gaps
- Computation times
- Numbers of variables, constraints, and nonzero constraint-matrix coefficients

Summary statistics should be interpreted alongside solver statuses and any pattern-enumeration or time-limit limits.

## Reading and Analyzing Results

After running the benchmark experiments, use `read_results.py` to inspect the saved results and generate summary statistics:

```bash
python read_results.py
```

The script reads `benchmark_results/results.csv` and:

- Prints the individual results for each instance and formulation.
- Calculates summary statistics for each model, including the number of instances solved optimally, the number with feasible solutions, mean and median runtime, mean model size, mean LP bound, and mean MIP gap.
- Saves the summary to `benchmark_results/summary.csv`.
- Identifies instances for which all three formulations proved optimal and checks whether their objective values agree.

The script requires `pandas` and assumes that `benchmark_results/results.csv` already exists.

## License

The original source code in this repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The MIT License applies to this project's original code, not automatically to third-party benchmark instances or other third-party materials. Those materials remain subject to their respective terms.

## References

Scholl, A., Klein, R., & Jürgens, C. (1997). BISON: A fast hybrid procedure for exactly solving the one-dimensional bin packing problem. *Computers & Operations Research, 24*(7), 627–645. https://doi.org/10.1016/S0305-0548(96)00082-2
