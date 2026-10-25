"""
ILP models for the one-dimensional Cutting Stock Problem (CSP) and Bin Packing
Problem (BPP), solved with Gurobi.

Reference
---------
J.M. Valerio de Carvalho (2002), "LP models for bin packing and cutting stock
problems", European Journal of Operational Research 141, 253-273.

Problem
-------
Given rolls of width W and m item types with widths w_i and demands b_i,
find the minimum number of rolls needed to cut all the demanded items.
(Bin packing is the special case b_i = 1 for all i; use `from_item_list`
to convert a plain list of item sizes, with repeated sizes, to (widths, demands).)

Models (numbers refer to the equations in the paper / slides)
-------------------------------------------------------------
1. solve_kantorovich    : (1)-(5)    x_ik = #times item i is cut from roll k,
                                      y_k = 1 if roll k is used.
2. solve_gilmore_gomory : (26)-(30)  lambda_p = #rolls cut with pattern p.
                                      All patterns are enumerated explicitly.
3. solve_arcflow        : (31)-(36)  x_de = #items of width e-d placed at
                                      position d..e of a roll (flow in a DAG).

Every solve_* function has the same interface:

    result = solve_xxx(widths, demands, W, time_limit=None, verbose=False, ...)

`time_limit` (seconds, wall clock) covers building the model (including the
pattern enumeration for Gilmore-Gomory) plus the MIP solve. The LP relaxation
bound (optional, `compute_lp_bound=True`) is computed *afterwards* and is not
counted in `result.runtime`.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    model: str                              # "kantorovich" | "gilmore-gomory" | "arcflow"
    status: str                             # "OPTIMAL", "TIME_LIMIT", ...
    objective: Optional[int]                # number of rolls (None if no solution found)
    best_bound: Optional[float]             # dual bound from the MIP solver
    gap: Optional[float]                    # relative MIP gap
    runtime: float                          # seconds: model building + MIP solve
    lp_bound: Optional[float]               # LP-relaxation value (None if not computed)
    num_vars: int = 0
    num_constrs: int = 0
    num_nonzeros: int = 0
    solution: Optional[List[List[int]]] = None   # one list of item widths per roll
    info: Dict[str, object] = field(default_factory=dict)  # K, #patterns, #arcs, ...

    @property
    def optimal(self) -> bool:
        return self.status == "OPTIMAL"


_STATUS_NAMES = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INTERRUPTED: "INTERRUPTED",
}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _aggregate(widths: Sequence[int], demands: Sequence[int], W: int
               ) -> Tuple[List[int], List[int]]:
    """Validate the instance, merge identical widths, drop zero demands and
    return (w, b) sorted by non-increasing width."""
    if len(widths) != len(demands):
        raise ValueError("widths and demands must have the same length")
    W = int(W)
    if W <= 0:
        raise ValueError("W must be a positive integer")
    agg: Dict[int, int] = defaultdict(int)
    for wi, bi in zip(widths, demands):
        wi, bi = int(wi), int(bi)
        if wi <= 0 or wi > W:
            raise ValueError(f"item width {wi} must satisfy 0 < w <= W={W}")
        if bi < 0:
            raise ValueError("demands must be non-negative")
        agg[wi] += bi
    items = sorted(((w, b) for w, b in agg.items() if b > 0), reverse=True)
    if not items:
        raise ValueError("instance has no items with positive demand")
    return [w for w, _ in items], [b for _, b in items]


def from_item_list(sizes: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Convert a plain list of item sizes (possibly with repeats), as found in
    bin packing benchmarks, into (widths, demands)."""
    cnt = Counter(int(s) for s in sizes)
    widths = sorted(cnt, reverse=True)
    return widths, [cnt[w] for w in widths]


def first_fit_decreasing(w: Sequence[int], b: Sequence[int], W: int) -> int:
    """First Fit Decreasing heuristic -> valid upper bound on the number of rolls."""
    residual: List[int] = []
    for wi, bi in sorted(zip(w, b), reverse=True):
        for _ in range(bi):
            for k, r in enumerate(residual):
                if r >= wi:
                    residual[k] = r - wi
                    break
            else:
                residual.append(W - wi)
    return len(residual)


def verify_solution(solution: List[List[int]], widths: Sequence[int],
                    demands: Sequence[int], W: int) -> bool:
    """Check that no roll exceeds W and that every demand is satisfied."""
    w, b = _aggregate(widths, demands, W)
    if any(sum(roll) > W for roll in solution):
        return False
    produced = Counter(x for roll in solution for x in roll)
    return all(produced[wi] >= bi for wi, bi in zip(w, b))


def _attr(model: gp.Model, name: str):
    try:
        return getattr(model, name)
    except (AttributeError, gp.GurobiError):
        return None


def _new_model(name: str, verbose: bool, threads: Optional[int]) -> gp.Model:
    model = gp.Model(name)
    model.Params.OutputFlag = 1 if verbose else 0
    if threads is not None:
        model.Params.Threads = int(threads)
    return model


def _timeout_result(name: str, t0: float, info: dict) -> Result:
    return Result(model=name, status="TIME_LIMIT", objective=None, best_bound=None,
                  gap=None, runtime=time.perf_counter() - t0, lp_bound=None, info=info)


def _optimize_and_report(model: gp.Model, name: str, t0: float,
                         time_limit: Optional[float], extract,
                         compute_lp_bound: bool, info: dict) -> Result:
    """Run the MIP with whatever time is left, then collect the statistics."""
    if time_limit is not None:
        remaining = time_limit - (time.perf_counter() - t0)
        if remaining <= 0:
            return _timeout_result(name, t0, info)
        model.Params.TimeLimit = remaining

    model.optimize()
    runtime = time.perf_counter() - t0

    has_sol = model.SolCount > 0
    obj = int(round(model.ObjVal)) if has_sol else None
    bound = _attr(model, "ObjBound") if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else None
    if model.Status == GRB.OPTIMAL and obj is not None:
        bound = float(obj) if bound is None else bound
    gap = None
    if obj is not None and bound is not None:
        gap = max(0.0, (obj - bound) / max(abs(obj), 1e-9))

    solution = extract() if has_sol else None

    lp_bound = None
    if compute_lp_bound:
        relaxed = model.relax()
        relaxed.Params.OutputFlag = 0
        if time_limit is not None:
            relaxed.Params.TimeLimit = time_limit
        relaxed.optimize()
        if relaxed.Status == GRB.OPTIMAL:
            lp_bound = relaxed.ObjVal

    return Result(
        model=name,
        status=_STATUS_NAMES.get(model.Status, f"STATUS_{model.Status}"),
        objective=obj, best_bound=bound, gap=gap, runtime=runtime, lp_bound=lp_bound,
        num_vars=model.NumVars, num_constrs=model.NumConstrs, num_nonzeros=model.NumNZs,
        solution=solution, info=info,
    )


# --------------------------------------------------------------------------- #
#  1. Kantorovich model
# --------------------------------------------------------------------------- #
def solve_kantorovich(widths: Sequence[int], demands: Sequence[int], W: int,
                      time_limit: Optional[float] = None, verbose: bool = False,
                      threads: Optional[int] = None, K: Optional[int] = None,
                      compute_lp_bound: bool = True) -> Result:
    """
        min  sum_k y_k                                   (1)
        s.t. sum_k x_ik >= b_i              for all i    (2)
             sum_i w_i x_ik <= W y_k        for all k    (3)
             y_k in {0,1}                                (4)
             x_ik >= 0 integer                           (5)

    K is a known upper bound on the number of rolls; by default it is the value
    of the First Fit Decreasing heuristic.
    """
    t0 = time.perf_counter()
    w, b = _aggregate(widths, demands, W)
    m = len(w)
    if K is None:
        K = first_fit_decreasing(w, b, W)

    model = _new_model("kantorovich", verbose, threads)
    y = model.addVars(K, vtype=GRB.BINARY, name="y")
    x = model.addVars(m, K, lb=0, vtype=GRB.INTEGER, name="x")

    model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)
    for i in range(m):                                            # (2)
        model.addConstr(gp.quicksum(x[i, k] for k in range(K)) >= b[i], name=f"demand_{i}")
    for k in range(K):                                            # (3)
        model.addConstr(gp.quicksum(w[i] * x[i, k] for i in range(m)) <= W * y[k],
                        name=f"capacity_{k}")

    def extract() -> List[List[int]]:
        rolls = []
        for k in range(K):
            roll: List[int] = []
            for i in range(m):
                roll += [w[i]] * int(round(x[i, k].X))
            if roll:
                rolls.append(sorted(roll, reverse=True))
        return rolls

    return _optimize_and_report(model, "kantorovich", t0, time_limit, extract,
                                compute_lp_bound, info={"K": K, "m": m})


# --------------------------------------------------------------------------- #
#  2. Gilmore-Gomory model
# --------------------------------------------------------------------------- #
class _EnumerationAbort(Exception):
    def __init__(self, message: str, patterns: list):
        super().__init__(message)
        self.patterns = patterns


def enumerate_patterns(w: Sequence[int], b: Sequence[int], W: int,
                       maximal_only: bool = True, max_patterns: int = 2_000_000,
                       deadline: Optional[float] = None) -> List[Tuple[int, ...]]:
    """Enumerate cutting patterns a = (a_1..a_m) with sum_i w_i a_i <= W (eq. 29-30).

    Because the demand constraints are '>=' and a_i > b_i is never useful, we cap
    a_i <= min(b_i, W // w_i). With maximal_only=True only patterns to which no
    further item can be added are kept: every solution of the full model can be
    converted into one using only such patterns, so the optimum and the LP bound
    are unchanged while the number of columns is much smaller.

    Raises _EnumerationAbort if max_patterns or the deadline is exceeded.
    """
    deadline = min(
        deadline if deadline is not None else float("inf"),
        time.perf_counter() + 50.0, #change here!
    )
    m = len(w)
    ub = [min(b[i], W // w[i]) for i in range(m)]
    a = [0] * m
    patterns: List[Tuple[int, ...]] = []
    counter = 0
    sys.setrecursionlimit(max(sys.getrecursionlimit(), m + 1000))

    def rec(i: int, rem: int) -> None:
        nonlocal counter
        counter += 1
        if deadline is not None and counter % 128 == 0 and time.perf_counter() > deadline:
            raise _EnumerationAbort("patterns", list(patterns))
        if i == m:
            if maximal_only and any(a[j] < ub[j] and w[j] <= rem for j in range(m)):
                return
            patterns.append(tuple(a))
            if len(patterns) > max_patterns:
                raise _EnumerationAbort("patterns", list(patterns))
            return
        for c in range(min(ub[i], rem // w[i]), -1, -1):
            a[i] = c
            rec(i + 1, rem - c * w[i])
        a[i] = 0

    rec(0, W)
    if not maximal_only:
        return patterns
    return patterns


def solve_gilmore_gomory(widths: Sequence[int], demands: Sequence[int], W: int,
                         time_limit: Optional[float] = None, verbose: bool = False,
                         threads: Optional[int] = None, maximal_only: bool = True,
                         max_patterns: int = 2_000_000,
                         compute_lp_bound: bool = True) -> Result:
    """
        min  sum_p lambda_p                              (26)
        s.t. sum_p a_i^p lambda_p >= b_i    for all i    (27)
             lambda_p >= 0 integer                       (28)

    where the set P of patterns (29)-(30) is generated explicitly beforehand.
    The number of patterns is exponential in m, so this is only practical for
    small/medium instances; if more than `max_patterns` columns would be
    needed (or the time limit is hit while enumerating), the returned status is
    "TOO_MANY_PATTERNS" / "TIME_LIMIT" and no solution is given.
    """
    t0 = time.perf_counter()
    w, b = _aggregate(widths, demands, W)
    m = len(w)
    #deadline = None if time_limit is None else t0 + time_limit
    aborted_status=None

    if time_limit is not None:
        # Strict partition: up to 50 seconds for patterns, remainder (max 10s) for Gurobi
        pattern_budget = min(50.0, time_limit) #change here!
        pattern_deadline = t0 + pattern_budget
    else:
        pattern_deadline = None

    try:
        patterns = enumerate_patterns(w, b, W, maximal_only, max_patterns, pattern_deadline)
    except _EnumerationAbort as exc:
        """res = _timeout_result("gilmore-gomory", t0, {"m": m})
        if str(exc) == "patterns":
            res.status = "TOO_MANY_PATTERNS"         
            
        return res"""
        # Capture the status from the exception
        if str(exc) == "patterns":
            aborted_status = "TOO_MANY_PATTERNS"
        else:
            aborted_status = "TIME_LIMIT"
            
        # Retrieve the partial patterns accumulated before the abort
        # Note: Ensure your _EnumerationAbort class attaches the partial patterns list (e.g., exc.patterns)
        patterns = getattr(exc, 'patterns', []) 
        
        # If absolutely no patterns were generated, we cannot solve
        if not patterns:
            res = _timeout_result("gilmore-gomory", t0, {"m": m})
            res.status = aborted_status
            return res

        #  Patch partial pool to prevent artificial INFEASIBLE status ---
        # Track which items have at least one valid pattern cutting them
        covered_items = set()
        for p in patterns:
            for i in range(m):
                if p[i] > 0:
                    covered_items.add(i)

        # Add basic fallback columns (one roll cutting exactly one item) 
        # for any unrepresented items so Gurobi can always find a feasible solution
        for i in range(m):
            if i not in covered_items:
                fallback_pattern = [0] * m
                fallback_pattern[i] = 1
                patterns.append(tuple(fallback_pattern))
        
    P = len(patterns)

    model = _new_model("gilmore_gomory", verbose, threads)
    lam = model.addVars(P, lb=0, vtype=GRB.INTEGER, name="lambda")
    model.setObjective(gp.quicksum(lam[p] for p in range(P)), GRB.MINIMIZE)
    for i in range(m):                                            # (27)
        model.addConstr(gp.quicksum(patterns[p][i] * lam[p] for p in range(P) if patterns[p][i])
                        >= b[i], name=f"demand_{i}")

    def extract() -> List[List[int]]:
        rolls = []
        for p in range(P):
            times = int(round(lam[p].X))
            if times > 0:
                roll: List[int] = []
                for i in range(m):
                    roll += [w[i]] * patterns[p][i]
                rolls += [sorted(roll, reverse=True)] * times
        return rolls

    res = _optimize_and_report(model, "gilmore-gomory", t0, time_limit, extract,
                                compute_lp_bound,
                                info={"patterns": P, "m": m, "maximal_only": maximal_only})

    # If the generation was truncated but Gurobi solved the sub-problem:
    if aborted_status is not None and res.status == "OPTIMAL":
        res.status = f"FEASIBLE_{aborted_status}"

    return res


# --------------------------------------------------------------------------- #
#  3. Arc-flow model
# --------------------------------------------------------------------------- #
def solve_arcflow(widths: Sequence[int], demands: Sequence[int], W: int,
                  time_limit: Optional[float] = None, verbose: bool = False,
                  threads: Optional[int] = None, reduce_graph: bool = True,
                  compute_lp_bound: bool = True) -> Result:
    """
    Graph G = (V, A): V = {0, 1, ..., W};
      * item arcs (d, d + w_i) for every item type i and every d with d + w_i <= W
      * loss arcs (d, d + 1) for the unused part of a roll.
    A path 0 -> W is a cutting pattern. With a single roll size (K = 1, unlimited
    supply) the model of the paper reduces to

        min  z      (= W z / W, i.e. the number of rolls)             (31)
        s.t. -sum_{(d,e)} x_de + sum_{(e,f)} x_ef =  z   if e = 0
                                                    -z   if e = W
                                                     0   otherwise    (32)
             sum_d x_{d,d+w_i} >= b_i             for all i           (33)
             x_de >= 0 integer, z >= 0 integer                        (35)-(36)

    reduce_graph=False builds the plain graph described above.
    reduce_graph=True keeps only the nodes d that are sums of item widths
    ("normal patterns"; loss arcs then link consecutive kept nodes and W). This
    gives the same optimum and LP bound with (often far) fewer variables.
    """
    t0 = time.perf_counter()
    w, b = _aggregate(widths, demands, W)
    m = len(w)

    # --- nodes ---
    if reduce_graph:
        reach = bytearray(W + 1)
        reach[0] = 1
        for d in range(W + 1):
            if reach[d]:
                for wi in w:
                    if d + wi <= W:
                        reach[d + wi] = 1
        nodes = [d for d in range(W + 1) if reach[d]]
        if nodes[-1] != W:
            nodes.append(W)
    else:
        nodes = list(range(W + 1))

    # --- arcs: (tail, head, label) with label = item index, or -1 for a loss arc ---
    arcs: List[Tuple[int, int, int]] = []
    for i in range(m):
        for d in nodes:
            if d + w[i] <= W:
                arcs.append((d, d + w[i], i))
    for u, v in zip(nodes[:-1], nodes[1:]):
        arcs.append((u, v, -1))

    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    item_arcs = defaultdict(list)
    for a in arcs:
        out_arcs[a[0]].append(a)
        in_arcs[a[1]].append(a)
        if a[2] >= 0:
            item_arcs[a[2]].append(a)

    model = _new_model("arcflow", verbose, threads)
    x = model.addVars(arcs, lb=0, vtype=GRB.INTEGER, name="x")
    z = model.addVar(lb=0, vtype=GRB.INTEGER, name="z")
    model.setObjective(z, GRB.MINIMIZE)                           # (31)

    for v in nodes:                                               # (32)
        lhs = gp.quicksum(x[a] for a in out_arcs[v]) - gp.quicksum(x[a] for a in in_arcs[v])
        if v == 0:
            model.addConstr(lhs == z, name=f"flow_{v}")
        elif v == W:
            model.addConstr(lhs == -z, name=f"flow_{v}")
        else:
            model.addConstr(lhs == 0, name=f"flow_{v}")
    for i in range(m):                                            # (33)
        model.addConstr(gp.quicksum(x[a] for a in item_arcs[i]) >= b[i], name=f"demand_{i}")

    def extract() -> List[List[int]]:
        """Decompose the integer flow into z paths from 0 to W."""
        remaining = {a: int(round(x[a].X)) for a in arcs}
        rolls = []
        for _ in range(int(round(z.X))):
            v, roll = 0, []
            while v != W:
                for a in out_arcs[v]:
                    if remaining[a] > 0:
                        remaining[a] -= 1
                        if a[2] >= 0:
                            roll.append(w[a[2]])
                        v = a[1]
                        break
                else:  # should never happen if flow conservation holds
                    raise RuntimeError("flow decomposition failed")
            if roll:
                rolls.append(sorted(roll, reverse=True))
        return rolls

    return _optimize_and_report(model, "arcflow", t0, time_limit, extract, compute_lp_bound,
                                info={"nodes": len(nodes), "arcs": len(arcs), "m": m,
                                      "reduce_graph": reduce_graph})


# --------------------------------------------------------------------------- #
#  Random instance generator + simple benchmark driver
# --------------------------------------------------------------------------- #
def generate_instance(m: int, W: int, w_min_frac: float = 0.05, w_max_frac: float = 0.5,
                      b_min: int = 1, b_max: int = 10, seed: Optional[int] = None
                      ) -> Tuple[List[int], List[int], int]:
    """Random CSP instance with m distinct widths drawn uniformly from
    [w_min_frac*W, w_max_frac*W] and demands uniform in [b_min, b_max].
    Returns (widths, demands, W). Set b_min = b_max = 1 for bin packing."""
    rng = random.Random(seed)
    lo = max(1, int(math.ceil(w_min_frac * W)))
    hi = min(W, int(math.floor(w_max_frac * W)))
    if hi - lo + 1 < m:
        raise ValueError("width range too small for m distinct widths")
    widths = sorted(rng.sample(range(lo, hi + 1), m), reverse=True)
    demands = [rng.randint(b_min, b_max) for _ in widths]
    return widths, demands, W


MODELS = {
    "kantorovich": solve_kantorovich,
    "gilmore-gomory": solve_gilmore_gomory,
    "arcflow": solve_arcflow,
}


def run_all(widths, demands, W, time_limit: Optional[float] = None, verbose: bool = False,
            threads: Optional[int] = None, check: bool = True) -> List[Result]:
    """Run the three models on one instance with the same time limit."""
    results = []
    for fn in MODELS.values():
        res = fn(widths, demands, W, time_limit=time_limit, verbose=verbose, threads=threads)
        if check and res.solution is not None:
            assert verify_solution(res.solution, widths, demands, W), \
                f"{res.model}: returned solution is infeasible!"
        results.append(res)
    return results


def print_results(results: List[Result], title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    hdr = f"{'model':<15}{'status':<18}{'obj':>6}{'bound':>9}{'LP':>10}{'time(s)':>10}{'vars':>10}{'constrs':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        f = lambda v, fmt: "-" if v is None else format(v, fmt)
        print(f"{r.model:<15}{r.status:<18}{f(r.objective, 'd'):>6}{f(r.best_bound, '.2f'):>9}"
              f"{f(r.lp_bound, '.4f'):>10}{r.runtime:>10.2f}{r.num_vars:>10}{r.num_constrs:>9}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare the Kantorovich, Gilmore-Gomory and "
                                             "arc-flow ILP models on random CSP instances.")
    ap.add_argument("--m", type=int, default=10, help="number of distinct item widths")
    ap.add_argument("--W", type=int, default=100, help="roll width")
    ap.add_argument("--b-max", type=int, default=10, help="max demand per item (1 = bin packing)")
    ap.add_argument("--n-instances", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # 1) Toy example from the slides (W = 9, items 4,3,2 each demanded twice): OPT = 2
    print_results(run_all([4, 3, 2], [2, 2, 2], 9, args.time_limit), "slides toy example (OPT = 2)")

    # 2) Random instances
    for s in range(args.n_instances):
        w, b, W = generate_instance(args.m, args.W, b_max=args.b_max, seed=args.seed + s)
        results = run_all(w, b, W, args.time_limit, args.verbose)
        print_results(results, f"random instance seed={args.seed + s}  m={args.m} W={W} "
                               f"total items={sum(b)}")
        objs = {r.objective for r in results if r.optimal}
        assert len(objs) <= 1, f"models disagree on the optimum: {objs}"


if __name__ == "__main__":
    main()

