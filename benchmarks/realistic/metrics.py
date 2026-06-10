from __future__ import annotations
import statistics
from typing import Iterable
from benchmarks.realistic.records import RunRow


def gap_pct(auto: float | None, oracle: float | None) -> float | None:
    """Relative excess of `auto` over `oracle`, in percent.

    Returns None if either is None or if oracle is 0 while auto is not
    (gap is undefined). When both are 0, returns 0.0 (trivially equal).
    """
    if auto is None or oracle is None:
        return None
    if oracle == 0.0:
        return 0.0 if auto == 0.0 else None
    return 100.0 * (auto - oracle) / abs(oracle)


def median_cost(rows: Iterable[RunRow]) -> float | None:
    feasible_costs = [r.cost for r in rows if r.feasible and r.cost is not None]
    if not feasible_costs:
        return None
    return statistics.median(feasible_costs)


def oracle_row(rows: Iterable[RunRow]) -> tuple[str | None, float | None]:
    """Best (lowest median cost) among oracle_* modes. Returns
    (algorithm_name, median_cost) or (None, None) if no oracle data."""
    by_algo: dict[str, list[RunRow]] = {}
    for r in rows:
        if not r.mode.startswith("oracle_"):
            continue
        by_algo.setdefault(r.algorithm, []).append(r)
    best_name: str | None = None
    best_cost: float | None = None
    for name, group in by_algo.items():
        med = median_cost(group)
        if med is None:
            continue
        if best_cost is None or med < best_cost:
            best_cost = med
            best_name = name
    return best_name, best_cost


def tunability_gain(rows: Iterable[RunRow]) -> tuple[float, str | None]:
    """How much the best oracle algorithm beats auto-headline, in
    percent. Returns (0.0, None) if no oracle beats auto."""
    rows_list = list(rows)
    auto_rows = [r for r in rows_list if r.mode == "auto_headline"]
    auto_med = median_cost(auto_rows)
    if auto_med is None:
        return 0.0, None
    best_name, best_cost = oracle_row(rows_list)
    if best_cost is None or best_cost >= auto_med:
        return 0.0, None
    gain = 100.0 * (auto_med - best_cost) / abs(auto_med) if auto_med != 0.0 else 0.0
    return gain, best_name


def is_no_cliff_failure(row: RunRow, tolerance: float) -> bool:
    """A single auto-headline / auto-stress run that constitutes a
    no-cliff bug: crash, not_found, or infeasible beyond tolerance.

    `nodes_expanded = -2` is the bench's existing crash sentinel
    (inherited from bench.py's run_one). Cost being None on a feasible
    flag should not happen, but treat it as failure too.
    """
    if not row.mode.startswith("auto_"):
        return False
    if row.missing_capability is not None:
        return False  # missing-capability suites are carved out
    if row.nodes_expanded == -2:
        return True  # crash sentinel
    if not row.feasible:
        return True
    if row.cost is None:
        return True
    return False
