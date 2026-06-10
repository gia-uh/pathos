import pytest
from benchmarks.realistic.records import RunRow
from benchmarks.realistic.metrics import (
    gap_pct, median_cost, oracle_row, is_no_cliff_failure,
    tunability_gain,
)
from benchmarks.realistic.lower_bounds import tsp_one_tree_lb, bin_packing_waste_lb


def _row(mode, algorithm, cost, feasible=True, elapsed=0.1):
    return RunRow(
        suite="X1", size=10, seed=1, mode=mode, algorithm=algorithm,
        cost=cost, feasible=feasible, elapsed=elapsed, nodes_expanded=0,
    )


def test_gap_pct_returns_zero_for_matching_costs():
    assert gap_pct(auto=10.0, oracle=10.0) == pytest.approx(0.0)


def test_gap_pct_returns_relative_excess():
    # auto=11, oracle=10 -> 10% gap
    assert gap_pct(auto=11.0, oracle=10.0) == pytest.approx(10.0)


def test_gap_pct_handles_none_oracle():
    assert gap_pct(auto=10.0, oracle=None) is None


def test_gap_pct_handles_none_auto():
    assert gap_pct(auto=None, oracle=10.0) is None


def test_gap_pct_handles_zero_oracle():
    # Avoid div-by-zero; use absolute difference scaled by 1 as fallback.
    assert gap_pct(auto=0.0, oracle=0.0) == pytest.approx(0.0)
    assert gap_pct(auto=1.0, oracle=0.0) is None  # undefined gap, log as None


def test_median_cost_ignores_infeasible_and_none():
    rows = [
        _row("auto_headline", "A", 10.0),
        _row("auto_headline", "A", 12.0),
        _row("auto_headline", "A", None, feasible=False),
    ]
    assert median_cost(rows) == pytest.approx(11.0)


def test_median_cost_returns_none_when_no_feasible():
    rows = [_row("auto_headline", "A", None, feasible=False)]
    assert median_cost(rows) is None


def test_oracle_row_picks_lowest_cost_feasible():
    rows = [
        _row("oracle_AStar", "AStar", 10.0),
        _row("oracle_AStar", "AStar", 12.0),
        _row("oracle_BFS", "BFS", 8.0),
        _row("oracle_BFS", "BFS", 9.0),
    ]
    name, cost = oracle_row(rows)
    assert name == "BFS"
    assert cost == pytest.approx(8.5)


def test_oracle_row_returns_none_when_no_oracle_runs():
    rows = [_row("auto_headline", "A", 10.0)]
    assert oracle_row(rows) == (None, None)


def test_tunability_gain_returns_zero_when_auto_already_best():
    rows = [
        _row("auto_headline", "A", 10.0),
        _row("oracle_A", "A", 10.0),
        _row("oracle_B", "B", 12.0),
    ]
    gain, who = tunability_gain(rows)
    assert gain == pytest.approx(0.0)
    assert who is None  # no candidate beats auto


def test_tunability_gain_names_the_beater():
    rows = [
        _row("auto_headline", "A", 10.0),
        _row("oracle_B", "B", 9.0),  # 10% better
    ]
    gain, who = tunability_gain(rows)
    assert gain == pytest.approx(10.0)
    assert who == "B"


def test_no_cliff_failure_detects_not_found():
    row = _row("auto_headline", "A", None, feasible=False)
    row.cost = None
    assert is_no_cliff_failure(row, tolerance=0.0) is True


def test_no_cliff_failure_passes_feasible_run():
    row = _row("auto_headline", "A", 10.0, feasible=True)
    assert is_no_cliff_failure(row, tolerance=0.0) is False


def test_tsp_one_tree_lb_is_a_lower_bound_on_tour_cost():
    # Triangle: 3 nodes, distances 1-1-1 -> LB <= 3 (optimal tour = 3).
    distances = {(0, 1): 1.0, (1, 0): 1.0,
                 (0, 2): 1.0, (2, 0): 1.0,
                 (1, 2): 1.0, (2, 1): 1.0}
    lb = tsp_one_tree_lb(nodes=[0, 1, 2], distances=distances)
    assert lb <= 3.0 + 1e-9


def test_bin_packing_waste_lb_is_ceil_total_div_capacity():
    items = [3.0, 3.0, 3.0, 3.0]   # total = 12
    capacity = 5.0                  # ceil(12/5) = 3 bins
    assert bin_packing_waste_lb(items=items, capacity=capacity) == 3
