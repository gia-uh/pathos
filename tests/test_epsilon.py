"""Tests for SearchResult.epsilon and the derived .optimal property.

epsilon semantics:
  - 1.0  → proven optimal (admissible algorithm finished)
  - >1.0 → ε-bounded suboptimal (cost ≤ ε × optimal); e.g. WeightedAStar(weight)
  - inf  → unbounded suboptimal; e.g. GreedyBestFirst
  - None → not applicable (e.g. metaheuristics with no quality bound,
           or not_found results)
"""
from __future__ import annotations

import math

from pathos.core.result import SearchResult


def test_epsilon_default_is_none():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="Foo", nodes_expanded=0, elapsed=0.0, found=True,
    )
    assert r.epsilon is None


def test_optimal_is_true_when_epsilon_one():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="AStar", nodes_expanded=0, elapsed=0.0, found=True,
        epsilon=1.0,
    )
    assert r.optimal is True


def test_optimal_is_false_when_epsilon_above_one():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="WeightedAStar", nodes_expanded=0, elapsed=0.0, found=True,
        epsilon=2.0,
    )
    assert r.optimal is False


def test_optimal_is_false_when_epsilon_inf():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="GreedyBestFirst", nodes_expanded=0, elapsed=0.0, found=True,
        epsilon=math.inf,
    )
    assert r.optimal is False


def test_optimal_is_false_when_epsilon_none():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="HillClimbing", nodes_expanded=0, elapsed=0.0, found=True,
        epsilon=None,
    )
    assert r.optimal is False


def test_not_found_has_epsilon_none():
    r = SearchResult.not_found("AStar", 0, 0.0)
    assert r.epsilon is None
    assert r.optimal is False


# ---------------------------------------------------------------------------
# Per-algorithm epsilon emission — path-search family
# ---------------------------------------------------------------------------

import pathos.algorithms  # noqa: F401, E402
from pathos import Space  # noqa: E402


def _trivial_unit_cost_chain() -> Space:
    space = Space().initial("a")

    @space.successors
    def expand(s):
        if s == "a":
            yield "go_b", "b"
        elif s == "b":
            yield "go_c", "c"

    @space.goal
    def is_goal(s):
        return s == "c"

    @space.evaluate
    def cost(s):
        return 1.0

    return space


def test_bfs_emits_epsilon_one_on_success():
    from pathos.algorithms.uninformed import BFS
    space = _trivial_unit_cost_chain()
    r = BFS(space).solve()
    assert r.found is True
    assert r.epsilon == 1.0
    assert r.optimal is True


def test_ucs_emits_epsilon_one_on_success():
    from pathos.algorithms.uninformed import UCS
    space = _trivial_unit_cost_chain()
    r = UCS(space).solve()
    assert r.found is True
    assert r.epsilon == 1.0


def test_iddfs_emits_epsilon_one_on_success():
    from pathos.algorithms.uninformed import IDDFS
    space = _trivial_unit_cost_chain()
    r = IDDFS(space).solve()
    assert r.found is True
    assert r.epsilon == 1.0


def test_dfs_emits_epsilon_inf_on_success():
    """DFS is non-optimal — see FINDINGS §3b. epsilon=inf is the
    correct quality bound."""
    from pathos.algorithms.uninformed import DFS
    space = _trivial_unit_cost_chain()
    r = DFS(space).solve()
    assert r.found is True
    assert r.epsilon == math.inf
