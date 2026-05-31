"""Regression test for FINDINGS §2b.

When a space declares a goal predicate, the auto-selector must prefer
algorithms that honour it. Before this fix, the highest-power_rank
compatible algorithm won regardless — on GraphSpace this picked
TabuSearch (rank 18) over BFS (10) / UCS (12) because GraphSpace
auto-declares @evaluate from edge weights, even though TabuSearch
ignores @goal and just optimises @evaluate.
"""
from __future__ import annotations

import pathos.algorithms  # noqa: F401
from pathos import CSPSpace, GraphSpace, Space, TourSpace
from pathos.algorithms.evolutionary import (
    DifferentialEvolution,
    GeneticAlgorithm,
    SimulatedAnnealing,
)
from pathos.algorithms.local import (
    HillClimbing,
    LocalBeamSearch,
    TabuSearch,
)

GOAL_IGNORING = {
    HillClimbing, TabuSearch, LocalBeamSearch,
    SimulatedAnnealing, GeneticAlgorithm, DifferentialEvolution,
}


def test_graphspace_with_goal_picks_goal_honoring():
    """GraphSpace auto-declares @evaluate from edge weights; if the user
    additionally declares a @goal, the auto-pick must NOT land on a
    local-search algorithm that ignores it."""
    graph = {"a": [("b", 1.0)], "b": [("c", 1.0)], "c": []}
    space = GraphSpace(graph=graph).initial("a")

    @space.goal
    def reached(n):
        return n == "c"

    picked = space.solver()._select()
    assert picked not in GOAL_IGNORING, (
        f"Auto-pick {picked.__name__} ignores @goal on a goal-bearing "
        f"GraphSpace — see FINDINGS §2b"
    )
    # Sanity: the picked algorithm actually solves it.
    result = space.solver().solve()
    assert result.found is True
    assert result.solution == "c"


def test_pure_optimization_still_picks_metaheuristic():
    """When no @goal is declared, local-search/metaheuristic should still
    win on power_rank — the preference is goal-aware, not blanket."""
    cities = list(range(5))
    distances = {(i, j): 1.0 for i in cities for j in cities if i != j}
    space = TourSpace(nodes=cities, distances=distances)

    @space.evaluate
    def cost(tour):
        return 1.0

    picked = space.solver()._select()
    # TourSpace with @evaluate only — no goal, so TabuSearch (highest
    # local-search rank) is still the right pick.
    assert picked is TabuSearch


def test_8puzzle_still_picks_astar():
    """A goal-bearing problem with a heuristic should still pick A*
    (highest-ranked goal-honoring algo with HEURISTIC)."""
    GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    space = Space().initial((1, 2, 3, 4, 5, 6, 0, 7, 8))

    @space.successors
    def expand(b):
        return iter(())

    @space.goal
    def is_goal(b):
        return b == GOAL

    @space.heuristic
    def h(b):
        return 0

    @space.evaluate
    def cost(b):
        return 1.0

    from pathos.algorithms.informed import AStar
    assert space.solver()._select() is AStar


def test_cspspace_still_picks_backtracking():
    """CSPSpace declares GOAL; among goal-honoring CSP-shaped algos
    Backtracking (rank 9) outranks FC (8) after the prior demote."""
    csp = CSPSpace(variables=list(range(4)))

    @csp.domain
    def dom(c):
        return [0, 1, 2, 3]

    @csp.constraint
    def c(a):
        return True

    from pathos.algorithms.csp import Backtracking
    assert csp.solver()._select() is Backtracking


def test_falls_back_to_full_pool_when_no_goal_honoring_compatible():
    """If a space has GOAL declared but no goal-honoring algorithm is
    compatible, the selector must NOT raise — it should fall through to
    the full compatible pool."""
    # Construct a pathological case: a goal-bearing space whose state is
    # unhashable (so BFS/DFS/IDDFS are rejected) and isn't dict-shaped
    # (so Backtracking/FC/MC are rejected) — but EVALUATE is declared.
    # Local-search becomes the only compatible option.
    space = Space().initial([1, 2, 3])  # list — unhashable

    @space.successors
    def expand(s):
        yield "shift", list(s[1:]) + [s[0]]

    @space.goal
    def is_goal(s):
        return s == [3, 2, 1]  # never reachable via rotation

    @space.evaluate
    def cost(s):
        return float(sum(s))

    # Should not raise — falls through to goal-ignoring local search.
    picked = space.solver()._select()
    assert picked in GOAL_IGNORING
