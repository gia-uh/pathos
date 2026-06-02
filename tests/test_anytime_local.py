"""Tests for AnytimeLocal cascade behaviour.

Confirms:
  - On pure-optimization spaces (SUCCESSORS+EVALUATE, no GOAL) under
    mode=auto, AnytimeLocal wins selection and finds an improving
    incumbent.
  - On goal-bearing spaces with a heuristic, AnytimeAStar still wins
    (AnytimeLocal cedes by not declaring GOAL in requires).
  - On goal-bearing spaces without heuristic, goal-honoring algorithms
    win (AnytimeLocal does not displace BFS/UCS just because EVALUATE
    is declared).
  - Under mode="exact" / "approximate", a pure-optimization space falls
    back to a single local-search algorithm rather than AnytimeLocal.
  - Cascade order is [HillClimbing, SimulatedAnnealing, TabuSearch]
    and keeps best incumbent across phases (lower cost wins).
  - On pre-armed cancel, returns not_found cleanly.
  - algorithm field is rewritten to "AnytimeLocal".
"""
from __future__ import annotations

import pathos.algorithms  # noqa: F401
from pathos import Space
from pathos.algorithms.local import (
    AnytimeLocal,
    HillClimbing,
    TabuSearch,
)
from pathos.algorithms.evolutionary import SimulatedAnnealing


def _bowl_space(low: int = 0, high: int = 10, target: int = 5) -> Space:
    """1D quadratic-bowl optimization. `target` is the minimizer; cost is
    (x - target) ** 2. Pure optimization — no GOAL declared."""
    space = Space().initial(low)

    @space.successors
    def expand(x):
        for dx in (-1, 1):
            nx = x + dx
            if low <= nx <= high:
                yield str(dx), nx

    @space.evaluate
    def cost(x):
        return float((x - target) ** 2)

    return space


def _goal_space_with_heuristic() -> Space:
    """A goal-bearing space with a heuristic — AnytimeAStar territory."""
    space = Space().initial(0)

    @space.successors
    def expand(s):
        if s < 5:
            yield "next", s + 1

    @space.goal
    def is_goal(s):
        return s == 5

    @space.heuristic
    def h(s):
        return float(5 - s)

    @space.evaluate
    def cost(s):
        return 1.0

    return space


def _goal_space_no_heuristic() -> Space:
    """Goal-bearing, no heuristic — goal-honoring uninformed algorithms
    win; AnytimeLocal must not displace them."""
    space = Space().initial(0)

    @space.successors
    def expand(s):
        if s < 5:
            yield "next", s + 1

    @space.goal
    def is_goal(s):
        return s == 5

    @space.evaluate
    def cost(s):
        return 1.0

    return space


# ---------------------------------------------------------------------------
# Selection: AnytimeLocal wins on pure-optimization spaces under mode=auto
# ---------------------------------------------------------------------------

def test_anytime_local_wins_selection_on_pure_optimization_space():
    space = _bowl_space()
    picked = space.solver()._select()
    assert picked is AnytimeLocal


def test_anytime_local_finds_optimum_on_bowl():
    space = _bowl_space()
    result = space.solver().solve()
    assert result.found is True
    assert result.algorithm == "AnytimeLocal"
    assert result.solution == 5
    assert result.cost == 0.0


def test_anytime_local_cedes_to_anytime_astar_on_heuristic_space():
    """AnytimeAStar requires HEURISTIC, AnytimeLocal does not. On a
    space with both GOAL and HEURISTIC, AnytimeAStar wins (it's the
    only goal-honoring meta-algorithm at score 1000)."""
    space = _goal_space_with_heuristic()
    picked = space.solver()._select()
    assert picked.__name__ == "AnytimeAStar"


def test_anytime_local_not_picked_on_goal_without_heuristic():
    """Goal-bearing, no heuristic. Goal-honoring filter in Solver._select
    excludes AnytimeLocal (no GOAL in requires), leaving BFS/DFS/UCS."""
    space = _goal_space_no_heuristic()
    picked = space.solver()._select()
    assert picked is not AnytimeLocal


def test_anytime_local_not_picked_under_mode_exact():
    """mode=exact: pure-optimization space falls back to a base local-
    search algorithm rather than AnytimeLocal."""
    space = _bowl_space()
    picked = space.solver(mode="exact")._select()
    assert picked is not AnytimeLocal


# ---------------------------------------------------------------------------
# Cascade body
# ---------------------------------------------------------------------------

def test_anytime_local_schedule_is_hc_sa_tabu():
    """Cascade order: fast-probe (HC) → escape (SA, Tabu)."""
    assert [cls for cls, _ in AnytimeLocal.SCHEDULE] == [
        HillClimbing, SimulatedAnnealing, TabuSearch,
    ]


def test_anytime_local_cancel_returns_not_found():
    """Pre-arm the cancel; AnytimeLocal bails before running any phase."""
    space = _bowl_space()
    space._request_cancel()
    result = AnytimeLocal(space).solve()
    assert result.found is False
    assert result.algorithm == "AnytimeLocal"


def test_anytime_local_algorithm_field_is_meta_not_phase():
    """Returned result's algorithm is 'AnytimeLocal' even though the
    underlying solver was HC / SA / Tabu."""
    space = _bowl_space()
    result = space.solver().solve()
    assert result.algorithm == "AnytimeLocal"


def test_anytime_local_keeps_best_incumbent_across_phases():
    """Lower cost wins. Drive directly to verify the meta picks the
    best of the phases (HC alone gets cost=0 on this convex bowl)."""
    space = _bowl_space()
    result = AnytimeLocal(space).solve()
    assert result.found is True
    # On a convex bowl HC reaches the optimum cost=0; SA/Tabu can't
    # improve below 0, so the incumbent stays at 0.
    assert result.cost == 0.0
