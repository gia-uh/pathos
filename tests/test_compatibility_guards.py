"""Regression tests for capability-lattice runtime crashes.

Before the compatible_with overrides in csp.py / evolutionary.py /
uninformed.py, the auto-solver would offer algorithms whose declared
`requires` were satisfied but whose runtime preconditions were not.
This file pins each previously-crashing combination through both:

  (a) cls.compatible_with(space) → False
  (b) space.solver(candidates=[cls]).solve() → RuntimeError
      (the empty-candidate path in Solver._select)
"""
from __future__ import annotations

import pytest

import pathos.algorithms  # noqa: F401 — register
from pathos import CSPSpace, Space, TourSpace
from pathos.algorithms.csp import Backtracking, ForwardChecking, MinConflicts
from pathos.algorithms.evolutionary import DifferentialEvolution
from pathos.algorithms.uninformed import BFS, DFS, IDDFS, UCS


def _cspspace_nqueens(n: int = 4) -> CSPSpace:
    csp = CSPSpace(variables=list(range(n)))

    @csp.domain
    def dom(c):
        return list(range(n))

    @csp.constraint
    def no_attack(a):
        items = list(a.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                c1, r1 = items[i]
                c2, r2 = items[j]
                if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                    return False
        return True

    return csp


def _tour_space() -> TourSpace:
    cities = list(range(5))
    distances = {(i, j): 1.0 for i in cities for j in cities if i != j}
    space = TourSpace(nodes=cities, distances=distances)

    @space.evaluate
    def cost(tour):
        return float(len(tour))

    return space


def _plain_puzzle_space() -> Space:
    GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    space = Space().initial(GOAL)

    @space.successors
    def expand(b):
        return iter(())

    @space.goal
    def is_goal(b):
        return b == GOAL

    @space.evaluate
    def step_cost(b):
        return 1.0

    return space


@pytest.mark.parametrize("cls", [BFS, DFS, IDDFS])
def test_uninformed_rejects_cspspace(cls):
    csp = _cspspace_nqueens()
    assert cls.compatible_with(csp) is False
    with pytest.raises(RuntimeError, match="No compatible algorithm"):
        csp.solver(candidates=[cls]).solve()


def test_ucs_rejects_cspspace_with_evaluate():
    csp = _cspspace_nqueens()

    @csp.evaluate
    def cost(a):
        return float(len(a))

    assert UCS.compatible_with(csp) is False
    with pytest.raises(RuntimeError):
        csp.solver(candidates=[UCS]).solve()


def test_de_rejects_tourspace():
    space = _tour_space()
    assert DifferentialEvolution.compatible_with(space) is False
    with pytest.raises(RuntimeError):
        space.solver(candidates=[DifferentialEvolution]).solve()


@pytest.mark.parametrize("cls", [Backtracking, ForwardChecking])
def test_csp_algorithms_reject_plain_space(cls):
    puzzle = _plain_puzzle_space()
    assert cls.compatible_with(puzzle) is False
    with pytest.raises(RuntimeError):
        puzzle.solver(candidates=[cls]).solve()


def test_minconflicts_rejects_plain_space():
    puzzle = _plain_puzzle_space()
    assert MinConflicts.compatible_with(puzzle) is False
    with pytest.raises(RuntimeError):
        puzzle.solver(candidates=[MinConflicts]).solve()


def test_csp_path_still_works():
    """Sanity: CSPSpace + Backtracking/FC/MC still auto-pick and solve."""
    csp = _cspspace_nqueens(4)
    assert Backtracking.compatible_with(csp) is True
    assert ForwardChecking.compatible_with(csp) is True
    result = csp.solver().solve()
    assert result.found
    assert len(result.solution) == 4
