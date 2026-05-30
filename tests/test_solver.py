import pytest
from pathos.core.space import Space
from pathos.core.solver import Solver
from pathos.algorithms.uninformed import BFS

def _simple_space():
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

    return space

def test_solver_selects_algorithm():
    space = _simple_space()
    solver = space.solver()
    result = solver.solve()
    assert result.found
    assert result.solution == "c"

def test_solver_explicit_algorithm():
    space = _simple_space()
    solver = space.solver(candidates=[BFS])
    result = solver.solve()
    assert result.found

def test_solver_warns_unused_capability():
    space = _simple_space()

    @space.heuristic
    def h(s): return 0.0

    with pytest.warns(UserWarning, match="heuristic"):
        solver = space.solver(candidates=[BFS])
        solver.solve()
