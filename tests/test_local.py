from pathos.core.space import Space
from pathos.algorithms.local import HillClimbing, TabuSearch, LocalBeamSearch

def _optimization_space(fn, neighbors_fn, initial):
    space = Space().initial(initial)

    @space.evaluate
    def cost(s): return fn(s)

    @space.successors
    def neighbors(s): yield from neighbors_fn(s)

    return space

# Minimize: find x in [0,10] minimizing (x-5)^2
def _int_neighbors(x):
    for dx in [-1, 1]:
        nx = x + dx
        if 0 <= nx <= 10:
            yield str(dx), nx

def test_hill_climbing_finds_optimum():
    space = _optimization_space(lambda x: (x - 5) ** 2, _int_neighbors, 0)
    result = HillClimbing(space).solve()
    assert result.found
    assert result.solution == 5

def test_tabu_search_finds_optimum():
    space = _optimization_space(lambda x: (x - 5) ** 2, _int_neighbors, 0)
    result = TabuSearch(space, max_iter=50, tabu_size=5).solve()
    assert result.found
    assert result.solution == 5

def test_local_beam_search_finds_optimum():
    space = _optimization_space(lambda x: (x - 5) ** 2, _int_neighbors, 0)
    result = LocalBeamSearch(space, k=3, max_iter=50).solve()
    assert result.found
    assert result.solution == 5
