from pathos.core.parallel import batch_map
from pathos.core.space import Space
from pathos.algorithms.evolutionary import GeneticAlgorithm, DifferentialEvolution
from pathos.algorithms.local import LocalBeamSearch
import random as _random


def _double(x):
    return x * 2


def test_batch_map_serial_matches_map():
    items = [1, 2, 3, 4, 5]
    assert batch_map(_double, items, n_workers=1) == [2, 4, 6, 8, 10]


def test_batch_map_parallel_matches_serial():
    items = list(range(10))
    serial = batch_map(_double, items, n_workers=1)
    parallel = batch_map(_double, items, n_workers=2)
    assert parallel == serial


def test_batch_map_empty():
    assert batch_map(_double, [], n_workers=2) == []


def test_algorithm_stores_n_workers():
    space = Space().initial(0).parallel(4)

    @space.evaluate
    def cost(x): return x

    ga = GeneticAlgorithm(space)
    assert ga._n_workers == 4


def test_algorithm_defaults_to_serial():
    space = Space().initial(0)

    @space.evaluate
    def cost(x): return x

    ga = GeneticAlgorithm(space)
    assert ga._n_workers == 1


# --- Picklable evaluate functions for parallel tests ---

def _ga_fitness(ind):
    return -sum(ind)


def _ga_crossover(p1, p2):
    pt = _random.randint(1, len(p1) - 1)
    return tuple(p1[:pt] + p2[pt:])


def _ga_mutate(ind):
    i = _random.randint(0, len(ind) - 1)
    lst = list(ind)
    lst[i] = 1 - lst[i]
    return tuple(lst)


def _de_cost(x):
    return sum(xi ** 2 for xi in x)


def _parabola(x):
    return (x - 5) ** 2


def _int_neighbors_parallel(x):
    for dx in [-1, 1]:
        nx = x + dx
        if 0 <= nx <= 10:
            yield str(dx), nx


def test_ga_parallel_correctness():
    space = (
        Space()
        .initial(lambda: tuple(_random.randint(0, 1) for _ in range(10)))
        .parallel(2)
    )
    space.evaluate(_ga_fitness)

    result = GeneticAlgorithm(
        space, pop_size=20, generations=100,
        crossover_fn=_ga_crossover, mutate_fn=_ga_mutate,
    ).solve()
    assert result.found
    assert -result.cost >= 8


def test_de_parallel_correctness():
    space = (
        Space()
        .initial(lambda: [_random.uniform(-5, 5) for _ in range(3)])
        .parallel(2)
    )
    space.evaluate(_de_cost)

    result = DifferentialEvolution(space, pop_size=20, generations=200).solve()
    assert result.found
    assert result.cost < 0.1


def test_local_beam_search_parallel_correctness():
    space = Space().initial(0).parallel(2)
    space.evaluate(_parabola)

    @space.successors
    def neighbors(x): yield from _int_neighbors_parallel(x)

    result = LocalBeamSearch(space, k=3, max_iter=50).solve()
    assert result.found
    assert result.solution == 5
