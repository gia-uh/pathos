"""Regression for FINDINGS §2 — GA quality on TSP with default operators.

Before the fix, GeneticAlgorithm with crossover_fn=None and mutate_fn=None
fell through to `child = deepcopy(p1)` and `if self.mutate_fn` → skip,
so children were exact copies of parents. GA degenerated to "pick best,
copy it forever" — no variation, no improvement. On TourSpace at 25
cities this produced tours 2.5× worse than TabuSearch.

After the fix, when the space declares @successors, GA defaults to a
random-neighbor operator for both crossover and mutation. GA now finds
the optimal tour on small TSP instances and a competitive (not absurd)
tour on larger ones.
"""
from __future__ import annotations

import random

import pathos.algorithms  # noqa: F401
from pathos import TourSpace
from pathos.algorithms.evolutionary import GeneticAlgorithm
from pathos.algorithms.local import HillClimbing


def _tsp_space(n_cities: int, seed: int = 0) -> TourSpace:
    rng = random.Random(seed)
    cities = list(range(n_cities))
    coords = {i: (rng.uniform(0, 100), rng.uniform(0, 100)) for i in cities}

    def dist(a, b):
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    distances = {(i, j): dist(i, j) for i in cities for j in cities if i != j}
    space = TourSpace(nodes=cities, distances=distances)

    @space.evaluate
    def tour_cost(tour):
        return sum(
            distances[(tour[i], tour[(i + 1) % len(tour)])]
            for i in range(len(tour))
        )

    return space


def test_ga_on_tsp_finds_competitive_tour():
    """GA at default settings on a 10-city TSP should find a tour within
    2× of HillClimbing's. Before the fix it returned 3-5× worse."""
    random.seed(42)
    space = _tsp_space(10, seed=42)
    hc_cost = HillClimbing(space).solve().cost

    # Reset the same problem for GA (each .solve() draws a fresh random
    # initial via the factory, so this is a fair comparison.)
    random.seed(42)
    space2 = _tsp_space(10, seed=42)
    ga_cost = GeneticAlgorithm(space2, pop_size=30, generations=50).solve().cost

    assert ga_cost is not None and hc_cost is not None
    assert ga_cost < 2.0 * hc_cost, (
        f"GA cost {ga_cost:.2f} is more than 2x HillClimbing's "
        f"{hc_cost:.2f} — default operators have regressed"
    )


def test_ga_without_successors_still_runs():
    """Pure-{evaluate} space (no @successors): GA degenerates to copy mode
    but must not crash. User should pass real operators for non-trivial
    state types — verified by the docstring."""
    from pathos.core.space import Space

    space = Space().initial(5.0)

    @space.evaluate
    def quadratic(x):
        return (x - 3.0) ** 2

    # No @successors — GA defaults to deepcopy, runs without crash.
    result = GeneticAlgorithm(space, pop_size=10, generations=5).solve()
    assert result.found is True  # no GOAL declared → True
    assert result.cost is not None
