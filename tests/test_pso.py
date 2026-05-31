"""Tests for ParticleSwarm — continuous-domain {evaluate}-only optimizer.

Verifies convergence on sphere (unimodal), compatibility guards
(rejects TourSpace/CSPSpace and non-vector states), bounds clamping,
and integration through space.solver() with explicit candidates.
"""
from __future__ import annotations

import random

import pathos.algorithms  # noqa: F401
from pathos import CSPSpace, Space, TourSpace
from pathos.algorithms.evolutionary import ParticleSwarm


def _sphere_space(dim: int = 5, seed: int = 0) -> Space:
    """f(x) = sum(x_i^2), optimum at the origin with f*=0."""
    rng = random.Random(seed)

    space = Space().initial(
        lambda: [rng.uniform(-5.0, 5.0) for _ in range(dim)]
    )

    @space.evaluate
    def f(x):
        return sum(xi * xi for xi in x)

    return space


def test_pso_converges_on_sphere():
    """30 particles × 100 generations on the 5-d sphere should land
    within a small ball of the origin. Optimum cost is 0; we accept
    < 0.5 (~2% of the initial random-cost magnitude)."""
    random.seed(42)
    space = _sphere_space(dim=5, seed=42)
    result = ParticleSwarm(space, pop_size=30, generations=100).solve()
    assert result.cost is not None
    assert result.cost < 0.5, (
        f"PSO failed to converge on the 5-d sphere: final cost {result.cost:.3f}"
    )
    # Sanity: returned solution is a numeric vector of the right dim.
    assert isinstance(result.solution, list)
    assert len(result.solution) == 5


def test_pso_respects_explicit_bounds():
    """If bounds are supplied, all positions in the returned solution
    must lie within them (PSO clamps each coordinate to [lo, hi])."""
    random.seed(42)
    space = _sphere_space(dim=3, seed=42)
    bounds = [(-1.0, 1.0)] * 3
    result = ParticleSwarm(
        space, pop_size=20, generations=20, bounds=bounds,
    ).solve()
    for v, (lo, hi) in zip(result.solution, bounds):
        assert lo <= v <= hi, f"coordinate {v} out of bounds [{lo}, {hi}]"


def test_pso_rejects_tourspace():
    """Same continuous-only carveout as DE."""
    cities = list(range(5))
    distances = {(i, j): 1.0 for i in cities for j in cities if i != j}
    space = TourSpace(nodes=cities, distances=distances)

    @space.evaluate
    def cost(t):
        return 1.0

    assert ParticleSwarm.compatible_with(space) is False


def test_pso_rejects_cspspace():
    csp = CSPSpace(variables=list(range(3)))

    @csp.domain
    def dom(c):
        return [0, 1, 2]

    @csp.constraint
    def c(a):
        return True

    assert ParticleSwarm.compatible_with(csp) is False


def test_pso_rejects_non_vector_state():
    """State must be a numeric vector; PSO has nothing to do with a
    scalar, a string, or a dict."""
    space = Space().initial("hello")

    @space.evaluate
    def f(s):
        return 1.0

    assert ParticleSwarm.compatible_with(space) is False


def test_pso_via_solver_with_explicit_candidates():
    """End-to-end: PSO is reachable through the solver path."""
    random.seed(42)
    space = _sphere_space(dim=4, seed=42)
    result = space.solver(candidates=[ParticleSwarm]).solve()
    assert result.algorithm == "ParticleSwarm"
    assert result.cost is not None
    assert result.cost < 2.0  # generous bound for shorter default run
