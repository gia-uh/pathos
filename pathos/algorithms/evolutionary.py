from __future__ import annotations
import time
import math
import random
import copy
from typing import Any, Callable
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register
from pathos.core.parallel import batch_map


@register
class SimulatedAnnealing(Algorithm):
    """Simulated Annealing — probabilistic local search with cooling schedule.

    Requires: successors, evaluate.

    Attributes:
        requires: Capability set needed.
        power_rank: 17.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.EVALUATE})
    power_rank = 17

    def __init__(self, space: Any, max_iter: int = 1000,
                 T0: float = 100.0, cooling: float = 0.99) -> None:
        super().__init__(space)
        self.max_iter = max_iter
        self.T0 = T0
        self.cooling = cooling

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        current = self.space._initial
        current_cost = self.space._evaluate(current)
        best, best_cost = current, current_cost
        T = self.T0

        for i in range(self.max_iter):
            neighbors = list(self.space._successors(current))
            if not neighbors:
                break
            _, candidate = random.choice(neighbors)
            candidate_cost = self.space._evaluate(candidate)
            delta = candidate_cost - current_cost
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current, current_cost = candidate, candidate_cost
                if current_cost < best_cost:
                    best, best_cost = current, current_cost
            T *= self.cooling

        return SearchResult(best, None, best_cost, "SimulatedAnnealing",
                            self.max_iter, time.perf_counter() - t0,
                            self._goal_reached(best))


@register
class GeneticAlgorithm(Algorithm):
    """Genetic Algorithm — population-based evolutionary optimization.

    Requires: evaluate.
    Caller must supply crossover_fn and mutate_fn for non-trivial state types.

    Attributes:
        requires: Capability set needed.
        power_rank: 14.
    """

    requires = frozenset({Capability.EVALUATE})
    power_rank = 14

    def __init__(self, space: Any, pop_size: int = 50, generations: int = 100,
                 crossover_fn: Callable[..., Any] | None = None,
                 mutate_fn: Callable[..., Any] | None = None,
                 mutation_rate: float = 0.1) -> None:
        super().__init__(space)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_fn = crossover_fn
        self.mutate_fn = mutate_fn
        self.mutation_rate = mutation_rate

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        population = [self.space._initial for _ in range(self.pop_size)]
        costs = batch_map(self.space._evaluate, population, self._n_workers)
        best_idx = min(range(len(population)), key=lambda i: costs[i])
        best, best_cost = population[best_idx], costs[best_idx]

        # Default operators when none supplied. If @successors is declared
        # we sample a random neighbor — works for any neighborhood-bearing
        # space (TSP via 2-opt, 8-puzzle via tile moves, GraphSpace via
        # adjacency). Falls back to deepcopy only when no successors are
        # available, which leaves GA in its no-variation degenerate mode
        # for pure-{evaluate} spaces — those should pass real operators.
        has_succ = Capability.SUCCESSORS in self.space.capabilities

        def _random_neighbor(state: Any) -> Any:
            neighbors = list(self.space._successors(state))
            if not neighbors:
                return state
            return random.choice(neighbors)[1]

        def _default_crossover(p1: Any, p2: Any) -> Any:
            return _random_neighbor(p1) if has_succ else copy.deepcopy(p1)

        def _default_mutate(child: Any) -> Any:
            return _random_neighbor(child) if has_succ else child

        crossover = self.crossover_fn or _default_crossover
        mutate = self.mutate_fn or _default_mutate

        for _ in range(self.generations):
            costs = batch_map(self.space._evaluate, population, self._n_workers)
            pairs = sorted(zip(costs, population))
            population = [x for _, x in pairs[: self.pop_size // 2]]
            while len(population) < self.pop_size:
                p1, p2 = random.sample(population, 2)
                child = crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = mutate(child)
                population.append(child)
            gen_costs = batch_map(self.space._evaluate, population, self._n_workers)
            gen_best_idx = min(range(len(population)), key=lambda i: gen_costs[i])
            if gen_costs[gen_best_idx] < best_cost:
                best, best_cost = population[gen_best_idx], gen_costs[gen_best_idx]

        return SearchResult(best, None, best_cost, "GeneticAlgorithm",
                            self.generations * self.pop_size, time.perf_counter() - t0,
                            self._goal_reached(best))


@register
class DifferentialEvolution(Algorithm):
    """Differential Evolution — vector-based evolutionary optimization for continuous spaces.

    Requires: evaluate. State must be a list/vector.

    Attributes:
        requires: Capability set needed.
        power_rank: 13.
    """

    requires = frozenset({Capability.EVALUATE})
    power_rank = 13

    @classmethod
    def compatible_with(cls, space: Any) -> bool:
        # DE generates real-valued perturbations; it isn't valid for spaces
        # whose state is a discrete permutation (TourSpace) or partial
        # assignment dict (CSPSpace).
        if not super().compatible_with(space):
            return False
        # Use string class names to avoid a reverse import from spaces/.
        for ancestor in type(space).__mro__:
            if ancestor.__name__ in {"TourSpace", "CSPSpace"}:
                return False
        return True

    def __init__(self, space: Any, pop_size: int = 20, generations: int = 100,
                 F: float = 0.8, CR: float = 0.9) -> None:
        super().__init__(space)
        self.pop_size = pop_size
        self.generations = generations
        self.F = F
        self.CR = CR

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        pop = [self.space._initial for _ in range(self.pop_size)]
        costs = batch_map(self.space._evaluate, pop, self._n_workers)
        best_idx = min(range(self.pop_size), key=lambda i: costs[i])
        best, best_cost = pop[best_idx], costs[best_idx]

        for _ in range(self.generations):
            trials = []
            for i in range(self.pop_size):
                x = pop[i]
                if not isinstance(x, list):
                    trials.append(x)
                    continue
                a, b, c = random.sample([j for j in range(self.pop_size) if j != i], 3)
                xa, xb, xc = pop[a], pop[b], pop[c]
                dim = len(x)
                j_rand = random.randint(0, dim - 1)
                trials.append([
                    xa[j] + self.F * (xb[j] - xc[j])
                    if random.random() < self.CR or j == j_rand else x[j]
                    for j in range(dim)
                ])
            trial_costs = batch_map(self.space._evaluate, trials, self._n_workers)
            for i in range(self.pop_size):
                if trial_costs[i] < costs[i]:
                    pop[i], costs[i] = trials[i], trial_costs[i]
                    if trial_costs[i] < best_cost:
                        best, best_cost = trials[i], trial_costs[i]

        return SearchResult(best, None, best_cost, "DifferentialEvolution",
                            self.generations * self.pop_size, time.perf_counter() - t0,
                            self._goal_reached(best))


def _is_numeric_vector(state: Any) -> bool:
    """True if state is a non-empty list/tuple of numbers (int/float, no bool)."""
    if not isinstance(state, (list, tuple)) or len(state) == 0:
        return False
    return all(
        isinstance(x, (int, float)) and not isinstance(x, bool)
        for x in state
    )


@register
class ParticleSwarm(Algorithm):
    """Particle Swarm Optimization — population of particles guided by
    personal best and global best with inertia.

    Each particle has a position vector and a velocity vector. The velocity
    is updated each generation by an inertia component, a cognitive pull
    toward the particle's own historical best, and a social pull toward the
    swarm's global best. Position then advances by the velocity.

    Requires: evaluate. State must be a numeric vector (list/tuple of
    int/float). Bounds are estimated from the initial swarm (per-dimension
    min/max expanded by 50%) unless supplied explicitly; positions are
    clamped to bounds on each step.

    Attributes:
        requires: Capability set needed.
        power_rank: 12 — sibling of DifferentialEvolution (13), kept below
            DE so a registered tie doesn't reshuffle existing auto-picks.
    """

    requires = frozenset({Capability.EVALUATE})
    power_rank = 12

    @classmethod
    def compatible_with(cls, space: Any) -> bool:
        if not super().compatible_with(space):
            return False
        # Same continuous-only carveout as DifferentialEvolution.
        for ancestor in type(space).__mro__:
            if ancestor.__name__ in {"TourSpace", "CSPSpace"}:
                return False
        return _is_numeric_vector(space._initial)

    def __init__(
        self,
        space: Any,
        pop_size: int = 30,
        generations: int = 100,
        w: float = 0.7,           # inertia weight
        c1: float = 1.5,          # cognitive coefficient
        c2: float = 1.5,          # social coefficient
        bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__(space)
        self.pop_size = pop_size
        self.generations = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds

    def _estimate_bounds(
        self, swarm: list[list[float]],
    ) -> list[tuple[float, float]]:
        """Per-dimension [min, max] across the initial swarm, expanded 50%
        in each direction so particles can drift outside the seed range."""
        dim = len(swarm[0])
        out: list[tuple[float, float]] = []
        for d in range(dim):
            vals = [p[d] for p in swarm]
            lo, hi = min(vals), max(vals)
            span = hi - lo if hi > lo else max(abs(hi), 1.0)
            out.append((lo - 0.5 * span, hi + 0.5 * span))
        return out

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        # Each call to space._initial invokes the factory; gives diversity
        # when the user's factory is randomized (the usual case).
        positions: list[list[float]] = [
            list(self.space._initial) for _ in range(self.pop_size)
        ]
        dim = len(positions[0])
        bounds = self.bounds or self._estimate_bounds(positions)

        # Velocities initialised in a fraction of the bound span.
        velocities: list[list[float]] = []
        for _ in range(self.pop_size):
            v = [
                random.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1)
                for lo, hi in bounds
            ]
            velocities.append(v)

        costs = batch_map(self.space._evaluate, positions, self._n_workers)
        personal_best = [list(p) for p in positions]
        personal_best_cost = list(costs)
        gbest_idx = min(range(self.pop_size), key=lambda i: costs[i])
        gbest = list(positions[gbest_idx])
        gbest_cost = costs[gbest_idx]

        for _ in range(self.generations):
            for i in range(self.pop_size):
                r1 = random.random()
                r2 = random.random()
                for d in range(dim):
                    cog = self.c1 * r1 * (personal_best[i][d] - positions[i][d])
                    soc = self.c2 * r2 * (gbest[d] - positions[i][d])
                    velocities[i][d] = self.w * velocities[i][d] + cog + soc
                    positions[i][d] = positions[i][d] + velocities[i][d]
                    lo, hi = bounds[d]
                    if positions[i][d] < lo:
                        positions[i][d] = lo
                        velocities[i][d] = 0.0
                    elif positions[i][d] > hi:
                        positions[i][d] = hi
                        velocities[i][d] = 0.0

            costs = batch_map(self.space._evaluate, positions, self._n_workers)
            for i in range(self.pop_size):
                if costs[i] < personal_best_cost[i]:
                    personal_best_cost[i] = costs[i]
                    personal_best[i] = list(positions[i])
                    if costs[i] < gbest_cost:
                        gbest_cost = costs[i]
                        gbest = list(positions[i])

        return SearchResult(
            gbest, None, gbest_cost, "ParticleSwarm",
            self.generations * self.pop_size, time.perf_counter() - t0,
            self._goal_reached(gbest),
        )
