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
                            self.max_iter, time.perf_counter() - t0, True)


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
                 crossover_fn: Callable | None = None,
                 mutate_fn: Callable | None = None,
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
        best = min(population, key=self.space._evaluate)
        best_cost = self.space._evaluate(best)

        for _ in range(self.generations):
            scored = sorted(population, key=self.space._evaluate)
            population = scored[: self.pop_size // 2]  # elitism
            while len(population) < self.pop_size:
                p1, p2 = random.sample(population, 2)
                if self.crossover_fn:
                    child = self.crossover_fn(p1, p2)
                else:
                    child = copy.deepcopy(p1)
                if self.mutate_fn and random.random() < self.mutation_rate:
                    child = self.mutate_fn(child)
                population.append(child)
            current_best = min(population, key=self.space._evaluate)
            current_cost = self.space._evaluate(current_best)
            if current_cost < best_cost:
                best, best_cost = current_best, current_cost

        return SearchResult(best, None, best_cost, "GeneticAlgorithm",
                            self.generations * self.pop_size, time.perf_counter() - t0, True)


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
        costs = [self.space._evaluate(x) for x in pop]
        best_idx = min(range(self.pop_size), key=lambda i: costs[i])
        best, best_cost = pop[best_idx], costs[best_idx]

        for _ in range(self.generations):
            for i in range(self.pop_size):
                a, b, c = random.sample([j for j in range(self.pop_size) if j != i], 3)
                x, xa, xb, xc = pop[i], pop[a], pop[b], pop[c]
                if not isinstance(x, list):
                    continue  # DE requires list/vector state
                dim = len(x)
                j_rand = random.randint(0, dim - 1)
                trial = [
                    xa[j] + self.F * (xb[j] - xc[j])
                    if random.random() < self.CR or j == j_rand else x[j]
                    for j in range(dim)
                ]
                trial_cost = self.space._evaluate(trial)
                if trial_cost < costs[i]:
                    pop[i], costs[i] = trial, trial_cost
                    if trial_cost < best_cost:
                        best, best_cost = trial, trial_cost

        return SearchResult(best, None, best_cost, "DifferentialEvolution",
                            self.generations * self.pop_size, time.perf_counter() - t0, True)
