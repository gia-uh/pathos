# tests/test_evolutionary.py
from pathos.core.space import Space
from pathos.algorithms.evolutionary import SimulatedAnnealing, GeneticAlgorithm, DifferentialEvolution

def _eval_space(fn, initial):
    space = Space().initial(initial)

    @space.evaluate
    def cost(s): return fn(s)

    return space

def test_simulated_annealing():
    # Minimize (x-5)^2 for integer x, SA on real space
    import random
    def neighbors(x):
        for _ in range(5):
            yield "perturb", x + random.uniform(-1, 1)

    space = Space().initial(0.0)

    @space.evaluate
    def cost(s): return (s - 5.0) ** 2

    @space.successors
    def nbrs(s): yield from neighbors(s)

    result = SimulatedAnnealing(space, max_iter=2000, T0=10.0, cooling=0.995).solve()
    assert result.found
    assert abs(result.solution - 5.0) < 0.5

def test_genetic_algorithm():
    # GA over binary strings, maximize number of 1s
    import random

    def decode(ind): return sum(ind)

    def crossover(p1, p2):
        pt = random.randint(1, len(p1) - 1)
        return tuple(p1[:pt] + p2[pt:])

    def mutate(ind):
        i = random.randint(0, len(ind) - 1)
        lst = list(ind)
        lst[i] = 1 - lst[i]
        return tuple(lst)

    space = Space().initial(lambda: tuple(random.randint(0, 1) for _ in range(10)))

    @space.evaluate
    def fitness(ind): return -decode(ind)  # minimize negative = maximize ones

    result = GeneticAlgorithm(
        space,
        pop_size=20,
        generations=100,
        crossover_fn=crossover,
        mutate_fn=mutate,
    ).solve()
    assert result.found
    assert -result.cost >= 8  # at least 8 ones

def test_differential_evolution():
    # DE on continuous: minimize sum of squares
    import random
    space = Space().initial(lambda: [random.uniform(-5, 5) for _ in range(3)])

    @space.evaluate
    def cost(x): return sum(xi ** 2 for xi in x)

    result = DifferentialEvolution(space, pop_size=20, generations=200, F=0.8, CR=0.9).solve()
    assert result.found
    assert result.cost < 0.1
