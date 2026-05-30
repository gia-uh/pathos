"""Traveling Salesman Problem with Simulated Annealing."""
import random
import pathos.algorithms  # ensure all algorithms are registered
from pathos import TourSpace

random.seed(42)

# 6-city distance matrix
cities = list(range(6))
coords = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in cities}


def dist(a, b):
    x1, y1 = coords[a]
    x2, y2 = coords[b]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


distances = {(i, j): dist(i, j) for i in cities for j in cities if i != j}

space = TourSpace(nodes=cities, distances=distances)


@space.evaluate
def tour_cost(tour):
    return sum(distances[(tour[i], tour[(i + 1) % len(tour)])] for i in range(len(tour)))


result = space.solver().solve()
print(f"Best tour: {result.solution}")
print(f"Cost: {result.cost:.2f}")
print(f"Algorithm: {result.algorithm}")
