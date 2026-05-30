"""Route planning with A* on a weighted graph."""
import pathos.algorithms  # ensure all algorithms are registered
from pathos import GraphSpace

# Road network: city -> [(neighbor, km)]
roads = {
    "Madrid":    [("Zaragoza", 325), ("Lisboa", 638), ("Sevilla", 534)],
    "Zaragoza":  [("Barcelona", 296), ("Madrid", 325)],
    "Barcelona": [("Zaragoza", 296), ("Valencia", 349)],
    "Valencia":  [("Barcelona", 349), ("Sevilla", 656)],
    "Sevilla":   [("Madrid", 534), ("Valencia", 656), ("Lisboa", 450)],
    "Lisboa":    [("Madrid", 638), ("Sevilla", 450)],
}

# Straight-line distances to Lisboa (heuristic)
sld_to_lisboa = {
    "Madrid": 502, "Zaragoza": 828, "Barcelona": 1067,
    "Valencia": 880, "Sevilla": 312, "Lisboa": 0,
}

space = GraphSpace(graph=roads).initial("Madrid")


@space.goal
def reached(city): return city == "Lisboa"


@space.heuristic
def h(city): return sld_to_lisboa.get(city, 0)


result = space.solver().solve()
cities = ["Madrid"] + [step[1] for step in (result.path or [])]
print(f"Route found: {' -> '.join(cities)}")
print(f"Algorithm: {result.algorithm}")
print(f"Distance: {result.cost:.0f} km")
print(f"Nodes expanded: {result.nodes_expanded}")
