# Getting Started

## Installation

```bash
pip install pathos-ai
```

Requires Python 3.11+.

## Core Concept

PATHOS is problem-centric. You declare *what your problem can do* using decorator hooks on a `Space` object. The auto-solver selects the most powerful compatible algorithm.

## Minimal Example: BFS

```python
from pathos import Space

space = Space().initial("A")

@space.successors
def expand(state):
    graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    for neighbor in graph.get(state, []):
        yield neighbor, neighbor

@space.goal
def is_goal(state): return state == "D"

result = space.solver().solve()
print(result.solution, result.path)
```

## Adding a Heuristic: A*

Adding `@space.heuristic` unlocks A* automatically:

```python
@space.heuristic
def h(state):
    return {"A": 2, "B": 1, "C": 1, "D": 0}.get(state, 0)

result = space.solver().solve()
print(result.algorithm)  # → "AStar"
```

## Route Planning

```python
from pathos import GraphSpace

space = GraphSpace(graph=road_network).initial("Madrid")

@space.goal
def reached(city): return city == "Lisboa"

@space.heuristic
def h(city): return straight_line_km(city, "Lisboa")

result = space.solver().solve()
```

## Constraint Satisfaction

```python
from pathos import CSPSpace

csp = CSPSpace(variables=["X", "Y", "Z"])

@csp.domain
def dom(var): return [1, 2, 3]

@csp.constraint
def all_different(assignment):
    vals = list(assignment.values())
    return len(vals) == len(set(vals))

result = csp.solver().solve()
```

## Adversarial Games

```python
from pathos import GameSpace

space = GameSpace().initial(board)

@space.successors
def moves(board): ...

@space.terminal
def is_over(board): ...

@space.utility
def score(board, player): ...

result = space.solver().solve()  # → uses Alpha-Beta
```

## Parallel Evaluation

Population-based algorithms (GA, DE, LocalBeamSearch) can evaluate candidates in parallel using Python's `multiprocessing` module. Call `.parallel(n)` on the Space to enable it:

```python
from pathos import Space
from pathos.algorithms import GeneticAlgorithm

# Module-level function — required for multiprocessing (must be picklable)
def fitness(genome):
    return -sum(genome)

space = (
    Space()
    .initial(lambda: [random.randint(0, 1) for _ in range(100)])
    .parallel(4)  # use 4 worker processes
)
space.evaluate(fitness)

result = GeneticAlgorithm(space, pop_size=200, generations=500).solve()
```

**Pickling constraint:** The evaluate (and successors) functions must be defined at module level, not as lambdas or inner functions, because worker processes receive them via `pickle`. This is a standard Python multiprocessing limitation.

The default is `.parallel(1)` — fully serial, no overhead.

## Capability → Algorithm Reference

| Capabilities | Best Algorithm |
|---|---|
| `evaluate` | Simulated Annealing |
| `successors + goal` | BFS |
| `successors + evaluate` | Hill Climbing |
| `successors + goal + heuristic + evaluate` | A* |
| `adversarial + terminal + utility` | Alpha-Beta |
| `csp constraints` | Backtracking |
