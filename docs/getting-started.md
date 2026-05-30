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

## Capability → Algorithm Reference

| Capabilities | Best Algorithm |
|---|---|
| `evaluate` | Simulated Annealing |
| `successors + goal` | BFS |
| `successors + evaluate` | Hill Climbing |
| `successors + goal + heuristic + evaluate` | A* |
| `adversarial + terminal + utility` | Alpha-Beta |
| `csp constraints` | Backtracking |
