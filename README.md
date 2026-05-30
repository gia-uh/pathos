# PATHOS — Python AI Search Library

[![CI](https://github.com/gia-uh/pathos/actions/workflows/ci.yml/badge.svg)](https://github.com/gia-uh/pathos/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pathos-ai.svg)](https://pypi.org/project/pathos-ai/)
[![Python](https://img.shields.io/pypi/pyversions/pathos-ai.svg)](https://pypi.org/project/pathos-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready classical AI search algorithms for Python. No machine learning. Pure search.

**[Documentation](https://gia-uh.github.io/pathos)** · [PyPI](https://pypi.org/project/pathos-ai/) · [Examples](examples/)

## Philosophy

Define your *problem*, not your algorithm. PATHOS inspects the capabilities you declare and selects the best algorithm automatically.

```python
from pathos import Space

space = Space().initial("Madrid")

@space.successors
def neighbors(city):
    for next_city, km in roads[city]:
        yield next_city, next_city

@space.goal
def reached(city): return city == "Lisboa"

@space.heuristic
def h(city): return straight_line_km(city, "Lisboa")

result = space.solver().solve()
# → Uses A* automatically (has successors + goal + heuristic)
print(result.path, result.cost, result.algorithm)
```

## Install

```bash
pip install pathos-ai
```

## Algorithm Families

| Declare | Algorithms Available |
|---------|---------------------|
| `@evaluate` | Simulated Annealing, Genetic Algorithm, DE |
| `@successors + @goal` | BFS, DFS, IDDFS |
| `@successors + @evaluate` | Hill Climbing, Tabu Search |
| `@successors + @goal + @heuristic` | A*, IDA*, Greedy Best-First |
| `@successors + @goal + @heuristic + @evaluate` | Weighted A*, UCS |
| `.adversarial() + @terminal + @utility` | Minimax, Alpha-Beta, MCTS |
| `CSPSpace + @constraint` | Backtracking, Forward Checking |

## Specialized Spaces

```python
from pathos import GraphSpace, CSPSpace, TourSpace, GameSpace

# Graph search (auto-provides successors from adjacency)
space = GraphSpace(graph=city_graph).initial("A")

# Constraint satisfaction (auto-provides successors + goal)
csp = CSPSpace(variables=["X", "Y", "Z"])

# Tour optimization (TSP — auto-provides 2-opt neighborhood)
tour = TourSpace(nodes=cities, distances=dist_matrix)

# Adversarial games (auto-sets adversarial mode)
game = GameSpace().initial(board)
```

## Direct Algorithm Access

```python
from pathos.algorithms import AStar, GeneticAlgorithm, AlphaBeta

result = AStar(space).solve()  # bypass auto-selection
```

## SearchResult

Every algorithm returns a uniform `SearchResult`:

```python
result.solution      # final state
result.path          # list of (action, state) steps
result.cost          # total cost
result.algorithm     # algorithm name
result.nodes_expanded
result.elapsed       # seconds
result.found         # bool
```

## Examples

- [Route Planning (A*)](examples/route_planning.py)
- [TSP (SA + GA)](examples/tsp.py)
- [N-Queens (CSP)](examples/nqueens.py)
- [Tic-tac-toe (Alpha-Beta)](examples/tictactoe.py)
- [8-Puzzle (A*)](examples/puzzle8.py)

## License

MIT — [gia-uh](https://github.com/gia-uh)
