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

## Parallel Evaluation

Population-based algorithms (GA, DE, LocalBeamSearch) support multiprocessing via `.parallel(n)`:

```python
# Evaluate all candidates in parallel across 4 processes
space = Space().initial(lambda: random_genome()).parallel(4)

# evaluate fn must be a module-level function (picklable)
def fitness(genome): return -sum(genome)
space.evaluate(fitness)

result = GeneticAlgorithm(space, pop_size=200, generations=500).solve()
```

Pass `n=1` (default) for serial execution. Falls back automatically when population size is 1.

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

## Performance

Reference numbers from `python -m benchmarks.bench --repeat 3` on an Intel
i7-6820HQ @ 2.70 GHz, Python 3.13. Algorithms are the ones auto-selected by
`space.solver()`. Reproduce with the same command; raw records dumped via
`--json`.

**N-Queens (Backtracking, CSPSpace)**

| N | elapsed (s, median) | nodes expanded |
|---|---:|---:|
| 6  | 0.0003 | 31 |
| 8  | 0.0022 | 113 |
| 10 | 0.0031 | 102 |
| 12 | 0.0149 | 261 |
| 14 | 0.1726 | 1 899 |
| 16 | 1.2363 | 10 052 |

**TSP (TabuSearch, TourSpace, 100 iters)**

| cities | elapsed (s, median) | tour cost (median) |
|---:|---:|---:|
| 5  | 0.0014 | 197.8 |
| 8  | 0.0060 | 253.3 |
| 12 | 0.0224 | 272.2 |
| 16 | 0.0439 | 354.9 |
| 20 | 0.0800 | 383.4 |
| 25 | 0.1593 | 409.3 |

**8-Puzzle (A\* + Manhattan)**

| scramble depth | elapsed (s, median) | nodes expanded | solution length |
|---:|---:|---:|---:|
| 10 | 0.0001 | 12    | 10 |
| 20 | 0.0042 | 461   | 20 |
| 30 | 0.0097 | 1 397 | 24 |
| 40 | 0.0127 | 1 728 | 26 |
| 50 | 0.0086 | 1 186 | 22 |

Solution length plateaus around 22–26 because the 8-puzzle state-space
diameter is ~31 — deeper scrambles don't make harder instances.

## Examples

- [Route Planning (A*)](examples/route_planning.py)
- [TSP (SA + GA)](examples/tsp.py)
- [N-Queens (CSP)](examples/nqueens.py)
- [Tic-tac-toe (Alpha-Beta)](examples/tictactoe.py)
- [8-Puzzle (A*)](examples/puzzle8.py)

## License

MIT — [gia-uh](https://github.com/gia-uh)
