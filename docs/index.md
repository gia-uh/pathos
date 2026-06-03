---
hide:
  - navigation
---

# PATHOS

**Production-ready classical AI search algorithms for Python — no machine learning, pure search.**

[![CI](https://github.com/gia-uh/pathos/actions/workflows/ci.yml/badge.svg)](https://github.com/gia-uh/pathos/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pathos-ai.svg)](https://pypi.org/project/pathos-ai/)
[![Python](https://img.shields.io/pypi/pyversions/pathos-ai.svg)](https://pypi.org/project/pathos-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gia-uh/pathos/blob/main/LICENSE)

PATHOS is **problem-centric**: you declare *what your problem can do* with
decorator hooks; the auto-solver picks the best compatible algorithm.

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

# Default mode is anytime — returns proven optimal if it finishes,
# best incumbent if the budget runs out. Never not_found if any
# feasible path exists.
result = space.solver(timeout=10).solve()
print(result.path, result.cost, result.algorithm, result.optimal)
```

## Install

```bash
pip install pathos-ai
```

Requires Python 3.11+. MIT license.

## Highlights

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } **Problem-centric**

    ---

    Declare your problem's structure with decorators. The solver figures out
    which algorithm fits.

    [:octicons-arrow-right-24: Getting started](getting-started.md)

-   :material-clock-fast:{ .lg .middle } **Anytime by default**

    ---

    `solver().solve()` runs a cascade tuned to your problem family
    (informed, local, CSP, adversarial) and always delivers the best
    incumbent within your budget — never `not_found` when a feasible
    answer exists.

    [:octicons-arrow-right-24: Modes & anytime](guides/modes-and-anytime.md)

-   :material-cancel:{ .lg .middle } **Cooperative cancellation**

    ---

    Every algorithm checks a cancel token in its main loop. Set a timeout
    and metaheuristics return their best individual seen — automatically.

    [:octicons-arrow-right-24: Cancel-token protocol](guides/cancel-token.md)

-   :material-library:{ .lg .middle } **Comprehensive coverage**

    ---

    Uninformed (BFS, DFS, IDDFS, UCS), informed (A\*, IDA\*, WAStar, Greedy,
    Bidirectional), local search (HC, Tabu, Beam), evolutionary (GA, DE, PSO,
    SA), adversarial (Minimax, AlphaBeta, MCTS), CSP (Backtracking, FC, MC).

    [:octicons-arrow-right-24: API reference](api/algorithms.md)

</div>

## Algorithm families

| Declare | Algorithms available |
|---------|---------------------|
| `@evaluate` | SA, GA, DE, PSO |
| `@successors + @goal` | BFS, DFS, IDDFS *(DFS is non-optimal — for shortest paths prefer BFS/UCS)* |
| `@successors + @evaluate` | **AnytimeLocal** (default under `mode="auto"`), HillClimbing, TabuSearch, LocalBeamSearch, SimulatedAnnealing |
| `@successors + @goal + @heuristic` | GreedyBestFirst |
| `@successors + @goal + @heuristic + @evaluate` | **AnytimeAStar** (default under `mode="auto"`), AStar, IDAstar, WeightedAStar, BidirectionalAStar, UCS |
| `.adversarial() + @terminal + @utility` | **AnytimeAdversarial** (default under `mode="auto"`), Minimax, AlphaBeta, Negamax, MCTS |
| `CSPSpace + @constraint` | **AnytimeCSP** (default under `mode="auto"`), Backtracking, ForwardChecking, MinConflicts |
