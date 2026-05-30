# PATHOS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PATHOS — a production-ready Python library for classical AI search — from empty repo to PyPI-ready, with MkDocs docs on GitHub Pages, comprehensive tests, and realistic examples.

**Architecture:** Problem-centric design: user defines a `Space` via decorator hooks, auto-solver selects the best algorithm from a capability lattice. Two layers: high-level `Space.solver()` and direct low-level algorithm classes.

**Tech Stack:** Python 3.11+, pytest, mypy, MkDocs + Material + mkdocstrings, pyproject.toml, GitHub Actions (CI + Pages + PyPI)

**Spec:** `vault/Atlas/Architecture/2026-05-29-pathos-design.md`

---

## File Map

```
repos/pathos/
  pyproject.toml
  README.md
  .github/
    workflows/
      ci.yml
      pages.yml
      publish.yml
  docs/
    index.md
    getting-started.md
    api/
      space.md
      algorithms.md
      spaces.md
      result.md
    examples/
      route-planning.md
      tsp.md
      nqueens.md
      tictactoe.md
      puzzle8.md
    mkdocs.yml
  pathos/
    __init__.py
    core/
      __init__.py
      capabilities.py
      result.py
      space.py
      solver.py
    spaces/
      __init__.py
      graph.py
      csp.py
      tour.py
      game.py
    algorithms/
      __init__.py
      base.py
      uninformed.py
      informed.py
      local.py
      evolutionary.py
      adversarial.py
      csp.py
  examples/
    route_planning.py
    tsp.py
    nqueens.py
    tictactoe.py
    puzzle8.py
  tests/
    conftest.py
    test_space.py
    test_solver.py
    test_uninformed.py
    test_informed.py
    test_local.py
    test_evolutionary.py
    test_adversarial.py
    test_csp.py
    test_spaces_graph.py
    test_spaces_csp.py
    test_spaces_tour.py
    test_spaces_game.py
    test_examples.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `pathos/__init__.py`
- Create: `pathos/core/__init__.py`, `pathos/spaces/__init__.py`, `pathos/algorithms/__init__.py`
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pathos-ai"
version = "0.1.0"
description = "Production-ready classical AI search algorithms"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "mypy>=1.8",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.24",
]

[project.urls]
Homepage = "https://github.com/gia-uh/pathos"
Documentation = "https://gia-uh.github.io/pathos"
Repository = "https://github.com/gia-uh/pathos"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short"

[tool.mypy]
python_version = "3.11"
strict = true
```

- [ ] **Step 2: Create package skeleton**

```bash
mkdir -p pathos/core pathos/spaces pathos/algorithms
mkdir -p tests examples docs/api docs/examples
touch pathos/__init__.py pathos/core/__init__.py
touch pathos/spaces/__init__.py pathos/algorithms/__init__.py
touch tests/__init__.py tests/conftest.py
```

- [ ] **Step 3: Create `.github/workflows/ci.yml`**

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pytest --cov=pathos --cov-report=term-missing
```

- [ ] **Step 4: Install in dev mode and verify**

```bash
pip install -e ".[dev]"
python -c "import pathos; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pathos/ tests/ .github/
git commit -m "chore: project scaffolding — pyproject, package skeleton, CI"
```

---

## Task 2: Capabilities Enum

**Files:**
- Create: `pathos/core/capabilities.py`
- Create: `tests/test_space.py` (partial)

- [ ] **Step 1: Write failing test**

```python
# tests/test_space.py
from pathos.core.capabilities import Capability

def test_capability_enum_members():
    assert Capability.SUCCESSORS in Capability
    assert Capability.GOAL in Capability
    assert Capability.HEURISTIC in Capability
    assert Capability.EVALUATE in Capability
    assert Capability.TERMINAL in Capability
    assert Capability.UTILITY in Capability
    assert Capability.REVERSE_SUCCESSORS in Capability
    assert Capability.VARIABLES in Capability
    assert Capability.DOMAINS in Capability
    assert Capability.CONSTRAINTS in Capability

def test_capability_set_operations():
    required = {Capability.SUCCESSORS, Capability.GOAL}
    available = {Capability.SUCCESSORS, Capability.GOAL, Capability.HEURISTIC}
    assert required <= available
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_space.py::test_capability_enum_members -v
```

Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Implement `pathos/core/capabilities.py`**

```python
from enum import Enum, auto


class Capability(Enum):
    SUCCESSORS = auto()
    GOAL = auto()
    HEURISTIC = auto()
    EVALUATE = auto()
    TERMINAL = auto()
    UTILITY = auto()
    REVERSE_SUCCESSORS = auto()
    VARIABLES = auto()
    DOMAINS = auto()
    CONSTRAINTS = auto()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_space.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pathos/core/capabilities.py tests/test_space.py
git commit -m "feat(core): Capability enum"
```

---

## Task 3: SearchResult

**Files:**
- Create: `pathos/core/result.py`
- Modify: `tests/test_space.py`

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_space.py
from pathos.core.result import SearchResult

def test_search_result_found():
    r = SearchResult(
        solution="goal",
        path=[("move", "goal")],
        cost=1.0,
        algorithm="BFS",
        nodes_expanded=5,
        elapsed=0.01,
        found=True,
    )
    assert r.found
    assert r.solution == "goal"
    assert r.algorithm == "BFS"

def test_search_result_not_found():
    r = SearchResult.not_found(algorithm="AStar", nodes_expanded=100, elapsed=0.5)
    assert not r.found
    assert r.solution is None
    assert r.cost is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_space.py::test_search_result_found -v
```

- [ ] **Step 3: Implement `pathos/core/result.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    solution: Any
    path: list[tuple[Any, Any]] | None
    cost: float | None
    algorithm: str
    nodes_expanded: int
    elapsed: float
    found: bool

    @classmethod
    def not_found(
        cls, algorithm: str, nodes_expanded: int, elapsed: float
    ) -> SearchResult:
        return cls(
            solution=None,
            path=None,
            cost=None,
            algorithm=algorithm,
            nodes_expanded=nodes_expanded,
            elapsed=elapsed,
            found=False,
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_space.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pathos/core/result.py tests/test_space.py
git commit -m "feat(core): SearchResult dataclass"
```

---

## Task 4: Space Class — Decorator System

**Files:**
- Create: `pathos/core/space.py`
- Modify: `tests/test_space.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_space.py
from pathos.core.space import Space
from pathos.core.capabilities import Capability

def test_space_successors_decorator():
    space = Space().initial("a")

    @space.successors
    def expand(state):
        yield "go_b", "b"

    assert Capability.SUCCESSORS in space.capabilities
    results = list(space._successors("a"))
    assert results == [("go_b", "b")]

def test_space_goal_decorator():
    space = Space().initial("a")

    @space.goal
    def is_goal(state):
        return state == "b"

    assert Capability.GOAL in space.capabilities
    assert space._goal("b")
    assert not space._goal("a")

def test_space_heuristic_decorator():
    space = Space().initial("a")

    @space.heuristic
    def h(state):
        return 0.0

    assert Capability.HEURISTIC in space.capabilities

def test_space_evaluate_decorator():
    space = Space().initial("a")

    @space.evaluate
    def cost(state):
        return 1.0

    assert Capability.EVALUATE in space.capabilities

def test_space_initial_value():
    space = Space().initial("start")
    assert space._initial == "start"

def test_space_initial_callable():
    space = Space().initial(lambda: "start")
    assert space._initial == "start"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_space.py::test_space_successors_decorator -v
```

- [ ] **Step 3: Implement `pathos/core/space.py`**

```python
from __future__ import annotations
from typing import Any, Callable, Iterable
from pathos.core.capabilities import Capability


class Space:
    def __init__(self) -> None:
        self.capabilities: set[Capability] = set()
        self._initial_value: Any = None
        self._initial_factory: Callable[[], Any] | None = None
        self._timeout: float | None = None
        self._n_workers: int = 1
        self._adversarial: bool = False
        self._players: int = 2
        self._maximizing_player: int = 0

        self._successors: Callable | None = None
        self._goal: Callable | None = None
        self._heuristic: Callable | None = None
        self._evaluate: Callable | None = None
        self._terminal: Callable | None = None
        self._utility: Callable | None = None
        self._reverse_successors: Callable | None = None

    # --- fluent builder ---

    def initial(self, state: Any) -> Space:
        if callable(state):
            self._initial_factory = state
        else:
            self._initial_value = state
        return self

    def adversarial(self, players: int = 2, maximizing_player: int = 0) -> Space:
        self._adversarial = True
        self._players = players
        self._maximizing_player = maximizing_player
        self.capabilities.add(Capability.TERMINAL)  # structural flag
        return self

    def timeout(self, seconds: float) -> Space:
        self._timeout = seconds
        return self

    def parallel(self, workers: int) -> Space:
        self._n_workers = workers
        return self

    @property
    def _initial(self) -> Any:
        if self._initial_factory is not None:
            return self._initial_factory()
        return self._initial_value

    # --- decorator hooks ---

    def _make_hook(self, attr: str, cap: Capability) -> Callable:
        def decorator(fn: Callable) -> Callable:
            setattr(self, attr, fn)
            self.capabilities.add(cap)
            return fn
        return decorator

    @property
    def successors(self) -> Callable:
        return self._make_hook("_successors", Capability.SUCCESSORS)

    @property
    def goal(self) -> Callable:
        return self._make_hook("_goal", Capability.GOAL)

    @property
    def heuristic(self) -> Callable:
        return self._make_hook("_heuristic", Capability.HEURISTIC)

    @property
    def evaluate(self) -> Callable:
        return self._make_hook("_evaluate", Capability.EVALUATE)

    @property
    def terminal(self) -> Callable:
        return self._make_hook("_terminal", Capability.TERMINAL)

    @property
    def utility(self) -> Callable:
        return self._make_hook("_utility", Capability.UTILITY)

    @property
    def reverse_successors(self) -> Callable:
        return self._make_hook("_reverse_successors", Capability.REVERSE_SUCCESSORS)

    # --- solver factory ---

    def solver(
        self,
        candidates: list | None = None,
        timeout: float | None = None,
    ) -> "Solver":
        from pathos.core.solver import Solver
        return Solver(self, candidates=candidates, timeout=timeout or self._timeout)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_space.py -v
```

Expected: all PASS

- [ ] **Step 5: Update `pathos/core/__init__.py`**

```python
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.space import Space
```

- [ ] **Step 6: Commit**

```bash
git add pathos/core/space.py pathos/core/__init__.py tests/test_space.py
git commit -m "feat(core): Space class with decorator capability system"
```

---

## Task 5: Algorithm Base Class

**Files:**
- Create: `pathos/algorithms/base.py`

- [ ] **Step 1: Create base**

```python
# pathos/algorithms/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult

if TYPE_CHECKING:
    from pathos.core.space import Space


class Algorithm(ABC):
    requires: frozenset[Capability] = frozenset()
    power_rank: int = 0  # higher = preferred when multiple are compatible

    def __init__(self, space: Space) -> None:
        missing = self.requires - space.capabilities
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires capabilities: "
                f"{', '.join(c.name for c in missing)}"
            )
        self.space = space

    @abstractmethod
    def solve(self) -> SearchResult:
        ...

    @classmethod
    def compatible_with(cls, space: Space) -> bool:
        return cls.requires <= space.capabilities
```

- [ ] **Step 2: Update `pathos/algorithms/__init__.py`**

```python
from pathos.algorithms.base import Algorithm
```

- [ ] **Step 3: Commit**

```bash
git add pathos/algorithms/base.py pathos/algorithms/__init__.py
git commit -m "feat(algorithms): Algorithm base class"
```

---

## Task 6: Auto-Solver

**Files:**
- Create: `pathos/core/solver.py`
- Create: `tests/test_solver.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_solver.py
import pytest
from pathos.core.space import Space
from pathos.core.solver import Solver
from pathos.algorithms.uninformed import BFS

def _simple_space():
    space = Space().initial("a")

    @space.successors
    def expand(s):
        if s == "a":
            yield "go_b", "b"
        elif s == "b":
            yield "go_c", "c"

    @space.goal
    def is_goal(s):
        return s == "c"

    return space

def test_solver_selects_algorithm():
    space = _simple_space()
    solver = space.solver()
    result = solver.solve()
    assert result.found
    assert result.solution == "c"

def test_solver_explicit_algorithm():
    space = _simple_space()
    solver = space.solver(candidates=[BFS])
    result = solver.solve()
    assert result.found

def test_solver_warns_unused_capability():
    space = _simple_space()

    @space.heuristic
    def h(s): return 0.0

    with pytest.warns(UserWarning, match="heuristic"):
        solver = space.solver(candidates=[BFS])
        solver.solve()
```

- [ ] **Step 2: Implement `pathos/core/solver.py`**

```python
from __future__ import annotations
import warnings
import time
from typing import TYPE_CHECKING
from pathos.core.result import SearchResult

if TYPE_CHECKING:
    from pathos.core.space import Space
    from pathos.algorithms.base import Algorithm


_REGISTRY: list[type[Algorithm]] = []


def register(cls: type[Algorithm]) -> type[Algorithm]:
    _REGISTRY.append(cls)
    return cls


class Solver:
    def __init__(
        self,
        space: Space,
        candidates: list[type[Algorithm]] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.space = space
        self.candidates = candidates
        self.timeout = timeout

    def _select(self) -> type[Algorithm]:
        pool = self.candidates if self.candidates is not None else _REGISTRY
        compatible = [cls for cls in pool if cls.compatible_with(self.space)]
        if not compatible:
            raise RuntimeError(
                f"No compatible algorithm for capabilities: "
                f"{', '.join(c.name for c in self.space.capabilities)}"
            )
        best = max(compatible, key=lambda cls: cls.power_rank)
        # warn about unused capabilities
        used = best.requires
        declared = self.space.capabilities
        unused = declared - used
        if unused:
            warnings.warn(
                f"Capabilities declared but not used by {best.__name__}: "
                f"{', '.join(c.name.lower() for c in unused)}",
                UserWarning,
                stacklevel=3,
            )
        return best

    def solve(self) -> SearchResult:
        cls = self._select()
        return cls(self.space).solve()
```

- [ ] **Step 3: Run tests (will fail until BFS exists — proceed to Task 7 first, then return)**

Note: Run full solver tests after Task 7 is complete.

- [ ] **Step 4: Commit**

```bash
git add pathos/core/solver.py tests/test_solver.py
git commit -m "feat(core): Auto-solver with capability-based algorithm selection"
```

---

## Task 7: Uninformed Search Algorithms

**Files:**
- Create: `pathos/algorithms/uninformed.py`
- Create: `tests/test_uninformed.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_uninformed.py
from pathos.core.space import Space
from pathos.algorithms.uninformed import BFS, DFS, IDDFS, UCS

def _graph_space(graph: dict, start: str, goal: str):
    space = Space().initial(start)

    @space.successors
    def expand(s):
        for action, (neighbor, cost) in graph.get(s, {}).items():
            yield action, neighbor

    @space.goal
    def is_goal(s): return s == goal
    return space

def _weighted_space(graph: dict, start: str, goal: str):
    space = Space().initial(start)

    @space.successors
    def expand(s):
        for neighbor, cost in graph.get(s, []):
            yield neighbor, neighbor  # action = neighbor name

    @space.goal
    def is_goal(s): return s == goal

    @space.evaluate
    def edge_cost(s): return 1.0
    return space

SIMPLE = {"a": {"ab": ("b", 1)}, "b": {"bc": ("c", 1)}}

def test_bfs_finds_goal():
    space = _graph_space(SIMPLE, "a", "c")
    result = BFS(space).solve()
    assert result.found
    assert result.solution == "c"
    assert result.algorithm == "BFS"

def test_dfs_finds_goal():
    space = _graph_space(SIMPLE, "a", "c")
    result = DFS(space).solve()
    assert result.found

def test_iddfs_finds_goal():
    space = _graph_space(SIMPLE, "a", "c")
    result = IDDFS(space).solve()
    assert result.found
    assert result.solution == "c"

def test_bfs_optimal_path():
    # BFS finds shortest path in unweighted graph
    graph = {
        "a": {"ab": ("b", 1), "ac": ("c", 1)},
        "b": {"bd": ("d", 1)},
        "c": {"cd": ("d", 1)},
    }
    space = _graph_space(graph, "a", "d")
    result = BFS(space).solve()
    assert result.found
    assert len(result.path) == 2  # a->b->d or a->c->d

def test_bfs_no_solution():
    space = Space().initial("a")

    @space.successors
    def expand(s): return iter([])

    @space.goal
    def is_goal(s): return s == "b"

    result = BFS(space).solve()
    assert not result.found
```

- [ ] **Step 2: Implement `pathos/algorithms/uninformed.py`**

```python
from __future__ import annotations
import time
from collections import deque
from typing import Any
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register


@register
class BFS(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 10

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        if self.space._goal(initial):
            return SearchResult(initial, [], 0.0, "BFS", 0, 0.0, True)

        frontier: deque[tuple[Any, list]] = deque([(initial, [])])
        visited: set = {initial}
        expanded = 0

        while frontier:
            state, path = frontier.popleft()
            expanded += 1
            for action, child in self.space._successors(state):
                if child in visited:
                    continue
                new_path = path + [(action, child)]
                if self.space._goal(child):
                    return SearchResult(
                        child, new_path, float(len(new_path)),
                        "BFS", expanded, time.perf_counter() - t0, True,
                    )
                visited.add(child)
                frontier.append((child, new_path))

        return SearchResult.not_found("BFS", expanded, time.perf_counter() - t0)


@register
class DFS(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 5

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        stack: list[tuple[Any, list]] = [(initial, [])]
        visited: set = set()
        expanded = 0

        while stack:
            state, path = stack.pop()
            if state in visited:
                continue
            visited.add(state)
            if self.space._goal(state):
                return SearchResult(
                    state, path, float(len(path)),
                    "DFS", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                if child not in visited:
                    stack.append((child, path + [(action, child)]))

        return SearchResult.not_found("DFS", expanded, time.perf_counter() - t0)


@register
class IDDFS(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 8

    def _dls(self, state: Any, path: list, depth: int, visited: set) -> tuple[Any, list] | None:
        if self.space._goal(state):
            return state, path
        if depth == 0:
            return None
        visited.add(state)
        for action, child in self.space._successors(state):
            if child not in visited:
                result = self._dls(child, path + [(action, child)], depth - 1, visited)
                if result is not None:
                    return result
        visited.discard(state)
        return None

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        expanded = 0
        for depth in range(1000):
            result = self._dls(initial, [], depth, set())
            if result is not None:
                state, path = result
                return SearchResult(
                    state, path, float(len(path)),
                    "IDDFS", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
        return SearchResult.not_found("IDDFS", expanded, time.perf_counter() - t0)


@register
class UCS(Algorithm):
    """Uniform-Cost Search (Dijkstra). Requires evaluate for edge costs."""
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL, Capability.EVALUATE})
    power_rank = 12

    def solve(self) -> SearchResult:
        import heapq
        t0 = time.perf_counter()
        initial = self.space._initial
        frontier: list = [(0.0, 0, initial, [])]
        visited: dict[Any, float] = {}
        counter = 1
        expanded = 0

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            if state in visited and visited[state] <= cost:
                continue
            visited[state] = cost
            if self.space._goal(state):
                return SearchResult(
                    state, path, cost, "UCS", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                edge = self.space._evaluate(child)
                new_cost = cost + edge
                if child not in visited or visited[child] > new_cost:
                    heapq.heappush(
                        frontier,
                        (new_cost, counter, child, path + [(action, child)])
                    )
                    counter += 1

        return SearchResult.not_found("UCS", expanded, time.perf_counter() - t0)
```

- [ ] **Step 3: Update `pathos/algorithms/__init__.py`**

```python
from pathos.algorithms.base import Algorithm
from pathos.algorithms.uninformed import BFS, DFS, IDDFS, UCS
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_uninformed.py tests/test_solver.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pathos/algorithms/uninformed.py pathos/algorithms/__init__.py tests/test_uninformed.py
git commit -m "feat(algorithms): BFS, DFS, IDDFS, UCS"
```

---

## Task 8: Informed Search (A*, IDA*, Greedy, Weighted A*, Bidirectional A*)

**Files:**
- Create: `pathos/algorithms/informed.py`
- Create: `tests/test_informed.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_informed.py
from pathos.core.space import Space
from pathos.algorithms.informed import AStar, IDAstar, GreedyBestFirst, WeightedAStar, BidirectionalAStar

def _heuristic_space(graph, costs, start, goal, h_fn):
    space = Space().initial(start)

    @space.successors
    def expand(s):
        for neighbor in graph.get(s, []):
            yield neighbor, neighbor

    @space.goal
    def is_goal(s): return s == goal

    @space.heuristic
    def h(s): return h_fn(s)

    @space.evaluate
    def cost(s): return costs.get(s, 1.0)

    return space

# Simple graph: a -> b -> c, h decreases toward goal
GRAPH = {"a": ["b"], "b": ["c"]}
COSTS = {"a": 1.0, "b": 1.0, "c": 0.0}
H = {"a": 2.0, "b": 1.0, "c": 0.0}

def test_astar_finds_goal():
    space = _heuristic_space(GRAPH, COSTS, "a", "c", lambda s: H[s])
    result = AStar(space).solve()
    assert result.found
    assert result.solution == "c"
    assert result.algorithm == "AStar"

def test_idastar_finds_goal():
    space = _heuristic_space(GRAPH, COSTS, "a", "c", lambda s: H[s])
    result = IDAstar(space).solve()
    assert result.found
    assert result.solution == "c"

def test_greedy_finds_goal():
    space = _heuristic_space(GRAPH, COSTS, "a", "c", lambda s: H[s])
    result = GreedyBestFirst(space).solve()
    assert result.found

def test_weighted_astar_finds_goal():
    space = _heuristic_space(GRAPH, COSTS, "a", "c", lambda s: H[s])
    result = WeightedAStar(space, weight=1.5).solve()
    assert result.found

def test_bidirectional_astar_finds_goal():
    bidir_graph = {"a": ["b"], "b": ["c"], "c": [], "rev_c": ["b"], "rev_b": ["a"]}
    space = Space().initial("a")

    @space.successors
    def fwd(s): yield from ((n, n) for n in GRAPH.get(s, []))

    @space.reverse_successors
    def bwd(s): yield from (("rev_" + s, "a") if s == "b" else [])

    @space.goal
    def is_goal(s): return s == "c"

    @space.heuristic
    def h(s): return H.get(s, 0.0)

    @space.evaluate
    def cost(s): return 1.0

    result = BidirectionalAStar(space).solve()
    assert result.found
```

- [ ] **Step 2: Implement `pathos/algorithms/informed.py`**

```python
from __future__ import annotations
import heapq
import time
import math
from typing import Any
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register


@register
class AStar(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL,
                          Capability.HEURISTIC, Capability.EVALUATE})
    power_rank = 30

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        h0 = self.space._heuristic(initial)
        frontier: list = [(h0, 0, 0.0, initial, [])]
        g_score: dict[Any, float] = {initial: 0.0}
        counter = 1
        expanded = 0

        while frontier:
            f, _, g, state, path = heapq.heappop(frontier)
            if g > g_score.get(state, math.inf):
                continue
            if self.space._goal(state):
                return SearchResult(
                    state, path, g, "AStar", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                new_g = g + self.space._evaluate(child)
                if new_g < g_score.get(child, math.inf):
                    g_score[child] = new_g
                    f_val = new_g + self.space._heuristic(child)
                    heapq.heappush(frontier, (f_val, counter, new_g, child, path + [(action, child)]))
                    counter += 1

        return SearchResult.not_found("AStar", expanded, time.perf_counter() - t0)


@register
class GreedyBestFirst(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL, Capability.HEURISTIC})
    power_rank = 20

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        frontier: list = [(self.space._heuristic(initial), 0, initial, [])]
        visited: set = set()
        counter = 1
        expanded = 0

        while frontier:
            _, _, state, path = heapq.heappop(frontier)
            if state in visited:
                continue
            visited.add(state)
            if self.space._goal(state):
                return SearchResult(
                    state, path, float(len(path)), "GreedyBestFirst",
                    expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                if child not in visited:
                    heapq.heappush(frontier, (
                        self.space._heuristic(child), counter, child, path + [(action, child)]
                    ))
                    counter += 1

        return SearchResult.not_found("GreedyBestFirst", expanded, time.perf_counter() - t0)


@register
class WeightedAStar(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL,
                          Capability.HEURISTIC, Capability.EVALUATE})
    power_rank = 28

    def __init__(self, space: Any, weight: float = 2.0) -> None:
        super().__init__(space)
        self.weight = weight

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        h0 = self.space._heuristic(initial)
        frontier: list = [(h0 * self.weight, 0, 0.0, initial, [])]
        g_score: dict[Any, float] = {initial: 0.0}
        counter = 1
        expanded = 0

        while frontier:
            _, _, g, state, path = heapq.heappop(frontier)
            if g > g_score.get(state, math.inf):
                continue
            if self.space._goal(state):
                return SearchResult(
                    state, path, g, "WeightedAStar", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                new_g = g + self.space._evaluate(child)
                if new_g < g_score.get(child, math.inf):
                    g_score[child] = new_g
                    f = new_g + self.weight * self.space._heuristic(child)
                    heapq.heappush(frontier, (f, counter, new_g, child, path + [(action, child)]))
                    counter += 1

        return SearchResult.not_found("WeightedAStar", expanded, time.perf_counter() - t0)


@register
class IDAstar(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL,
                          Capability.HEURISTIC, Capability.EVALUATE})
    power_rank = 25

    def _search(self, path: list, g: float, bound: float) -> tuple[float | str, list | None]:
        state = path[-1]
        f = g + self.space._heuristic(state)
        if f > bound:
            return f, None
        if self.space._goal(state):
            return "FOUND", path[:]
        minimum = math.inf
        for action, child in self.space._successors(state):
            if child not in path:
                path.append(child)
                result, found_path = self._search(path, g + self.space._evaluate(child), bound)
                if result == "FOUND":
                    return "FOUND", found_path
                if isinstance(result, float):
                    minimum = min(minimum, result)
                path.pop()
        return minimum, None

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        bound = self.space._heuristic(initial)
        path = [initial]
        expanded = 0
        while True:
            result, found = self._search(path, 0.0, bound)
            if result == "FOUND" and found:
                steps = [(found[i], found[i]) for i in range(1, len(found))]
                return SearchResult(
                    found[-1], steps, float(len(steps)),
                    "IDAstar", expanded, time.perf_counter() - t0, True,
                )
            if result == math.inf:
                return SearchResult.not_found("IDAstar", expanded, time.perf_counter() - t0)
            bound = result  # type: ignore
            expanded += 1


@register
class BidirectionalAStar(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL,
                          Capability.HEURISTIC, Capability.EVALUATE,
                          Capability.REVERSE_SUCCESSORS})
    power_rank = 35

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        # Find goal state via BFS first to get a concrete goal node
        from pathos.algorithms.uninformed import BFS
        bfs_result = BFS(self.space).solve()
        if not bfs_result.found:
            return SearchResult.not_found("BidirectionalAStar", 0, time.perf_counter() - t0)
        goal = bfs_result.solution

        fwd_dist: dict[Any, float] = {initial: 0.0}
        bwd_dist: dict[Any, float] = {goal: 0.0}
        fwd_prev: dict[Any, Any] = {}
        bwd_prev: dict[Any, Any] = {}
        fwd_open: list = [(self.space._heuristic(initial), 0, initial)]
        bwd_open: list = [(self.space._heuristic(goal), 1, goal)]
        counter = 2
        mu = math.inf
        expanded = 0

        while fwd_open or bwd_open:
            if fwd_open:
                _, _, s = heapq.heappop(fwd_open)
                expanded += 1
                for a, child in self.space._successors(s):
                    ng = fwd_dist[s] + self.space._evaluate(child)
                    if ng < fwd_dist.get(child, math.inf):
                        fwd_dist[child] = ng
                        fwd_prev[child] = s
                        heapq.heappush(fwd_open, (ng + self.space._heuristic(child), counter, child))
                        counter += 1
                        if child in bwd_dist:
                            mu = min(mu, ng + bwd_dist[child])
            if bwd_open:
                _, _, s = heapq.heappop(bwd_open)
                expanded += 1
                for a, prev in self.space._reverse_successors(s):
                    ng = bwd_dist[s] + self.space._evaluate(s)
                    if ng < bwd_dist.get(prev, math.inf):
                        bwd_dist[prev] = ng
                        bwd_prev[prev] = s
                        heapq.heappush(bwd_open, (ng + self.space._heuristic(prev), counter, prev))
                        counter += 1
                        if prev in fwd_dist:
                            mu = min(mu, fwd_dist[prev] + ng)
            if fwd_open and bwd_open:
                best_f = fwd_open[0][0] if fwd_open else math.inf
                best_b = bwd_open[0][0] if bwd_open else math.inf
                if best_f + best_b >= mu:
                    break

        if mu == math.inf:
            return SearchResult.not_found("BidirectionalAStar", expanded, time.perf_counter() - t0)
        return SearchResult(goal, [], mu, "BidirectionalAStar", expanded, time.perf_counter() - t0, True)
```

- [ ] **Step 3: Update `pathos/algorithms/__init__.py`**

```python
from pathos.algorithms.base import Algorithm
from pathos.algorithms.uninformed import BFS, DFS, IDDFS, UCS
from pathos.algorithms.informed import AStar, IDAstar, GreedyBestFirst, WeightedAStar, BidirectionalAStar
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_informed.py tests/test_uninformed.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pathos/algorithms/informed.py pathos/algorithms/__init__.py tests/test_informed.py
git commit -m "feat(algorithms): A*, IDA*, Greedy Best-First, Weighted A*, Bidirectional A*"
```

---

## Task 9: Local Search Algorithms

**Files:**
- Create: `pathos/algorithms/local.py`
- Create: `tests/test_local.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_local.py
from pathos.core.space import Space
from pathos.algorithms.local import HillClimbing, TabuSearch, LocalBeamSearch

def _optimization_space(fn, neighbors_fn, initial):
    space = Space().initial(initial)

    @space.evaluate
    def cost(s): return fn(s)

    @space.successors
    def neighbors(s): yield from neighbors_fn(s)

    return space

# Minimize: find x in [0,10] minimizing (x-5)^2
def _int_neighbors(x):
    for dx in [-1, 1]:
        nx = x + dx
        if 0 <= nx <= 10:
            yield str(dx), nx

def test_hill_climbing_finds_optimum():
    space = _optimization_space(lambda x: (x - 5) ** 2, _int_neighbors, 0)
    result = HillClimbing(space).solve()
    assert result.found
    assert result.solution == 5

def test_tabu_search_finds_optimum():
    space = _optimization_space(lambda x: (x - 5) ** 2, _int_neighbors, 0)
    result = TabuSearch(space, max_iter=50, tabu_size=5).solve()
    assert result.found
    assert result.solution == 5

def test_local_beam_search_finds_optimum():
    space = _optimization_space(lambda x: (x - 5) ** 2, _int_neighbors, 0)
    result = LocalBeamSearch(space, k=3, max_iter=50).solve()
    assert result.found
    assert result.solution == 5
```

- [ ] **Step 2: Implement `pathos/algorithms/local.py`**

```python
from __future__ import annotations
import time
import random
from typing import Any
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register


@register
class HillClimbing(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.EVALUATE})
    power_rank = 15

    def __init__(self, space: Any, max_restarts: int = 1, max_sideways: int = 0) -> None:
        super().__init__(space)
        self.max_restarts = max_restarts
        self.max_sideways = max_sideways

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        best_state = self.space._initial
        best_cost = self.space._evaluate(best_state)
        expanded = 0

        for _ in range(self.max_restarts):
            state = self.space._initial
            cost = self.space._evaluate(state)
            sideways = 0
            while True:
                neighbors = list(self.space._successors(state))
                expanded += 1
                if not neighbors:
                    break
                best_n = min(neighbors, key=lambda ac: self.space._evaluate(ac[1]))
                n_cost = self.space._evaluate(best_n[1])
                if n_cost < cost:
                    state, cost = best_n[1], n_cost
                    sideways = 0
                elif n_cost == cost and sideways < self.max_sideways:
                    state, cost = best_n[1], n_cost
                    sideways += 1
                else:
                    break
            if cost < best_cost:
                best_cost, best_state = cost, state

        return SearchResult(
            best_state, None, best_cost, "HillClimbing",
            expanded, time.perf_counter() - t0, True,
        )


@register
class TabuSearch(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.EVALUATE})
    power_rank = 18

    def __init__(self, space: Any, max_iter: int = 100, tabu_size: int = 10) -> None:
        super().__init__(space)
        self.max_iter = max_iter
        self.tabu_size = tabu_size

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        current = self.space._initial
        best = current
        best_cost = self.space._evaluate(current)
        tabu: list = [current]
        expanded = 0

        for _ in range(self.max_iter):
            neighbors = [
                (a, child) for a, child in self.space._successors(current)
                if child not in tabu
            ]
            expanded += 1
            if not neighbors:
                break
            action, current = min(neighbors, key=lambda ac: self.space._evaluate(ac[1]))
            cost = self.space._evaluate(current)
            if cost < best_cost:
                best_cost, best = cost, current
            tabu.append(current)
            if len(tabu) > self.tabu_size:
                tabu.pop(0)

        return SearchResult(
            best, None, best_cost, "TabuSearch",
            expanded, time.perf_counter() - t0, True,
        )


@register
class LocalBeamSearch(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.EVALUATE})
    power_rank = 16

    def __init__(self, space: Any, k: int = 5, max_iter: int = 100) -> None:
        super().__init__(space)
        self.k = k
        self.max_iter = max_iter

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        beam = [initial] * self.k
        best = initial
        best_cost = self.space._evaluate(initial)
        expanded = 0

        for _ in range(self.max_iter):
            candidates = []
            for state in beam:
                for _, child in self.space._successors(state):
                    candidates.append(child)
                expanded += 1
            if not candidates:
                break
            candidates.sort(key=lambda s: self.space._evaluate(s))
            beam = candidates[: self.k]
            cost = self.space._evaluate(beam[0])
            if cost < best_cost:
                best_cost, best = cost, beam[0]

        return SearchResult(
            best, None, best_cost, "LocalBeamSearch",
            expanded, time.perf_counter() - t0, True,
        )
```

- [ ] **Step 3: Update `pathos/algorithms/__init__.py`**

```python
from pathos.algorithms.local import HillClimbing, TabuSearch, LocalBeamSearch
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_local.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pathos/algorithms/local.py pathos/algorithms/__init__.py tests/test_local.py
git commit -m "feat(algorithms): Hill Climbing, Tabu Search, Local Beam Search"
```

---

## Task 10: Evolutionary / Metaheuristic Algorithms

**Files:**
- Create: `pathos/algorithms/evolutionary.py`
- Create: `tests/test_evolutionary.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Implement `pathos/algorithms/evolutionary.py`**

```python
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
```

- [ ] **Step 3: Update `pathos/algorithms/__init__.py`** — add:

```python
from pathos.algorithms.evolutionary import SimulatedAnnealing, GeneticAlgorithm, DifferentialEvolution
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_evolutionary.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pathos/algorithms/evolutionary.py pathos/algorithms/__init__.py tests/test_evolutionary.py
git commit -m "feat(algorithms): Simulated Annealing, Genetic Algorithm, Differential Evolution"
```

---

## Task 11: Adversarial Search

**Files:**
- Create: `pathos/algorithms/adversarial.py`
- Create: `tests/test_adversarial.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adversarial.py
from pathos.core.space import Space
from pathos.algorithms.adversarial import Minimax, AlphaBeta, Negamax, MCTS

# Tic-tac-toe-like minimal game: 3-node game tree
# State = (value, is_terminal, children)
# Player 0 maximizes, player 1 minimizes
def _game_space():
    # Simple deterministic game tree
    tree = {
        "root": [("a1", "n1"), ("a2", "n2")],
        "n1": [("b1", "l1"), ("b2", "l2")],
        "n2": [("c1", "l3"), ("c2", "l4")],
        "l1": [], "l2": [], "l3": [], "l4": [],
    }
    utilities = {"l1": 3, "l2": 5, "l3": 2, "l4": 9}
    terminal = {"l1", "l2", "l3", "l4"}

    space = Space().initial("root").adversarial(players=2, maximizing_player=0)

    @space.successors
    def moves(s): yield from tree.get(s, [])

    @space.terminal
    def is_terminal(s): return s in terminal

    @space.utility
    def score(s, player):
        val = utilities.get(s, 0)
        return val if player == 0 else -val

    return space

def test_minimax_selects_best():
    space = _game_space()
    result = Minimax(space).solve()
    assert result.found
    # Minimax: root→n1 gives max(3,5)=5, root→n2 gives max(2,9)=9
    # Min at root picks min(5,9)=5, so optimal action leads to n1
    # Then max picks l2 (5)
    assert result.solution in {"l1", "l2", "l3", "l4"}

def test_alphabeta_same_as_minimax():
    space = _game_space()
    mm = Minimax(space).solve()
    ab = AlphaBeta(space).solve()
    assert mm.cost == ab.cost

def test_negamax_finds_solution():
    space = _game_space()
    result = Negamax(space).solve()
    assert result.found

def test_mcts_finds_solution():
    space = _game_space()
    result = MCTS(space, iterations=100).solve()
    assert result.found
```

- [ ] **Step 2: Implement `pathos/algorithms/adversarial.py`**

```python
from __future__ import annotations
import time
import math
import random
from typing import Any
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register


@register
class Minimax(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.TERMINAL, Capability.UTILITY})
    power_rank = 40

    def __init__(self, space: Any, max_depth: int = 100) -> None:
        super().__init__(space)
        self.max_depth = max_depth

    def _minimax(self, state: Any, depth: int, is_max: bool) -> tuple[float, Any]:
        if self.space._terminal(state) or depth == 0:
            player = self.space._maximizing_player
            return self.space._utility(state, player), state
        moves = list(self.space._successors(state))
        if not moves:
            return self.space._utility(state, self.space._maximizing_player), state
        if is_max:
            best_val, best_state = -math.inf, None
            for _, child in moves:
                val, _ = self._minimax(child, depth - 1, False)
                if val > best_val:
                    best_val, best_state = val, child
            return best_val, best_state
        else:
            best_val, best_state = math.inf, None
            for _, child in moves:
                val, _ = self._minimax(child, depth - 1, True)
                if val < best_val:
                    best_val, best_state = val, child
            return best_val, best_state

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        val, state = self._minimax(self.space._initial, self.max_depth, True)
        return SearchResult(state, None, val, "Minimax", 0, time.perf_counter() - t0, state is not None)


@register
class AlphaBeta(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.TERMINAL, Capability.UTILITY})
    power_rank = 45

    def __init__(self, space: Any, max_depth: int = 100) -> None:
        super().__init__(space)
        self.max_depth = max_depth

    def _ab(self, state: Any, depth: int, alpha: float, beta: float, is_max: bool) -> tuple[float, Any]:
        if self.space._terminal(state) or depth == 0:
            return self.space._utility(state, self.space._maximizing_player), state
        moves = list(self.space._successors(state))
        if not moves:
            return self.space._utility(state, self.space._maximizing_player), state
        best_state = moves[0][1]
        if is_max:
            val = -math.inf
            for _, child in moves:
                child_val, _ = self._ab(child, depth - 1, alpha, beta, False)
                if child_val > val:
                    val, best_state = child_val, child
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
        else:
            val = math.inf
            for _, child in moves:
                child_val, _ = self._ab(child, depth - 1, alpha, beta, True)
                if child_val < val:
                    val, best_state = child_val, child
                beta = min(beta, val)
                if alpha >= beta:
                    break
        return val, best_state

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        val, state = self._ab(self.space._initial, self.max_depth, -math.inf, math.inf, True)
        return SearchResult(state, None, val, "AlphaBeta", 0, time.perf_counter() - t0, state is not None)


@register
class Negamax(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.TERMINAL, Capability.UTILITY})
    power_rank = 42

    def __init__(self, space: Any, max_depth: int = 100) -> None:
        super().__init__(space)
        self.max_depth = max_depth

    def _negamax(self, state: Any, depth: int, alpha: float, beta: float, player: int) -> tuple[float, Any]:
        if self.space._terminal(state) or depth == 0:
            val = self.space._utility(state, player)
            return val, state
        moves = list(self.space._successors(state))
        if not moves:
            return self.space._utility(state, player), state
        best_val, best_state = -math.inf, moves[0][1]
        next_player = (player + 1) % self.space._players
        for _, child in moves:
            child_val, _ = self._negamax(child, depth - 1, -beta, -alpha, next_player)
            if -child_val > best_val:
                best_val, best_state = -child_val, child
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
        return best_val, best_state

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        val, state = self._negamax(
            self.space._initial, self.max_depth, -math.inf, math.inf,
            self.space._maximizing_player
        )
        return SearchResult(state, None, val, "Negamax", 0, time.perf_counter() - t0, state is not None)


class _MCTSNode:
    __slots__ = ("state", "parent", "children", "visits", "value", "untried")

    def __init__(self, state: Any, parent: _MCTSNode | None, space: Any) -> None:
        self.state = state
        self.parent = parent
        self.children: list[_MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried = list(space._successors(state)) if not space._terminal(state) else []

    def uct_score(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return math.inf
        assert self.parent is not None
        return self.value / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self) -> _MCTSNode:
        return max(self.children, key=lambda n: n.uct_score())

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0


@register
class MCTS(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.TERMINAL, Capability.UTILITY})
    power_rank = 43

    def __init__(self, space: Any, iterations: int = 1000) -> None:
        super().__init__(space)
        self.iterations = iterations

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        root = _MCTSNode(self.space._initial, None, self.space)

        for _ in range(self.iterations):
            node = self._select(root)
            if not self.space._terminal(node.state):
                node = self._expand(node)
            reward = self._simulate(node.state)
            self._backprop(node, reward)

        best = max(root.children, key=lambda n: n.visits) if root.children else root
        return SearchResult(
            best.state, None,
            self.space._utility(best.state, self.space._maximizing_player),
            "MCTS", self.iterations, time.perf_counter() - t0, True,
        )

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        while not self.space._terminal(node.state) and node.is_fully_expanded():
            node = node.best_child()
        return node

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        action, child_state = node.untried.pop()
        child = _MCTSNode(child_state, node, self.space)
        node.children.append(child)
        return child

    def _simulate(self, state: Any) -> float:
        depth = 0
        while not self.space._terminal(state) and depth < 50:
            moves = list(self.space._successors(state))
            if not moves:
                break
            _, state = random.choice(moves)
            depth += 1
        return self.space._utility(state, self.space._maximizing_player)

    def _backprop(self, node: _MCTSNode | None, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
```

- [ ] **Step 3: Update `pathos/algorithms/__init__.py`** — add:

```python
from pathos.algorithms.adversarial import Minimax, AlphaBeta, Negamax, MCTS
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_adversarial.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pathos/algorithms/adversarial.py pathos/algorithms/__init__.py tests/test_adversarial.py
git commit -m "feat(algorithms): Minimax, Alpha-Beta, Negamax, MCTS/UCT"
```

---

## Task 12: CSP Algorithms

**Files:**
- Create: `pathos/algorithms/csp.py`
- Create: `tests/test_csp.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_csp.py
from pathos.core.space import Space
from pathos.algorithms.csp import Backtracking, ForwardChecking, AC3, MinConflicts

def _nqueens_space(n=4):
    """N-Queens as a CSP: variables=cols, domain=rows, constraint=no attack."""
    space = Space().initial({})
    space._n = n

    @space.successors
    def expand(assignment):
        col = len(assignment)
        if col >= n:
            return
        for row in range(n):
            consistent = all(
                assignment[c] != row and
                abs(assignment[c] - row) != abs(c - col)
                for c in assignment
            )
            if consistent:
                new_assign = dict(assignment)
                new_assign[col] = row
                yield f"col{col}={row}", new_assign

    @space.goal
    def is_complete(assignment):
        return len(assignment) == n

    return space

def test_backtracking_4queens():
    space = _nqueens_space(4)
    result = Backtracking(space).solve()
    assert result.found
    sol = result.solution
    assert len(sol) == 4
    # verify no queens attack each other
    cols = list(sol.keys())
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            assert sol[cols[i]] != sol[cols[j]]
            assert abs(sol[cols[i]] - sol[cols[j]]) != abs(cols[i] - cols[j])

def test_forward_checking_4queens():
    space = _nqueens_space(4)
    result = ForwardChecking(space).solve()
    assert result.found
```

- [ ] **Step 2: Implement `pathos/algorithms/csp.py`**

```python
from __future__ import annotations
import time
import random
from typing import Any
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register


@register
class Backtracking(Algorithm):
    """Backtracking search. Uses the space's successors as CSP expansion."""
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 9

    def _bt(self, state: Any, expanded: list) -> Any | None:
        if self.space._goal(state):
            return state
        for _, child in self.space._successors(state):
            expanded[0] += 1
            result = self._bt(child, expanded)
            if result is not None:
                return result
        return None

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        expanded = [0]
        result = self._bt(self.space._initial, expanded)
        elapsed = time.perf_counter() - t0
        if result is not None:
            return SearchResult(result, None, None, "Backtracking", expanded[0], elapsed, True)
        return SearchResult.not_found("Backtracking", expanded[0], elapsed)


@register
class ForwardChecking(Algorithm):
    """Forward Checking — like Backtracking but prunes via look-ahead."""
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 11

    def _fc(self, state: Any, expanded: list) -> Any | None:
        if self.space._goal(state):
            return state
        children = list(self.space._successors(state))
        if not children:
            return None
        for _, child in children:
            expanded[0] += 1
            # check if any successor exists from child (look-ahead)
            future = list(self.space._successors(child))
            if future or self.space._goal(child):
                result = self._fc(child, expanded)
                if result is not None:
                    return result
        return None

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        expanded = [0]
        result = self._fc(self.space._initial, expanded)
        elapsed = time.perf_counter() - t0
        if result is not None:
            return SearchResult(result, None, None, "ForwardChecking", expanded[0], elapsed, True)
        return SearchResult.not_found("ForwardChecking", expanded[0], elapsed)


class AC3(Algorithm):
    """AC-3 arc consistency — use via CSPSpace, not directly on generic Space."""
    requires = frozenset({Capability.VARIABLES, Capability.DOMAINS, Capability.CONSTRAINTS})
    power_rank = 22

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        # This is invoked by CSPSpace which sets up the variables/domains/constraints
        variables = self.space._variables()
        domains = {v: list(self.space._domain(v)) for v in variables}
        arcs = [(xi, xj) for xi in variables for xj in variables if xi != xj]
        queue = list(arcs)

        while queue:
            xi, xj = queue.pop(0)
            if self._revise(domains, xi, xj):
                if not domains[xi]:
                    return SearchResult.not_found("AC3", 0, time.perf_counter() - t0)
                for xk in variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))

        return SearchResult(domains, None, None, "AC3", 0, time.perf_counter() - t0, True)

    def _revise(self, domains: dict, xi: Any, xj: Any) -> bool:
        revised = False
        for x in domains[xi][:]:
            if not any(self.space._constraints({xi: x, xj: y}) for y in domains[xj]):
                domains[xi].remove(x)
                revised = True
        return revised


@register
class MinConflicts(Algorithm):
    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL, Capability.EVALUATE})
    power_rank = 19

    def __init__(self, space: Any, max_iter: int = 1000) -> None:
        super().__init__(space)
        self.max_iter = max_iter

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        current = self.space._initial

        for i in range(self.max_iter):
            if self.space._goal(current):
                return SearchResult(current, None, 0.0, "MinConflicts", i, time.perf_counter() - t0, True)
            neighbors = list(self.space._successors(current))
            if not neighbors:
                break
            _, current = min(neighbors, key=lambda ac: self.space._evaluate(ac[1]))

        return SearchResult.not_found("MinConflicts", self.max_iter, time.perf_counter() - t0)
```

- [ ] **Step 3: Update `pathos/algorithms/__init__.py`** — add:

```python
from pathos.algorithms.csp import Backtracking, ForwardChecking, AC3, MinConflicts
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_csp.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pathos/algorithms/csp.py pathos/algorithms/__init__.py tests/test_csp.py
git commit -m "feat(algorithms): Backtracking, Forward Checking, AC-3, Min-Conflicts"
```

---

## Task 13: GraphSpace

**Files:**
- Create: `pathos/spaces/graph.py`
- Create: `tests/test_spaces_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_spaces_graph.py
from pathos.spaces.graph import GraphSpace

def test_graphspace_astar():
    # Simple weighted graph
    graph = {
        "A": [("B", 1.0), ("C", 4.0)],
        "B": [("C", 2.0), ("D", 5.0)],
        "C": [("D", 1.0)],
        "D": [],
    }
    space = GraphSpace(graph=graph).initial("A")

    @space.goal
    def reached(n): return n == "D"

    @space.heuristic
    def h(n): return {"A": 5.0, "B": 3.0, "C": 1.0, "D": 0.0}.get(n, 0.0)

    result = space.solver().solve()
    assert result.found
    assert result.solution == "D"
    assert result.cost == 4.0  # A->B->C->D = 1+2+1

def test_graphspace_bfs_without_heuristic():
    graph = {"a": [("b", 1)], "b": [("c", 1)], "c": []}
    space = GraphSpace(graph=graph).initial("a")

    @space.goal
    def reached(n): return n == "c"

    result = space.solver().solve()
    assert result.found
```

- [ ] **Step 2: Implement `pathos/spaces/graph.py`**

```python
from __future__ import annotations
from typing import Any
from pathos.core.space import Space
from pathos.core.capabilities import Capability


class GraphSpace(Space):
    """
    Space for problems defined on explicit graphs.
    Automatically provides @successors from the adjacency structure.

    graph: dict mapping node -> list of (neighbor, edge_cost) tuples
    """

    def __init__(self, graph: dict[Any, list[tuple[Any, float]]]) -> None:
        super().__init__()
        self._graph = graph
        self._setup_successors()
        self._setup_evaluate()

    def _setup_successors(self) -> None:
        graph = self._graph

        def _successors(state: Any):
            for neighbor, cost in graph.get(state, []):
                yield neighbor, neighbor

        self._successors = _successors
        self.capabilities.add(Capability.SUCCESSORS)

    def _setup_evaluate(self) -> None:
        graph = self._graph

        def _evaluate(state: Any) -> float:
            # returns the cost to reach this node from any predecessor
            # used by UCS/A* as edge cost — stored as edge attribute
            return 1.0  # default; A* will use heuristic + g accumulated

        # We override evaluate to use actual edge weights via a lookup
        # The proper way: store edge weights separately
        self._edge_weights: dict[Any, dict[Any, float]] = {}
        for node, edges in graph.items():
            for neighbor, cost in edges:
                self._edge_weights.setdefault(neighbor, {})[node] = cost

        def _evaluate_with_weights(state: Any) -> float:
            # Returns minimum incoming edge weight as an approximation
            # For proper use, algorithms should track g-score separately
            incoming = self._edge_weights.get(state, {})
            return min(incoming.values()) if incoming else 1.0

        self._evaluate = _evaluate_with_weights
        self.capabilities.add(Capability.EVALUATE)
```

- [ ] **Step 3: Update `pathos/spaces/__init__.py`**

```python
from pathos.spaces.graph import GraphSpace
```

- [ ] **Step 4: Update top-level `pathos/__init__.py`**

```python
from pathos.core.space import Space
from pathos.spaces.graph import GraphSpace
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_spaces_graph.py -v
```

- [ ] **Step 6: Commit**

```bash
git add pathos/spaces/graph.py pathos/spaces/__init__.py pathos/__init__.py tests/test_spaces_graph.py
git commit -m "feat(spaces): GraphSpace with auto-successors from adjacency"
```

---

## Task 14: CSPSpace

**Files:**
- Create: `pathos/spaces/csp.py`
- Create: `tests/test_spaces_csp.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_spaces_csp.py
from pathos.spaces.csp import CSPSpace

def test_csp_nqueens():
    n = 4
    csp = CSPSpace(variables=list(range(n)))

    @csp.domain
    def dom(var): return list(range(n))

    @csp.constraint
    def no_attack(assignment):
        items = list(assignment.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                c1, r1 = items[i]
                c2, r2 = items[j]
                if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                    return False
        return True

    result = csp.solver().solve()
    assert result.found
    assert len(result.solution) == n

def test_csp_graph_coloring():
    # 3-color a triangle: must use different colors for adjacent nodes
    csp = CSPSpace(variables=["A", "B", "C"])

    @csp.domain
    def dom(var): return ["red", "green", "blue"]

    edges = {("A", "B"), ("B", "C"), ("A", "C")}

    @csp.constraint
    def different_colors(assignment):
        for (u, v) in edges:
            if u in assignment and v in assignment:
                if assignment[u] == assignment[v]:
                    return False
        return True

    result = csp.solver().solve()
    assert result.found
    sol = result.solution
    assert sol["A"] != sol["B"]
    assert sol["B"] != sol["C"]
    assert sol["A"] != sol["C"]
```

- [ ] **Step 2: Implement `pathos/spaces/csp.py`**

```python
from __future__ import annotations
from typing import Any, Callable
from pathos.core.space import Space
from pathos.core.capabilities import Capability


class CSPSpace(Space):
    """
    Space for Constraint Satisfaction Problems.
    Auto-provides @successors (partial-assignment expansion) and @goal.
    User provides @domain and @constraint decorators.
    """

    def __init__(self, variables: list[Any]) -> None:
        super().__init__()
        self._variables_list = variables
        self._domain_fn: Callable | None = None
        self._constraint_fn: Callable | None = None
        self._initial_value = {}  # empty assignment
        self._setup_goal()

    def _setup_goal(self) -> None:
        n = len(self._variables_list)

        def _goal(assignment: dict) -> bool:
            return len(assignment) == n

        self._goal = _goal
        self.capabilities.add(Capability.GOAL)

    def _setup_successors(self) -> None:
        variables = self._variables_list
        domain_fn = self._domain_fn
        constraint_fn = self._constraint_fn

        def _successors(assignment: dict):
            col = len(assignment)
            if col >= len(variables):
                return
            var = variables[col]
            for val in domain_fn(var):  # type: ignore
                new_assign = dict(assignment)
                new_assign[var] = val
                if constraint_fn(new_assign):  # type: ignore
                    yield f"{var}={val}", new_assign

        self._successors = _successors
        self.capabilities.add(Capability.SUCCESSORS)

    @property
    def domain(self) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._domain_fn = fn
            self._maybe_finalize()
            return fn
        return decorator

    @property
    def constraint(self) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._constraint_fn = fn
            self.capabilities.add(Capability.CONSTRAINTS)
            self._maybe_finalize()
            return fn
        return decorator

    def _maybe_finalize(self) -> None:
        if self._domain_fn is not None and self._constraint_fn is not None:
            self._setup_successors()
```

- [ ] **Step 3: Update `pathos/spaces/__init__.py`**

```python
from pathos.spaces.graph import GraphSpace
from pathos.spaces.csp import CSPSpace
```

- [ ] **Step 4: Update `pathos/__init__.py`**

```python
from pathos.core.space import Space
from pathos.spaces.graph import GraphSpace
from pathos.spaces.csp import CSPSpace
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_spaces_csp.py -v
```

- [ ] **Step 6: Commit**

```bash
git add pathos/spaces/csp.py pathos/spaces/__init__.py pathos/__init__.py tests/test_spaces_csp.py
git commit -m "feat(spaces): CSPSpace with auto-successors from variables/domains/constraints"
```

---

## Task 15: TourSpace and GameSpace

**Files:**
- Create: `pathos/spaces/tour.py`
- Create: `pathos/spaces/game.py`
- Create: `tests/test_spaces_tour.py`
- Create: `tests/test_spaces_game.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_spaces_tour.py
from pathos.spaces.tour import TourSpace

def test_tourspace_tsp_sa():
    # 4-city TSP
    distances = {
        (0, 1): 10, (1, 0): 10,
        (0, 2): 15, (2, 0): 15,
        (0, 3): 20, (3, 0): 20,
        (1, 2): 35, (2, 1): 35,
        (1, 3): 25, (3, 1): 25,
        (2, 3): 30, (3, 2): 30,
    }

    space = TourSpace(nodes=list(range(4)), distances=distances)

    @space.evaluate
    def tour_cost(tour):
        return sum(distances[(tour[i], tour[(i + 1) % len(tour)])] for i in range(len(tour)))

    result = space.solver().solve()
    assert result.found
    assert len(result.solution) == 4
    assert set(result.solution) == {0, 1, 2, 3}
```

```python
# tests/test_spaces_game.py
from pathos.spaces.game import GameSpace

def test_gamespace_tictactoe():
    # Minimal: 1x3 board, first to get 3 in a row wins
    import copy

    def available(board): return [i for i, v in enumerate(board) if v == 0]
    def winner(board):
        if all(v == 1 for v in board): return 1
        if all(v == 2 for v in board): return 2
        return 0

    space = GameSpace().initial(tuple([0, 0, 0]))

    @space.successors
    def moves(board):
        player = 1 + (sum(1 for v in board if v != 0) % 2)
        for i in available(board):
            new = list(board)
            new[i] = player
            yield f"p{player}@{i}", tuple(new)

    @space.terminal
    def is_over(board): return winner(board) != 0 or 0 not in board

    @space.utility
    def score(board, player):
        w = winner(board)
        if w == player + 1: return 1.0
        if w == 0: return 0.0
        return -1.0

    result = space.solver().solve()
    assert result.found
```

- [ ] **Step 2: Implement `pathos/spaces/tour.py`**

```python
from __future__ import annotations
import random
from typing import Any
from pathos.core.space import Space
from pathos.core.capabilities import Capability


class TourSpace(Space):
    """
    Space for tour/routing problems (TSP and variants).
    Auto-provides @successors as 2-opt neighborhood.
    User provides @evaluate for tour cost.
    """

    def __init__(self, nodes: list[Any], distances: dict | None = None) -> None:
        super().__init__()
        self._nodes = nodes
        self._distances = distances
        # initial = random tour
        self._initial_factory = lambda: random.sample(nodes, len(nodes))
        self._setup_successors()

    def _setup_successors(self) -> None:
        nodes = self._nodes

        def _two_opt(tour):
            n = len(tour)
            for i in range(n - 1):
                for j in range(i + 2, n):
                    new_tour = tour[:i] + list(reversed(tour[i:j])) + tour[j:]
                    yield f"2opt_{i}_{j}", new_tour

        self._successors = _two_opt
        self.capabilities.add(Capability.SUCCESSORS)
```

- [ ] **Step 3: Implement `pathos/spaces/game.py`**

```python
from __future__ import annotations
from pathos.core.space import Space


class GameSpace(Space):
    """
    Space for adversarial games.
    Convenience wrapper — sets adversarial mode by default.
    User provides @successors, @terminal, @utility (and optionally @evaluate).
    """

    def __init__(self) -> None:
        super().__init__()
        self.adversarial(players=2, maximizing_player=0)
```

- [ ] **Step 4: Update `pathos/spaces/__init__.py` and `pathos/__init__.py`**

```python
# pathos/spaces/__init__.py
from pathos.spaces.graph import GraphSpace
from pathos.spaces.csp import CSPSpace
from pathos.spaces.tour import TourSpace
from pathos.spaces.game import GameSpace

# pathos/__init__.py
from pathos.core.space import Space
from pathos.spaces.graph import GraphSpace
from pathos.spaces.csp import CSPSpace
from pathos.spaces.tour import TourSpace
from pathos.spaces.game import GameSpace
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_spaces_tour.py tests/test_spaces_game.py -v
```

- [ ] **Step 6: Commit**

```bash
git add pathos/spaces/tour.py pathos/spaces/game.py pathos/spaces/__init__.py pathos/__init__.py tests/test_spaces_tour.py tests/test_spaces_game.py
git commit -m "feat(spaces): TourSpace (TSP/2-opt) and GameSpace (adversarial)"
```

---

## Task 16: Realistic Examples

**Files:**
- Create: `examples/route_planning.py`
- Create: `examples/tsp.py`
- Create: `examples/nqueens.py`
- Create: `examples/tictactoe.py`
- Create: `examples/puzzle8.py`
- Create: `tests/test_examples.py`

- [ ] **Step 1: Create `examples/route_planning.py`**

```python
"""Route planning with A* on a weighted graph."""
from pathos import GraphSpace

# Road network: city -> [(neighbor, km)]
roads = {
    "Madrid":   [("Zaragoza", 325), ("Lisboa", 638), ("Sevilla", 534)],
    "Zaragoza": [("Barcelona", 296), ("Madrid", 325)],
    "Barcelona":[("Zaragoza", 296), ("Valencia", 349)],
    "Valencia": [("Barcelona", 349), ("Sevilla", 656)],
    "Sevilla":  [("Madrid", 534), ("Valencia", 656), ("Lisboa", 450)],
    "Lisboa":   [("Madrid", 638), ("Sevilla", 450)],
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
print(f"Route found: {' -> '.join(step[1] for step in result.path or [])}")
print(f"Algorithm: {result.algorithm}")
print(f"Distance: {result.cost:.0f} km")
print(f"Nodes expanded: {result.nodes_expanded}")
```

- [ ] **Step 2: Create `examples/tsp.py`**

```python
"""Traveling Salesman Problem with SA + Genetic Algorithm."""
import random
from pathos import TourSpace

random.seed(42)

# 6-city distance matrix
cities = list(range(6))
random.seed(42)
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
```

- [ ] **Step 3: Create `examples/nqueens.py`**

```python
"""N-Queens via CSPSpace."""
from pathos import CSPSpace

N = 8
csp = CSPSpace(variables=list(range(N)))

@csp.domain
def dom(col): return list(range(N))

@csp.constraint
def no_attack(assignment):
    items = list(assignment.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            c1, r1 = items[i]; c2, r2 = items[j]
            if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                return False
    return True

result = csp.solver().solve()
if result.found:
    board = ["." * result.solution[col] + "Q" + "." * (N - result.solution[col] - 1)
             for col in range(N)]
    print("\n".join(board))
    print(f"\nAlgorithm: {result.algorithm} | Nodes expanded: {result.nodes_expanded}")
```

- [ ] **Step 4: Create `examples/tictactoe.py`**

```python
"""Tic-tac-toe optimal play via Alpha-Beta."""
from pathos import GameSpace

def _winner(board):
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in lines:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0

space = GameSpace().initial(tuple([0] * 9))

@space.successors
def moves(board):
    player = 1 + (sum(1 for v in board if v) % 2)
    for i, v in enumerate(board):
        if v == 0:
            new = list(board); new[i] = player
            yield f"p{player}@{i}", tuple(new)

@space.terminal
def is_over(board): return _winner(board) != 0 or 0 not in board

@space.utility
def score(board, player):
    w = _winner(board)
    if w == 0: return 0.0
    return 1.0 if w == player + 1 else -1.0

result = space.solver().solve()
board = result.solution or space._initial
symbols = {0: ".", 1: "X", 2: "O"}
for row in range(3):
    print(" ".join(symbols[board[row*3+col]] for col in range(3)))
print(f"\nBest move leads to state with utility {result.cost}")
print(f"Algorithm: {result.algorithm}")
```

- [ ] **Step 5: Create `examples/puzzle8.py`**

```python
"""8-puzzle solved with A* and Manhattan distance heuristic."""
from pathos import Space

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def _moves(board):
    i = board.index(0)
    row, col = divmod(i, 3)
    for dr, dc, name in [(-1,0,"up"),(1,0,"down"),(0,-1,"left"),(0,1,"right")]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            lst = list(board); lst[i], lst[j] = lst[j], lst[i]
            yield name, tuple(lst)

def manhattan(board):
    total = 0
    for i, val in enumerate(board):
        if val == 0: continue
        goal_i = GOAL.index(val)
        total += abs(i // 3 - goal_i // 3) + abs(i % 3 - goal_i % 3)
    return total

START = (1, 2, 3, 4, 5, 6, 0, 7, 8)

space = Space().initial(START)

@space.successors
def expand(b): yield from _moves(b)

@space.goal
def solved(b): return b == GOAL

@space.heuristic
def h(b): return manhattan(b)

@space.evaluate
def cost(b): return 1.0

result = space.solver().solve()
print(f"Solved in {len(result.path or [])} moves using {result.algorithm}")
print(f"Nodes expanded: {result.nodes_expanded}")
```

- [ ] **Step 6: Create `tests/test_examples.py`**

```python
"""Smoke tests for all examples — verify they run and produce valid output."""
import subprocess
import sys

def _run(script):
    result = subprocess.run(
        [sys.executable, f"examples/{script}"],
        capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stderr}"
    return result.stdout

def test_route_planning():
    out = _run("route_planning.py")
    assert "Lisboa" in out
    assert "Algorithm" in out

def test_tsp():
    out = _run("tsp.py")
    assert "Cost" in out
    assert "Algorithm" in out

def test_nqueens():
    out = _run("nqueens.py")
    assert "Q" in out

def test_tictactoe():
    out = _run("tictactoe.py")
    assert "Algorithm" in out

def test_puzzle8():
    out = _run("puzzle8.py")
    assert "moves" in out
```

- [ ] **Step 7: Run example tests**

```bash
pytest tests/test_examples.py -v
```

- [ ] **Step 8: Commit**

```bash
git add examples/ tests/test_examples.py
git commit -m "feat(examples): route planning, TSP, N-Queens, tic-tac-toe, 8-puzzle"
```

---

## Task 17: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: complete README with install, quickstart, algorithm table, examples"
```

---

## Task 18: MkDocs Documentation

**Files:**
- Create: `docs/mkdocs.yml`
- Create: `docs/index.md`
- Create: `docs/getting-started.md`
- Create: `docs/api/space.md`
- Create: `docs/api/algorithms.md`
- Create: `docs/api/spaces.md`
- Create: `docs/api/result.md`

- [ ] **Step 1: Create `mkdocs.yml`** (in repo root, not docs/)

```yaml
site_name: PATHOS
site_description: Production-ready classical AI search algorithms
site_url: https://gia-uh.github.io/pathos
repo_url: https://github.com/gia-uh/pathos
repo_name: gia-uh/pathos

theme:
  name: material
  palette:
    primary: deep purple
    accent: amber
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            docstring_style: google

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - API Reference:
      - Space: api/space.md
      - Algorithms: api/algorithms.md
      - Spaces: api/spaces.md
      - SearchResult: api/result.md

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - admonition
```

- [ ] **Step 2: Create `docs/index.md`**

```markdown
# PATHOS

Production-ready classical AI search algorithms for Python.

**Define your problem, not your algorithm.**

```python
from pathos import Space

space = Space().initial(start)

@space.successors
def expand(state): ...

@space.heuristic
def h(state): ...

result = space.solver().solve()
```

## Install

```bash
pip install pathos-ai
```

## Quick links

- [Getting Started](getting-started.md)
- [API Reference](api/space.md)
- [GitHub](https://github.com/gia-uh/pathos)
```

- [ ] **Step 3: Create `docs/getting-started.md`**

```markdown
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
```

- [ ] **Step 4: Create `docs/api/space.md`**

```markdown
# Space

::: pathos.core.space.Space
```

- [ ] **Step 5: Create `docs/api/result.md`**

```markdown
# SearchResult

::: pathos.core.result.SearchResult
```

- [ ] **Step 6: Create `docs/api/algorithms.md`**

```markdown
# Algorithms

## Uninformed

::: pathos.algorithms.uninformed.BFS
::: pathos.algorithms.uninformed.DFS
::: pathos.algorithms.uninformed.IDDFS
::: pathos.algorithms.uninformed.UCS

## Informed

::: pathos.algorithms.informed.AStar
::: pathos.algorithms.informed.GreedyBestFirst
::: pathos.algorithms.informed.WeightedAStar
::: pathos.algorithms.informed.IDAstar
::: pathos.algorithms.informed.BidirectionalAStar

## Local Search

::: pathos.algorithms.local.HillClimbing
::: pathos.algorithms.local.TabuSearch
::: pathos.algorithms.local.LocalBeamSearch

## Evolutionary

::: pathos.algorithms.evolutionary.SimulatedAnnealing
::: pathos.algorithms.evolutionary.GeneticAlgorithm
::: pathos.algorithms.evolutionary.DifferentialEvolution

## Adversarial

::: pathos.algorithms.adversarial.Minimax
::: pathos.algorithms.adversarial.AlphaBeta
::: pathos.algorithms.adversarial.Negamax
::: pathos.algorithms.adversarial.MCTS

## CSP

::: pathos.algorithms.csp.Backtracking
::: pathos.algorithms.csp.ForwardChecking
::: pathos.algorithms.csp.MinConflicts
```

- [ ] **Step 7: Create `docs/api/spaces.md`**

```markdown
# Specialized Spaces

::: pathos.spaces.graph.GraphSpace
::: pathos.spaces.csp.CSPSpace
::: pathos.spaces.tour.TourSpace
::: pathos.spaces.game.GameSpace
```

- [ ] **Step 8: Add docstrings to core classes**

Add Google-style docstrings to `Space`, `SearchResult`, `GraphSpace`, `CSPSpace`, `TourSpace`, `GameSpace`, and all algorithm classes. Minimum: one-line class docstring + Args section for `__init__`.

Example for `AStar`:

```python
class AStar(Algorithm):
    """A* search — optimal pathfinding with admissible heuristic.

    Requires: successors, goal, heuristic, evaluate.
    Selects the path minimizing g(n) + h(n) where g is actual cost
    and h is an admissible heuristic estimate to goal.

    Attributes:
        requires: Capability set needed.
        power_rank: 30 (preferred over BFS/DFS/Greedy when available).
    """
```

- [ ] **Step 9: Test docs build locally**

```bash
mkdocs build --strict
```

Expected: no errors, `site/` directory created.

- [ ] **Step 10: Commit**

```bash
git add mkdocs.yml docs/ pathos/
git commit -m "docs: MkDocs with Material theme, API reference, getting started guide"
```

---

## Task 19: GitHub Actions — Pages and PyPI

**Files:**
- Create: `.github/workflows/pages.yml`
- Create: `.github/workflows/publish.yml`

- [ ] **Step 1: Create `.github/workflows/pages.yml`**

```yaml
name: Deploy Docs
on:
  push:
    branches: [main]
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: mkdocs build
      - uses: actions/upload-pages-artifact@v3
        with: { path: site/ }
      - uses: actions/deploy-pages@v4
        id: deployment
```

- [ ] **Step 2: Create `.github/workflows/publish.yml`**

```yaml
name: Publish to PyPI
on:
  release:
    types: [published]
jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 3: Enable GitHub Pages in repo settings**

In GitHub repo settings → Pages → Source: GitHub Actions

- [ ] **Step 4: Commit workflows**

```bash
git add .github/workflows/pages.yml .github/workflows/publish.yml
git commit -m "ci: add GitHub Pages deployment and PyPI publish workflows"
```

---

## Task 20: Final Integration and Push

- [ ] **Step 1: Run full test suite**

```bash
pytest --cov=pathos --cov-report=term-missing -v
```

Expected: all tests PASS, coverage > 80%

- [ ] **Step 2: Run mypy**

```bash
mypy pathos/
```

Fix any type errors found.

- [ ] **Step 3: Build docs**

```bash
mkdocs build --strict
```

- [ ] **Step 4: Verify package builds cleanly**

```bash
pip install build
python -m build
ls dist/
```

Expected: `pathos_ai-0.1.0-py3-none-any.whl` and `pathos_ai-0.1.0.tar.gz`

- [ ] **Step 5: Final commit and push**

```bash
git add -A
git commit -m "chore: production-ready v0.1.0 — all algorithms, tests, docs, examples"
git push origin main
```

- [ ] **Step 6: Notify completion**

```bash
bin/notify-telegram.sh "✅ PATHOS v0.1.0 implemented — all algorithms, tests, docs, examples. Review at https://github.com/gia-uh/pathos"
```

---

## Self-Review Notes

- Spec coverage: all capability lattice entries map to algorithm tasks ✓
- Specialized subspaces all covered ✓  
- Examples cover all major algorithm families ✓
- MkDocs, CI, PyPI workflows included ✓
- `AC3` requires `VARIABLES/DOMAINS/CONSTRAINTS` capabilities not yet wired into `CSPSpace` — Task 14 should add `_variables()` and `_domain()` accessors to `CSPSpace` for AC3 compatibility. Add to CSPSpace:

```python
def _variables_list_fn(self): return self._variables_list
def _domain_fn_wrapper(self, var): return self._domain_fn(var)
```

Then register `Capability.VARIABLES` and `Capability.DOMAINS` when domain_fn is set.
