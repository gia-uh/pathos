# Multiprocessing Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `concurrent.futures.ProcessPoolExecutor` into GA, DE, and LocalBeamSearch so calling `space.parallel(n)` actually evaluates candidates in parallel.

**Architecture:** Add a `batch_map(fn, items, n_workers)` utility in `pathos/core/parallel.py` that falls back to serial when `n_workers=1`. Store `_n_workers` in `Algorithm.__init__` from `space._n_workers`. Replace serial evaluation loops in the three target algorithms with `batch_map` calls. Existing tests are unaffected (they don't call `.parallel()`, so `n_workers` stays 1). New parallel-specific tests use module-level evaluate functions (required for pickle).

**Tech Stack:** `concurrent.futures` (stdlib only, no new deps), pytest

---

## File Structure

- **Create:** `pathos/core/parallel.py` — `batch_map` utility wrapping ProcessPoolExecutor
- **Modify:** `pathos/algorithms/base.py:15-22` — store `self._n_workers` in `__init__`
- **Modify:** `pathos/algorithms/evolutionary.py` — parallel eval in GA (lines 84-108) and DE (lines 133-160)
- **Modify:** `pathos/algorithms/local.py:132-157` — parallel eval in LocalBeamSearch
- **Create:** `tests/test_parallel.py` — parallel correctness tests with picklable evaluate functions

---

### Task 1: batch_map utility

**Files:**
- Create: `pathos/core/parallel.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_parallel.py`:

```python
from pathos.core.parallel import batch_map


def _double(x):
    return x * 2


def test_batch_map_serial_matches_map():
    items = [1, 2, 3, 4, 5]
    assert batch_map(_double, items, n_workers=1) == [2, 4, 6, 8, 10]


def test_batch_map_parallel_matches_serial():
    items = list(range(10))
    serial = batch_map(_double, items, n_workers=1)
    parallel = batch_map(_double, items, n_workers=2)
    assert parallel == serial


def test_batch_map_empty():
    assert batch_map(_double, [], n_workers=2) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/apiad/Workspace/repos/pathos
pytest tests/test_parallel.py -v
```

Expected: `ImportError` — `batch_map` does not exist yet.

- [ ] **Step 3: Create `pathos/core/parallel.py`**

```python
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable


def batch_map(fn: Callable[[Any], Any], items: list[Any], n_workers: int) -> list[Any]:
    if n_workers <= 1 or len(items) <= 1:
        return list(map(fn, items))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(fn, items))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parallel.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/apiad/Workspace/repos/pathos
git add pathos/core/parallel.py tests/test_parallel.py
git commit -m "feat(core): add batch_map utility for parallel evaluation"
```

---

### Task 2: Wire _n_workers into Algorithm base

**Files:**
- Modify: `pathos/algorithms/base.py:15-22`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_parallel.py`:

```python
from pathos.core.space import Space
from pathos.algorithms.evolutionary import GeneticAlgorithm


def test_algorithm_stores_n_workers():
    space = Space().initial(0).parallel(4)

    @space.evaluate
    def cost(x): return x

    ga = GeneticAlgorithm(space)
    assert ga._n_workers == 4


def test_algorithm_defaults_to_serial():
    space = Space().initial(0)

    @space.evaluate
    def cost(x): return x

    ga = GeneticAlgorithm(space)
    assert ga._n_workers == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_parallel.py::test_algorithm_stores_n_workers tests/test_parallel.py::test_algorithm_defaults_to_serial -v
```

Expected: `AttributeError: 'GeneticAlgorithm' object has no attribute '_n_workers'`

- [ ] **Step 3: Update `pathos/algorithms/base.py`**

Replace lines 15-22:

```python
    def __init__(self, space: Space) -> None:
        missing = self.requires - space.capabilities
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires capabilities: "
                f"{', '.join(c.name for c in missing)}"
            )
        self.space = space
        self._n_workers: int = space._n_workers
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parallel.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full suite to catch regressions**

```bash
pytest tests/ -v
```

Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pathos/algorithms/base.py tests/test_parallel.py
git commit -m "feat(algorithms): propagate n_workers from Space into Algorithm base"
```

---

### Task 3: Parallel GA population evaluation

**Files:**
- Modify: `pathos/algorithms/evolutionary.py:1-10` (imports), `84-108` (GA.solve)
- Modify: `tests/test_parallel.py` (add GA parallel test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_parallel.py`:

```python
import random as _random
from pathos.algorithms.evolutionary import GeneticAlgorithm, DifferentialEvolution


def _ga_fitness(ind):
    return -sum(ind)


def _ga_crossover(p1, p2):
    pt = _random.randint(1, len(p1) - 1)
    return tuple(p1[:pt] + p2[pt:])


def _ga_mutate(ind):
    i = _random.randint(0, len(ind) - 1)
    lst = list(ind)
    lst[i] = 1 - lst[i]
    return tuple(lst)


def test_ga_parallel_correctness():
    """GA with parallel=2 should find a solution of equal quality to serial."""
    space = (
        Space()
        .initial(lambda: tuple(_random.randint(0, 1) for _ in range(10)))
        .parallel(2)
    )
    space.evaluate(_ga_fitness)

    result = GeneticAlgorithm(
        space, pop_size=20, generations=100,
        crossover_fn=_ga_crossover, mutate_fn=_ga_mutate,
    ).solve()
    assert result.found
    assert -result.cost >= 8
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_parallel.py::test_ga_parallel_correctness -v
```

Expected: PASS but only because `_n_workers` is stored, not yet used. The test should pass accidentally since serial GA works fine — but the POINT is to verify it still passes after we change the implementation. Run it now to establish baseline.

> Note: if the test passes here, it's fine — we're confirming the GA works. The key check is that it still passes after the Task 3 implementation changes.

- [ ] **Step 3: Update `pathos/algorithms/evolutionary.py`**

Add import at top (after existing imports):

```python
from pathos.core.parallel import batch_map
```

Replace `GeneticAlgorithm.solve` (lines 84-108):

```python
    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        population = [self.space._initial for _ in range(self.pop_size)]
        costs = batch_map(self.space._evaluate, population, self._n_workers)
        best_idx = min(range(len(population)), key=lambda i: costs[i])
        best, best_cost = population[best_idx], costs[best_idx]

        for _ in range(self.generations):
            costs = batch_map(self.space._evaluate, population, self._n_workers)
            pairs = sorted(zip(costs, population))
            population = [x for _, x in pairs[: self.pop_size // 2]]
            while len(population) < self.pop_size:
                p1, p2 = random.sample(population, 2)
                if self.crossover_fn:
                    child = self.crossover_fn(p1, p2)
                else:
                    child = copy.deepcopy(p1)
                if self.mutate_fn and random.random() < self.mutation_rate:
                    child = self.mutate_fn(child)
                population.append(child)
            gen_costs = batch_map(self.space._evaluate, population, self._n_workers)
            gen_best_idx = min(range(len(population)), key=lambda i: gen_costs[i])
            if gen_costs[gen_best_idx] < best_cost:
                best, best_cost = population[gen_best_idx], gen_costs[gen_best_idx]

        return SearchResult(best, None, best_cost, "GeneticAlgorithm",
                            self.generations * self.pop_size, time.perf_counter() - t0, True)
```

- [ ] **Step 4: Run parallel test and full suite**

```bash
pytest tests/test_parallel.py::test_ga_parallel_correctness tests/test_evolutionary.py -v
```

Expected: all PASS. The existing serial GA test passes because `n_workers=1` makes `batch_map` fall through to `list(map(...))`.

- [ ] **Step 5: Commit**

```bash
git add pathos/algorithms/evolutionary.py tests/test_parallel.py
git commit -m "feat(algorithms): parallel population evaluation in GeneticAlgorithm"
```

---

### Task 4: Parallel DE trial vector evaluation

**Files:**
- Modify: `pathos/algorithms/evolutionary.py:133-160` (DE.solve)
- Modify: `tests/test_parallel.py` (add DE parallel test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_parallel.py`:

```python
def _de_cost(x):
    return sum(xi ** 2 for xi in x)


def test_de_parallel_correctness():
    """DE with parallel=2 should minimize sum-of-squares close to zero."""
    import random as r
    space = (
        Space()
        .initial(lambda: [r.uniform(-5, 5) for _ in range(3)])
        .parallel(2)
    )
    space.evaluate(_de_cost)

    result = DifferentialEvolution(space, pop_size=20, generations=200).solve()
    assert result.found
    assert result.cost < 0.1
```

- [ ] **Step 2: Run test to verify baseline**

```bash
pytest tests/test_parallel.py::test_de_parallel_correctness -v
```

- [ ] **Step 3: Replace `DifferentialEvolution.solve` (lines 133-160)**

The key change: collect all trial vectors for the generation, batch-evaluate them, then do synchronous replacement.

```python
    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        pop = [self.space._initial for _ in range(self.pop_size)]
        costs = batch_map(self.space._evaluate, pop, self._n_workers)
        best_idx = min(range(self.pop_size), key=lambda i: costs[i])
        best, best_cost = pop[best_idx], costs[best_idx]

        for _ in range(self.generations):
            trials = []
            for i in range(self.pop_size):
                x = pop[i]
                if not isinstance(x, list):
                    trials.append(x)
                    continue
                a, b, c = random.sample([j for j in range(self.pop_size) if j != i], 3)
                xa, xb, xc = pop[a], pop[b], pop[c]
                dim = len(x)
                j_rand = random.randint(0, dim - 1)
                trials.append([
                    xa[j] + self.F * (xb[j] - xc[j])
                    if random.random() < self.CR or j == j_rand else x[j]
                    for j in range(dim)
                ])
            trial_costs = batch_map(self.space._evaluate, trials, self._n_workers)
            for i in range(self.pop_size):
                if trial_costs[i] < costs[i]:
                    pop[i], costs[i] = trials[i], trial_costs[i]
                    if trial_costs[i] < best_cost:
                        best, best_cost = trials[i], trial_costs[i]

        return SearchResult(best, None, best_cost, "DifferentialEvolution",
                            self.generations * self.pop_size, time.perf_counter() - t0, True)
```

- [ ] **Step 4: Run parallel test and full suite**

```bash
pytest tests/test_parallel.py::test_de_parallel_correctness tests/test_evolutionary.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pathos/algorithms/evolutionary.py tests/test_parallel.py
git commit -m "feat(algorithms): parallel trial evaluation in DifferentialEvolution"
```

---

### Task 5: Parallel LocalBeamSearch candidate evaluation

**Files:**
- Modify: `pathos/algorithms/local.py:1-8` (imports), `132-157` (LocalBeamSearch.solve)
- Modify: `tests/test_parallel.py` (add LocalBeamSearch parallel test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_parallel.py`:

```python
from pathos.algorithms.local import LocalBeamSearch


def _parabola(x):
    return (x - 5) ** 2


def _int_neighbors_parallel(x):
    for dx in [-1, 1]:
        nx = x + dx
        if 0 <= nx <= 10:
            yield str(dx), nx


def test_local_beam_search_parallel_correctness():
    """LocalBeamSearch with parallel=2 should still find the minimum."""
    space = Space().initial(0).parallel(2)
    space.evaluate(_parabola)

    @space.successors
    def neighbors(x): yield from _int_neighbors_parallel(x)

    result = LocalBeamSearch(space, k=3, max_iter=50).solve()
    assert result.found
    assert result.solution == 5
```

- [ ] **Step 2: Run test to verify baseline**

```bash
pytest tests/test_parallel.py::test_local_beam_search_parallel_correctness -v
```

- [ ] **Step 3: Update `pathos/algorithms/local.py`**

Add import at top (after existing imports):

```python
from pathos.core.parallel import batch_map
```

Replace `LocalBeamSearch.solve` (lines 132-157):

```python
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
            candidate_costs = batch_map(self.space._evaluate, candidates, self._n_workers)
            ranked = sorted(zip(candidate_costs, candidates))[: self.k]
            beam = [s for _, s in ranked]
            if ranked[0][0] < best_cost:
                best_cost, best = ranked[0][0], beam[0]

        return SearchResult(
            best, None, best_cost, "LocalBeamSearch",
            expanded, time.perf_counter() - t0, True,
        )
```

- [ ] **Step 4: Run parallel test and full suite**

```bash
pytest tests/test_parallel.py::test_local_beam_search_parallel_correctness tests/test_local.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pathos/algorithms/local.py tests/test_parallel.py
git commit -m "feat(algorithms): parallel candidate evaluation in LocalBeamSearch"
```

---

### Task 6: Full test suite verification

**Files:**
- No changes — validation only

- [ ] **Step 1: Run entire test suite**

```bash
cd /home/apiad/Workspace/repos/pathos
pytest tests/ -v
```

Expected: all tests PASS. Note the test count: existing tests + 7 new parallel tests.

- [ ] **Step 2: Verify `parallel()` is documented in the Space class**

Read `pathos/core/space.py` and confirm `parallel()` has a docstring or inline comment explaining the pickling requirement. If not, add a one-line comment:

```python
    def parallel(self, workers: int) -> Space:
        # evaluate fn must be picklable (module-level) when workers > 1
        self._n_workers = workers
        return self
```

Edit `pathos/core/space.py` if the comment is missing.

- [ ] **Step 3: Commit if space.py was changed**

```bash
git add pathos/core/space.py
git commit -m "docs(core): note pickling constraint on parallel() evaluate fn"
```

---

## Self-Review

**Spec coverage:**
- ✅ `batch_map` utility in `pathos/core/parallel.py`
- ✅ `_n_workers` wired in `Algorithm.__init__`
- ✅ GA: parallel population eval per generation
- ✅ DE: parallel trial vector eval (synchronous generation variant)
- ✅ LocalBeamSearch: parallel candidate eval
- ✅ Tests for all three parallel paths
- ✅ Existing serial tests unaffected (n_workers=1 fallback)

**Algorithms NOT in this slice (intentional):**
- SA: single-trajectory, no natural population to parallelize
- HillClimbing / TabuSearch: single-trajectory
- Multi-start variants (SA, HC): next slice — different pattern (run N independent instances)
- A\*, Bidirectional A\*, CSP: different parallelism pattern (not population-based)

**Pickle constraint:** All new parallel tests use module-level evaluate functions. The `parallel()` docstring notes the constraint. Existing tests use closures/lambdas but call no `.parallel()`, so they always hit the `n_workers=1` serial path.

**Semantic change in DE:** The original DE uses greedy per-individual replacement within a generation (each trial immediately replaces if better). The parallel version collects all trials, evaluates in batch, then applies replacements — a "synchronous generation" variant. Both are valid DE implementations; the parallel variant is standard in literature.
