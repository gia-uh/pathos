# ScheduleSpace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ScheduleSpace` (peer subspace to `GraphSpace`/`CSPSpace`/`TourSpace`/`GameSpace`) plus the `weighted_minmax` fairness helper, so users can model node × time-slot scheduling problems with capacity constraints and fairness objectives, and the existing `AnytimeLocal` cascade solves them.

**Architecture:** Pure-stdlib, zero new dependencies. State is `frozenset[tuple[int, int]]` (the set of `(slot, entity_index)` cells that are ON). Capacity violations are folded into `_evaluate` as a `λ·overshoot` penalty term, so the emitted capability set is `{EVALUATE, SUCCESSORS}` and the existing `HillClimbing` / `SimulatedAnnealing` / `TabuSearch` / `AnytimeLocal` algorithms accept it verbatim. No algorithm files are touched. `SearchResult` gains one optional field (`slack`) so users can verify per-slot feasibility on the returned schedule.

**Tech Stack:** Python 3.11+, stdlib only. pytest for tests.

**Spec:** `docs/superpowers/specs/2026-06-09-schedule-space-power-grid-design.md`.

---

## File Structure

- **Create:** `pathos/spaces/schedule.py` — `ScheduleSpace` class.
- **Create:** `pathos/fairness.py` — `weighted_minmax(weights, space)` helper.
- **Create:** `tests/test_schedule_space.py` — unit + property + integration tests.
- **Create:** `tests/test_fairness.py` — `weighted_minmax` unit tests.
- **Create:** `examples/power_grid.py` — worked example.
- **Modify:** `pathos/core/result.py` — add optional `slack` field to `SearchResult`.
- **Modify:** `pathos/__init__.py` — re-export `ScheduleSpace`.

Tasks 1–9 each produce a self-contained, committable change.

---

### Task 1: Add `slack` field to `SearchResult`

`SearchResult` currently has no field for per-slot residual capacity. Adding it now (as `slack: list[float] | None = None`, defaulting to `None`) keeps every existing call site valid and lets ScheduleSpace populate it without touching algorithm code.

**Files:**
- Modify: `pathos/core/result.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_result_slack.py`:

```python
from pathos.core.result import SearchResult


def test_searchresult_has_optional_slack_field_defaulting_to_none():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="Dummy", nodes_expanded=0, elapsed=0.0, found=True,
    )
    assert r.slack is None


def test_searchresult_slack_field_can_be_set():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="Dummy", nodes_expanded=0, elapsed=0.0, found=True,
        slack=[1.5, 2.0, -0.5],
    )
    assert r.slack == [1.5, 2.0, -0.5]


def test_searchresult_not_found_factory_still_works():
    r = SearchResult.not_found("Dummy", 0, 0.0)
    assert r.slack is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/apiad/Workspace/repos/pathos
pytest tests/test_result_slack.py -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'slack'` on the second test.

- [ ] **Step 3: Add the `slack` field to `SearchResult`**

Modify `pathos/core/result.py` — append the new field after `epsilon`:

```python
@dataclass
class SearchResult:
    """Result returned by every PATHOS algorithm after a solve() call.

    Attributes:
        solution: The goal state (or best state found for local search).
        path: List of (action, state) tuples from initial to solution,
            or None for algorithms that don't track paths.
        cost: Accumulated cost to reach the solution, or None if not applicable.
        algorithm: Name of the algorithm that produced this result.
        nodes_expanded: Number of nodes expanded during search.
        elapsed: Wall-clock seconds taken by solve().
        found: True if a solution was found, False if search exhausted.
        epsilon: Suboptimality bound on `cost`. 1.0 = proven optimal,
            >1.0 = cost ≤ ε × optimal (bounded suboptimal),
            inf = unbounded suboptimal (e.g. greedy),
            None = not applicable (e.g. metaheuristics with no quality bound,
            or not_found results).
        slack: Per-slot residual capacity for ScheduleSpace solutions.
            Negative entries indicate the penalty-folded local search
            accepted an infeasible schedule. None for every other subspace.
    """
    solution: Any
    path: list[tuple[Any, Any]] | None
    cost: float | None
    algorithm: str
    nodes_expanded: int
    elapsed: float
    found: bool
    epsilon: float | None = None
    slack: list[float] | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_result_slack.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
pytest -q
```

Expected: every prior test still passes.

- [ ] **Step 6: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add pathos/core/result.py tests/test_result_slack.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(result): optional slack field for ScheduleSpace per-slot headroom"
```

---

### Task 2: Build `ScheduleSpace` skeleton (constructor + validation)

The skeleton accepts `entities`, `slots`, optional `downstream`, and `penalty`; runs constructor validation; exposes `_initial` as an empty frozenset. No decorators yet — those come in Task 3. The class sits at `pathos/spaces/schedule.py`, peer to `csp.py`/`graph.py`/`tour.py`/`game.py`, and inherits from `pathos.core.space.Space`.

**Files:**
- Create: `pathos/spaces/schedule.py`
- Create: `tests/test_schedule_space.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schedule_space.py`:

```python
import pytest
from pathos.spaces.schedule import ScheduleSpace


def test_constructor_accepts_entities_and_slots():
    s = ScheduleSpace(entities=["a", "b", "c"], slots=4)
    assert s._entities == ("a", "b", "c")
    assert s._slots == 4


def test_constructor_rejects_empty_entities():
    with pytest.raises(ValueError, match="entities"):
        ScheduleSpace(entities=[], slots=4)


def test_constructor_rejects_zero_or_negative_slots():
    with pytest.raises(ValueError, match="slots"):
        ScheduleSpace(entities=["a"], slots=0)
    with pytest.raises(ValueError, match="slots"):
        ScheduleSpace(entities=["a"], slots=-1)


def test_constructor_default_downstream_is_identity():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    assert list(s._downstream("a")) == ["a"]
    assert list(s._downstream("b")) == ["b"]


def test_constructor_custom_downstream():
    leaves = {"sub1": ["x", "y"], "sub2": ["z"]}
    s = ScheduleSpace(entities=["sub1", "sub2"], slots=2,
                       downstream=lambda e: leaves[e])
    assert list(s._downstream("sub1")) == ["x", "y"]
    assert list(s._downstream("sub2")) == ["z"]


def test_constructor_default_penalty_is_1000():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s._penalty == 1e3


def test_constructor_custom_penalty():
    s = ScheduleSpace(entities=["a"], slots=1, penalty=42.0)
    assert s._penalty == 42.0


def test_initial_state_is_empty_frozenset():
    s = ScheduleSpace(entities=["a", "b"], slots=3)
    assert s._initial == frozenset()


def test_capabilities_empty_before_any_decorator():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s.capabilities == set()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: `ModuleNotFoundError: No module named 'pathos.spaces.schedule'`.

- [ ] **Step 3: Create `pathos/spaces/schedule.py`**

```python
from __future__ import annotations
from typing import Any, Callable, Hashable, Iterable, Sequence
from pathos.core.space import Space


class ScheduleSpace(Space):
    """Space for discrete-time scheduling problems with capacity constraints
    and fairness objectives.

    Each (slot, entity) cell is a binary decision: on or off. State is a
    frozenset of (slot_index, entity_index) tuples for the ON cells.
    Capacity violations are folded into _evaluate as a penalty so the
    existing AnytimeLocal cascade (HC -> SA -> Tabu) solves it directly.

    See `docs/superpowers/specs/2026-06-09-schedule-space-power-grid-design.md`.
    """

    def __init__(
        self,
        entities: Sequence[Hashable],
        slots: int,
        downstream: Callable[[Hashable], Iterable[Hashable]] | None = None,
        penalty: float = 1e3,
    ) -> None:
        super().__init__()
        entities_tuple = tuple(entities)
        if not entities_tuple:
            raise ValueError("entities must be non-empty")
        if slots <= 0:
            raise ValueError(f"slots must be positive, got {slots}")
        self._entities: tuple[Hashable, ...] = entities_tuple
        self._slots: int = slots
        self._downstream: Callable[[Hashable], Iterable[Hashable]] = (
            downstream if downstream is not None else (lambda e: (e,))
        )
        self._penalty: float = penalty
        self._initial_value = frozenset()
        # User-attached decorators populated in Task 3.
        self._demand_fn: Callable[[Hashable, int], float] | None = None
        self._capacity_fn: Callable[[int], float] | None = None
        self._fairness_fn: Callable[[tuple[tuple[bool, ...], ...]], float] | None = None
        # Target band, set via .target(). Default: upper bound only.
        self._tolerance: float = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add pathos/spaces/schedule.py tests/test_schedule_space.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(spaces): ScheduleSpace skeleton — constructor + validation"
```

---

### Task 3: Add `@demand`, `@capacity`, `@fairness` decorators

Each decorator stores the user's function, errors on re-attachment, and (for `@fairness`) emits the `EVALUATE` capability. `SUCCESSORS` is emitted in Task 4 alongside the neighborhood. We don't emit `EVALUATE` until `@fairness` is attached because `_evaluate` reads the fairness function.

**Files:**
- Modify: `pathos/spaces/schedule.py`
- Modify: `tests/test_schedule_space.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schedule_space.py`:

```python
from pathos.core.capabilities import Capability


def _attach_demand_capacity_fairness(s):
    @s.demand
    def _d(e, t): return 1.0
    @s.capacity
    def _c(t): return 1.0
    @s.fairness
    def _f(schedule): return 0.5
    return _d, _c, _f


def test_demand_decorator_stores_function():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.demand
    def my_demand(entity, slot): return 7.0
    assert s._demand_fn is my_demand


def test_capacity_decorator_stores_function():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.capacity
    def my_capacity(slot): return 99.0
    assert s._capacity_fn is my_capacity


def test_fairness_decorator_stores_function_and_emits_evaluate():
    s = ScheduleSpace(entities=["a"], slots=2)
    assert Capability.EVALUATE not in s.capabilities
    @s.fairness
    def my_fairness(schedule): return 0.3
    assert s._fairness_fn is my_fairness
    assert Capability.EVALUATE in s.capabilities


def test_demand_decorator_raises_on_reattach():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.demand
    def _d1(e, t): return 1.0
    with pytest.raises(RuntimeError, match="demand"):
        @s.demand
        def _d2(e, t): return 2.0


def test_capacity_decorator_raises_on_reattach():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.capacity
    def _c1(t): return 1.0
    with pytest.raises(RuntimeError, match="capacity"):
        @s.capacity
        def _c2(t): return 2.0


def test_fairness_decorator_raises_on_reattach():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.fairness
    def _f1(schedule): return 0.5
    with pytest.raises(RuntimeError, match="fairness"):
        @s.fairness
        def _f2(schedule): return 0.6


def test_fairness_decorator_returns_the_callable_unchanged():
    """Standard pathos pattern — decorator returns fn so user can also call it."""
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.fairness
    def f(schedule): return 0.5
    assert f is s._fairness_fn
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: failures on the decorator-related tests with `AttributeError: 'ScheduleSpace' object has no attribute 'demand'` (Space's `Space.demand` etc. don't exist either — only the basic ones).

- [ ] **Step 3: Add the decorator properties to `pathos/spaces/schedule.py`**

Append to `ScheduleSpace` (after `__init__`):

```python
    # --- decorator hooks ---

    @property
    def demand(self) -> Callable[..., Any]:
        def decorator(fn: Callable[..., float]) -> Callable[..., float]:
            if self._demand_fn is not None:
                raise RuntimeError("@demand already defined on this space")
            self._demand_fn = fn
            return fn
        return decorator

    @property
    def capacity(self) -> Callable[..., Any]:
        def decorator(fn: Callable[..., float]) -> Callable[..., float]:
            if self._capacity_fn is not None:
                raise RuntimeError("@capacity already defined on this space")
            self._capacity_fn = fn
            return fn
        return decorator

    @property
    def fairness(self) -> Callable[..., Any]:
        from pathos.core.capabilities import Capability
        def decorator(fn: Callable[..., float]) -> Callable[..., float]:
            if self._fairness_fn is not None:
                raise RuntimeError("@fairness already defined on this space")
            self._fairness_fn = fn
            self.capabilities.add(Capability.EVALUATE)
            return fn
        return decorator
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: all (originally-9 + new-7 = 16) passing.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add pathos/spaces/schedule.py tests/test_schedule_space.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(spaces): ScheduleSpace @demand/@capacity/@fairness decorators"
```

---

### Task 4: Implement `_to_matrix` view + 1-bit-flip `_successors`

`_to_matrix(state)` materialises the frozenset into a `tuple[tuple[bool, ...], ...]` of shape `(T, N)`. `_successors(state)` yields one neighbor per cell with that cell toggled. Both run pure-Python; no third-party deps. `_successors` is set as `self._successors` and `Capability.SUCCESSORS` is added to `self.capabilities`. This makes the space immediately compatible with `HillClimbing` / `TabuSearch` / `LocalBeamSearch` once `_evaluate` is wired in Task 5.

**Files:**
- Modify: `pathos/spaces/schedule.py`
- Modify: `tests/test_schedule_space.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schedule_space.py`:

```python
def test_to_matrix_empty_state_is_all_false():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    m = s._to_matrix(frozenset())
    assert m == ((False, False), (False, False))


def test_to_matrix_partial_state():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    # ON cells: (slot=0, entity_idx=1), (slot=1, entity_idx=0)
    m = s._to_matrix(frozenset({(0, 1), (1, 0)}))
    assert m == ((False, True), (True, False))


def test_successors_count_is_T_times_N():
    s = ScheduleSpace(entities=["a", "b", "c"], slots=4)
    _attach_demand_capacity_fairness(s)
    state = frozenset()
    children = list(s._successors(state))
    assert len(children) == 12   # 4 * 3


def test_successors_each_action_toggles_one_cell():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    _attach_demand_capacity_fairness(s)
    state = frozenset({(0, 0)})
    children = dict(s._successors(state))
    # action label is "(slot, entity_idx)" for the toggled cell
    assert frozenset({(0, 0), (0, 1)}) in children.values()  # turn (0,1) ON
    assert frozenset() in children.values()                    # turn (0,0) OFF
    assert frozenset({(0, 0), (1, 0)}) in children.values()  # turn (1,0) ON
    assert frozenset({(0, 0), (1, 1)}) in children.values()  # turn (1,1) ON


def test_successors_capability_emitted_after_demand_capacity_fairness():
    s = ScheduleSpace(entities=["a"], slots=2)
    _attach_demand_capacity_fairness(s)
    assert Capability.SUCCESSORS in s.capabilities
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: `AttributeError: '_to_matrix'` and successors-related failures.

- [ ] **Step 3: Add `_to_matrix` and successor wiring**

In `pathos/spaces/schedule.py`, add `_to_matrix` and a `_setup_successors` method. Wire `_setup_successors` to fire when all three decorators are attached (similar to `CSPSpace._maybe_finalize`).

```python
    # --- internal helpers ---

    def _to_matrix(self, state: frozenset[tuple[int, int]]) -> tuple[tuple[bool, ...], ...]:
        """Convert a frozenset of (slot, entity_idx) ON cells to a (T, N) matrix."""
        n = len(self._entities)
        return tuple(
            tuple((t, e) in state for e in range(n))
            for t in range(self._slots)
        )

    def _setup_successors(self) -> None:
        from pathos.core.capabilities import Capability
        T, N = self._slots, len(self._entities)

        def _successors(state: frozenset[tuple[int, int]]) -> Any:
            for t in range(T):
                for e in range(N):
                    cell = (t, e)
                    if cell in state:
                        yield f"off({t},{e})", state - {cell}
                    else:
                        yield f"on({t},{e})", state | {cell}

        self._successors = _successors
        self.capabilities.add(Capability.SUCCESSORS)
```

Update each decorator to call `_maybe_finalize` when all three are present:

```python
    def _maybe_finalize(self) -> None:
        if (
            self._demand_fn is not None
            and self._capacity_fn is not None
            and self._fairness_fn is not None
            and self._successors is None
        ):
            self._setup_successors()
```

Then in each of `demand`, `capacity`, `fairness` decorator closures, call `self._maybe_finalize()` after attaching. For example, in `demand`:

```python
            self._demand_fn = fn
            self._maybe_finalize()
            return fn
```

Do the same in `capacity` and `fairness` (after the existing `self.capabilities.add(...)` in `fairness`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: all tests in the file (~20) passing.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add pathos/spaces/schedule.py tests/test_schedule_space.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(spaces): ScheduleSpace _to_matrix view + 1-flip successors"
```

---

### Task 5: Implement `_evaluate` (fairness − penalty)

`_evaluate(state)` is called by `HillClimbing`/`SimulatedAnnealing`/`TabuSearch`/`LocalBeamSearch` to score candidates. Lower = better (those algorithms minimise). We return `-fairness(matrix) + penalty · overshoot` where `overshoot = sum_t max(0, sum_e demand(entity_e, t) - capacity(t))`. With the default `penalty=1e3` and fairness in `[0, 1]`, any feasible schedule out-scores any schedule with a 1-unit overshoot.

The lower band from `.target(tolerance)` (added in Task 6) also contributes to `overshoot`. Task 5 implements upper-bound-only; Task 6 layers in the lower band.

**Files:**
- Modify: `pathos/spaces/schedule.py`
- Modify: `tests/test_schedule_space.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schedule_space.py`:

```python
def test_evaluate_pure_fairness_when_feasible():
    s = ScheduleSpace(entities=["a", "b"], slots=2, penalty=1e3)
    @s.demand
    def _d(e, t): return 1.0
    @s.capacity
    def _c(t): return 10.0    # huge, never violated
    @s.fairness
    def _f(schedule): return 0.42
    state = frozenset({(0, 0), (1, 1)})
    # fairness = 0.42, no violation -> _evaluate = -0.42
    assert s._evaluate(state) == pytest.approx(-0.42)


def test_evaluate_penalises_capacity_overshoot():
    s = ScheduleSpace(entities=["a", "b"], slots=1, penalty=1e3)
    @s.demand
    def _d(e, t): return 5.0
    @s.capacity
    def _c(t): return 3.0     # any single entity ON overshoots by 2
    @s.fairness
    def _f(schedule): return 0.0
    state = frozenset({(0, 0)})   # one entity ON, demand=5, cap=3 -> overshoot=2
    # _evaluate = -0 + 1000*2 = 2000
    assert s._evaluate(state) == pytest.approx(2000.0)


def test_evaluate_overshoot_summed_across_slots():
    s = ScheduleSpace(entities=["a"], slots=3, penalty=10.0)
    @s.demand
    def _d(e, t): return 5.0
    @s.capacity
    def _c(t): return 3.0    # overshoot=2 per slot with entity ON
    @s.fairness
    def _f(schedule): return 0.0
    state = frozenset({(0, 0), (1, 0), (2, 0)})  # ON every slot
    # overshoot = 2 + 2 + 2 = 6; _evaluate = 60.0
    assert s._evaluate(state) == pytest.approx(60.0)


def test_evaluate_off_slot_contributes_zero_overshoot():
    s = ScheduleSpace(entities=["a"], slots=2, penalty=10.0)
    @s.demand
    def _d(e, t): return 5.0
    @s.capacity
    def _c(t): return 3.0
    @s.fairness
    def _f(schedule): return 0.0
    state = frozenset()  # all off, no demand, no overshoot
    assert s._evaluate(state) == pytest.approx(0.0)


def test_evaluate_rejects_negative_capacity():
    s = ScheduleSpace(entities=["a"], slots=1)
    @s.demand
    def _d(e, t): return 1.0
    @s.capacity
    def _c(t): return -1.0
    @s.fairness
    def _f(schedule): return 0.0
    with pytest.raises(ValueError, match="capacity"):
        s._evaluate(frozenset())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: `AttributeError: '_evaluate'` (or returns None) — the function doesn't exist yet.

- [ ] **Step 3: Add `_evaluate` and wire it as the EVALUATE capability**

Replace the `fairness` decorator body in `pathos/spaces/schedule.py` so it now installs `_evaluate` (not just stores the user function):

```python
    @property
    def fairness(self) -> Callable[..., Any]:
        from pathos.core.capabilities import Capability
        def decorator(fn: Callable[..., float]) -> Callable[..., float]:
            if self._fairness_fn is not None:
                raise RuntimeError("@fairness already defined on this space")
            self._fairness_fn = fn

            def _evaluate(state: frozenset[tuple[int, int]]) -> float:
                matrix = self._to_matrix(state)
                fairness_score = self._fairness_fn(matrix)
                overshoot = self._overshoot(state)
                return -float(fairness_score) + self._penalty * overshoot

            self._evaluate = _evaluate
            self.capabilities.add(Capability.EVALUATE)
            self._maybe_finalize()
            return fn
        return decorator

    def _overshoot(self, state: frozenset[tuple[int, int]]) -> float:
        """Total capacity-violation overshoot summed across slots.

        For each slot, compute (load - capacity) and clip below at 0.
        Lower band from .target() is layered in by Task 6.
        """
        if self._demand_fn is None or self._capacity_fn is None:
            return 0.0
        total = 0.0
        for t in range(self._slots):
            cap_t = self._capacity_fn(t)
            if cap_t < 0:
                raise ValueError(
                    f"@capacity returned negative value {cap_t} at slot {t}",
                )
            load_t = sum(
                self._demand_fn(self._entities[e], t)
                for tt, e in state if tt == t
            )
            total += max(0.0, load_t - cap_t)
        return total
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: all tests passing.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add pathos/spaces/schedule.py tests/test_schedule_space.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(spaces): ScheduleSpace _evaluate — fairness minus capacity penalty"
```

---

### Task 6: `.target(tolerance)` lower band + `.neighborhood(k)` builder

`.target(tolerance)` adds a per-slot lower bound: `load[t] ≥ capacity[t] · (1 - tolerance)`. Implemented as an additional overshoot term: `max(0, capacity[t] · (1 - tolerance) - load[t])`, added to the existing upper-bound overshoot in `_overshoot`. `.neighborhood(k)` records `k`; v1 only exercises k=1 in tests, but the field is plumbed so users can switch.

**Files:**
- Modify: `pathos/spaces/schedule.py`
- Modify: `tests/test_schedule_space.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schedule_space.py`:

```python
def test_target_default_is_zero_tolerance():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s._tolerance == 0.0


def test_target_sets_tolerance():
    s = ScheduleSpace(entities=["a"], slots=1).target(tolerance=0.1)
    assert s._tolerance == 0.1


def test_target_returns_self_for_chaining():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s.target(tolerance=0.05) is s


def test_target_rejects_tolerance_below_0():
    s = ScheduleSpace(entities=["a"], slots=1)
    with pytest.raises(ValueError, match="tolerance"):
        s.target(tolerance=-0.1)


def test_target_rejects_tolerance_above_1():
    s = ScheduleSpace(entities=["a"], slots=1)
    with pytest.raises(ValueError, match="tolerance"):
        s.target(tolerance=1.5)


def test_target_lower_band_penalises_under_use():
    s = ScheduleSpace(entities=["a"], slots=1, penalty=10.0).target(tolerance=0.5)
    @s.demand
    def _d(e, t): return 5.0
    @s.capacity
    def _c(t): return 10.0    # tolerance=0.5 -> lower band = 5.0
    @s.fairness
    def _f(schedule): return 0.0
    state = frozenset()  # load=0, lower band undershoots by 5
    # _evaluate = -0 + 10 * 5 = 50
    assert s._evaluate(state) == pytest.approx(50.0)


def test_neighborhood_default_is_1():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s._k == 1


def test_neighborhood_sets_k():
    s = ScheduleSpace(entities=["a"], slots=1).neighborhood(k=2)
    assert s._k == 2


def test_neighborhood_returns_self_for_chaining():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s.neighborhood(k=2) is s


def test_neighborhood_rejects_k_below_1():
    s = ScheduleSpace(entities=["a"], slots=1)
    with pytest.raises(ValueError, match="k"):
        s.neighborhood(k=0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: `AttributeError: 'target'` and `'neighborhood'`.

- [ ] **Step 3: Add the builders and extend `_overshoot`**

In `pathos/spaces/schedule.py`, initialise `self._k = 1` at the bottom of `__init__`, then append the builders:

```python
    # --- fluent builders ---

    def target(self, tolerance: float) -> "ScheduleSpace":
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError(
                f"tolerance must be in [0, 1], got {tolerance}"
            )
        self._tolerance = tolerance
        return self

    def neighborhood(self, k: int) -> "ScheduleSpace":
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        return self
```

Update `_overshoot` to include the lower band — replace the per-slot body with:

```python
        for t in range(self._slots):
            cap_t = self._capacity_fn(t)
            if cap_t < 0:
                raise ValueError(
                    f"@capacity returned negative value {cap_t} at slot {t}",
                )
            load_t = sum(
                self._demand_fn(self._entities[e], t)
                for tt, e in state if tt == t
            )
            total += max(0.0, load_t - cap_t)
            if self._tolerance > 0.0:
                lower = cap_t * (1.0 - self._tolerance)
                total += max(0.0, lower - load_t)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: all tests passing.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add pathos/spaces/schedule.py tests/test_schedule_space.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(spaces): ScheduleSpace .target() lower band + .neighborhood(k) builder"
```

---

### Task 7: `weighted_minmax` fairness helper

Create `pathos/fairness.py` with the single helper `weighted_minmax(weights, space)`. It returns a callable that takes a `(T, N)` matrix and returns `min over leaves of weights[leaf] * uptime_fraction(leaf)`, where `uptime_fraction(leaf, matrix)` is the share of slots in which the entity feeding `leaf` (per `space._downstream`) was ON. Leaves with `weights[leaf] == 0` (critical loads) are skipped because they never tolerate blackout — the user surfaces that as an infeasibility via capacity sizing, not as a fairness penalty.

Wait, re-read the spec: `weights` is a map leaf → tolerance (higher = more tolerant of blackouts). The fairness score is `min(weights[leaf] · uptime_fraction(leaf))`. Higher weight magnifies the importance of that leaf's uptime in the min — critical (weight=0) means uptime can be 0 without penalty (the term is 0). That doesn't quite line up with the spec's "hospitals = 0 = never blackout" framing.

Actually re-reading more carefully: the spec text "0.0 = critical (hospital), 0.5 = industrial, 1.0 = residential" implies **lower weight = more critical**. So the objective is `min over leaves of (1 - weights[leaf]) · uptime_fraction(leaf)` to MAXIMISE — wait, that's still off.

Resolving the ambiguity: define `weights[leaf]` as the **importance** (0 = don't care, 1 = critical). Then `weighted_minmax` returns `min over leaves of weights[leaf] * uptime_fraction(leaf)`. To MAXIMISE: high-importance leaves with poor uptime drag the min down. Critical leaves (weight ~ 1) MUST have high uptime; residential (weight ~ 0) doesn't matter. Update the example accordingly in Task 9.

**Files:**
- Create: `pathos/fairness.py`
- Create: `tests/test_fairness.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fairness.py`:

```python
import pytest
from pathos.spaces.schedule import ScheduleSpace
from pathos.fairness import weighted_minmax


def test_weighted_minmax_returns_a_callable():
    s = ScheduleSpace(entities=["a"], slots=1)
    helper = weighted_minmax(weights={}, space=s)
    assert callable(helper)


def test_weighted_minmax_all_on_returns_min_weight():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    # downstream defaults to identity: leaves = entities
    helper = weighted_minmax(weights={"a": 0.5, "b": 1.0}, space=s)
    # All ON for both entities -> uptime_fraction = 1.0 each
    # term[a] = 0.5 * 1.0 = 0.5; term[b] = 1.0 * 1.0 = 1.0; min = 0.5
    matrix = ((True, True), (True, True))
    assert helper(matrix) == pytest.approx(0.5)


def test_weighted_minmax_partial_uptime():
    s = ScheduleSpace(entities=["a", "b"], slots=4)
    helper = weighted_minmax(weights={"a": 1.0, "b": 1.0}, space=s)
    # a ON for slot 0 only (uptime 0.25); b ON every slot (uptime 1.0)
    matrix = (
        (True, True),
        (False, True),
        (False, True),
        (False, True),
    )
    # term[a] = 0.25, term[b] = 1.0; min = 0.25
    assert helper(matrix) == pytest.approx(0.25)


def test_weighted_minmax_zero_weight_leaf_is_skipped():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    helper = weighted_minmax(weights={"a": 0.0, "b": 1.0}, space=s)
    # a always OFF (uptime 0), but weight 0 -> skipped
    # b always OFF (uptime 0), weight 1 -> term = 0
    matrix = ((False, False), (False, False))
    assert helper(matrix) == pytest.approx(0.0)
    # a always OFF (skipped); b ON every slot -> term = 1.0; min = 1.0
    matrix2 = ((False, True), (False, True))
    assert helper(matrix2) == pytest.approx(1.0)


def test_weighted_minmax_uses_downstream():
    leaves = {"sub1": ["x", "y"], "sub2": ["z"]}
    s = ScheduleSpace(entities=["sub1", "sub2"], slots=2,
                       downstream=lambda e: leaves[e])
    helper = weighted_minmax(weights={"x": 1.0, "y": 1.0, "z": 1.0}, space=s)
    # sub1 ON every slot, sub2 OFF every slot
    matrix = ((True, False), (True, False))
    # x and y feed from sub1 -> uptime 1.0 each; z feeds from sub2 -> uptime 0.0
    # terms: x=1.0, y=1.0, z=0.0; min = 0.0
    assert helper(matrix) == pytest.approx(0.0)


def test_weighted_minmax_no_weighted_leaves_returns_inf():
    """Edge case: weights map is empty. Min over empty set = +inf
    so the fairness term doesn't influence _evaluate."""
    s = ScheduleSpace(entities=["a"], slots=1)
    helper = weighted_minmax(weights={}, space=s)
    matrix = ((True,),)
    assert helper(matrix) == float("inf")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fairness.py -v
```

Expected: `ModuleNotFoundError: No module named 'pathos.fairness'`.

- [ ] **Step 3: Implement `pathos/fairness.py`**

```python
from __future__ import annotations
from typing import Callable, Hashable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from pathos.spaces.schedule import ScheduleSpace


def weighted_minmax(
    weights: Mapping[Hashable, float],
    space: "ScheduleSpace",
) -> Callable[[tuple[tuple[bool, ...], ...]], float]:
    """Return a fairness callable bound to `space`.

    The returned function takes a (T, N) tuple-of-tuples bool schedule
    (with the matrix layout produced by `ScheduleSpace._to_matrix`) and
    returns

        min over leaves L of  weights[L] * uptime_fraction(L, schedule)

    where uptime_fraction(L, schedule) is the share of slots in which
    the entity feeding L (per `space._downstream`) was on. Higher =
    fairer.

    Leaves with weight 0 are skipped. Missing leaves (not in `weights`)
    are skipped. If no leaf has positive weight, returns +inf so the
    fairness term does not influence selection.
    """
    # Pre-compute leaf -> entity_index for all entities once.
    entity_index = {e: i for i, e in enumerate(space._entities)}
    leaf_to_idx: list[tuple[Hashable, int, float]] = []
    for entity, idx in entity_index.items():
        for leaf in space._downstream(entity):
            w = weights.get(leaf, 0.0)
            if w > 0.0:
                leaf_to_idx.append((leaf, idx, w))

    if not leaf_to_idx:
        def _empty(schedule: tuple[tuple[bool, ...], ...]) -> float:
            return float("inf")
        return _empty

    def _fairness(schedule: tuple[tuple[bool, ...], ...]) -> float:
        T = len(schedule)
        worst = float("inf")
        for _leaf, idx, w in leaf_to_idx:
            on_count = sum(1 for t in range(T) if schedule[t][idx])
            uptime = on_count / T
            term = w * uptime
            if term < worst:
                worst = term
        return worst

    return _fairness
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_fairness.py -v
```

Expected: all 6 passing.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add pathos/fairness.py tests/test_fairness.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(fairness): weighted_minmax helper for ScheduleSpace"
```

---

### Task 8: Wire `slack` into ScheduleSpace solver return + public re-export

`ScheduleSpace.solver()` is inherited from `Space`. Tasks 4–6 set up `_evaluate` and `_successors`, so `space.solver().solve()` already calls `AnytimeLocal` (under `mode="auto"`, which is the `Space` default). The remaining gap: `SearchResult.slack` is never populated, and `result.solution` is a frozenset rather than a matrix. We override `solver()` on `ScheduleSpace` to wrap the result.

We also re-export `ScheduleSpace` at the package top level so `from pathos import ScheduleSpace` works (mirrors `GraphSpace`/`CSPSpace`/etc.).

**Files:**
- Modify: `pathos/spaces/schedule.py`
- Modify: `pathos/__init__.py`
- Modify: `tests/test_schedule_space.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schedule_space.py`:

```python
import dataclasses
from pathos import ScheduleSpace as _PublicScheduleSpace  # re-export check
from pathos.spaces.schedule import ScheduleSpace as _PrivateScheduleSpace


def test_schedule_space_re_exported_at_top_level():
    assert _PublicScheduleSpace is _PrivateScheduleSpace


def test_solver_returns_matrix_solution_and_slack_for_feasible_problem():
    s = ScheduleSpace(entities=["a"], slots=2, penalty=1e3)
    @s.demand
    def _d(e, t): return 1.0
    @s.capacity
    def _c(t): return 5.0   # huge headroom
    @s.fairness
    def _f(schedule): return 0.5
    result = s.solver(timeout=2).solve()
    assert result.found
    # solution is a (T, N) tuple-of-tuples, not a frozenset
    assert isinstance(result.solution, tuple)
    assert len(result.solution) == 2
    assert all(isinstance(row, tuple) for row in result.solution)
    # slack is a per-slot list, one entry per slot
    assert result.slack is not None
    assert len(result.slack) == 2


def test_solver_slack_reflects_capacity_minus_load():
    s = ScheduleSpace(entities=["a"], slots=1, penalty=1e3)
    @s.demand
    def _d(e, t): return 2.0
    @s.capacity
    def _c(t): return 10.0
    @s.fairness
    def _f(schedule):
        # ON => uptime fraction 1; prefer ON
        return float(schedule[0][0])
    result = s.solver(timeout=2).solve()
    assert result.found
    # If the algorithm chose ON: slack = 10 - 2 = 8; if OFF: slack = 10
    assert result.slack[0] in (pytest.approx(8.0), pytest.approx(10.0))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schedule_space.py -v -k "slack or re_export"
```

Expected: import error on the public re-export; the slack-related tests fail because `result.solution` is a frozenset.

- [ ] **Step 3: Override `solver()` on `ScheduleSpace`**

Append to `pathos/spaces/schedule.py`:

```python
    # --- solver factory override: wrap result with matrix solution + slack ---

    def solver(self, candidates=None, timeout=None, mode=None):
        import dataclasses
        inner = super().solver(candidates=candidates, timeout=timeout, mode=mode)
        space = self

        class _ScheduleSolver:
            def __init__(self, inner_solver):
                self._inner = inner_solver
            def solve(self):
                raw = self._inner.solve()
                if not raw.found:
                    return raw
                state = raw.solution
                # Build slack per slot: capacity[t] - load[t]
                slack: list[float] = []
                for t in range(space._slots):
                    cap_t = space._capacity_fn(t)
                    load_t = sum(
                        space._demand_fn(space._entities[e], t)
                        for tt, e in state if tt == t
                    )
                    slack.append(cap_t - load_t)
                matrix = space._to_matrix(state)
                return dataclasses.replace(raw, solution=matrix, slack=slack)

        return _ScheduleSolver(inner)
```

Re-export `ScheduleSpace` from `pathos/__init__.py`:

```python
from pathos.core.space import Space
from pathos.spaces.graph import GraphSpace
from pathos.spaces.csp import CSPSpace
from pathos.spaces.tour import TourSpace
from pathos.spaces.game import GameSpace
from pathos.spaces.schedule import ScheduleSpace
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schedule_space.py -v
```

Expected: every test in the file (~30) passing.

- [ ] **Step 5: Run the full suite to confirm no regressions**

```bash
pytest -q
```

Expected: all prior tests still pass, plus the new ones.

- [ ] **Step 6: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add pathos/spaces/schedule.py pathos/__init__.py tests/test_schedule_space.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(spaces): ScheduleSpace solver — matrix solution + slack, public re-export"
```

---

### Task 9: Worked example + integration smoke test

Translate the spec's worked example into `examples/power_grid.py` (executable, uses stdlib + pathos only, fixed RNG seed). Add an integration test that imports and runs it as a smoke check.

**Files:**
- Create: `examples/power_grid.py`
- Modify: `tests/test_schedule_space.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_schedule_space.py`:

```python
def test_example_power_grid_runs_end_to_end():
    """Smoke: the example runs, picks AnytimeLocal, returns a feasible
    schedule. Locks no exact float values; catches API drift only."""
    import importlib.util
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    example_path = repo_root / "examples" / "power_grid.py"
    assert example_path.exists(), f"Missing example at {example_path}"
    spec = importlib.util.spec_from_file_location("power_grid_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run()
    assert result.found
    assert result.algorithm == "AnytimeLocal"
    # All slots should be feasible (slack >= 0) after the AnytimeLocal cascade.
    assert result.slack is not None
    assert all(s >= -1e-6 for s in result.slack), (
        f"Infeasible schedule returned: slack min = {min(result.slack)}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_schedule_space.py -v -k example_power_grid
```

Expected: `AssertionError: Missing example at .../examples/power_grid.py`.

- [ ] **Step 3: Create `examples/power_grid.py`**

```python
"""Power-grid blackout scheduling — ScheduleSpace worked example.

Run directly:
    python examples/power_grid.py
"""
from __future__ import annotations
import random
from pathos import ScheduleSpace
from pathos.fairness import weighted_minmax


def run():
    rng = random.Random(42)
    n_substations = 20
    n_slots = 168

    substations = [f"sub_{i}" for i in range(n_substations)]
    leaves_per_sub = {
        s: [f"{s}_leaf_{j}" for j in range(rng.randint(2, 4))]
        for s in substations
    }
    all_leaves = [
        leaf for leaves in leaves_per_sub.values() for leaf in leaves
    ]
    # Higher importance = critical (hospital). Lower = tolerant (residential).
    weights = {leaf: rng.choice([1.0, 0.5, 0.1]) for leaf in all_leaves}

    base_load = {
        (s, t): rng.uniform(50, 150)
        for s in substations for t in range(n_slots)
    }
    supply = [
        sum(base_load[s, t] for s in substations) * 0.7
        for t in range(n_slots)
    ]

    space = (
        ScheduleSpace(
            entities=substations,
            slots=n_slots,
            downstream=lambda s: leaves_per_sub[s],
            penalty=1e3,
        )
        .target(tolerance=0.05)
        .mode("auto")
    )

    @space.demand
    def demand(sub, slot):
        return base_load[sub, slot]

    @space.capacity
    def capacity(slot):
        return supply[slot]

    @space.fairness
    def fairness(schedule):
        return weighted_minmax(weights, space)(schedule)

    return space.solver(timeout=5).solve()


if __name__ == "__main__":
    result = run()
    print(f"Algorithm: {result.algorithm}")
    print(f"Found feasible schedule: {result.found}")
    print(f"Objective (-fairness + lambda*violations): {result.cost:.3f}")
    print(f"Mean slack: {sum(result.slack) / len(result.slack):.1f}")
    print(f"Min slack:  {min(result.slack):+.1f}  (negative = capacity overshoot)")
```

- [ ] **Step 4: Run the example manually**

```bash
cd /home/apiad/Workspace/repos/pathos
python examples/power_grid.py
```

Expected: prints `Algorithm: AnytimeLocal`, `Found feasible schedule: True`, finite objective, slack stats. If the slack-min is significantly negative (e.g. `< -10`), the `λ=1e3` penalty is too small for this problem size — escalate to `1e4` in the example and document it.

- [ ] **Step 5: Run the integration test to verify it passes**

```bash
pytest tests/test_schedule_space.py -v -k example_power_grid
```

Expected: pass.

- [ ] **Step 6: Run the full test suite**

```bash
pytest -q
```

Expected: every test in the repo passes.

- [ ] **Step 7: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add examples/power_grid.py tests/test_schedule_space.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(examples): power_grid — end-to-end ScheduleSpace + AnytimeLocal demo"
```

---

### Task 10: Push to origin + spec self-review note

Push all commits. Verify `git status -sb` is clean.

**Files:** none (commit/push only)

- [ ] **Step 1: Push all commits**

```bash
git -C /home/apiad/Workspace/repos/pathos push origin main
```

Expected: 9 new commits land on `gia-uh/pathos@main`.

- [ ] **Step 2: Confirm clean tree**

```bash
git -C /home/apiad/Workspace/repos/pathos status -sb
```

Expected: `## main...origin/main` and no `??`/`M`/`A` lines.

- [ ] **Step 3: Final smoke**

```bash
cd /home/apiad/Workspace/repos/pathos && python -c "
from pathos import ScheduleSpace
from pathos.fairness import weighted_minmax
s = ScheduleSpace(entities=['x'], slots=2)
@s.demand
def _d(e, t): return 1.0
@s.capacity
def _c(t): return 5.0
@s.fairness
def _f(m): return 0.5
r = s.solver(timeout=2).solve()
print('OK:', r.found, r.algorithm)
"
```

Expected: `OK: True AnytimeLocal`.

---

## Self-Review Notes

**Spec coverage check:** every section of the spec maps to at least one task:
- "Constructor" → Task 2
- "Decorators" → Task 3
- "Builder methods (.target, .neighborhood)" → Task 6
- "Auto-emitted capabilities {EVALUATE, SUCCESSORS}" → Tasks 3 + 4
- "Auto-solver behaviour (AnytimeLocal)" → Task 8 (verified by Task 9 integration)
- "SearchResult.slack" → Tasks 1 + 8
- "File structure" → Tasks 1 (result.py), 2 (schedule.py), 7 (fairness.py), 8 (__init__.py), 9 (example)
- "Fairness helper signature" → Task 7
- "Worked example" → Task 9
- "Error handling table" → Tasks 2 (slots/entities), 3 (decorator re-attach), 5 (negative capacity), 6 (tolerance bounds)
- "Testing strategy / unit / property / integration" → Tasks 2–9 (TDD throughout)
- "Risk: λ calibration" → Task 9 step 4 explicitly checks slack feasibility

**Type consistency check:**
- `_demand_fn: Callable[[Hashable, int], float] | None` — same signature in Task 2 + Task 5.
- `_capacity_fn: Callable[[int], float] | None` — same.
- `_fairness_fn: Callable[[tuple[tuple[bool, ...], ...]], float] | None` — same.
- `_initial = frozenset()` (Task 2) → `_successors` yields `frozenset` neighbors (Task 4) → `_evaluate` accepts `frozenset` (Task 5) — consistent.
- `_to_matrix(state) -> tuple[tuple[bool, ...], ...]` — same shape consumed by `weighted_minmax` (Task 7) and returned as `result.solution` (Task 8).
- Capability set: `{EVALUATE, SUCCESSORS}` matches spec, both registered through `_make_hook`-style adds in Tasks 3 and 4.

**Placeholder scan:** No "TBD", "TODO", "implement later", or "similar to Task N". Every step contains the actual code to write or the actual command to run with its expected output.

---

Plan complete and saved to `docs/superpowers/plans/2026-06-09-schedule-space.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
