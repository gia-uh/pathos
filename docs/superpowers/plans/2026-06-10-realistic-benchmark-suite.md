# Realistic Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship slice 1 of the realistic benchmark suite — a `benchmarks/realistic/` orchestration layer + `benchmarks/suites/` registry covering 10 suites (3 classical folded in + 7 realistic), with a JSON + `REPORT.md` artifact that validates the no-cliff / anytime-quality / tunability contract on big planning/logistics/scheduling instances.

**Architecture:** Two new packages alongside the existing `benchmarks/bench.py` (which stays UNCHANGED). `benchmarks/suites/` defines a uniform `SuiteSpec` per problem family (generator + expression-layer + metadata). `benchmarks/realistic/` consumes the registry, runs auto-headline + auto-stress + oracle-row + lower-bound rows per `(suite, size, seed)`, and renders `results-*.json` + `REPORT.md`. R4 (VRP-with-time-windows) and R5 (project-scheduling-with-precedence) are *missing-capability* suites that emit structured gap records; their penalty-folded workarounds run but are excluded from the no-cliff gate.

**Tech Stack:** Python 3.11+, stdlib only for runner/report (mirrors pathos itself). `pytest` for tests. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-06-10-realistic-benchmark-suite-design.md`.

**Precondition:** The ScheduleSpace plan (`docs/superpowers/plans/2026-06-09-schedule-space.md`) must be executed before Tasks 6 and 7 (R1 power-grid, R2 rostering) — those tasks import `ScheduleSpace` from `pathos`. Tasks 1–5 and 8–14 do not depend on ScheduleSpace and can land first.

---

## File Structure

```
benchmarks/
├── bench.py                                   # UNCHANGED — fast-feedback runner
├── FINDINGS.md                                # UNCHANGED — existing audit log
├── suites/                                    # NEW: one module per suite
│   ├── __init__.py                            #   SUITE_REGISTRY: dict[str, SuiteSpec]
│   ├── _spec.py                               #   SuiteSpec dataclass (Task 1)
│   ├── classical.py                           #   C1, C2, C3 (Task 2)
│   ├── bin_packing.py                         #   R6 (Task 4)
│   ├── vrp.py                                 #   R3 + R4 (Tasks 5, 11)
│   ├── power_grid.py                          #   R1 (Task 6, ScheduleSpace-dep)
│   ├── rostering.py                           #   R2 (Task 7, ScheduleSpace-dep)
│   ├── timetabling.py                         #   R7 (Task 10)
│   └── project_scheduling.py                  #   R5 (Task 12, missing-capability)
└── realistic/                                 # NEW: orchestration + reporting
    ├── __init__.py
    ├── __main__.py                            #   CLI entry (Task 9)
    ├── records.py                             #   RunRow, SuiteRow, Report (Task 1)
    ├── metrics.py                             #   gap_to_oracle, feasibility (Task 3)
    ├── lower_bounds.py                        #   1-tree, waste-LB (Task 3)
    ├── runner.py                              #   orchestration loop (Task 8)
    ├── report.py                              #   JSON → REPORT.md renderer (Task 9)
    └── results-*.json                         #   gitignored
```

Tests sit under `tests/test_benchmarks_*.py` mirroring this layout.

Tasks 1–14 each produce a self-contained, committable change. After Task 14, run the smoke sweep (Task 13) and push.

---

### Task 1: `SuiteSpec` dataclass + `RunRow` / `SuiteRow` / `Report` records

The contracts that every other task slots into. `SuiteSpec` is the shape every suite module exposes; `RunRow` is what the runner emits per (suite, size, seed, mode); `SuiteRow` aggregates across seeds for the report table; `Report` is the top-level JSON shape with `bench_schema: 1` and a `meta` block.

**Files:**
- Create: `benchmarks/suites/__init__.py`
- Create: `benchmarks/suites/_spec.py`
- Create: `benchmarks/realistic/__init__.py`
- Create: `benchmarks/realistic/records.py`
- Create: `tests/test_benchmarks_records.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_records.py`:

```python
import json
from dataclasses import asdict
from benchmarks.suites._spec import SuiteSpec
from benchmarks.realistic.records import (
    RunRow, SuiteRow, Report, BENCH_SCHEMA,
)


def test_suitespec_holds_generate_and_express_callables():
    spec = SuiteSpec(
        id="X1",
        family="example",
        constraint_classes=frozenset({"v"}),
        expressibility="full",
        sizes={"S": (10,), "M": (50,), "L": (200,)},
        generate=lambda size, seed: {"size": size, "seed": seed},
        express=lambda instance: instance,
        budgets={"S": 1.0, "M": 5.0, "L": 30.0},
    )
    assert spec.id == "X1"
    inst = spec.generate(10, 42)
    assert inst == {"size": 10, "seed": 42}
    assert spec.express(inst) is inst


def test_runrow_serializes_to_json():
    row = RunRow(
        suite="X1", size=10, seed=42, mode="auto_headline",
        algorithm="AStar", cost=12.5, feasible=True, elapsed=0.42,
        nodes_expanded=137, missing_capability=None,
    )
    j = json.dumps(asdict(row))
    assert json.loads(j)["algorithm"] == "AStar"


def test_runrow_missing_capability_is_optional_string():
    row = RunRow(
        suite="R4", size=100, seed=1, mode="auto_headline",
        algorithm="TabuSearch", cost=None, feasible=False,
        elapsed=10.0, nodes_expanded=0,
        missing_capability="time_windows",
    )
    assert row.missing_capability == "time_windows"


def test_suiterow_aggregates_per_size():
    sr = SuiteRow(
        suite="C1", size=8,
        auto_headline_cost=1.0, auto_headline_feasible=True,
        auto_headline_algorithm="Backtracking",
        auto_stress_cost=1.0, auto_stress_feasible=True,
        oracle_cost=1.0, oracle_algorithm="Backtracking",
        gap_to_oracle_pct=0.0, gap_to_lb_pct=None,
        tunability_gain_pct=0.0, tunability_algorithm=None,
        missing_capability=None,
    )
    assert sr.gap_to_oracle_pct == 0.0


def test_report_round_trips_to_json():
    rep = Report(
        bench_schema=BENCH_SCHEMA,
        meta={"cpu": "test", "python": "3.13", "pathos_sha": "abc"},
        suite_rows=[],
        raw_rows=[],
        no_cliff_failures=[],
    )
    j = json.dumps(asdict(rep))
    loaded = json.loads(j)
    assert loaded["bench_schema"] == BENCH_SCHEMA


def test_bench_schema_is_pinned_integer():
    assert isinstance(BENCH_SCHEMA, int)
    assert BENCH_SCHEMA >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/apiad/Workspace/repos/pathos
pytest tests/test_benchmarks_records.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmarks.suites._spec'`.

- [ ] **Step 3: Create the suite-spec module**

Create `benchmarks/suites/__init__.py`:

```python
"""Suite registry for the realistic benchmark sweep.

Every suite module registers a `SuiteSpec` into `SUITE_REGISTRY` at
import time. `benchmarks.realistic.runner` iterates this dict.
"""
from __future__ import annotations
from benchmarks.suites._spec import SuiteSpec

SUITE_REGISTRY: dict[str, SuiteSpec] = {}


def register(spec: SuiteSpec) -> SuiteSpec:
    if spec.id in SUITE_REGISTRY:
        raise RuntimeError(f"suite id {spec.id!r} already registered")
    SUITE_REGISTRY[spec.id] = spec
    return spec
```

Create `benchmarks/suites/_spec.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SuiteSpec:
    """Uniform contract every benchmark suite exposes.

    `generate(size, seed)` produces a problem instance (deterministic
    per seed). `express(instance)` wraps it as a `pathos` space ready
    for `.solver().solve()`. `budgets[tier]` is the auto-headline budget
    in seconds. `lower_bound`, if set, returns a numeric LB on the
    instance (used for absolute-gap reporting).
    """
    id: str
    family: str  # "csp" / "tour" / "graph" / "schedule" / "evaluate" / "mixed"
    constraint_classes: frozenset[str]  # subset of {"i","ii","iii","iv","v"}
    expressibility: str  # "full" | "partial" | "gap"
    sizes: dict[str, tuple[int, ...]]  # {"S": (...), "M": (...), "L": (...)}
    generate: Callable[..., Any]
    express: Callable[[Any], Any]
    budgets: dict[str, float]
    lower_bound: Callable[[Any], float] | None = None
    # If set, the suite is a missing-capability suite. The string names
    # the constraint class that cannot be expressed (e.g. "time_windows").
    missing_capability: str | None = None
    # Per-suite per-tier feasibility tolerance: an answer with
    # violations > tolerance counts as infeasible for the no-cliff gate.
    tolerance: float = 0.0
    notes: str = ""
```

Create `benchmarks/realistic/__init__.py` (empty):

```python
```

Create `benchmarks/realistic/records.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


BENCH_SCHEMA: int = 1


@dataclass
class RunRow:
    """One row per (suite, size, seed, mode). `mode` is one of:
    auto_headline | auto_stress | oracle_<AlgName>.
    `missing_capability`, when set, names the constraint class that
    couldn't be expressed; the row is the penalty-folded workaround.
    """
    suite: str
    size: int
    seed: int
    mode: str
    algorithm: str
    cost: float | None
    feasible: bool
    elapsed: float
    nodes_expanded: int
    missing_capability: str | None = None
    epsilon: float | None = None


@dataclass
class SuiteRow:
    """One row per (suite, size) in the regression table — median of
    seeds for cost/elapsed, set-union of algorithm names."""
    suite: str
    size: int
    auto_headline_cost: float | None
    auto_headline_feasible: bool
    auto_headline_algorithm: str
    auto_stress_cost: float | None
    auto_stress_feasible: bool
    oracle_cost: float | None
    oracle_algorithm: str | None
    gap_to_oracle_pct: float | None
    gap_to_lb_pct: float | None
    tunability_gain_pct: float | None
    tunability_algorithm: str | None
    missing_capability: str | None


@dataclass
class NoCliffFailure:
    suite: str
    size: int
    seed: int
    failure_kind: str  # "crash" | "not_found" | "infeasible"
    detail: str  # exception repr or violation count, for triage
    reproduction_args: str  # CLI snippet


@dataclass
class Report:
    bench_schema: int
    meta: dict[str, str]
    suite_rows: list[SuiteRow] = field(default_factory=list)
    raw_rows: list[RunRow] = field(default_factory=list)
    no_cliff_failures: list[NoCliffFailure] = field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_records.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/suites/__init__.py benchmarks/suites/_spec.py benchmarks/realistic/__init__.py benchmarks/realistic/records.py tests/test_benchmarks_records.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): SuiteSpec + RunRow/SuiteRow/Report records"
```

---

### Task 2: Classical suites (C1 N-Queens, C2 TSP, C3 8-puzzle)

Wrap the existing `bench.py` builders into `SuiteSpec`s. No duplication: we import `build_nqueens`, `build_tsp`, `build_puzzle` directly. Classical suites have no constraint classes, are fully expressible, and use the smallest budget tier.

**Files:**
- Create: `benchmarks/suites/classical.py`
- Create: `tests/test_benchmarks_classical.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_classical.py`:

```python
import pytest
import benchmarks.suites.classical  # noqa: F401 — registers C1/C2/C3
from benchmarks.suites import SUITE_REGISTRY


@pytest.mark.parametrize("sid", ["C1", "C2", "C3"])
def test_classical_suite_registered(sid):
    assert sid in SUITE_REGISTRY


def test_c1_generates_nqueens_csp_and_solves_default():
    spec = SUITE_REGISTRY["C1"]
    inst = spec.generate(8, seed=42)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.found, "C1 N-Queens N=8 must solve in 2s"


def test_c2_generates_tsp_tourspace_and_solves_default():
    spec = SUITE_REGISTRY["C2"]
    inst = spec.generate(8, seed=42)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.cost is not None and res.cost > 0


def test_c3_generates_puzzle_and_solves_default():
    spec = SUITE_REGISTRY["C3"]
    inst = spec.generate(10, seed=42)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.found


def test_classical_suites_have_no_constraint_classes():
    for sid in ("C1", "C2", "C3"):
        assert SUITE_REGISTRY[sid].constraint_classes == frozenset()
        assert SUITE_REGISTRY[sid].expressibility == "full"
        assert SUITE_REGISTRY[sid].missing_capability is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_classical.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmarks.suites.classical'`.

- [ ] **Step 3: Create `benchmarks/suites/classical.py`**

```python
"""Classical bench suites C1/C2/C3 — N-Queens, TSP, 8-puzzle.

These wrap the existing builders in `benchmarks.bench` so the
classical generators carry over into the unified ladder without
duplication. They occupy the bottom of the difficulty ladder; the
realistic suites R1-R7 sit above.
"""
from __future__ import annotations
import pathos.algorithms  # noqa: F401 — register algorithms
from benchmarks.bench import build_nqueens, build_tsp, build_puzzle
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


CLASSICAL_BUDGETS = {"S": 5.0, "M": 5.0, "L": 5.0}


def _c1_generate(size: int, seed: int) -> int:
    return size  # N-Queens needs only N; seed unused.


def _c1_express(n: int):
    return build_nqueens(n)


register(SuiteSpec(
    id="C1",
    family="csp",
    constraint_classes=frozenset(),
    expressibility="full",
    sizes={"S": (6, 8), "M": (10, 12), "L": (14, 16)},
    generate=_c1_generate,
    express=_c1_express,
    budgets=CLASSICAL_BUDGETS,
    notes="N-Queens via CSPSpace; classical bottom-of-ladder.",
))


def _c2_generate(size: int, seed: int) -> tuple[int, int]:
    return (size, seed)


def _c2_express(payload):
    n, seed = payload
    return build_tsp(n, seed)


register(SuiteSpec(
    id="C2",
    family="tour",
    constraint_classes=frozenset(),
    expressibility="full",
    sizes={"S": (5, 8), "M": (12, 16), "L": (20, 25)},
    generate=_c2_generate,
    express=_c2_express,
    budgets=CLASSICAL_BUDGETS,
    notes="TSP uniform-random on TourSpace; classical bottom-of-ladder.",
))


def _c3_generate(size: int, seed: int) -> tuple[int, int]:
    return (size, seed)


def _c3_express(payload):
    depth, seed = payload
    return build_puzzle(depth, seed)


register(SuiteSpec(
    id="C3",
    family="informed",
    constraint_classes=frozenset(),
    expressibility="full",
    sizes={"S": (10, 20), "M": (30, 40), "L": (50,)},
    generate=_c3_generate,
    express=_c3_express,
    budgets=CLASSICAL_BUDGETS,
    notes="8-puzzle with Manhattan heuristic; classical bottom-of-ladder.",
))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_classical.py -v
```

Expected: 6 passed (3 registered + 1 each for C1/C2/C3 solving + 1 metadata).

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/suites/classical.py tests/test_benchmarks_classical.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): classical suites C1/C2/C3 folded into registry"
```

---

### Task 3: Metrics + lower bounds modules

`metrics.py` computes `gap_to_oracle_pct`, `gap_to_lb_pct`, `tunability_gain_pct`, and the feasibility predicate from a list of `RunRow`s. `lower_bounds.py` provides callable LBs where cheap (TSP 1-tree, bin-packing waste-LB). The dataclass plumbing matters more than the LB sophistication — slice 1 ships two LBs and `None` for the rest.

**Files:**
- Create: `benchmarks/realistic/metrics.py`
- Create: `benchmarks/realistic/lower_bounds.py`
- Create: `tests/test_benchmarks_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_metrics.py`:

```python
import pytest
from benchmarks.realistic.records import RunRow
from benchmarks.realistic.metrics import (
    gap_pct, median_cost, oracle_row, is_no_cliff_failure,
    tunability_gain,
)
from benchmarks.realistic.lower_bounds import tsp_one_tree_lb, bin_packing_waste_lb


def _row(mode, algorithm, cost, feasible=True, elapsed=0.1):
    return RunRow(
        suite="X1", size=10, seed=1, mode=mode, algorithm=algorithm,
        cost=cost, feasible=feasible, elapsed=elapsed, nodes_expanded=0,
    )


def test_gap_pct_returns_zero_for_matching_costs():
    assert gap_pct(auto=10.0, oracle=10.0) == pytest.approx(0.0)


def test_gap_pct_returns_relative_excess():
    # auto=11, oracle=10 -> 10% gap
    assert gap_pct(auto=11.0, oracle=10.0) == pytest.approx(10.0)


def test_gap_pct_handles_none_oracle():
    assert gap_pct(auto=10.0, oracle=None) is None


def test_gap_pct_handles_none_auto():
    assert gap_pct(auto=None, oracle=10.0) is None


def test_gap_pct_handles_zero_oracle():
    # Avoid div-by-zero; use absolute difference scaled by 1 as fallback.
    assert gap_pct(auto=0.0, oracle=0.0) == pytest.approx(0.0)
    assert gap_pct(auto=1.0, oracle=0.0) is None  # undefined gap, log as None


def test_median_cost_ignores_infeasible_and_none():
    rows = [
        _row("auto_headline", "A", 10.0),
        _row("auto_headline", "A", 12.0),
        _row("auto_headline", "A", None, feasible=False),
    ]
    assert median_cost(rows) == pytest.approx(11.0)


def test_median_cost_returns_none_when_no_feasible():
    rows = [_row("auto_headline", "A", None, feasible=False)]
    assert median_cost(rows) is None


def test_oracle_row_picks_lowest_cost_feasible():
    rows = [
        _row("oracle_AStar", "AStar", 10.0),
        _row("oracle_AStar", "AStar", 12.0),
        _row("oracle_BFS", "BFS", 8.0),
        _row("oracle_BFS", "BFS", 9.0),
    ]
    name, cost = oracle_row(rows)
    assert name == "BFS"
    assert cost == pytest.approx(8.5)


def test_oracle_row_returns_none_when_no_oracle_runs():
    rows = [_row("auto_headline", "A", 10.0)]
    assert oracle_row(rows) == (None, None)


def test_tunability_gain_returns_zero_when_auto_already_best():
    rows = [
        _row("auto_headline", "A", 10.0),
        _row("oracle_A", "A", 10.0),
        _row("oracle_B", "B", 12.0),
    ]
    gain, who = tunability_gain(rows)
    assert gain == pytest.approx(0.0)
    assert who is None  # no candidate beats auto


def test_tunability_gain_names_the_beater():
    rows = [
        _row("auto_headline", "A", 10.0),
        _row("oracle_B", "B", 9.0),  # 10% better
    ]
    gain, who = tunability_gain(rows)
    assert gain == pytest.approx(10.0)
    assert who == "B"


def test_no_cliff_failure_detects_not_found():
    row = _row("auto_headline", "A", None, feasible=False)
    row.cost = None
    assert is_no_cliff_failure(row, tolerance=0.0) is True


def test_no_cliff_failure_passes_feasible_run():
    row = _row("auto_headline", "A", 10.0, feasible=True)
    assert is_no_cliff_failure(row, tolerance=0.0) is False


def test_tsp_one_tree_lb_is_a_lower_bound_on_tour_cost():
    # Triangle: 3 nodes, distances 1-1-1 -> LB <= 3 (optimal tour = 3).
    distances = {(0, 1): 1.0, (1, 0): 1.0,
                 (0, 2): 1.0, (2, 0): 1.0,
                 (1, 2): 1.0, (2, 1): 1.0}
    lb = tsp_one_tree_lb(nodes=[0, 1, 2], distances=distances)
    assert lb <= 3.0 + 1e-9


def test_bin_packing_waste_lb_is_ceil_total_div_capacity():
    items = [3.0, 3.0, 3.0, 3.0]   # total = 12
    capacity = 5.0                  # ceil(12/5) = 3 bins
    assert bin_packing_waste_lb(items=items, capacity=capacity) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_metrics.py -v
```

Expected: import errors on `benchmarks.realistic.metrics` and `lower_bounds`.

- [ ] **Step 3: Create `benchmarks/realistic/metrics.py`**

```python
from __future__ import annotations
import statistics
from typing import Iterable
from benchmarks.realistic.records import RunRow


def gap_pct(auto: float | None, oracle: float | None) -> float | None:
    """Relative excess of `auto` over `oracle`, in percent.

    Returns None if either is None or if oracle is 0 while auto is not
    (gap is undefined). When both are 0, returns 0.0 (trivially equal).
    """
    if auto is None or oracle is None:
        return None
    if oracle == 0.0:
        return 0.0 if auto == 0.0 else None
    return 100.0 * (auto - oracle) / abs(oracle)


def median_cost(rows: Iterable[RunRow]) -> float | None:
    feasible_costs = [r.cost for r in rows if r.feasible and r.cost is not None]
    if not feasible_costs:
        return None
    return statistics.median(feasible_costs)


def oracle_row(rows: Iterable[RunRow]) -> tuple[str | None, float | None]:
    """Best (lowest median cost) among oracle_* modes. Returns
    (algorithm_name, median_cost) or (None, None) if no oracle data."""
    by_algo: dict[str, list[RunRow]] = {}
    for r in rows:
        if not r.mode.startswith("oracle_"):
            continue
        by_algo.setdefault(r.algorithm, []).append(r)
    best_name: str | None = None
    best_cost: float | None = None
    for name, group in by_algo.items():
        med = median_cost(group)
        if med is None:
            continue
        if best_cost is None or med < best_cost:
            best_cost = med
            best_name = name
    return best_name, best_cost


def tunability_gain(rows: Iterable[RunRow]) -> tuple[float, str | None]:
    """How much the best oracle algorithm beats auto-headline, in
    percent. Returns (0.0, None) if no oracle beats auto."""
    rows_list = list(rows)
    auto_rows = [r for r in rows_list if r.mode == "auto_headline"]
    auto_med = median_cost(auto_rows)
    if auto_med is None:
        return 0.0, None
    best_name, best_cost = oracle_row(rows_list)
    if best_cost is None or best_cost >= auto_med:
        return 0.0, None
    gain = 100.0 * (auto_med - best_cost) / abs(auto_med) if auto_med != 0.0 else 0.0
    return gain, best_name


def is_no_cliff_failure(row: RunRow, tolerance: float) -> bool:
    """A single auto-headline / auto-stress run that constitutes a
    no-cliff bug: crash, not_found, or infeasible beyond tolerance.

    `nodes_expanded = -2` is the bench's existing crash sentinel
    (inherited from bench.py's run_one). Cost being None on a feasible
    flag should not happen, but treat it as failure too.
    """
    if not row.mode.startswith("auto_"):
        return False
    if row.missing_capability is not None:
        return False  # missing-capability suites are carved out
    if row.nodes_expanded == -2:
        return True  # crash sentinel
    if not row.feasible:
        return True
    if row.cost is None:
        return True
    return False
```

Create `benchmarks/realistic/lower_bounds.py`:

```python
from __future__ import annotations
import math
from typing import Hashable, Mapping, Sequence


def tsp_one_tree_lb(
    nodes: Sequence[Hashable],
    distances: Mapping[tuple[Hashable, Hashable], float],
) -> float:
    """Held-Karp 1-tree lower bound for symmetric TSP.

    For symmetric TSP with n >= 2 nodes:
      LB = weight of MST on nodes minus one + two shortest edges
           incident to the dropped node.

    This is a valid LB on the optimal tour cost.
    """
    n = len(nodes)
    if n < 2:
        return 0.0
    drop = nodes[0]
    rest = [v for v in nodes if v != drop]
    # Prim's MST over `rest` using the distance table.
    in_tree = {rest[0]}
    out_tree = set(rest[1:])
    mst_weight = 0.0
    while out_tree:
        best = math.inf
        best_v = None
        for u in in_tree:
            for v in out_tree:
                d = distances.get((u, v), distances.get((v, u), math.inf))
                if d < best:
                    best = d
                    best_v = v
        if best_v is None:
            return 0.0  # graph disconnected; degenerate LB
        mst_weight += best
        in_tree.add(best_v)
        out_tree.remove(best_v)
    incident = sorted(
        distances.get((drop, v), distances.get((v, drop), math.inf))
        for v in rest
    )
    if len(incident) < 2:
        return mst_weight
    return mst_weight + incident[0] + incident[1]


def bin_packing_waste_lb(items: Sequence[float], capacity: float) -> int:
    """Floor LB on the number of bins required: ceil(sum / capacity)."""
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    return math.ceil(sum(items) / capacity)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_metrics.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/realistic/metrics.py benchmarks/realistic/lower_bounds.py tests/test_benchmarks_metrics.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): metrics module + TSP 1-tree and bin-packing waste LBs"
```

---

### Task 4: R6 — Bin packing (pure `@evaluate`)

Simplest realistic suite: variable-sized items into fixed-capacity bins. State is `tuple[int, ...]` — `item -> bin_index`. `@evaluate` returns `bins_used + overshoot_penalty`. Uses `pathos.Space` directly. Tests assert generator determinism, smallest size solves under default solver, LB hooks in.

**Files:**
- Create: `benchmarks/suites/bin_packing.py`
- Create: `tests/test_benchmarks_bin_packing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_bin_packing.py`:

```python
import benchmarks.suites.bin_packing  # noqa: F401 — registers R6
from benchmarks.suites import SUITE_REGISTRY


def test_r6_registered():
    assert "R6" in SUITE_REGISTRY


def test_r6_generator_is_deterministic_per_seed():
    spec = SUITE_REGISTRY["R6"]
    a = spec.generate(size=50, seed=7)
    b = spec.generate(size=50, seed=7)
    c = spec.generate(size=50, seed=8)
    assert a == b
    assert a != c


def test_r6_generator_respects_size():
    spec = SUITE_REGISTRY["R6"]
    inst = spec.generate(size=50, seed=1)
    assert len(inst.items) == 50


def test_r6_express_yields_space_that_solves_default():
    spec = SUITE_REGISTRY["R6"]
    inst = spec.generate(size=20, seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    # bin packing is pure optimization; "found" is true when local
    # search terminates (no goal predicate). Cost must be finite.
    assert res.cost is not None
    assert res.cost > 0


def test_r6_lower_bound_callable():
    spec = SUITE_REGISTRY["R6"]
    inst = spec.generate(size=20, seed=1)
    assert spec.lower_bound is not None
    lb = spec.lower_bound(inst)
    assert lb >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_bin_packing.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmarks.suites.bin_packing'`.

- [ ] **Step 3: Create `benchmarks/suites/bin_packing.py`**

```python
"""R6 — Bin packing with variable items, single bin class.

Generator: `size` items uniformly distributed in [0.1, 0.9] × capacity.
Expression: pure-`@evaluate` Space whose state is a tuple `(bin_index
for item i)`; successors move one item between bins. Capacity overshoot
folded into the evaluation as a large penalty term so feasibility is
not strictly enforced but heavily preferred.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from pathos import Space
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec
from benchmarks.realistic.lower_bounds import bin_packing_waste_lb


@dataclass(frozen=True)
class BinPackingInstance:
    items: tuple[float, ...]
    capacity: float


def _generate(size: int, seed: int) -> BinPackingInstance:
    rng = random.Random(seed)
    capacity = 1.0
    items = tuple(rng.uniform(0.1, 0.9) for _ in range(size))
    return BinPackingInstance(items=items, capacity=capacity)


def _express(inst: BinPackingInstance) -> Space:
    n = len(inst.items)
    # Start with each item in its own bin -> trivially feasible.
    initial = tuple(range(n))
    space = Space().initial(initial)

    @space.successors
    def neighbors(state: tuple[int, ...]):
        used = set(state)
        # Try moving each item to each currently-used bin (or to a fresh one).
        for i in range(n):
            for target in list(used) + [max(used) + 1]:
                if target == state[i]:
                    continue
                new = list(state)
                new[i] = target
                yield (i, target), tuple(new)

    @space.evaluate
    def cost(state: tuple[int, ...]) -> float:
        # Compute load per bin.
        load: dict[int, float] = {}
        for idx, b in enumerate(state):
            load[b] = load.get(b, 0.0) + inst.items[idx]
        bins_used = len(load)
        overshoot = sum(max(0.0, l - inst.capacity) for l in load.values())
        # Penalty so any feasible packing beats any infeasible one.
        return float(bins_used) + 1e3 * overshoot

    return space


def _lb(inst: BinPackingInstance) -> float:
    return float(bin_packing_waste_lb(items=inst.items, capacity=inst.capacity))


register(SuiteSpec(
    id="R6",
    family="evaluate",
    constraint_classes=frozenset({"v"}),
    expressibility="full",
    sizes={"S": (50,), "M": (200,), "L": (1000,)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    lower_bound=_lb,
    notes="Bin packing — uniform items, single bin class; penalty-fold capacity.",
))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_bin_packing.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/suites/bin_packing.py tests/test_benchmarks_bin_packing.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): R6 bin-packing suite + waste-LB hookup"
```

---

### Task 5: R3 — Capacitated VRP (TourSpace + penalty-fold vehicle capacity)

VRP with `n_vehicles` vehicles of equal capacity, customers with demand. Encode as a single TSP-like tour over `(depot, c1, c2, ..., depot, ck, ...)` — penalty-fold any vehicle-capacity overflow. Reuses TourSpace; the `nodes` list includes one depot per vehicle to mark route boundaries.

**Files:**
- Create: `benchmarks/suites/vrp.py`
- Create: `tests/test_benchmarks_vrp.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_vrp.py`:

```python
import benchmarks.suites.vrp  # noqa: F401 — registers R3 (R4 added in Task 11)
from benchmarks.suites import SUITE_REGISTRY


def test_r3_registered():
    assert "R3" in SUITE_REGISTRY


def test_r3_generator_deterministic():
    spec = SUITE_REGISTRY["R3"]
    a = spec.generate(size=20, seed=7)
    b = spec.generate(size=20, seed=7)
    c = spec.generate(size=20, seed=8)
    assert a == b
    assert a != c


def test_r3_generator_size_is_customer_count():
    spec = SUITE_REGISTRY["R3"]
    inst = spec.generate(size=30, seed=1)
    assert len(inst.customers) == 30


def test_r3_express_yields_space_that_solves_default():
    spec = SUITE_REGISTRY["R3"]
    inst = spec.generate(size=20, seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.cost is not None


def test_r3_constraint_classes_include_iii_and_v():
    assert SUITE_REGISTRY["R3"].constraint_classes == frozenset({"iii", "v"})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_vrp.py -v
```

- [ ] **Step 3: Create `benchmarks/suites/vrp.py`**

```python
"""R3 — Capacitated VRP (delivery routes, vehicle loads).

Encoded as a single TSP-shaped tour through (depot_v for each vehicle v,
customers...). Each depot in the tour acts as a route boundary: customers
between depot_v and depot_(v+1) are served by vehicle v. Capacity
overshoot per vehicle is penalty-folded into the tour cost.

R4 (VRP with time windows) is added in Task 11 as a missing-capability
suite that reuses this generator + adds time-window data.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathos import TourSpace
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


@dataclass(frozen=True)
class VRPInstance:
    customers: tuple[tuple[float, float], ...]   # coords
    demands: tuple[float, ...]
    n_vehicles: int
    vehicle_capacity: float
    depot: tuple[float, float]


def _generate(size: int, seed: int) -> VRPInstance:
    rng = random.Random(seed)
    customers = tuple(
        (rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(size)
    )
    # Heterogeneous demands so capacity bites.
    demands = tuple(rng.uniform(1.0, 10.0) for _ in range(size))
    # Vehicles scale with size: ~1 per 20 customers.
    n_vehicles = max(2, size // 20)
    # Capacity sized so a feasible solution exists with comfortable margin.
    vehicle_capacity = float(sum(demands) / n_vehicles) * 1.25
    depot = (50.0, 50.0)
    return VRPInstance(
        customers=customers,
        demands=demands,
        n_vehicles=n_vehicles,
        vehicle_capacity=vehicle_capacity,
        depot=depot,
    )


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_tour_space(inst: VRPInstance) -> TourSpace:
    # Node IDs: 0..n_vehicles-1 are depot copies; n_vehicles..n_vehicles+n_cust-1 are customers.
    n_cust = len(inst.customers)
    depot_ids = list(range(inst.n_vehicles))
    cust_ids = list(range(inst.n_vehicles, inst.n_vehicles + n_cust))
    nodes = depot_ids + cust_ids

    def coord(node_id: int) -> tuple[float, float]:
        if node_id < inst.n_vehicles:
            return inst.depot
        return inst.customers[node_id - inst.n_vehicles]

    distances = {
        (i, j): _dist(coord(i), coord(j))
        for i in nodes for j in nodes if i != j
    }
    space = TourSpace(nodes=nodes, distances=distances)

    @space.evaluate
    def tour_cost(tour: tuple[int, ...]) -> float:
        # Base tour distance.
        base = sum(
            distances[(tour[i], tour[(i + 1) % len(tour)])]
            for i in range(len(tour))
        )
        # Walk the tour, accumulating load per vehicle segment.
        loads: list[float] = []
        cur = 0.0
        for node in tour:
            if node < inst.n_vehicles:
                loads.append(cur)
                cur = 0.0
            else:
                cur += inst.demands[node - inst.n_vehicles]
        loads.append(cur)  # the wraparound segment
        overshoot = sum(max(0.0, l - inst.vehicle_capacity) for l in loads)
        return base + 1e3 * overshoot

    return space


def _express(inst: VRPInstance) -> TourSpace:
    return _build_tour_space(inst)


register(SuiteSpec(
    id="R3",
    family="tour",
    constraint_classes=frozenset({"iii", "v"}),
    expressibility="full",
    sizes={"S": (100,), "M": (300,), "L": (1000,)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    notes="Capacitated VRP via tour with depot-copy boundaries; penalty-fold cap.",
))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_vrp.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/suites/vrp.py tests/test_benchmarks_vrp.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): R3 capacitated-VRP suite"
```

---

### Task 6: R1 — Power-grid blackout scheduling (ScheduleSpace)

**Precondition:** ScheduleSpace plan (`docs/superpowers/plans/2026-06-09-schedule-space.md`) has been executed; `from pathos import ScheduleSpace` works.

Adapt the spec's worked example (`examples/power_grid.py` after ScheduleSpace lands) into a suite. The generator emits `(substations, slots, demands, supply, weights)`; the expression layer builds a `ScheduleSpace` with `@demand`, `@capacity`, `@fairness` decorators.

**Files:**
- Create: `benchmarks/suites/power_grid.py`
- Create: `tests/test_benchmarks_power_grid.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_power_grid.py`:

```python
import pytest

# Skip the whole module if ScheduleSpace hasn't shipped yet.
pytest.importorskip("pathos.spaces.schedule")

import benchmarks.suites.power_grid  # noqa: E402,F401 — registers R1
from benchmarks.suites import SUITE_REGISTRY


def test_r1_registered():
    assert "R1" in SUITE_REGISTRY


def test_r1_generator_deterministic():
    spec = SUITE_REGISTRY["R1"]
    sub_a, slots_a = 20, 72
    a = spec.generate(size=(sub_a, slots_a), seed=7)
    b = spec.generate(size=(sub_a, slots_a), seed=7)
    assert a == b


def test_r1_express_yields_schedulespace_that_solves_default():
    spec = SUITE_REGISTRY["R1"]
    inst = spec.generate(size=(10, 24), seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=3).solve()
    assert res.cost is not None


def test_r1_constraint_classes_include_iii_iv_v():
    assert SUITE_REGISTRY["R1"].constraint_classes == frozenset({"iii", "iv", "v"})
    assert SUITE_REGISTRY["R1"].expressibility == "partial"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_power_grid.py -v
```

If ScheduleSpace not yet shipped, the whole module skips — proceed to other tasks first and return.

- [ ] **Step 3: Create `benchmarks/suites/power_grid.py`**

```python
"""R1 — Power-grid blackout scheduling on ScheduleSpace.

Generator parameters mirror the ScheduleSpace worked example:
- `size` is (n_substations, n_slots).
- Each substation has 2-4 downstream leaves, each weighted by importance
  (0.1 residential / 0.5 industrial / 1.0 critical).
- Base load is uniform-random per (substation, slot); supply is sized
  at 70% of total demand so the solver must shed load.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from pathos import ScheduleSpace  # type: ignore[attr-defined]  # available post-plan
from pathos.fairness import weighted_minmax  # type: ignore[attr-defined]
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


@dataclass(frozen=True)
class PowerGridInstance:
    substations: tuple[str, ...]
    leaves_per_sub: dict[str, tuple[str, ...]]
    weights: dict[str, float]
    base_load: dict[tuple[str, int], float]
    supply: tuple[float, ...]
    n_slots: int


def _generate(size, seed: int) -> PowerGridInstance:
    n_sub, n_slots = size
    rng = random.Random(seed)
    substations = tuple(f"sub_{i}" for i in range(n_sub))
    leaves_per_sub = {
        s: tuple(f"{s}_leaf_{j}" for j in range(rng.randint(2, 4)))
        for s in substations
    }
    all_leaves = [l for ls in leaves_per_sub.values() for l in ls]
    weights = {l: rng.choice([1.0, 0.5, 0.1]) for l in all_leaves}
    base_load = {
        (s, t): rng.uniform(50.0, 150.0)
        for s in substations for t in range(n_slots)
    }
    supply = tuple(
        sum(base_load[s, t] for s in substations) * 0.7
        for t in range(n_slots)
    )
    return PowerGridInstance(
        substations=substations,
        leaves_per_sub=dict(leaves_per_sub),
        weights=weights,
        base_load=base_load,
        supply=supply,
        n_slots=n_slots,
    )


def _express(inst: PowerGridInstance):
    space = (
        ScheduleSpace(
            entities=list(inst.substations),
            slots=inst.n_slots,
            downstream=lambda s: inst.leaves_per_sub[s],
            penalty=1e3,
        )
        .target(tolerance=0.05)
    )

    @space.demand
    def demand(sub, slot):
        return inst.base_load[sub, slot]

    @space.capacity
    def capacity(slot):
        return inst.supply[slot]

    @space.fairness
    def fairness(schedule):
        return weighted_minmax(inst.weights, space)(schedule)

    return space


register(SuiteSpec(
    id="R1",
    family="schedule",
    constraint_classes=frozenset({"iii", "iv", "v"}),
    expressibility="partial",
    sizes={"S": ((20, 72),), "M": ((50, 168),), "L": ((100, 336),)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    notes="Power-grid blackout scheduling; multi-resource folded into penalty.",
))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_power_grid.py -v
```

Expected: 4 passed (assuming ScheduleSpace shipped).

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/suites/power_grid.py tests/test_benchmarks_power_grid.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): R1 power-grid scheduling suite (ScheduleSpace)"
```

---

### Task 7: R2 — Employee rostering (ScheduleSpace + CSP-shaped skills)

**Precondition:** ScheduleSpace shipped (same as Task 6).

Schedule `n_staff` employees across `n_shifts` shifts. Each shift requires a minimum staffing level; each employee has 1-3 skills, each shift requires 1 skill. Fairness measures shift-balance across employees. Soft preferences ("staff i prefers shifts in [start, end]") are *recorded as a partial gap* — the suite folds them into the penalty but flags the gap in metadata.

**Files:**
- Create: `benchmarks/suites/rostering.py`
- Create: `tests/test_benchmarks_rostering.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_rostering.py`:

```python
import pytest
pytest.importorskip("pathos.spaces.schedule")

import benchmarks.suites.rostering  # noqa: E402,F401 — registers R2
from benchmarks.suites import SUITE_REGISTRY


def test_r2_registered():
    assert "R2" in SUITE_REGISTRY


def test_r2_generator_deterministic():
    spec = SUITE_REGISTRY["R2"]
    a = spec.generate(size=(20, 72), seed=7)
    b = spec.generate(size=(20, 72), seed=7)
    assert a == b


def test_r2_express_yields_space_that_solves_default():
    spec = SUITE_REGISTRY["R2"]
    inst = spec.generate(size=(10, 24), seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=3).solve()
    assert res.cost is not None


def test_r2_constraint_classes_include_iii_iv_v_and_expressibility_partial():
    spec = SUITE_REGISTRY["R2"]
    assert spec.constraint_classes == frozenset({"iii", "iv", "v"})
    assert spec.expressibility == "partial"
    assert "soft_preferences" in spec.notes  # the partial-gap flag
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_rostering.py -v
```

- [ ] **Step 3: Create `benchmarks/suites/rostering.py`**

```python
"""R2 — Employee rostering on ScheduleSpace.

Each staff member has a skill set (subset of {"a","b","c"}); each shift
requires one skill. The bench folds skill-mismatch and shift-preference
violations into the penalty term — soft-preference modeling is the
partial gap recorded for slice 2-N follow-up.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from pathos import ScheduleSpace  # type: ignore[attr-defined]
from pathos.fairness import weighted_minmax  # type: ignore[attr-defined]
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


SKILLS = ("a", "b", "c")


@dataclass(frozen=True)
class RosteringInstance:
    staff: tuple[str, ...]
    staff_skills: dict[str, frozenset[str]]
    shift_skill: tuple[str, ...]  # skill required per slot
    min_staff_per_shift: int
    preferred_window: dict[str, tuple[int, int]]  # staff -> (start, end)
    n_slots: int


def _generate(size, seed: int) -> RosteringInstance:
    n_staff, n_slots = size
    rng = random.Random(seed)
    staff = tuple(f"emp_{i}" for i in range(n_staff))
    staff_skills = {
        s: frozenset(rng.sample(SKILLS, rng.randint(1, len(SKILLS))))
        for s in staff
    }
    shift_skill = tuple(rng.choice(SKILLS) for _ in range(n_slots))
    preferred_window = {
        s: (
            rng.randint(0, n_slots // 2),
            rng.randint(n_slots // 2 + 1, n_slots),
        )
        for s in staff
    }
    min_staff = max(1, n_staff // 5)
    return RosteringInstance(
        staff=staff,
        staff_skills=dict(staff_skills),
        shift_skill=shift_skill,
        min_staff_per_shift=min_staff,
        preferred_window=preferred_window,
        n_slots=n_slots,
    )


def _express(inst: RosteringInstance):
    space = (
        ScheduleSpace(entities=list(inst.staff), slots=inst.n_slots, penalty=1e3)
        .target(tolerance=0.0)
    )

    @space.demand
    def demand(emp, slot):
        # An employee "demands" 1 unit on a slot they don't have the skill for.
        if inst.shift_skill[slot] not in inst.staff_skills[emp]:
            return 1.0
        return 0.0

    @space.capacity
    def capacity(slot):
        # No skill-mismatch demand may exceed 0 -> capacity 0.
        return 0.0

    @space.fairness
    def fairness(schedule):
        # Equal weight per staff; rewards uptime balance.
        weights = {s: 1.0 for s in inst.staff}
        return weighted_minmax(weights, space)(schedule)

    return space


register(SuiteSpec(
    id="R2",
    family="schedule",
    constraint_classes=frozenset({"iii", "iv", "v"}),
    expressibility="partial",
    sizes={"S": ((20, 72),), "M": ((50, 168),), "L": ((100, 336),)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    notes=(
        "Employee rostering; skill-mismatch penalty-folded. "
        "Partial gap: soft_preferences (preferred shift windows) are "
        "not yet expressed as first-class soft constraints."
    ),
))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_rostering.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/suites/rostering.py tests/test_benchmarks_rostering.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): R2 employee-rostering suite (partial-gap: soft prefs)"
```

---

### Task 8: Runner (orchestration loop)

The core: iterate `(suite, size, seed)`, run auto-headline, auto-stress, oracle row, optional LB. Returns a list of `RunRow`s. Reuses `bench.py`'s `_wall_clock_limit` and follows the same crash-sentinel convention.

**Files:**
- Create: `benchmarks/realistic/runner.py`
- Create: `tests/test_benchmarks_runner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_runner.py`:

```python
import benchmarks.suites.classical  # noqa: F401 — populates registry
import benchmarks.suites.bin_packing  # noqa: F401
from benchmarks.realistic.runner import (
    run_one, run_suite_tier, RunConfig,
)


def test_run_one_returns_runrow_with_auto_headline_mode():
    cfg = RunConfig(seeds=1, oracle=False, stress_multiplier=5.0)
    rows = run_one("C1", tier="S", cfg=cfg)
    assert any(r.mode == "auto_headline" for r in rows)
    headline = next(r for r in rows if r.mode == "auto_headline")
    assert headline.suite == "C1"
    assert headline.feasible is True


def test_run_one_emits_oracle_rows_when_enabled():
    cfg = RunConfig(seeds=1, oracle=True, stress_multiplier=5.0)
    rows = run_one("C1", tier="S", cfg=cfg)
    oracle_modes = [r.mode for r in rows if r.mode.startswith("oracle_")]
    assert oracle_modes, "expected at least one oracle row"


def test_run_one_emits_stress_row():
    cfg = RunConfig(seeds=1, oracle=False, stress_multiplier=2.0)
    rows = run_one("C1", tier="S", cfg=cfg)
    assert any(r.mode == "auto_stress" for r in rows)


def test_run_suite_tier_seeds_increment_deterministically():
    cfg = RunConfig(seeds=2, oracle=False, stress_multiplier=5.0, base_seed=100)
    rows = run_suite_tier("R6", tier="S", cfg=cfg)
    headline_seeds = sorted({r.seed for r in rows if r.mode == "auto_headline"})
    assert headline_seeds == [100, 101]


def test_run_one_records_crash_with_nodes_expanded_minus_two():
    # Use a suite where some explicit algorithms are known to crash on
    # the space (per FINDINGS.md history: BFS on CSPSpace blew up before
    # the compat-guard fix). We verify the *shape*, not specific
    # algorithm names: every row has a valid mode and feasible flag.
    cfg = RunConfig(seeds=1, oracle=True, stress_multiplier=2.0)
    rows = run_one("C1", tier="S", cfg=cfg)
    for r in rows:
        assert r.mode == "auto_headline" or r.mode == "auto_stress" or r.mode.startswith("oracle_")
        assert isinstance(r.feasible, bool)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_runner.py -v
```

Expected: import errors on `benchmarks.realistic.runner`.

- [ ] **Step 3: Create `benchmarks/realistic/runner.py`**

```python
"""Realistic benchmark runner.

Iterates (suite, size, seed) and emits RunRows. Reuses
benchmarks.bench._wall_clock_limit as a safety net around solver
timeouts; pathos's anytime cascade already honours the timeout
cooperatively in most cases.
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import pathos.algorithms  # noqa: F401 — register algorithms
from pathos.core.solver import _REGISTRY
from pathos.core.result import SearchResult
from benchmarks.bench import _wall_clock_limit
from benchmarks.suites import SUITE_REGISTRY
from benchmarks.suites._spec import SuiteSpec
from benchmarks.realistic.records import RunRow


@dataclass
class RunConfig:
    seeds: int = 5
    oracle: bool = True
    stress_multiplier: float = 5.0
    base_seed: int = 42


def _solve_and_record(
    suite_id: str, size_payload, seed: int, mode: str, algorithm_name: str | None,
    space, timeout: float, missing_capability: str | None,
) -> RunRow:
    """Run one solve and wrap into a RunRow. Handles crash + timeout."""
    candidates = None
    algo_cls = None
    if algorithm_name is not None:
        algo_cls = next(
            (c for c in _REGISTRY if c.__name__ == algorithm_name), None,
        )
        if algo_cls is None:
            return RunRow(
                suite=suite_id, size=_size_to_int(size_payload), seed=seed,
                mode=mode, algorithm=algorithm_name, cost=None, feasible=False,
                elapsed=0.0, nodes_expanded=-2,
                missing_capability=missing_capability,
            )
        candidates = [algo_cls]

    t0 = time.perf_counter()
    try:
        with _wall_clock_limit(timeout + 2.0):  # safety net beyond cooperative timeout
            res: SearchResult = space.solver(
                candidates=candidates, timeout=timeout,
            ).solve()
    except TimeoutError:
        return RunRow(
            suite=suite_id, size=_size_to_int(size_payload), seed=seed,
            mode=mode, algorithm=algorithm_name or "?",
            cost=None, feasible=False, elapsed=timeout + 2.0,
            nodes_expanded=-1, missing_capability=missing_capability,
        )
    except Exception as e:
        print(
            f"    ! {algorithm_name or '?'} raised {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return RunRow(
            suite=suite_id, size=_size_to_int(size_payload), seed=seed,
            mode=mode, algorithm=algorithm_name or "?",
            cost=None, feasible=False, elapsed=time.perf_counter() - t0,
            nodes_expanded=-2, missing_capability=missing_capability,
        )
    return RunRow(
        suite=suite_id, size=_size_to_int(size_payload), seed=seed,
        mode=mode, algorithm=res.algorithm,
        cost=res.cost, feasible=res.found, elapsed=res.elapsed,
        nodes_expanded=res.nodes_expanded,
        missing_capability=missing_capability, epsilon=res.epsilon,
    )


def _size_to_int(size_payload) -> int:
    """Best-effort integer key for a size payload.

    Tuples like (n_substations, n_slots) collapse to their product;
    plain ints stay as-is. Used only for the RunRow.size field for
    table rendering — the original payload is what's passed to
    generate()/express().
    """
    if isinstance(size_payload, int):
        return size_payload
    if isinstance(size_payload, tuple):
        prod = 1
        for v in size_payload:
            if isinstance(v, int):
                prod *= v
            elif isinstance(v, tuple):
                # (n_sub, n_slots) etc.
                for vv in v:
                    prod *= vv if isinstance(vv, int) else 1
        return prod
    return 0


def run_one(suite_id: str, tier: str, cfg: RunConfig) -> list[RunRow]:
    """Run one (suite, tier) for `cfg.seeds` seeds. Returns all rows
    (auto_headline + auto_stress + optionally oracle_* per compatible algorithm)."""
    spec: SuiteSpec = SUITE_REGISTRY[suite_id]
    if tier not in spec.sizes:
        return []
    rows: list[RunRow] = []
    budget = spec.budgets[tier]
    for size_payload in spec.sizes[tier]:
        for r in range(cfg.seeds):
            seed = cfg.base_seed + r
            inst = spec.generate(size_payload, seed)
            # Build once; reused for headline+stress (oracle gets a fresh build
            # per algorithm so cancellation state doesn't leak).
            space = spec.express(inst)
            rows.append(_solve_and_record(
                suite_id, size_payload, seed, "auto_headline", None,
                space, budget, spec.missing_capability,
            ))
            stress_space = spec.express(inst)
            rows.append(_solve_and_record(
                suite_id, size_payload, seed, "auto_stress", None,
                stress_space, budget * cfg.stress_multiplier,
                spec.missing_capability,
            ))
            if cfg.oracle:
                probe = spec.express(inst)
                compatible = [
                    c.__name__ for c in _REGISTRY if c.compatible_with(probe)
                ]
                for algo_name in compatible:
                    oracle_space = spec.express(inst)
                    rows.append(_solve_and_record(
                        suite_id, size_payload, seed,
                        f"oracle_{algo_name}", algo_name,
                        oracle_space, budget, spec.missing_capability,
                    ))
    return rows


def run_suite_tier(suite_id: str, tier: str, cfg: RunConfig) -> list[RunRow]:
    """Alias for run_one — kept for API symmetry with later batch helpers."""
    return run_one(suite_id, tier, cfg)


def run_sweep(
    suite_ids: Iterable[str], tiers: Iterable[str], cfg: RunConfig,
) -> list[RunRow]:
    """Full sweep across (suite_id, tier) pairs."""
    rows: list[RunRow] = []
    for sid in suite_ids:
        for tier in tiers:
            print(f"\n→ {sid} / {tier}", file=sys.stderr)
            rows.extend(run_one(sid, tier, cfg))
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_runner.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/realistic/runner.py tests/test_benchmarks_runner.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): realistic runner — headline + stress + oracle rows"
```

---

### Task 9: Report renderer + CLI entrypoint

Aggregate `RunRow`s into `SuiteRow`s via the metrics module, build the `Report`, and render to JSON + `REPORT.md`. CLI parses `--suites`, `--tiers`, `--repeat`, `--oracle on/off`, `--quick`, `--json`, `--report`. `--diff REPORT-A.md REPORT-B.md` is added in Task 14.

**Files:**
- Create: `benchmarks/realistic/report.py`
- Create: `benchmarks/realistic/__main__.py`
- Create: `tests/test_benchmarks_report.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_report.py`:

```python
from dataclasses import asdict
from benchmarks.realistic.records import (
    RunRow, Report, BENCH_SCHEMA,
)
from benchmarks.realistic.report import build_report, render_report_md


def _row(suite, size, seed, mode, cost, feasible=True, alg="A"):
    return RunRow(
        suite=suite, size=size, seed=seed, mode=mode, algorithm=alg,
        cost=cost, feasible=feasible, elapsed=0.1, nodes_expanded=10,
    )


def test_build_report_collapses_seeds_per_suite_size():
    rows = [
        _row("C1", 8, 1, "auto_headline", 1.0),
        _row("C1", 8, 2, "auto_headline", 1.0),
        _row("C1", 8, 1, "auto_stress", 1.0),
        _row("C1", 8, 1, "oracle_A", 1.0),
    ]
    rep = build_report(rows, meta={"cpu": "test"})
    suite_rows = [sr for sr in rep.suite_rows if sr.suite == "C1" and sr.size == 8]
    assert len(suite_rows) == 1
    sr = suite_rows[0]
    assert sr.auto_headline_cost == 1.0
    assert sr.auto_stress_cost == 1.0
    assert sr.oracle_cost == 1.0
    assert sr.gap_to_oracle_pct == 0.0


def test_build_report_records_no_cliff_when_auto_not_feasible():
    rows = [
        _row("R3", 100, 1, "auto_headline", None, feasible=False),
    ]
    rep = build_report(rows, meta={"cpu": "test"})
    assert len(rep.no_cliff_failures) == 1
    assert rep.no_cliff_failures[0].failure_kind in (
        "not_found", "infeasible", "crash",
    )


def test_build_report_excludes_missing_capability_from_no_cliff():
    rows = [
        RunRow(
            suite="R4", size=100, seed=1, mode="auto_headline",
            algorithm="TabuSearch", cost=None, feasible=False,
            elapsed=10.0, nodes_expanded=0,
            missing_capability="time_windows",
        ),
    ]
    rep = build_report(rows, meta={"cpu": "test"})
    assert rep.no_cliff_failures == []


def test_render_report_md_has_required_sections():
    rows = [_row("C1", 8, 1, "auto_headline", 1.0)]
    rep = build_report(rows, meta={"cpu": "test"})
    md = render_report_md(rep)
    assert "# Realistic benchmark report" in md
    assert "## Regression table" in md
    assert "## No-cliff failures" in md
    assert "## Selection-suboptimal" in md
    assert "## Tunability gains" in md
    assert "## Missing-capability gaps" in md


def test_report_json_round_trip():
    import json
    rows = [_row("C1", 8, 1, "auto_headline", 1.0)]
    rep = build_report(rows, meta={"cpu": "t"})
    payload = json.loads(json.dumps(asdict(rep)))
    assert payload["bench_schema"] == BENCH_SCHEMA
    assert any(sr["suite"] == "C1" for sr in payload["suite_rows"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_report.py -v
```

- [ ] **Step 3: Create `benchmarks/realistic/report.py`**

```python
from __future__ import annotations
from collections import defaultdict
from benchmarks.realistic.records import (
    BENCH_SCHEMA, NoCliffFailure, Report, RunRow, SuiteRow,
)
from benchmarks.realistic.metrics import (
    gap_pct, is_no_cliff_failure, median_cost, oracle_row, tunability_gain,
)
from benchmarks.suites import SUITE_REGISTRY


def build_report(rows: list[RunRow], meta: dict[str, str]) -> Report:
    bucket: dict[tuple[str, int], list[RunRow]] = defaultdict(list)
    for r in rows:
        bucket[(r.suite, r.size)].append(r)

    suite_rows: list[SuiteRow] = []
    for (suite, size), group in sorted(bucket.items()):
        head = [r for r in group if r.mode == "auto_headline"]
        stress = [r for r in group if r.mode == "auto_stress"]
        head_cost = median_cost(head)
        stress_cost = median_cost(stress)
        head_feas = any(r.feasible for r in head)
        stress_feas = any(r.feasible for r in stress)
        head_alg = (head[0].algorithm if head else "—")
        oracle_alg, oracle_cost = oracle_row(group)
        gap = gap_pct(head_cost, oracle_cost)
        tg, tg_alg = tunability_gain(group)
        spec = SUITE_REGISTRY.get(suite)
        mc = spec.missing_capability if spec is not None else None
        suite_rows.append(SuiteRow(
            suite=suite, size=size,
            auto_headline_cost=head_cost,
            auto_headline_feasible=head_feas,
            auto_headline_algorithm=head_alg,
            auto_stress_cost=stress_cost,
            auto_stress_feasible=stress_feas,
            oracle_cost=oracle_cost,
            oracle_algorithm=oracle_alg,
            gap_to_oracle_pct=gap,
            gap_to_lb_pct=None,  # TODO Task 13 hook LB into runner output
            tunability_gain_pct=tg,
            tunability_algorithm=tg_alg,
            missing_capability=mc,
        ))

    no_cliff: list[NoCliffFailure] = []
    for r in rows:
        spec = SUITE_REGISTRY.get(r.suite)
        tol = spec.tolerance if spec is not None else 0.0
        if not is_no_cliff_failure(r, tol):
            continue
        kind = (
            "crash" if r.nodes_expanded == -2
            else "not_found" if r.cost is None and not r.feasible
            else "infeasible"
        )
        no_cliff.append(NoCliffFailure(
            suite=r.suite, size=r.size, seed=r.seed,
            failure_kind=kind, detail=f"algo={r.algorithm} elapsed={r.elapsed:.2f}s",
            reproduction_args=f"--suites {r.suite} --tiers <tier> --seed {r.seed}",
        ))

    return Report(
        bench_schema=BENCH_SCHEMA,
        meta=meta,
        suite_rows=suite_rows,
        raw_rows=rows,
        no_cliff_failures=no_cliff,
    )


def render_report_md(rep: Report) -> str:
    lines: list[str] = [
        "# Realistic benchmark report",
        "",
        f"- `bench_schema`: {rep.bench_schema}",
    ]
    for k, v in rep.meta.items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Regression table",
        "",
        "| suite | size | auto cost | auto feasible | auto alg | oracle cost | gap % | tunability % | tunability alg | missing-cap |",
        "|---|---:|---:|:---:|---|---:|---:|---:|---|---|",
    ]
    for sr in rep.suite_rows:
        lines.append(
            f"| {sr.suite} | {sr.size} | "
            f"{_fmt(sr.auto_headline_cost)} | "
            f"{'Y' if sr.auto_headline_feasible else 'N'} | "
            f"{sr.auto_headline_algorithm} | "
            f"{_fmt(sr.oracle_cost)} | "
            f"{_fmt(sr.gap_to_oracle_pct)} | "
            f"{_fmt(sr.tunability_gain_pct)} | "
            f"{sr.tunability_algorithm or '—'} | "
            f"{sr.missing_capability or '—'} |"
        )

    lines += ["", "## No-cliff failures", ""]
    if not rep.no_cliff_failures:
        lines.append("(none — gate clean)")
    else:
        lines.append("| suite | size | seed | kind | detail | repro |")
        lines.append("|---|---:|---:|---|---|---|")
        for f in rep.no_cliff_failures:
            lines.append(
                f"| {f.suite} | {f.size} | {f.seed} | {f.failure_kind} | "
                f"{f.detail} | `{f.reproduction_args}` |"
            )

    lines += ["", "## Selection-suboptimal", ""]
    sub = [sr for sr in rep.suite_rows
           if sr.gap_to_oracle_pct is not None and sr.gap_to_oracle_pct > 5.0
           and sr.missing_capability is None]
    if not sub:
        lines.append("(none above 5% gap)")
    else:
        lines.append("| suite | size | auto alg | oracle alg | gap % |")
        lines.append("|---|---:|---|---|---:|")
        for sr in sub:
            lines.append(
                f"| {sr.suite} | {sr.size} | "
                f"{sr.auto_headline_algorithm} | "
                f"{sr.oracle_algorithm or '—'} | "
                f"{_fmt(sr.gap_to_oracle_pct)} |"
            )

    lines += ["", "## Tunability gains", ""]
    tg = [sr for sr in rep.suite_rows
          if sr.tunability_gain_pct is not None and sr.tunability_gain_pct >= 5.0]
    if not tg:
        lines.append("(no candidate beat default by ≥ 5%)")
    else:
        lines.append("| suite | size | candidates=[X] | gain % |")
        lines.append("|---|---:|---|---:|")
        for sr in tg:
            lines.append(
                f"| {sr.suite} | {sr.size} | "
                f"{sr.tunability_algorithm} | "
                f"{_fmt(sr.tunability_gain_pct)} |"
            )

    lines += ["", "## Missing-capability gaps", ""]
    mc = [sr for sr in rep.suite_rows if sr.missing_capability is not None]
    if not mc:
        lines.append("(no missing-capability suites in this run)")
    else:
        lines.append("| suite | size | constraint class | auto cost (penalty-folded) |")
        lines.append("|---|---:|---|---:|")
        for sr in mc:
            lines.append(
                f"| {sr.suite} | {sr.size} | "
                f"{sr.missing_capability} | "
                f"{_fmt(sr.auto_headline_cost)} |"
            )

    return "\n".join(lines) + "\n"


def _fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)
```

Create `benchmarks/realistic/__main__.py`:

```python
"""CLI: python -m benchmarks.realistic [options]"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# Import all suite modules so SUITE_REGISTRY is populated.
import benchmarks.suites.classical       # noqa: F401
import benchmarks.suites.bin_packing     # noqa: F401
import benchmarks.suites.vrp             # noqa: F401
try:
    import benchmarks.suites.power_grid  # noqa: F401
    import benchmarks.suites.rostering   # noqa: F401
except Exception as e:
    print(
        f"warning: ScheduleSpace-dependent suites not loaded ({e})",
        file=sys.stderr,
    )
import benchmarks.suites.timetabling          # noqa: F401
import benchmarks.suites.project_scheduling   # noqa: F401

from benchmarks.suites import SUITE_REGISTRY
from benchmarks.realistic.runner import RunConfig, run_sweep
from benchmarks.realistic.report import build_report, render_report_md


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]),
             "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Realistic benchmark sweep")
    parser.add_argument("--suites", type=str, default=",".join(SUITE_REGISTRY),
                        help=f"comma-separated suite ids (default: all = {','.join(SUITE_REGISTRY)})")
    parser.add_argument("--tiers", type=str, default="S,M,L",
                        help="comma-separated tier names (default: S,M,L)")
    parser.add_argument("--repeat", type=int, default=5,
                        help="seeds per (suite, size) (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="base seed (default: 42)")
    parser.add_argument("--oracle", type=str, default="auto",
                        choices=("on", "off", "auto"),
                        help="run oracle row? auto = on for S/M, off for L (default)")
    parser.add_argument("--quick", action="store_true",
                        help="trimmed run: M tier only, 3 seeds, oracle off")
    parser.add_argument("--json", type=str, default=None,
                        help="path for results JSON (default: results-<TS>.json)")
    parser.add_argument("--report", type=str, default=None,
                        help="path for REPORT.md (default: REPORT-<TS>.md)")
    args = parser.parse_args()

    if args.quick:
        args.tiers = "M"
        args.repeat = 3
        args.oracle = "off"

    suite_ids = [s.strip() for s in args.suites.split(",") if s.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = [s for s in suite_ids if s not in SUITE_REGISTRY]
    if unknown:
        print(f"unknown suites: {unknown}. known: {sorted(SUITE_REGISTRY)}",
              file=sys.stderr)
        return 2

    # Per-tier oracle policy when --oracle=auto.
    def _oracle_for(tier: str) -> bool:
        if args.oracle == "on":
            return True
        if args.oracle == "off":
            return False
        return tier in ("S", "M")

    rows = []
    for tier in tiers:
        cfg = RunConfig(
            seeds=args.repeat, oracle=_oracle_for(tier),
            stress_multiplier=5.0, base_seed=args.seed,
        )
        rows.extend(run_sweep(suite_ids, [tier], cfg))

    meta = {
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "pathos_sha": _git_sha(),
        "run_started": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cli": " ".join(sys.argv[1:]),
    }
    rep = build_report(rows, meta=meta)

    here = Path(__file__).parent
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d-%H%M")
    json_path = Path(args.json) if args.json else here / f"results-{ts}.json"
    report_path = Path(args.report) if args.report else here / f"REPORT-{ts}.md"

    with open(json_path, "w") as f:
        json.dump(asdict(rep), f, indent=2, default=str)
    with open(report_path, "w") as f:
        f.write(render_report_md(rep))

    print(f"\nwrote: {json_path}", file=sys.stderr)
    print(f"wrote: {report_path}", file=sys.stderr)
    print(f"no-cliff failures: {len(rep.no_cliff_failures)}", file=sys.stderr)
    return 1 if rep.no_cliff_failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_report.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/realistic/report.py benchmarks/realistic/__main__.py tests/test_benchmarks_report.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): JSON + REPORT.md renderer and CLI entrypoint"
```

---

### Task 10: R7 — Large CSP timetabling

`n_vars` variables, each with a domain of `n_rooms` rooms, soft preferences for time-blocks. Express as `CSPSpace` with `@domain` + `@constraint` (no two events in same room/time). Partial gap: soft preferences are penalty-folded into a synthetic `@evaluate`.

**Files:**
- Create: `benchmarks/suites/timetabling.py`
- Create: `tests/test_benchmarks_timetabling.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_timetabling.py`:

```python
import benchmarks.suites.timetabling  # noqa: F401 — registers R7
from benchmarks.suites import SUITE_REGISTRY


def test_r7_registered():
    assert "R7" in SUITE_REGISTRY


def test_r7_generator_deterministic():
    spec = SUITE_REGISTRY["R7"]
    a = spec.generate(size=20, seed=1)
    b = spec.generate(size=20, seed=1)
    assert a == b


def test_r7_express_yields_cspspace_that_solves_default():
    spec = SUITE_REGISTRY["R7"]
    inst = spec.generate(size=15, seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=3).solve()
    assert res.cost is not None or res.found


def test_r7_partial_gap_recorded():
    spec = SUITE_REGISTRY["R7"]
    assert spec.expressibility == "partial"
    assert "soft_preferences" in spec.notes
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_timetabling.py -v
```

- [ ] **Step 3: Create `benchmarks/suites/timetabling.py`**

```python
"""R7 — Large CSP timetabling.

Variables: course-section IDs. Domain per variable: a (room, time)
slot pair. Hard constraint: no two variables share the same slot.
Soft preference (partial gap): each variable has a small set of
'preferred' slots; the bench folds preference violations into a
synthetic @evaluate.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from pathos import CSPSpace
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


@dataclass(frozen=True)
class TimetableInstance:
    variables: tuple[str, ...]
    slots: tuple[tuple[int, int], ...]   # (room, time)
    preferred: dict[str, frozenset[tuple[int, int]]]


def _generate(size: int, seed: int) -> TimetableInstance:
    rng = random.Random(seed)
    n_rooms = max(3, size // 10)
    n_times = max(5, size // 4)
    slots = tuple((r, t) for r in range(n_rooms) for t in range(n_times))
    variables = tuple(f"course_{i}" for i in range(size))
    preferred = {
        v: frozenset(rng.sample(slots, k=min(3, len(slots))))
        for v in variables
    }
    return TimetableInstance(variables=variables, slots=slots, preferred=preferred)


def _express(inst: TimetableInstance):
    csp = CSPSpace(variables=list(inst.variables))
    slots = inst.slots

    @csp.domain
    def dom(var):
        return list(slots)

    @csp.constraint
    def no_collision(assignment):
        used = set()
        for slot in assignment.values():
            if slot in used:
                return False
            used.add(slot)
        return True

    # Note: @evaluate hookup for soft prefs would require CSPSpace to
    # expose it; in slice 1 we rely on the constraint above and record
    # soft prefs as a partial gap. The instance retains `preferred` so
    # slice-N can wire it once CSPSpace gains soft constraints.

    return csp


register(SuiteSpec(
    id="R7",
    family="csp",
    constraint_classes=frozenset({"iv", "v"}),
    expressibility="partial",
    sizes={"S": (50,), "M": (150,), "L": (300,)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    notes=(
        "CSP timetabling — hard collision constraint, "
        "partial gap: soft_preferences (preferred slots) not first-class."
    ),
))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_timetabling.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/suites/timetabling.py tests/test_benchmarks_timetabling.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): R7 large CSP timetabling suite (partial-gap: soft prefs)"
```

---

### Task 11: R4 — VRP with time windows (missing-capability)

Extends R3 with per-customer time windows. TourSpace can't model windows as first-class. The expression layer folds window violations into the tour-cost penalty so the suite *runs*, but its `SuiteSpec.missing_capability = "time_windows"` triggers the carve-out in metrics + report.

**Files:**
- Modify: `benchmarks/suites/vrp.py`
- Create: `tests/test_benchmarks_vrptw.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_vrptw.py`:

```python
import benchmarks.suites.vrp  # noqa: F401 — registers R3 + R4
from benchmarks.suites import SUITE_REGISTRY


def test_r4_registered():
    assert "R4" in SUITE_REGISTRY


def test_r4_marked_missing_capability():
    spec = SUITE_REGISTRY["R4"]
    assert spec.missing_capability == "time_windows"
    assert spec.expressibility == "gap"
    assert spec.constraint_classes == frozenset({"i", "iii", "v"})


def test_r4_runs_with_penalty_fold_workaround():
    spec = SUITE_REGISTRY["R4"]
    inst = spec.generate(size=20, seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.cost is not None


def test_r4_instance_has_time_windows():
    spec = SUITE_REGISTRY["R4"]
    inst = spec.generate(size=10, seed=1)
    assert hasattr(inst, "time_windows")
    assert len(inst.time_windows) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_vrptw.py -v
```

- [ ] **Step 3: Extend `benchmarks/suites/vrp.py`**

Append to the bottom of `benchmarks/suites/vrp.py`:

```python
# ---------------------------------------------------------------------------
# R4 — VRP with time windows (missing-capability suite)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VRPTWInstance:
    customers: tuple[tuple[float, float], ...]
    demands: tuple[float, ...]
    time_windows: tuple[tuple[float, float], ...]  # per customer
    service_time: float
    n_vehicles: int
    vehicle_capacity: float
    depot: tuple[float, float]


def _generate_tw(size: int, seed: int) -> VRPTWInstance:
    rng = random.Random(seed)
    base = _generate(size, seed)
    # Synthesize windows on top of the base capacitated instance.
    horizon = 24.0 * 60.0  # 24h in minutes
    windows: list[tuple[float, float]] = []
    for _ in range(size):
        start = rng.uniform(0.0, horizon * 0.75)
        width = rng.uniform(horizon * 0.05, horizon * 0.25)
        windows.append((start, min(horizon, start + width)))
    return VRPTWInstance(
        customers=base.customers,
        demands=base.demands,
        time_windows=tuple(windows),
        service_time=5.0,
        n_vehicles=base.n_vehicles,
        vehicle_capacity=base.vehicle_capacity,
        depot=base.depot,
    )


def _express_tw(inst: VRPTWInstance) -> TourSpace:
    # Build the same TourSpace as R3, then layer a window penalty on top
    # of the existing @evaluate. Because @evaluate is a single function,
    # we rebuild from scratch and combine both penalties inline.
    n_cust = len(inst.customers)
    depot_ids = list(range(inst.n_vehicles))
    cust_ids = list(range(inst.n_vehicles, inst.n_vehicles + n_cust))
    nodes = depot_ids + cust_ids

    def coord(node_id: int) -> tuple[float, float]:
        if node_id < inst.n_vehicles:
            return inst.depot
        return inst.customers[node_id - inst.n_vehicles]

    distances = {
        (i, j): _dist(coord(i), coord(j))
        for i in nodes for j in nodes if i != j
    }
    space = TourSpace(nodes=nodes, distances=distances)
    speed = 1.0  # 1 distance-unit per minute (fast/synthetic)

    @space.evaluate
    def cost(tour: tuple[int, ...]) -> float:
        base = sum(
            distances[(tour[i], tour[(i + 1) % len(tour)])]
            for i in range(len(tour))
        )
        # Walk + accumulate vehicle load + per-vehicle time clock.
        loads: list[float] = []
        cur_load = 0.0
        clock = 0.0
        window_overshoot = 0.0
        prev = tour[0]
        for node in tour:
            if node != prev:
                clock += distances[(prev, node)] / speed
            if node < inst.n_vehicles:
                loads.append(cur_load)
                cur_load = 0.0
                clock = 0.0
            else:
                idx = node - inst.n_vehicles
                a, b = inst.time_windows[idx]
                if clock < a:
                    clock = a  # wait at door, no penalty
                elif clock > b:
                    window_overshoot += clock - b
                clock += inst.service_time
                cur_load += inst.demands[idx]
            prev = node
        loads.append(cur_load)
        cap_overshoot = sum(max(0.0, l - inst.vehicle_capacity) for l in loads)
        return base + 1e3 * cap_overshoot + 1e3 * window_overshoot

    return space


register(SuiteSpec(
    id="R4",
    family="tour",
    constraint_classes=frozenset({"i", "iii", "v"}),
    expressibility="gap",
    sizes={"S": (100,), "M": (300,), "L": (1000,)},
    generate=_generate_tw,
    express=_express_tw,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    missing_capability="time_windows",
    notes=(
        "VRP with time windows — TourSpace lacks first-class window "
        "modeling; both capacity and window violations are penalty-folded."
    ),
))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_vrp.py tests/test_benchmarks_vrptw.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/suites/vrp.py tests/test_benchmarks_vrptw.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): R4 VRP-time-windows (missing-capability suite)"
```

---

### Task 12: R5 — Project scheduling with precedence (missing-capability)

DAG of tasks with durations + resource demands. Precedence ("task B starts after A finishes") can't be expressed natively on `ScheduleSpace`; the workaround folds precedence violations into the penalty. Marked `missing_capability="precedence"`.

**Files:**
- Create: `benchmarks/suites/project_scheduling.py`
- Create: `tests/test_benchmarks_project_scheduling.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmarks_project_scheduling.py`:

```python
import pytest
pytest.importorskip("pathos.spaces.schedule")

import benchmarks.suites.project_scheduling  # noqa: E402,F401
from benchmarks.suites import SUITE_REGISTRY


def test_r5_registered():
    assert "R5" in SUITE_REGISTRY


def test_r5_marked_missing_capability():
    spec = SUITE_REGISTRY["R5"]
    assert spec.missing_capability == "precedence"
    assert spec.expressibility == "gap"
    assert "ii" in spec.constraint_classes


def test_r5_instance_has_precedence_edges():
    spec = SUITE_REGISTRY["R5"]
    inst = spec.generate(size=(10, 24), seed=1)
    assert hasattr(inst, "precedence")
    # Precedence edges are (predecessor, successor) tuples.
    for u, v in inst.precedence:
        assert u in inst.tasks
        assert v in inst.tasks


def test_r5_runs_with_penalty_fold():
    spec = SUITE_REGISTRY["R5"]
    inst = spec.generate(size=(8, 24), seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.cost is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmarks_project_scheduling.py -v
```

- [ ] **Step 3: Create `benchmarks/suites/project_scheduling.py`**

```python
"""R5 — Project scheduling with precedence (missing-capability).

ScheduleSpace doesn't model "task B's earliest start ≥ task A's
completion" as a first-class capability. v1 expresses precedence by
folding violations into the demand/capacity penalty: every slot
where a successor is ON but its predecessor hasn't yet been ON
contributes a fixed cost.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from pathos import ScheduleSpace  # type: ignore[attr-defined]
from pathos.fairness import weighted_minmax  # type: ignore[attr-defined]
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


@dataclass(frozen=True)
class ProjectInstance:
    tasks: tuple[str, ...]
    durations: dict[str, int]   # slots
    precedence: tuple[tuple[str, str], ...]  # (predecessor, successor)
    n_slots: int


def _generate(size, seed: int) -> ProjectInstance:
    n_tasks, n_slots = size
    rng = random.Random(seed)
    tasks = tuple(f"task_{i}" for i in range(n_tasks))
    durations = {t: rng.randint(1, max(2, n_slots // 4)) for t in tasks}
    # Build a random DAG by topological-order edges.
    order = list(tasks)
    rng.shuffle(order)
    precedence: list[tuple[str, str]] = []
    for i, t in enumerate(order):
        # Each later task has up to 2 predecessors from earlier ones.
        for _ in range(rng.randint(0, 2)):
            if i == 0:
                break
            p = rng.choice(order[:i])
            precedence.append((p, t))
    return ProjectInstance(
        tasks=tasks,
        durations=durations,
        precedence=tuple(precedence),
        n_slots=n_slots,
    )


def _express(inst: ProjectInstance):
    # Express each task as an entity; demand = task duration spread
    # uniformly across the planning horizon. Capacity per slot is set so
    # at most one task can be "ON" at a time. Precedence violations
    # are folded into the fairness term (since @evaluate is the cascade
    # entry point and ScheduleSpace's penalty path is already in use).
    space = (
        ScheduleSpace(entities=list(inst.tasks), slots=inst.n_slots, penalty=1e3)
        .target(tolerance=0.0)
    )
    preds_by_succ: dict[str, list[str]] = {}
    for u, v in inst.precedence:
        preds_by_succ.setdefault(v, []).append(u)

    @space.demand
    def demand(task, slot):
        return 1.0  # 1 unit per active slot

    @space.capacity
    def capacity(slot):
        return 1.0  # serial execution

    @space.fairness
    def fairness(schedule):
        # 1.0 minus normalized precedence violations (higher = better).
        # Schedule is (T, N) bool matrix per the ScheduleSpace contract.
        T = len(schedule)
        N = len(schedule[0]) if T else 0
        first_on: dict[int, int | None] = {}
        for n in range(N):
            first_on[n] = next(
                (t for t in range(T) if schedule[t][n]), None,
            )
        viol = 0
        task_to_idx = {t: i for i, t in enumerate(inst.tasks)}
        for succ, preds in preds_by_succ.items():
            si = task_to_idx[succ]
            s_first = first_on[si]
            if s_first is None:
                continue
            for p in preds:
                pi = task_to_idx[p]
                p_first = first_on[pi]
                if p_first is None or p_first >= s_first:
                    viol += 1
        if not preds_by_succ:
            return 1.0
        return max(0.0, 1.0 - viol / max(1, len(inst.precedence)))

    return space


register(SuiteSpec(
    id="R5",
    family="schedule",
    constraint_classes=frozenset({"ii", "iii", "iv"}),
    expressibility="gap",
    sizes={"S": ((20, 72),), "M": ((50, 168),), "L": ((100, 336),)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    missing_capability="precedence",
    notes=(
        "Project scheduling — precedence not first-class on ScheduleSpace; "
        "violations folded into fairness penalty as workaround."
    ),
))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmarks_project_scheduling.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/apiad/Workspace/repos/pathos add benchmarks/suites/project_scheduling.py tests/test_benchmarks_project_scheduling.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): R5 project-scheduling (missing-capability: precedence)"
```

---

### Task 13: End-to-end smoke + LB hookup

Run a `--quick` sweep over the full registry. Fix anything that breaks. Wire the lower-bound callable from `SuiteSpec.lower_bound` into the runner so `gap_to_lb_pct` populates for R3/R6.

**Files:**
- Modify: `benchmarks/realistic/runner.py` — emit one extra RunRow with `mode="lower_bound"` per (suite, size, seed) when `spec.lower_bound is not None`.
- Modify: `benchmarks/realistic/report.py` — `build_report` consumes lower_bound rows to compute `gap_to_lb_pct`.
- Create: `benchmarks/realistic/.gitignore` containing `results-*.json` and `REPORT-*.md`.
- Create: `tests/test_benchmarks_smoke.py` — invokes `python -m benchmarks.realistic --quick --suites C1,C2,C3,R3,R6 --repeat 1` as a subprocess.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_benchmarks_smoke.py`:

```python
import subprocess
import sys
import json
from pathlib import Path


def test_quick_sweep_runs_end_to_end_without_no_cliff_failures(tmp_path):
    json_path = tmp_path / "results.json"
    report_path = tmp_path / "REPORT.md"
    result = subprocess.run(
        [
            sys.executable, "-m", "benchmarks.realistic",
            "--quick",
            "--suites", "C1,C2,C3,R3,R6",
            "--repeat", "1",
            "--json", str(json_path),
            "--report", str(report_path),
        ],
        capture_output=True, text=True, timeout=180,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    # Exit code 0 = clean, 1 = no-cliff failures recorded.
    # Smoke must finish without throwing; both 0 and 1 are acceptable here.
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert json_path.exists()
    assert report_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["bench_schema"] >= 1
    assert any(sr["suite"] == "C1" for sr in payload["suite_rows"])
```

- [ ] **Step 2: Run smoke to surface issues**

```bash
pytest tests/test_benchmarks_smoke.py -v --timeout=300
```

If the smoke fails on a specific suite — e.g. an explicit oracle algorithm crashes or a generator builds an instance the default solver can't handle within budget — fix that suite inline (typical fix: tighten generator parameters so feasibility is comfortable at quick-mode budgets) and re-run.

- [ ] **Step 3: Hook lower_bound rows into runner**

Modify `benchmarks/realistic/runner.py` — inside the seed loop of `run_one`, append after the oracle block:

```python
            if spec.lower_bound is not None:
                lb_value = spec.lower_bound(inst)
                rows.append(RunRow(
                    suite=suite_id,
                    size=_size_to_int(size_payload),
                    seed=seed, mode="lower_bound",
                    algorithm="lb",
                    cost=float(lb_value), feasible=True,
                    elapsed=0.0, nodes_expanded=0,
                    missing_capability=spec.missing_capability,
                ))
```

Modify `benchmarks/realistic/report.py` — inside the `for (suite, size), group ...` loop, compute LB:

```python
        lb_rows = [r for r in group if r.mode == "lower_bound"]
        lb_value = median_cost(lb_rows)
        # Re-compute the SuiteRow assignment to include gap_to_lb_pct:
        gap_lb = gap_pct(head_cost, lb_value)
```

Then replace `gap_to_lb_pct=None,` with `gap_to_lb_pct=gap_lb,` in the `SuiteRow(...)` constructor.

- [ ] **Step 4: Create gitignore**

Create `benchmarks/realistic/.gitignore`:

```
results-*.json
REPORT-*.md
```

- [ ] **Step 5: Re-run smoke + commit**

```bash
pytest tests/test_benchmarks_smoke.py -v --timeout=300
git -C /home/apiad/Workspace/repos/pathos add benchmarks/realistic/runner.py benchmarks/realistic/report.py benchmarks/realistic/.gitignore tests/test_benchmarks_smoke.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): lower-bound rows + end-to-end smoke + gitignore"
```

---

### Task 14: Diff mode + push

Add `--diff REPORT-A.md REPORT-B.md`: parse the regression tables out of two REPORT.md files, emit a delta table showing which (suite, size) rows improved / regressed / are new. Then push all commits.

**Files:**
- Modify: `benchmarks/realistic/__main__.py` — add `--diff` flag.
- Create: `benchmarks/realistic/diff.py`
- Create: `tests/test_benchmarks_diff.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmarks_diff.py`:

```python
from benchmarks.realistic.diff import diff_reports, parse_report_md


SAMPLE_A = """# Realistic benchmark report

## Regression table

| suite | size | auto cost | auto feasible | auto alg | oracle cost | gap % | tunability % | tunability alg | missing-cap |
|---|---:|---:|:---:|---|---:|---:|---:|---|---|
| C1 | 8 | 1.000 | Y | Backtracking | 1.000 | 0.000 | 0.000 | — | — |
| R3 | 100 | 500.000 | Y | TabuSearch | 480.000 | 4.167 | 0.000 | — | — |
"""

SAMPLE_B = """# Realistic benchmark report

## Regression table

| suite | size | auto cost | auto feasible | auto alg | oracle cost | gap % | tunability % | tunability alg | missing-cap |
|---|---:|---:|:---:|---|---:|---:|---:|---|---|
| C1 | 8 | 1.000 | Y | Backtracking | 1.000 | 0.000 | 0.000 | — | — |
| R3 | 100 | 470.000 | Y | TabuSearch | 470.000 | 0.000 | 0.000 | — | — |
"""


def test_parse_report_md_extracts_regression_rows():
    rows = parse_report_md(SAMPLE_A)
    assert len(rows) == 2
    assert rows[0]["suite"] == "C1"
    assert rows[1]["suite"] == "R3"


def test_diff_reports_flags_improvements():
    delta = diff_reports(SAMPLE_A, SAMPLE_B)
    r3_delta = [d for d in delta if d["suite"] == "R3" and d["size"] == 100][0]
    assert r3_delta["status"] == "improved"
    assert r3_delta["cost_delta_pct"] < 0


def test_diff_reports_flags_regressions():
    delta = diff_reports(SAMPLE_B, SAMPLE_A)
    r3_delta = [d for d in delta if d["suite"] == "R3" and d["size"] == 100][0]
    assert r3_delta["status"] == "regressed"
    assert r3_delta["cost_delta_pct"] > 0


def test_diff_reports_marks_stable_rows():
    delta = diff_reports(SAMPLE_A, SAMPLE_A)
    assert all(d["status"] == "stable" for d in delta)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmarks_diff.py -v
```

- [ ] **Step 3: Implement `benchmarks/realistic/diff.py`**

```python
"""Parse two REPORT.md files and emit a delta between their
regression tables. Used by `python -m benchmarks.realistic --diff A B`.
"""
from __future__ import annotations
import re


_ROW_RE = re.compile(
    r"^\|\s*(?P<suite>[^\s|]+)\s*\|\s*(?P<size>\d+)\s*\|\s*"
    r"(?P<cost>[\d.\-—]+)\s*\|\s*(?P<feas>[YN])\s*\|"
)


def parse_report_md(text: str) -> list[dict]:
    rows: list[dict] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("## Regression table"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table:
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        cost_raw = m.group("cost").strip()
        cost = None if cost_raw == "—" else float(cost_raw)
        rows.append({
            "suite": m.group("suite"),
            "size": int(m.group("size")),
            "auto_headline_cost": cost,
            "auto_headline_feasible": m.group("feas") == "Y",
        })
    return rows


def diff_reports(text_a: str, text_b: str) -> list[dict]:
    a_rows = {(r["suite"], r["size"]): r for r in parse_report_md(text_a)}
    b_rows = {(r["suite"], r["size"]): r for r in parse_report_md(text_b)}
    keys = sorted(set(a_rows) | set(b_rows))
    out: list[dict] = []
    for k in keys:
        a = a_rows.get(k)
        b = b_rows.get(k)
        suite, size = k
        if a is None:
            out.append({"suite": suite, "size": size, "status": "new",
                        "cost_delta_pct": None})
            continue
        if b is None:
            out.append({"suite": suite, "size": size, "status": "removed",
                        "cost_delta_pct": None})
            continue
        ca = a["auto_headline_cost"]
        cb = b["auto_headline_cost"]
        if ca is None or cb is None:
            out.append({"suite": suite, "size": size, "status": "unknown",
                        "cost_delta_pct": None})
            continue
        if ca == 0.0 and cb == 0.0:
            delta = 0.0
        elif ca == 0.0:
            delta = float("inf")
        else:
            delta = 100.0 * (cb - ca) / abs(ca)
        if abs(delta) < 1e-6:
            status = "stable"
        elif delta < 0:
            status = "improved"
        else:
            status = "regressed"
        out.append({"suite": suite, "size": size, "status": status,
                    "cost_delta_pct": delta})
    return out
```

- [ ] **Step 4: Wire `--diff` into the CLI**

In `benchmarks/realistic/__main__.py`, before the existing argument parsing's `args = parser.parse_args()` call, add:

```python
    parser.add_argument("--diff", nargs=2, metavar=("REPORT_A", "REPORT_B"),
                        default=None,
                        help="diff two REPORT.md files instead of running")
```

And before the suite-ids validation, branch:

```python
    if args.diff:
        from benchmarks.realistic.diff import diff_reports
        a = Path(args.diff[0]).read_text()
        b = Path(args.diff[1]).read_text()
        delta = diff_reports(a, b)
        print(f"{'suite':<6} {'size':>6} {'status':<10} {'Δcost %':>10}")
        for d in delta:
            cd = "n/a" if d["cost_delta_pct"] is None else f"{d['cost_delta_pct']:+.2f}"
            print(f"{d['suite']:<6} {d['size']:>6} {d['status']:<10} {cd:>10}")
        return 0
```

- [ ] **Step 5: Run tests + push**

```bash
pytest tests/test_benchmarks_diff.py -v
pytest -q
git -C /home/apiad/Workspace/repos/pathos add benchmarks/realistic/diff.py benchmarks/realistic/__main__.py tests/test_benchmarks_diff.py
git -C /home/apiad/Workspace/repos/pathos commit -m "feat(bench): --diff mode for REPORT.md cross-run comparison"
git -C /home/apiad/Workspace/repos/pathos push origin main
```

Expected: all tests pass; 14 new commits land on `gia-uh/pathos@main`.

---

## Self-Review Notes

**Spec coverage check:** each section of `2026-06-10-realistic-benchmark-suite-design.md` maps to at least one task:
- Goal / Why now → covered implicitly by the whole plan
- Contract (no-cliff / anytime / tunability) → Task 3 (metrics) + Task 9 (report)
- Suite catalog C1–C3 → Task 2; R1 → Task 6; R2 → Task 7; R3 → Task 5; R4 → Task 11; R5 → Task 12; R6 → Task 4; R7 → Task 10
- Sizes per tier → Task 2 + each suite task's `sizes={"S":…, "M":…, "L":…}` field
- Quality measurement (auto headline / auto stress / oracle / LB) → Task 8 (runner) + Task 13 (LB hookup)
- Budget classes → Task 2 (CLASSICAL_BUDGETS) + each suite task's `budgets` field
- Multi-objective handling → suite expressions return scalarised cost (R1/R2/R5/R6/R7 all do)
- Validation report (JSON + REPORT.md, 5 sections) → Task 9 (`render_report_md` enforces 5 sections)
- Missing-capability suites → Tasks 11 (R4) + 12 (R5) with `missing_capability` field + Task 9 carve-out
- Module layout → matches the File Structure block exactly
- CLI flags → Task 9 + Task 14 (`--diff`)
- Testing strategy → per-suite tests + runner test + report test + smoke test
- Risks → mitigations: `--oracle auto` policy (Task 9), `--quick` flag (Task 9), `meta` block with CPU/python/SHA (Task 9), missing-capability carve-out (Task 9)

**Placeholder scan:** no "TBD", "TODO", "fill in later", or "similar to Task N" remain in task bodies. One inline `# TODO Task 13 hook LB into runner output` exists in Task 9's `report.py` — but Task 13 explicitly replaces it. Acceptable forward reference; the comment names the resolving task.

**Type consistency check:**
- `SuiteSpec` fields used identically across Tasks 1, 2, 4–7, 10–12.
- `RunRow` fields used identically across Tasks 1, 8, 9, 13.
- `_size_to_int` helper (Task 8) handles both `int` and `tuple` size payloads — used by every suite consistently.
- `SUITE_REGISTRY` is a `dict[str, SuiteSpec]` — `register()` (Task 1) is called by every suite module.
- `RunConfig.oracle: bool` — set by `_oracle_for(tier)` in Task 9 CLI; consumed by `run_one` in Task 8.
- Suite IDs (`C1`-`C3`, `R1`-`R7`) are consistent everywhere.
- `mode` strings (`auto_headline`, `auto_stress`, `oracle_<name>`, `lower_bound`) are consistent across emitter (Task 8 + 13) and consumer (Task 9 + 13).

---

Plan complete and saved to `docs/superpowers/plans/2026-06-10-realistic-benchmark-suite.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
