---
title: "ScheduleSpace — discrete scheduling with capacity constraints and fairness objectives"
date: 2026-06-09
type: design
status: draft
tags: [pathos, schedule-space, power-grid, csp, fairness]
---

# ScheduleSpace — Design Spec

## Motivation

The driving use case is **power-grid blackout scheduling**: a graph of substations
with per-slot demand, a per-slot capacity bound below total demand, and the need
to distribute mandatory blackouts *fairly* across nodes (weighted by priority —
hospitals should never be shed, residential can be cycled). Decide, over a
finite horizon of T slots, which substations are on or off at each slot, so the
overall blackout pattern is as equitable as possible while respecting capacity.

This problem shape is broader than power grids. It captures any **discrete-time
scheduling problem with a capacity constraint and a distributive objective**:
rolling blackouts, brownout schedules, demand-response cycles, fair-share
batch scheduling on shared resources. None of pathos's current subspaces
(`GraphSpace`, `CSPSpace`, `TourSpace`, `GameSpace`) handle this shape
cleanly — `CSPSpace` lacks the fairness/optimization dimension, `GraphSpace`
has no temporal axis, and raw `Space` + `@evaluate` works but throws away
the constraint structure that lets CSP algorithms shine.

## Goals

- Add a new specialized subspace, **`ScheduleSpace`**, peer to the four
  existing subspaces.
- Model one-shot discrete scheduling: decision variable is the full T×N
  binary matrix (slots × entities), produced in a single solve.
- Emit a capability set rich enough that the auto-solver picks meaningfully
  across small / medium / large regimes — Backtracking, Min-Conflicts,
  Simulated Annealing, Tabu Search, Local Beam, GA, DE all eligible.
- Reuse the **`AnytimeCSP`** cascade that shipped in v0.2.0 — no new
  meta-algorithm.
- Ship one batteries-included fairness helper (`weighted_minmax`) and keep
  the `@fairness` decorator open so users provide arbitrary scalar objectives.

## Non-goals (v1)

- Physical flow constraints (DC OPF, line congestion). That is LP / convex
  optimization territory, not classical search.
- Sequential / rolling-horizon decision-making. Requires a temporal-state
  primitive pathos does not have; out of scope for this spec.
- Multiple coupled `@capacity` axes (e.g., per-region transmission limits in
  addition to total power). v1 ships single-capacity-per-slot.
- A built-in fairness library beyond `weighted_minmax`. Gini, variance,
  Jain's index, etc. are one-line callables the user can hand-roll inside
  `@fairness`.

## Architecture

### State representation

A `ScheduleSpace` state is a `numpy.ndarray` of shape `(T, N)` with dtype
`bool`. Row `t` is the on/off vector for slot `t`; column `e` is the
on/off vector for entity `e` across the horizon. `True` = on, `False` =
blacked out.

NumPy under the hood because every fairness/capacity computation is a
vectorisable reduction. The capability layer exposes the matrix as the
solution; algorithm internals may convert to per-variable representations
where needed (e.g., Min-Conflicts uses a flat `T·N`-length variable list
indexed `(t, e) → t·N + e`).

### Constructor

```python
ScheduleSpace(
    entities: Sequence[Hashable],          # cut-set decision vars
    slots: int,                            # T time slots
    graph: networkx.Graph | None = None,   # optional grid topology
    downstream: Callable[[entity], Iterable[leaf]] | None = None,
)
```

- `entities` is the list of binary decision variables — one decision per
  entity per slot. For grids these are substations or feeders (the
  cut-set), not individual customers.
- `slots` is the horizon T. Slot semantics (hour, 15-minute interval,
  day) are user-defined; ScheduleSpace treats them as opaque indices.
- `graph` is optional and used only for fairness expansion. When omitted,
  fairness is computed over `entities` directly.
- `downstream(entity) → leaves` maps each entity to the set of nodes whose
  experience of blackout is determined by that entity's on/off state.
  When `graph` is provided and `downstream` is omitted, ScheduleSpace
  derives `downstream` as the set of `graph` nodes reachable from the
  entity once all *other* entities are cut. This is a one-time
  preprocessing step at `ScheduleSpace.__init__`.

### Decorators (capabilities the user attaches)

```python
@space.demand
def demand(entity, slot) -> float:
    """Per-entity per-slot demand. Required."""

@space.capacity
def capacity(slot) -> float:
    """Per-slot total capacity. Required."""

@space.fairness
def fairness(schedule: np.ndarray) -> float:
    """Scalar to MAXIMISE. schedule is (T, N) bool. Required."""
```

All three decorators are required. The auto-solver raises a clear error at
`space.solver()` if any is missing, naming the missing decorator.

### Builder methods

```python
space.target(tolerance: float = 0.0)
    # Enforces capacity[t] · (1 - tolerance) ≤ load[t] ≤ capacity[t]
    # 0 ≤ tolerance ≤ 1. tolerance=0 means upper-bound only.

space.neighborhood(k: int = 1)
    # k-bit-flip successors. Default k=1. Larger k = bigger
    # search neighborhood, slower per step. Reserved for tuning;
    # not exercised in v1 examples.
```

Both return `self` (fluent chaining).

### Auto-emitted capabilities

| User decorator / builder | Emitted capability | Algorithms unlocked |
|---|---|---|
| `@demand` + `@capacity` (+ `.target()`) | `constraints` (per-slot capacity bound, optionally with lower band) | Backtracking, Forward Checking, AC-3, Min-Conflicts |
| `@fairness` | `evaluate` | Hill Climbing, SA, Tabu, Local Beam, GA, DE, Random Search |
| Built-in (from constructor) | `variables` (T·N binary) | (CSP family enabler) |
| Built-in | `domains` ({0,1}) | (CSP family enabler) |
| Built-in (via `.neighborhood(k)`) | `successors` (k-bit-flip neighborhood, default k=1) | Hill Climbing, SA, Tabu, Local Beam |

The emitted capability set is `{evaluate, constraints, variables, domains,
successors}` — the richest of any existing subspace, by design.

## Auto-solver behaviour

Selection runs the same `score_for(space)` machinery as v0.2.0. The new
algorithm-side overrides for ScheduleSpace:

| Algorithm | `score_for(space)` rule |
|---|---|
| `Backtracking` + `ForwardChecking` | High when `T · N ≤ 200`. Exact; complete search of the 2^(T·N) space terminates fast in this regime regardless of demand/capacity numeric type. |
| `MinConflicts` | High when `200 < T · N ≤ 10_000`. Capacity violations are the conflict count; bitmap-native. |
| `SimulatedAnnealing`, `TabuSearch` | High when `T · N > 10_000` *or* when fairness is non-smooth (we cannot detect this generically — user can force via `solver(candidates=…)`). |
| `GeneticAlgorithm`, `DifferentialEvolution` | Eligible but demoted (bitmap crossover gives poor signal on this objective). Available if the user forces them. |
| `AnytimeCSP` (meta) | Wins under `mode="auto"` (the default). Cascade `[MinConflicts, Backtracking]`, returns best incumbent within budget. **Already shipped in v0.2.0 — no new code.** |

The score thresholds (200, 10_000) are starting points based on rough order-of-magnitude reasoning — they will be tuned against the v1 example and any benchmarks added later.

## SearchResult extensions

ScheduleSpace returns the standard `SearchResult` with one optional addition:

```python
@dataclass
class SearchResult:
    # ... existing fields ...
    slack: np.ndarray | None = None  # per-slot residual capacity, shape (T,)
                                      # None for non-Schedule problems
```

`slack[t] = capacity[t] - load[t]` for the returned schedule. Useful for grid
operators to see headroom per slot. Populated only by ScheduleSpace; remains
`None` for every other subspace.

## File structure

```
pathos/
  spaces/
    schedule.py          # NEW — ScheduleSpace
  fairness.py            # NEW — weighted_minmax(weights, space) helper
  core/
    result.py            # MODIFIED — add optional `slack` field to SearchResult
  algorithms/
    csp.py               # MODIFIED — score_for overrides for ScheduleSpace
    local.py             # MODIFIED — score_for overrides for ScheduleSpace
    evolutionary.py      # MODIFIED — score_for demotion for ScheduleSpace

examples/
  power_grid.py          # NEW — worked example on a synthetic grid

tests/
  test_schedule_space.py # NEW — unit + property + integration tests
```

Top-level public API additions:

```python
from pathos import ScheduleSpace            # peer to GraphSpace, CSPSpace, ...
from pathos.fairness import weighted_minmax
```

## Fairness helper

```python
def weighted_minmax(
    weights: Mapping[Hashable, float],
    space: ScheduleSpace,
) -> Callable[[np.ndarray], float]:
    """Returns a fairness callable bound to `space`.

    The returned function takes a (T, N) bool schedule and returns
        min over leaves of  weights[leaf] * uptime_fraction(leaf, schedule)
    where uptime_fraction(leaf, schedule) is the share of slots in which the
    entity feeding `leaf` (per space.downstream) was on. Higher = fairer.

    `weights` keys should cover every leaf the user cares about; missing
    leaves are skipped (treated as fully tolerant).
    """
```

The helper is the **only** built-in fairness function in v1. Other shapes
(variance, Gini, Jain's) are one-line user callables.

## API in full — worked example

```python
import numpy as np
import networkx as nx
from pathos import ScheduleSpace
from pathos.fairness import weighted_minmax

# Synthetic grid: 20 substations on a small radial topology
grid = nx.balanced_tree(r=3, h=3)
substations = [n for n in grid.nodes if grid.degree(n) > 1]
leaves = [n for n in grid.nodes if grid.degree(n) == 1]
priority = {leaf: np.random.choice([0.0, 0.5, 1.0]) for leaf in leaves}
# 0.0 = critical (hospital), 0.5 = industrial, 1.0 = residential

space = (
    ScheduleSpace(entities=substations, slots=168, graph=grid)
    .target(tolerance=0.05)
)

@space.demand
def demand(sub, slot):
    return base_load[sub][slot]   # kW, deterministic forecast

@space.capacity
def capacity(slot):
    return available_supply[slot]  # kW, total power available this slot

@space.fairness
def fairness(schedule):
    # schedule: (168, 20) bool — True = substation is on
    # weighted_minmax returns a closure bound to `space`, which provides
    # the entity → downstream leaves mapping internally.
    return weighted_minmax(priority, space)(schedule)

result = space.solver().solve()

assert result.found
print(f"Algorithm: {result.algorithm}")          # likely "AnytimeCSP[MinConflicts]"
print(f"Min weighted uptime: {-result.cost:.3f}")
print(f"Slack per slot: {result.slack.mean():.1f} kW avg headroom")
```

## Error handling

| Condition | Behaviour |
|---|---|
| Missing `@demand`, `@capacity`, or `@fairness` at `solver()` time | `RuntimeError` naming the missing decorator. |
| Decorator attached twice (same space) | `RuntimeError("@demand already defined on this space")`. |
| `tolerance` outside `[0, 1]` | `ValueError` in `.target()`. |
| `entities` contains node not in `graph` (when `graph` provided) | `ValueError` at `__init__`. |
| `capacity(t)` returns negative value | `ValueError` at first evaluation; surfaces in solve. |
| Infeasible problem (no schedule satisfies capacity even with all-off — e.g., demand=0 but lower band requires load>0) | `SearchResult(found=False)` with `cost=None`; `algorithm` still names what tried. |
| `target()` band tightens to infeasibility | `UserWarning`, fall back to upper-bound-only constraint. |

## Testing strategy

### Unit
- ScheduleSpace emits exactly `{evaluate, constraints, variables, domains, successors}`.
- Each decorator raises on re-attachment.
- `target(tolerance=…)` rejects `<0` and `>1`.
- When `graph` is provided, `downstream` is auto-derived; when omitted,
  entities self-map.
- The neighborhood is k-bit flips (k=1 default), verified by enumerating
  successors of a tiny state.

### Property (Hypothesis or hand-rolled)
- For any returned schedule and any t:
  `sum(demand(e, t) for e in entities if schedule[t, e]) ≤ capacity(t)`,
  and (with tolerance>0) `≥ capacity(t) · (1 - tolerance)`.
- Min-Conflicts iterations are monotone-improving in conflict count (with
  high probability, allowing for the random-restart tail).
- Fairness output is invariant to entity reordering (the objective is
  symmetric in entities — `weighted_minmax` respects this).

### Integration
- **Tiny (exact).** 5 substations × 24 slots, integer demand/capacity.
  Auto-solver picks Backtracking, finds the brute-force-verified optimum.
- **Medium (anytime).** 20 substations × 168 slots (the worked example).
  Auto-solver picks `AnytimeCSP[MinConflicts]`. Lands within a configured
  optimality gap of a Min-Conflicts-only baseline run with a much larger
  budget.
- **Example as smoke test.** `examples/power_grid.py` runs in CI with a
  fixed seed; the printed algorithm name and a coarse fairness threshold
  are asserted, catching API drift without locking exact float values.

## Risks and open questions

- **1-bit-flip neighborhood may be too local.** Fairness landscapes have
  long flat plateaus (changing one slot for one entity rarely changes
  the worst-case weighted uptime). Mitigation: `.neighborhood(k=2)` is in
  the builder API from day one — not exercised by the v1 example, but
  available as a tuning knob. If the medium-integration test reveals
  poor convergence, escalate to k=2 in the default.
- **Min-Conflicts performance on T·N variables.** Current pathos
  Min-Conflicts iterates over a Python list of variables; on
  T·N = 168·20 = 3 360 vars this should be fine, but if profiling shows
  it dominates wall-clock, refactor to ndarray indexing during
  implementation. Tracked as an implementation-phase decision, not a
  spec-blocker.
- **Auto-derivation of `downstream` when `graph` is provided.** "Set of
  nodes reachable from entity once all other entities are cut" is
  well-defined for trees but ambiguous on meshed graphs (multiple
  feeders may serve the same leaf). v1 documents the tree-grid
  assumption; meshed-grid handling is deferred.

## Open question for v1.1 / future

- Built-in fairness helpers beyond `weighted_minmax` (Gini, variance, Jain's).
- Meshed-grid `downstream` semantics (max-flow based, or user-required).
- Per-region capacity axes (multiple `@capacity` decorators with region tags).
- Sequential-decision mode for online operation against live demand.

## Repository

This spec ships under `repos/pathos/`. Linked from the pathos design spec
([[2026-05-29-pathos-design]]) once implementation lands.
