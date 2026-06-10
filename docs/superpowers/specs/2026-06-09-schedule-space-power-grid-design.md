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
- Emit `{EVALUATE, SUCCESSORS}` so the local-search /
  metaheuristic family — HillClimbing, SimulatedAnnealing, TabuSearch,
  LocalBeamSearch, GA, DE — is eligible.
- Reuse the **`AnytimeLocal`** cascade `[HillClimbing,
  SimulatedAnnealing, TabuSearch]` that shipped in v0.2.0 — no new
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
- **CSP-family compatibility.** `Backtracking`, `ForwardChecking`,
  `MinConflicts`, and `AnytimeCSP` are gated by `_is_csp_shaped(space)`
  (`isinstance(space._initial, dict)` in `pathos/algorithms/csp.py:13`),
  which expects partial-assignment dicts. ScheduleSpace's state is a
  hashable wrapper over the complete schedule, not a partial assignment,
  so those algorithms reject it cleanly. v1 ships with capacity violations
  folded into `_evaluate` as a penalty term, not surfaced via the
  `CONSTRAINTS` capability. Native CSP-family compatibility (`MinConflicts`
  on `ScheduleSpace`, explicit `@constraint` capability with pruning) is
  v1.1 work — see "Open questions" below.

## Architecture

### State representation

A `ScheduleSpace` state is a `frozenset[tuple[int, int]]` — the set of
`(slot, entity_index)` cells that are **on**. Stdlib-only, hashable,
equality-comparable. This shape is required because `TabuSearch` keeps a
tabu list and does `child not in tabu` (`pathos/algorithms/local.py:120`);
raw `numpy.ndarray` instances raise on equality-in-list. A frozenset
gives O(1) membership, O(1) hashing on cached `hash`, and dirt-cheap
"flip one cell" via `s ^ {(t, e)}`.

NumPy is used internally for fairness/capacity arithmetic: the helper
`ScheduleSpace._to_array(state) -> np.ndarray` converts the frozenset to
a `(T, N)` bool ndarray on demand. Fairness callables receive the ndarray
form (clean for the user), while algorithms see the frozenset (cheap to
hash and copy).

The returned `SearchResult.solution` is the **ndarray form** of the final
state — users get a `(T, N)` bool matrix, not a frozenset.

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
| `@fairness` (combined with `@demand` + `@capacity` + `.target()` internally) | `EVALUATE` — internal `_evaluate(state)` returns `-fairness(state) + λ · violations(state)` where `violations` is total capacity overshoot summed over slots. Lower is better; algorithms minimise. | HillClimbing, SimulatedAnnealing, TabuSearch, LocalBeamSearch, GA, DE, RandomSearch |
| Built-in (via `.neighborhood(k)`) | `SUCCESSORS` — k-bit-flip neighborhood (default k=1) | (same list) |

The emitted capability set is `{EVALUATE, SUCCESSORS}` — minimal and
consumed by the entire local-search / metaheuristic family without
modification.

**No `GOAL`**: ScheduleSpace deliberately does not declare GOAL.
`Solver._select` (`solver.py:50`) applies a goal-honoring filter when GOAL
is present, which would exclude `AnytimeLocal` (whose `requires` doesn't
include GOAL). Skipping GOAL keeps `AnytimeLocal` in the candidate pool.
Capacity satisfaction is a property of the returned schedule that the
user checks via `SearchResult.slack`, not a goal predicate.

**`λ` (capacity penalty weight)** is configurable on the constructor
(`penalty=`), default `1e3`. Large enough that any feasible schedule
out-scores any infeasible one for realistic fairness values in `[0, 1]`,
small enough that gradients near the feasible boundary stay informative.

## Auto-solver behaviour

ScheduleSpace declares `mode="auto"` by default (inherited from `Space`).
Under `mode="auto"`, the auto-solver picks `AnytimeLocal` — already
shipped in v0.2.0 at `pathos/algorithms/local.py:188`. Its cascade is
`[HillClimbing(max_restarts=3), SimulatedAnnealing(max_iter=500, T0=100,
cooling=0.99), TabuSearch(max_iter=200, tabu_size=20)]` and it returns
the lowest-cost incumbent across all phases under the wall-clock budget.

No new meta-algorithm and no per-algorithm `score_for` overrides are
needed for v1. The existing cascade adapts to budget naturally.

Users opting into `mode="exact"` or `mode="approximate"` get the base
algorithm pick — typically `TabuSearch` (`power_rank=18`) above
`LocalBeamSearch` (16) and `HillClimbing` (15). Users can still force
`GA` or `DE` via `space.solver(candidates=[GeneticAlgorithm])`.

**Out of scope for v1:** size-aware `score_for` thresholds (a sensible
v1.1 once we have ScheduleSpace benchmarks to calibrate against). The
v0.2.0 AnytimeLocal cascade is the v1 selection policy verbatim.

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
  __init__.py            # MODIFIED — re-export ScheduleSpace
  spaces/
    schedule.py          # NEW — ScheduleSpace
  fairness.py            # NEW — weighted_minmax(weights, space) helper
  core/
    result.py            # MODIFIED — add optional `slack` field to SearchResult

examples/
  power_grid.py          # NEW — worked example on a synthetic grid

tests/
  test_schedule_space.py # NEW — unit + property + integration tests
  test_fairness.py       # NEW — weighted_minmax unit tests
```

No algorithm files are touched. The existing AnytimeLocal cascade
(HC → SA → Tabu) consumes ScheduleSpace verbatim.

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
print(f"Algorithm: {result.algorithm}")          # "AnytimeLocal"
print(f"Best objective (-fairness + λ·violations): {result.cost:.3f}")
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
- ScheduleSpace emits exactly `{EVALUATE, SUCCESSORS}` once all three
  decorators are attached (none earlier).
- Each decorator raises on re-attachment.
- `target(tolerance=…)` rejects `<0` and `>1`.
- When `graph` is provided, `downstream` is auto-derived; when omitted,
  entities self-map.
- The neighborhood is k-bit flips (k=1 default), verified by enumerating
  successors of a tiny state.

### Property (hand-rolled)
- For any state in the search trajectory, `ScheduleSpace._evaluate(s)` =
  `-fairness(_to_array(s)) + λ·max(0, sum_e demand·on - cap)` summed per
  slot. Verified by parameterising over random states.
- Fairness output is invariant to entity reordering (the objective is
  symmetric in entities — `weighted_minmax` respects this).
- 1-flip neighborhood enumerates exactly `T·N` successors of any state.

### Integration
- **Tiny.** 4 substations × 6 slots, integer demand/capacity. Run
  `space.solver().solve()`; assert `result.found`, `result.algorithm ==
  "AnytimeLocal"`, and that all per-slot capacity constraints are satisfied
  in the returned ndarray (no `λ·violations` term left in `result.cost`).
- **Medium (smoke).** 20 substations × 168 slots, fixed RNG seed
  (`np.random.default_rng(42)`). Run with `space.solver(timeout=5).solve()`.
  Assert `result.found`, capacity bound holds on every slot, and the
  weighted-min-max fairness is at least `0.50` (a coarse threshold; locks
  no exact float, catches gross regressions).
- **Example as smoke test.** `examples/power_grid.py` runs in CI with the
  same RNG seed; the printed algorithm name (`AnytimeLocal`) and the
  feasibility check are asserted.

## Risks and open questions

- **1-bit-flip neighborhood may be too local.** Fairness landscapes have
  long flat plateaus (changing one slot for one entity rarely changes
  the worst-case weighted uptime). Mitigation: `.neighborhood(k=2)` is in
  the builder API from day one — not exercised by the v1 example, but
  available as a tuning knob. If the medium-integration test reveals
  poor convergence, escalate to k=2 in the default.
- **Penalty weight `λ` calibration.** The default `1e3` is a heuristic.
  Wrong `λ` can let SA/Tabu accept infeasible schedules in exchange for
  fairness gains. Mitigation: the integration test asserts feasibility
  on the returned schedule; if it fails, bump `λ` and retry. Long-term
  fix (v1.1): adaptive penalty or native `CONSTRAINTS` capability.
- **Auto-derivation of `downstream` when `graph` is provided.** "Set of
  nodes reachable from entity once all other entities are cut" is
  well-defined for trees but ambiguous on meshed graphs (multiple
  feeders may serve the same leaf). v1 documents the tree-grid
  assumption; meshed-grid handling is deferred.
- **Frozenset state copy cost.** Each successor allocates a fresh
  frozenset. At T·N = 3 360 the per-state hash + copy is sub-millisecond
  but every `_evaluate` call rebuilds the ndarray view. If profiling
  shows this dominates wall-clock, add a small `lru_cache`-backed
  ndarray cache keyed on `id(state)` (frozenset id is stable for
  immutable values). Implementation-phase decision, not a spec-blocker.

## Open questions for v1.1 / future

- **Native CSP-family compatibility.** Either relax `_is_csp_shaped` in
  `pathos/algorithms/csp.py:13` to accept ScheduleSpace's frozenset state
  (or a new shape predicate `_is_schedule_shaped`), and adapt
  `MinConflicts` to operate on `frozenset[(t,e)]` states. Would let
  `AnytimeCSP[MinConflicts, Backtracking]` win selection for
  ScheduleSpace AND surface `CONSTRAINTS` natively instead of folding
  capacity into `_evaluate` as a penalty.
- Built-in fairness helpers beyond `weighted_minmax` (Gini, variance,
  Jain's).
- Meshed-grid `downstream` semantics (max-flow based, or user-required).
- Per-region capacity axes (multiple `@capacity` decorators with region
  tags).
- Sequential-decision mode for online operation against live demand.
- Size-aware `score_for` overrides once we have ScheduleSpace benchmarks.

## Repository

This spec ships under `repos/pathos/`. Linked from the pathos design spec
([[2026-05-29-pathos-design]]) once implementation lands.
