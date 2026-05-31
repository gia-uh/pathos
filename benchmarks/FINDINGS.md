# Benchmark findings

**Status update (after commits 087059b / 2e3efaa / c4f0014):** §1a–d, §2a, and
the `.timeout()` part of §0 are FIXED. Remaining open: §2b, §2c, §3a, §3b.

Bugs and surprises surfaced by `python -m benchmarks.bench --all-algorithms`.
Numbers below come from a 3-repeat run on Intel i7-6820HQ @ 2.70 GHz, Python
3.13. Reproduce with `--all-algorithms --repeat 3 --timeout 10 --json …`.

Each item is a real library issue, not a bench artifact. Listed roughly by
severity: capability-lattice gaps first (these are runtime crashes triggered
by the auto-selector's promise), then quality/perf surprises (auto-pick
losing to a sibling), then reporting issues.

---

## 1. Capability-lattice gaps — algorithms claim compatibility, crash at runtime

**FIXED in c4f0014** — each algorithm below gained a `compatible_with`
override that adds the missing precondition (hashable state / not
TourSpace+CSPSpace / dict state). 9 regression tests at
`tests/test_compatibility_guards.py`. Auto-selector and explicit
`candidates=[X]` both honour the new gate.

`Algorithm.compatible_with(space)` returns `True` when `cls.requires ⊆
space.capabilities`. Several algorithms have requirements that are necessary
but not sufficient: they additionally need hashable states, integer
neighborhoods, recursion-safe depths, or the CSP shape. The lattice doesn't
model these, so the algorithm gets selected (or hand-picked via `candidates`)
and crashes the moment `solve()` is called.

### 1a. BFS / DFS / IDDFS on `CSPSpace`

```
TypeError: unhashable type: 'dict'
```

CSP state is an assignment dict like `{0: 4, 1: 6, …}`. BFS/DFS/IDDFS put
states into a `set` for visited-tracking. Capability requirements (`GOAL +
SUCCESSORS`) are met, so the algorithms are offered — and immediately blow up.

Fix: either (a) add a `HASHABLE_STATE` capability the uninformed algorithms
require, or (b) wrap CSP states in a `frozenset(items)` adapter before
visited-tracking, or (c) have CSPSpace expose states as tuples.

### 1b. DifferentialEvolution on `TourSpace`

```
KeyError: (9, 7.2)
```

DE is a continuous optimizer — it generates real-valued vectors. TourSpace's
distance dict is keyed on integer city IDs. DE only requires `EVALUATE`, which
TourSpace satisfies, so it gets offered. The first mutation produces float
values; `distances[(city, mutated_float)]` misses.

Fix: DE should require a `CONTINUOUS` capability (or `BOUNDS` declaring real
intervals), and TourSpace should declare `DISCRETE/PERMUTATION` instead.

### 1c. Backtracking / ForwardChecking on plain `Space` (e.g. 8-puzzle)

```
RecursionError: maximum recursion depth exceeded
```

These CSP algorithms only require `GOAL + SUCCESSORS`, which plain `Space`
satisfies. They get offered for 8-puzzle and recurse forever because the
puzzle's branching factor and state space aren't CSP-shaped.

Fix: require a CSP-specific capability (`VARIABLES`, `DOMAINS`, or
`CONSTRAINTS`).

### 1d. MinConflicts on plain `Space`

Doesn't crash — silently returns `found=False` for every run. Same root cause
as 1c: requirements are satisfied by non-CSP spaces but the algorithm has no
useful behavior there.

---

## 2. Auto-pick is wrong — a sibling algorithm strictly dominates

`Solver._select()` picks `max(compatible, key=power_rank)`. Several
`power_rank` values are calibrated against an assumed but unverified ordering.
The head-to-head shows the auto-pick losing on real workloads.

### 2a. ForwardChecking loses to Backtracking on n-queens

**FIXED in 2e3efaa** — FC `power_rank` demoted 11 → 8 (below Backtracking).
Real domain-pruning still TODO; comment in `pathos/algorithms/csp.py`
documents the situation. README numbers refreshed.

Original observation:

| N  | Backtracking | ForwardChecking | nodes (both) |
|---:|---:|---:|---:|
| 12 | 0.012s | 0.020s | 261 |
| 14 | 0.147s | 0.237s | 1 899 |
| 16 | 1.249s | 1.926s | 10 052 |

Node counts are *identical*. So FC isn't actually pruning — it's running plain
backtracking with an extra layer of overhead. This matches the persistent
warning `Capabilities declared but not used by ForwardChecking: domains,
constraints, variables` seen in `tests/test_spaces_csp.py`: FC isn't reading
the `@domain`/`@constraint` decorations to do forward-checking work.

Fix: actually implement forward checking (prune neighbouring variables'
domains on each assignment), OR demote FC's `power_rank` below Backtracking
until it does.

### 2b. TabuSearch picked over HillClimbing on TSP — 5× slower, no quality gain

| cities | HC time | TS time | HC cost | TS cost |
|---:|---:|---:|---:|---:|
|  5 | 0.0001 | 0.0014 | 197.76 | 197.76 |
|  8 | 0.0002 | 0.0065 | 253.28 | 253.28 |
| 12 | 0.0018 | 0.0249 | 272.25 | 272.25 |
| 16 | 0.0061 | 0.0513 | 354.89 | 350.34 |
| 20 | 0.0170 | 0.1144 | 436.41 | 383.37 |
| 25 | 0.0386 | 0.2891 | 411.35 | 409.27 |

HillClimbing wins on speed everywhere and ties on quality through 12 cities.
TabuSearch only starts paying off at 16+ cities, and even there the margin
is modest. The fixed `power_rank` (TS=18 > HC=15) is too coarse — it doesn't
account for problem size.

Fix: either (a) accept that `power_rank` is a heuristic and let users override
with `solver(candidates=[…])`, or (b) make the rank size-aware via a hook on
the space.

### 2c. AStar picked over WeightedAStar / GreedyBestFirst on 8-puzzle

| scramble | AStar time | WAStar time | Greedy time | AStar cost | WAStar cost | Greedy cost |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.0042 | 0.0016 | 0.0020 | 20 | 20 | 46 |
| 30 | 0.0136 | 0.0041 | 0.0025 | 24 | 24 | 46 |
| 40 | 0.0171 | 0.0053 | 0.0035 | 26 | 30 | 44 |
| 50 | 0.0098 | 0.0092 | 0.0030 | 22 | 28 | 62 |

WeightedAStar matches A\*'s optimal cost at scramble=20,30 at 2-3× the speed.
Greedy is faster still but trades quality. A\* keeps the rank because it's
admissible — fine if optimality matters, but the auto-pick currently has no
way for users to say "ε-optimal is fine, give me speed."

Not a bug; a missing axis. Worth documenting.

---

## 3. Reporting issues

### 3a. Local-search algos report `found=True, cost=1.0` on puzzle8

HillClimbing/TabuSearch/LocalBeamSearch/SimulatedAnnealing/GA/DE all report
`found=True` and `cost=1.0` on every 8-puzzle run — but they never reach
GOAL. They ignore the `@goal` predicate and minimize the `@evaluate` function
(which returns `1.0` per step in `examples/puzzle8.py`). The "found" flag is
set when the algorithm terminates by its own stopping rule, not when GOAL
holds.

Fix: when a space has both `GOAL` and `EVALUATE`, local-search algorithms
should check the goal predicate on their best state before claiming
`found=True`.

### 3b. DFS returns wildly suboptimal paths

| scramble | DFS path cost | optimal |
|---:|---:|---:|
| 10 | 4 022   | 10 |
| 30 | 11 018  | 24 |
| 40 | 14 712  | 26 |
| 50 | 15 796  | 22 |

Expected for uninformed DFS but worth a one-line README caveat next to the
algorithm table, since the cost number looks alarming in benchmark output.

---

## Suggested triage order

1. **1a, 1b, 1c, 1d** — these are 4 separate runtime crashes induced by the
   auto-selector. Fix the capability model. High signal-to-effort.
2. **2a** — FC bug is small and concrete: either fix the pruning or demote
   the rank.
3. **3a** — small reporting fix; one-line guard at end of local-search.
4. **2b, 2c** — deeper questions about how `power_rank` should work; punt
   until 1 and 2a are done.
5. **3b** — README caveat or leave as-is.
