# Spec → code audit

Cross-references the design spec at `vault/Atlas/Architecture/2026-05-29-pathos-design.md`
against the repo on `main` as of commit `45dee75`. Goal: surface what's
planned but not yet implemented, what's implemented but missing from the
spec, and where the two have drifted.

Format: ✓ in place · ◐ partial · ✗ missing · ⚠ spec/code disagree.

---

## Core architecture

| Item | Status | Notes |
|---|:---:|---|
| `Space` base class | ✓ | `pathos/core/space.py` |
| Capability enum | ✓ | `pathos/core/capabilities.py` — 11 entries |
| Auto-solver (filter + power-rank) | ✓ | `pathos/core/solver.py:_select` |
| `SearchResult` dataclass | ✓ | `pathos/core/result.py` — matches spec fields exactly |

## Fluent builder methods

| Method | Status | Notes |
|---|:---:|---|
| `.initial(state)` | ✓ | |
| `.adversarial(players, maximizing_player)` | ✓ | |
| `.parallel(workers)` | ⚠ | **Spec says "API reserved, not implemented" — but it IS implemented** (recent commits 83af022, ebbc4e5, f7a57ad, a0ea240, bdfd1f4). Spec is stale. |
| `.timeout(seconds)` | ✓ | **Fixed in 087059b**: Solver.solve wraps the algorithm run in a SIGALRM-based wall-clock guard; returns `SearchResult.not_found(...)` on expiry. Both `space.solver(timeout=N)` and `space.timeout(N).solver()` honour it. 4 regression tests in `tests/test_timeout.py`. |

## Decorator hooks

All seven hooks are wired in `Space._make_hook`: `@space.successors`, `@goal`,
`@heuristic`, `@evaluate`, `@terminal`, `@utility`, `@reverse_successors`. ✓

Spec claim "validates the function signature at definition time" — **not
implemented**: `_make_hook` just stores the function. No `inspect.signature`
check anywhere in `pathos/`. Either drop the claim from the spec or add a
validation pass.

---

## Algorithm registry vs spec families

### Uninformed (spec: 4)
- BFS ✓ · DFS ✓ · IDDFS ✓ · UCS ✓

### Informed (spec: 5)
- A\* ✓ · IDA\* ✓ · Greedy Best-First ✓ · Weighted A\* ✓ · Bidirectional A\* ✓

### Local Search (spec: 3, one with sub-variants)
- Hill Climbing ◐ — registry has a single `HillClimbing`; spec promises
  "steepest + stochastic" variants. No `kind=` parameter in
  `pathos/algorithms/local.py`. Either expose a parameter / two classes,
  or drop the sub-variant promise.
- Tabu Search ✓
- Local Beam Search ✓

### Evolutionary / Metaheuristic (spec: 4)
- Genetic Algorithm ✓ · Simulated Annealing ✓ · Differential Evolution ✓
- **PSO ✗** — listed in spec, no class in registry.

### Adversarial (spec: 4)
- Minimax ✓ · Negamax ✓ · Alpha-Beta ✓ · MCTS ✓
- Spec also describes "MCTS/UCT with heuristic cutoff" requiring
  `{adversarial, successors, terminal, utility, evaluate}`. Current MCTS
  requires only `{SUCCESSORS, TERMINAL, UTILITY}`. Heuristic-cutoff
  variant is not modeled as a separate algorithm (and current MCTS
  doesn't seem to consume `@evaluate` if declared).

### CSP (spec: 4)
- Backtracking ✓ · Forward Checking ⚠ (see FINDINGS 2a — doesn't actually
  prune) · Min-Conflicts ✓
- **AC-3 ◐** — class `AC3` exists in `pathos/algorithms/csp.py` but is
  not `@register`'d, so it's invisible to the auto-selector. Likely
  intentional (AC-3 is preprocessing) but neither documented nor
  exposed via the package API. Either expose explicitly or document
  the rationale.

### Cross-family combos (spec lists 3)
- `{evaluate} + {successors}` → Memetic GA, Iterated Local Search — **both ✗**
- `{constraints} + {evaluate}` → Constraint Optimization (WalkSAT) — **✗**
- `{successors, evaluate, heuristic}` → Weighted A\* with real cost
  function — folded into `WeightedAStar` ✓

### Capability-lattice integrity
The lattice declared in the spec maps capability-sets to family-level
algorithm lists. The actual `requires` on each class is finer — but, as
documented in `FINDINGS.md` §1, **the declared `requires` is necessary but
not sufficient** for several algorithms: BFS/DFS/IDDFS need hashable state,
DE needs continuous domains, BT/FC/MC need CSP shape. The spec doesn't
acknowledge these implicit assumptions; the runtime crashes are the
result. Either the spec's capability enum needs finer-grained shapes
(`HASHABLE_STATE`, `CONTINUOUS`, `CSP_SHAPE`) or each algorithm needs an
explicit `compatible_with` override that adds the missing check.

---

## Specialized subspaces

| Subspace | Status | Notes |
|---|:---:|---|
| `Space` | ✓ | |
| `GraphSpace` | ✓ | Auto-`successors` from adjacency dict, plus auto-`evaluate` from edge weights (latter is a nice bonus the spec doesn't mention) |
| `CSPSpace` | ✓ | Auto-`successors` + `goal` from variables/domains/constraints |
| `TourSpace` | ◐ | Auto-`successors` is **2-opt only**; spec promises "2-opt/3-opt". Either drop the 3-opt promise or add it. |
| `GameSpace` | ✓ | `adversarial(players=2)` default |

---

## Package structure vs spec layout

Spec maps to:
```
pathos/
  core/{space,solver,result,capabilities}.py
  spaces/{graph,csp,tour,game}.py
  algorithms/{uninformed,informed,local,evolutionary,adversarial,csp}.py
  backends/{pure,numpy}.py
```

Actual:
- `pathos/core/` — matches, **plus** `parallel.py` (new, not in spec — supports the parallel-evaluation feature)
- `pathos/spaces/` — matches ✓
- `pathos/algorithms/` — matches, plus `base.py` (Algorithm ABC, fine to omit from spec)
- `pathos/backends/` — **does not exist**. Spec calls for `backends/pure.py` as the v1 default. Either create the dir (today everything is "pure" inline) or remove the backends concept from v1 spec — it only matters once NumPy/Rust backends start landing.

---

## SearchResult

Spec fields and actual `@dataclass SearchResult` fields are identical:
`solution, path, cost, algorithm, nodes_expanded, elapsed, found`. ✓

---

## v1 scope checklist (spec section "v1 Scope")

| Spec item | Status |
|---|:---:|
| All algorithm families pure Python | ◐ — PSO and AC-3 absent; HC sub-variants absent; cross-family combos absent |
| `Space`, `GraphSpace`, `CSPSpace`, `TourSpace`, `GameSpace` | ✓ |
| Auto-solver with power-rank selection | ✓ (but several mis-ranks — see FINDINGS §2) |
| `SearchResult` uniform return type | ✓ |
| Warning system for unused capabilities | ✓ — `Solver._select` emits `UserWarning` when capabilities are unused |
| Type stubs for mypy/pyright compatibility | ✓ — entire codebase is `mypy --strict` clean (commit 9ab0bc8) |

"Out of scope for v1" items: NumPy/Rust backends ✓ (not implemented).
Parallel execution ⚠ — spec says reserved, code says shipped.

---

## Summary

**True gaps (planned, not implemented):**
1. PSO algorithm
2. AC-3 algorithm (or as preprocessing utility)
3. Memetic GA, Iterated Local Search, WalkSAT (cross-family combos)
4. Hill Climbing steepest/stochastic sub-variants
5. TourSpace 3-opt neighborhood
6. Function-signature validation in decorators
7. `.timeout()` actually being respected by algorithms
8. `backends/` directory (or remove from spec until v1.1)

**Spec is stale (code is ahead):**
1. Parallel evaluation IS implemented — spec still says reserved
2. GraphSpace auto-evaluates edge costs — spec doesn't mention this
3. MCTS doesn't model heuristic-cutoff as a distinct variant

**Capability-lattice model is incomplete** (see FINDINGS.md §1): several
required-but-unmodeled state shapes (hashable, continuous, CSP) cause
the auto-selector to offer crashing algorithms.

**Status after 2026-05-30 session (commits 087059b, 2e3efaa, c4f0014, 3c5246c):**
Spec sync ✓, FC rank demote ✓, 4 lattice-crash guards ✓, `.timeout()` wired ✓.
Genuinely open: PSO, HC sub-variants, TourSpace 3-opt, function-signature
validation in decorators, the deeper power_rank-vs-size question (2b, 2c),
and reporting fixes for local-search `found=True` on puzzle8 (FINDINGS §3a).
