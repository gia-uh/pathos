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
| `.mode(mode)` | ✓ | **Renamed from `.optimality(mode)` in the anytime-cascade commit cycle.** Three values: `"exact"`, `"approximate"`, `"auto"` (default). `"auto"` makes AnytimeAStar win selection and auto-applies a 3600s default timeout. `solver(mode=…)` kwarg is per-call (non-mutating). Tests in `tests/test_mode.py` (14) and `tests/test_mode_auto.py` (10). |
| `.timeout(seconds)` (with mode=auto) | ✓ | **Cooperative cancellation in the anytime-cascade commit cycle.** SIGALRM sets `space._cancel_token`; cooperating algorithms return best-so-far cleanly. 2s watchdog (SIGVTALRM) raises TimeoutError as backstop for algorithms not yet wired (IDA*, CSP). |

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
- **PSO ✓** — `ParticleSwarm` added in commit 52f4e51. Pure-Python implementation requiring numeric vector state; same continuous-only carveout as DE.

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
| Auto-solver with power-rank selection | ✓ — context-aware `score_for(space)` + `mode` knob; FINDINGS §2 fully closed |
| Anytime cascade meta-algorithm for A*-family | ✓ — `AnytimeAStar` registered; 6-phase cascade with cancel-token cooperation |
| Anytime cascade meta-algorithm for CSP-family | ✓ — `AnytimeCSP` registered; `[MinConflicts (if EVALUATE), Backtracking]` cascade |
| Anytime cascade meta-algorithm for local-search family | ✓ — `AnytimeLocal` registered; `[HillClimbing, SimulatedAnnealing, TabuSearch]` cascade |
| Anytime cascade meta-algorithm for adversarial family | ✓ — `AnytimeAdversarial` registered; iterative deepening over AlphaBeta (2-player) or Negamax (3+ player) with PV-first move ordering |
| Cancel-token primitive | ✓ — `pathos/core/cancel.py`; checked by 10 algorithms in v1 |
| SearchResult.epsilon | ✓ — admissible algorithms emit 1.0, WeightedAStar emits weight, Greedy emits inf |
| `SearchResult` uniform return type | ✓ |
| Warning system for unused capabilities | ✓ — `Solver._select` emits `UserWarning` when capabilities are unused |
| Type stubs for mypy/pyright compatibility | ✓ — entire codebase is `mypy --strict` clean (commit 9ab0bc8) |

"Out of scope for v1" items: NumPy/Rust backends ✓ (not implemented).
Parallel execution ⚠ — spec says reserved, code says shipped.

---

## Summary

**True gaps (planned, not implemented):**
1. ~~PSO algorithm~~ ✓ commit 52f4e51
2. AC-3 algorithm (or as preprocessing utility)
3. Memetic GA, Iterated Local Search, WalkSAT (cross-family combos)
4. Hill Climbing steepest/stochastic sub-variants
5. TourSpace 3-opt neighborhood
6. Function-signature validation in decorators
7. ~~`.timeout()` actually being respected by algorithms~~ ✓ commit 087059b
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

**Status after 2026-06-02 session — benchmark audit fully green.**
§2c closed by the `optimality` knob (commit 8ba0aec). All FINDINGS items now
either FIXED or explicitly out-of-scope.

**Status after 2026-06-02 anytime-cascade ship (13 commits).**
`optimality` knob renamed to `mode={"exact", "approximate", "auto"}`,
default `"auto"`. `AnytimeAStar` meta-algorithm registered with 6-phase
cascade. `CancelToken` primitive added; 10 algorithms (HC, TabuSearch,
LocalBeamSearch, SA, GA, DE, PSO, BFS, DFS, IDDFS, UCS, AStar, WAStar,
Greedy, BidirA*) check it at top of main loop. SIGALRM handler sets the
token instead of raising; 2s SIGVTALRM watchdog backstops for non-
cooperating algorithms (IDA*, CSP).

**Status after 2026-06-02 AnytimeCSP ship.** Cascade pattern extended to
the CSP family: `AnytimeCSP` wins selection under `mode="auto"` for any
CSP-shaped space (initial state is a dict). Cascade is
`[MinConflicts (max_iter=200, only if EVALUATE present), Backtracking]`.
Algorithm base gained an `optional` class attr so meta-algorithms can
declare capabilities they consume dynamically — Solver treats
`requires | optional` as "used" for the unused-capability warning. Tests
at `tests/test_anytime_csp.py` (9).

**Status after 2026-06-02 AnytimeLocal ship.** Cascade pattern extended
to the local-search family: `AnytimeLocal` wins selection under
`mode="auto"` for any pure-optimization space (`{SUCCESSORS, EVALUATE}`,
no GOAL). Cascade is `[HillClimbing (max_restarts=3),
SimulatedAnnealing (max_iter=500), TabuSearch (max_iter=200,
tabu_size=20)]` — fast-probe followed by escape phases. Lower-cost
incumbent wins across phases (AnytimeAStar's `_is_better` rule, not
AnytimeCSP's first-found exit). `AnytimeLocal.requires` omits GOAL so
the goal-honoring filter in `Solver._select` cedes goal-bearing spaces
to `AnytimeAStar` / uninformed algorithms. Tests at
`tests/test_anytime_local.py` (9). Genuinely open work:
AnytimeAdversarial meta-algorithm (spec sketch), per-phase budget
enforcement, IDA* cancel-token integration (recursive shape).

**Status after 2026-06-03 AnytimeAdversarial ship.** Cascade pattern
extended to the adversarial family — fourth and final algorithm family
covered under `mode="auto"`. `AnytimeAdversarial` registered; routes
to `AlphaBeta` for 2-player and `Negamax` for 3+-player spaces via
`_phase_class()`. Iterative deepening from depth 1 to `max_depth`,
threading the previous depth's principal variation as `pv_hint` into
the next phase's `AlphaBeta` / `Negamax` constructor — AB pruning
becomes substantially more effective with correct move ordering.
Cancel-token cooperation added to `AlphaBeta._ab`, `Minimax._minimax`,
`Negamax._negamax` (top of recursion → `(nan, None)` sentinel → root
`solve()` returns `not_found`) and `MCTS.solve` (top of iteration loop
→ best-so-far from partial tree). All three of AB/Minimax/Negamax now
populate `SearchResult.path` with `[(action, state), ...]` — contract
change documented in
[2026-06-03-pathos-anytime-adversarial-design](../../vault/Atlas/Architecture/2026-06-03-pathos-anytime-adversarial-design.md).

Bundled mode-contract fixes: `MCTS.score_for` bumps to 53 under
`mode="approximate"` (closes the prior gap where approximate did
nothing for game spaces); `Negamax.score_for` bumps to 52 when
`space._players > 2` (closes a pre-existing bug where `AlphaBeta`
silently won `"exact"` selection on multi-player games despite not
honouring `_players`). Tests at `tests/test_anytime_adversarial.py`,
`tests/test_adversarial_cancel.py`, `tests/test_adversarial_pv.py`,
`tests/test_adversarial_mode.py`.

**Genuinely open work after this ship:** per-phase budget enforcement
(deep depths can still blow a tight total budget if the cancel checks
inside `_ab` don't fire often enough for cheap leaf evals); IDA*
cancel-token integration (recursive shape, deferred); transposition
tables, aspiration windows, killer-move heuristic for AB (all listed
as separate follow-ups in the design spec's "Out of scope" section).
