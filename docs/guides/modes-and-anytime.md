# Modes & Anytime delivery

PATHOS gives you three execution modes that trade off optimality against
wall-clock time. The default (`"auto"`) is **anytime** — it always returns the
best answer it can find within your budget, instead of `not_found`.

## The three modes

| Mode | When to use | What runs |
|---|---|---|
| `"auto"` (default) | You want the best answer the budget allows. | [`AnytimeAStar`][pathos.algorithms.informed.AnytimeAStar] cascade: `[Greedy, WAStar(5), WAStar(3), WAStar(2), WAStar(1.5), AStar]` |
| `"exact"` | You need optimality and have unbounded time. | Admissible algorithm picked by `score_for`. No implicit timeout. |
| `"approximate"` | You want a single-shot bounded-suboptimal answer, no cascade overhead. | `WeightedAStar` (or sibling). No implicit timeout. |

```python
# Default — anytime cascade with 1h budget
space.solver().solve()

# Anytime with explicit budget
space.solver(timeout=60).solve()

# Single-shot admissible (proves optimal)
space.solver(mode="exact").solve()

# Single-shot bounded-suboptimal (fast)
space.solver(mode="approximate").solve()
```

`mode="auto"` automatically injects a `3600s` default timeout if neither
`Space.timeout()` nor `solver(timeout=…)` sets one. The other modes have
**no** implicit timeout — they run to natural completion.

## What `AnytimeAStar` does

When `mode="auto"` is active on an A\*-family space
(`{SUCCESSORS, GOAL, HEURISTIC, EVALUATE}`), the solver selects
[`AnytimeAStar`][pathos.algorithms.informed.AnytimeAStar]. Its `solve()`
runs six phases in sequence, keeping the best incumbent across phases:

```
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ GreedyBestFirst │→ │ WAStar(w=5)  │→ │ WAStar(w=3)  │→ │ WAStar(w=2)  │
│ (plant first    │  │ ε ≤ 5        │  │ ε ≤ 3        │  │ ε ≤ 2        │
│  incumbent fast)│  │              │  │              │  │              │
└─────────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
                                                                  │
                                                                  ▼
                          ┌──────────────┐  ┌─────────────────┐
                          │ WAStar(w=1.5)│→ │ AStar           │
                          │ ε ≤ 1.5      │  │ ε = 1 (optimal) │
                          │              │  │                 │
                          └──────────────┘  └─────────────────┘
```

The cascade order is **fast-first**: `GreedyBestFirst` finishes in
milliseconds and plants an incumbent; each `WeightedAStar` phase tightens the
ε-bound; the final `AStar` proves optimality if the budget permits. Between
phases the cascade checks `space._cancel_requested()` — if the global
timeout has fired, the cascade returns whatever incumbent it has so far
instead of `not_found`.

!!! tip "The anytime guarantee"
    Even with a microsecond budget, `GreedyBestFirst` typically finishes
    before the timeout. You get a feasible (suboptimal) path **always** on
    A\*-family spaces.

## Reading `SearchResult.epsilon`

The new `epsilon` field on `SearchResult` tells you exactly how good the
answer is:

| `epsilon` | Meaning | Set by |
|---|---|---|
| `1.0` | Proven optimal | `AStar`, `IDAstar`, `BidirectionalAStar`, `BFS`, `UCS`, `IDDFS` |
| `>1.0` | Cost ≤ ε × optimal | `WeightedAStar(weight=ε)` |
| `inf` | Unbounded suboptimal | `GreedyBestFirst`, `DFS` |
| `None` | Not applicable | Metaheuristics (`HC`, `SA`, `Tabu`, `GA`, `DE`, `PSO`, `LocalBeam`) |

Use the `SearchResult.optimal` property (derived from `epsilon == 1.0`)
for the common "did we prove optimality" check:

```python
result = space.solver(timeout=10).solve()
if result.optimal:
    print(f"Proven optimal: cost {result.cost}")
elif result.epsilon and result.epsilon != float("inf"):
    print(f"Within {result.epsilon}× optimal: cost {result.cost}")
elif result.found:
    print(f"Feasible (no quality bound): cost {result.cost}")
else:
    print("No incumbent found")
```

## When `mode="auto"` is *not* what you want

- **You explicitly need provable optimality and can afford the time.**
  Use `mode="exact"`. The cascade's earlier phases produce identical
  optimal results on small problems, but the overhead of 5 wasted phases
  before `AStar` can add 100-200ms on tiny instances.
- **The problem isn't A\*-family** (no heuristic declared, or CSP-shaped).
  `AnytimeAStar.score_for` returns `-inf` outside auto — you fall back to
  the base algorithm pick anyway. No harm, but also no anytime benefit.
- **You need a specific algorithm pinned.** Pass
  `candidates=[YourAlgorithm]` — the cascade is bypassed.

```python
# Pin a specific algorithm; mode doesn't change selection
result = space.solver(candidates=[AStar], mode="auto").solve()
assert result.algorithm == "AStar"
```

## Other family cascades

The cascade isn't a special case in `Solver` — it's just an `Algorithm` whose
`score_for` wins under `mode="auto"` for a specific capability shape. The
same pattern covers all four families:

| Family | Meta-algorithm | Capability shape | Cascade |
|---|---|---|---|
| A\* (informed path) | [`AnytimeAStar`][pathos.algorithms.informed.AnytimeAStar] | `{SUCCESSORS, GOAL, HEURISTIC, EVALUATE}` | `[Greedy, WAStar(5,3,2,1.5), AStar]` — six phases, lowest-cost incumbent wins. |
| Local search | [`AnytimeLocal`][pathos.algorithms.local.AnytimeLocal] | `{SUCCESSORS, EVALUATE}` (no `GOAL`) | `[HillClimbing, SimulatedAnnealing, TabuSearch]` — fast-probe + two escape phases. Lowest-cost incumbent wins. |
| CSP | [`AnytimeCSP`][pathos.algorithms.csp.AnytimeCSP] | `CSPSpace` (initial state is a dict) | `[MinConflicts (only if @evaluate), Backtracking]` — first phase to return a consistent complete assignment wins. |
| Adversarial | [`AnytimeAdversarial`][pathos.algorithms.adversarial.AnytimeAdversarial] | `.adversarial() + @terminal + @utility` | Iterative deepening from depth 1 to `max_depth` over `AlphaBeta` (2-player) or `Negamax` (3+ player), threading the previous depth's principal variation as `pv_hint` for move ordering. |

All four register themselves with `register`, return a large `score_for`
under `mode="auto"` on their capability set, and run their own cascade in
`solve()` — no special case in `Solver`. Selection between them is unambiguous
because their capability shapes don't overlap: `AnytimeLocal.requires` omits
`GOAL`, so it cedes goal-bearing spaces to `AnytimeAStar`; `AnytimeCSP`
matches on CSP shape, etc.

The cancel-token primitive is universal — every metaheuristic, path-search
algorithm, and adversarial recursion (`Minimax`, `AlphaBeta`, `Negamax`,
`MCTS`) checks it, so they all become naturally anytime as soon as a
meta-algorithm composes them.
