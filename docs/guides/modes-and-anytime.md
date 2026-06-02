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

## Extending to other algorithm families (v2)

The cascade isn't a special case in the `Solver` — it's just an `Algorithm`
whose `score_for` wins under a specific mode. Future meta-algorithms
(`AnytimeLocal`, `AnytimeCSP`, `AnytimeAdversarial`) will slot in the same
way: register themselves, return a large score under `mode="auto"` on their
capability set, run their own cascade in `solve()`.

The cancel-token primitive is universal — every metaheuristic and path-search
algorithm already checks it, so they all become naturally anytime as soon as a
meta-algorithm composes them.
