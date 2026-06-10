# Realistic Benchmark Suite — Design

**Status:** spec · 2026-06-10
**Slice:** 1 of N (this spec ships only the bench scaffolding; constraint-class space extensions are sequenced as slices 2-N driven by what slice 1 surfaces).

## Goal

Validate that `pathos`'s default algorithm selection produces *usable* answers on big, real-life planning / logistics / scheduling instances within a wall-clock budget — and stays usable when the user reaches for `candidates=[…]` or `mode=` to tune.

The existing `benchmarks/bench.py` covers classical-paper sizes (N-Queens up to 16, TSP up to 25 cities, 8-puzzle scramble up to 50) and has driven roughly ten selection fixes documented in `benchmarks/FINDINGS.md`. This spec extends that loop to real-job conditions: 100–1000-entity instances, multi-resource capacity, heterogeneous entities, soft / multi-objective scoring, and constraint shapes (time windows, precedence) that today's spaces can't express — recorded as structured `missing-capability` gaps that prioritize later space extensions.

The new suite **subsumes** the classical bench: classical generators carry over as the easy bottom rung of a unified difficulty ladder, so selection regression coverage stays continuous from textbook problems to real-job scale.

## Why now

An upcoming real job requires `pathos` to act as the default solver on planning / scheduling / logistics workloads at sizes well past the current bench's reach. The contract that needs to hold:

- `space.solver(timeout=T).solve()` returns a feasible, high-quality answer in T seconds without the user knowing which algorithm is right.
- `space.solver(candidates=[X])` lets a knowledgeable user beat the default when they need to.

The current bench can't certify either at real-job scale, and several constraint classes the real job needs (time windows, precedence, multi-resource, multi-objective, soft preferences) aren't first-class in any pathos space today. This spec is the probe that turns the gap list from "we think" into "we know, ranked by impact."

## Non-goals

- Adversarial suites (game-tree search at depth). Out of scope.
- Replacing `benchmarks/bench.py`. The fast-feedback dev loop stays as-is.
- Extending TourSpace / ScheduleSpace / CSPSpace to model time-windows / precedence / soft-constraints. Those gaps surface from this suite; they ship as slices 2-N, one spec each.
- Designing a unified constraint-protocol layer. Same reason.
- Classical-paper-size scaling beyond what `bench.py` already covers.
- Real-instance dataset replay (e.g. Solomon VRPTW, Toronto timetabling). The bench's generators ship synthetic "uniform" flavors only; real-instance replay slots in later as a `flavor` parameter once the synthetic baseline is calibrated.

## Contract validated

Three guarantees, layered:

1. **No-cliff gate (hard).** Default solver never returns `not_found`, never crashes, never returns infeasible with violations > the suite's per-suite tolerance. Any single instance failing = a recorded bug for triage. *Carve-out:* suites flagged `missing_capability` (R4, R5; see below) are excluded from this gate unless they crash — their feasibility is bounded by the penalty-folding workaround, not by `pathos`'s selection.
2. **Anytime-quality headline (warning).** Default solver's objective ≤ 1.05 × the best objective among all compatible algorithms run at the same budget, taking the median across seeds per (suite, size). 5 % is the threshold; cases above it surface as selection-suboptimal flags.
3. **Tunability gain (informational).** When `candidates=[X]` for some explicit algorithm X beats the default by ≥ 5 %, the bench logs (suite, size, X, gain). This is not a bug — it's the catalog of "default reasonable, but the knowledgeable user can do better."

These three are reported separately. (1) gates the build (any failure = release-blocker until triaged or waived). (2) and (3) are tracked over time via the regression artifact, not gated.

## Suite catalog

Ten suites total: three classical from `bench.py` (the bottom of the ladder) plus seven realistic. Constraint classes use the i–v labels from the brainstorming session:

- i. time windows / deadlines
- ii. precedence / sequencing
- iii. capacity bands / multi-resource
- iv. soft constraints / multi-objective
- v. heterogeneous entities

| ID | Suite | Family | Tier | Stresses | Expressible today? |
|---|---|---|---|---|---|
| C1 | N-Queens | `CSPSpace` | classical | — | full |
| C2 | TSP (uniform random) | `TourSpace` | classical | — | full |
| C3 | 8-puzzle (informed search) | `Space` + `@heuristic` | classical | — | full |
| R1 | Power-grid blackout scheduling | `ScheduleSpace` | S / M / L | iii, v, iv | partial — multi-resource via penalty-folding |
| R2 | Employee rostering (shifts × staff, skill match, fairness) | `ScheduleSpace` + `CSPSpace` | S / M / L | iii, v, iv | partial — fairness via penalty, soft prefs are gap iv |
| R3 | Capacitated VRP (delivery routes, vehicle loads) | `TourSpace` + `GraphSpace` | S / M / L | iii, v | full — vehicle cap via penalty-folding |
| R4 | VRP with time windows | `TourSpace` | S / M / L | i, iii, v | **gap** — recorded as `missing_capability(time_windows)` |
| R5 | Project scheduling with precedence | `ScheduleSpace` + `CSPSpace` | S / M / L | ii, iii, iv | **gap** — recorded as `missing_capability(precedence)` |
| R6 | Bin packing (variable items, multi-bin) | pure `@evaluate` | S / M / L | v, iv | full |
| R7 | Large CSP timetabling | `CSPSpace` | S / M / L | iv, v | partial — soft prefs via penalty-folding |

### Sizes per realistic tier

- TSP / VRP variants: S = 100 / M = 300 / L = 1000 cities (for VRP, additionally 5 / 15 / 50 vehicles).
- Scheduling (ScheduleSpace-shaped): S = 20 × 72 / M = 50 × 168 / L = 100 × 336 (entities × slots).
- CSP timetabling: S = 50 / M = 150 / L = 300 variables.
- Bin packing: S = 50 / M = 200 / L = 1000 items.

Classical sizes stay at the current `bench.py` defaults (N-Queens 6 – 16, TSP 5 – 25, 8-puzzle scramble 10 – 50). They exist as ladder rungs for selection-continuity regression, not as headline metrics.

## Quality measurement

Per (suite, size, seed):

- **Auto headline** — `space.solver(timeout=T_head).solve()` → record `{cost, feasible, elapsed, algorithm}`.
- **Auto stress** — same with `T_stress = 5 × T_head`. Headline measures the contract; stress measures graceful degradation.
- **Oracle row** — for every algorithm `A` in `space.compatible_algorithms()`, run `space.solver(candidates=[A], timeout=T_head).solve()` → take the best `{cost, feasible}` and record which algorithm hit it.
- **Lower bound** (optional, where cheap) — TSP / VRP 1-tree LB; bin-packing waste LB; CSP arc-consistency LB. Where computable, also report `gap_to_lb`.

### Budget classes (T_head per tier)

- Classical: 5 s
- S realistic: 10 s
- M realistic: 30 s
- L realistic: 150 s

Default seed count = 5 per (suite, size). Approximate full-sweep wall-clock: 5 seeds × 10 suites × 3 tiers × 30 s mean × oracle-multiplier ≈ 1 – 2 hours on the reference machine. CLI flags (`--suites`, `--tiers`, `--repeat`, `--oracle off`) trim the sweep for quick iterations.

### Multi-objective handling

R1, R2, R5, R6, R7 are multi-objective. The bench reports objective values as the scalarized cost the space's `@evaluate` returns (which is already how pathos's algorithms minimize). Secondary objectives (fairness, slack, lateness) are logged alongside but do not enter the gap calculation. Rationale: the bench measures what `pathos` actually optimizes; reframing scalarization is a separate concern.

## Validation report

Per run, two artifacts in `benchmarks/realistic/`:

- `results-YYYY-MM-DD-HHMM.json` — full structured record: every (suite, size, seed, mode) row with cost / feasible / elapsed / algorithm / LB. Includes top-level `bench_schema: 1` for forward-compatible diff tooling, plus a `meta` block with CPU model, Python version, `pathos` git SHA.
- `REPORT.md` — auto-rendered markdown with five sections:

  1. **Regression table.** One row per (suite, size). Columns: `auto_headline_cost`, `auto_headline_feasible`, `oracle_cost`, `gap_to_oracle_pct`, `gap_to_lb_pct` (if LB available), `tunability_gain_pct`, `auto_algorithm`, `oracle_algorithm`.
  2. **No-cliff failures.** Hard bugs surfaced this run. Each row: (suite, size, seed, failure_kind, reproduction_args). Empty section if clean.
  3. **Selection-suboptimal flags.** Cases where `gap_to_oracle_pct > 5`. Each row names the auto-pick algorithm vs the oracle-best algorithm.
  4. **Tunability-gain log.** Cases where `candidates=[X]` beat the default by ≥ 5 %. Names `X` and the gain.
  5. **Missing-capability gaps.** One row per (suite, size) marked as `missing_capability`, with the constraint class, the penalty-folding workaround used, and a one-line "what a first-class capability would look like" sketch. These rows are the prioritized backlog for slices 2-N.

`REPORT.md` is generated by `benchmarks/realistic/report.py` from the JSON. Diff two REPORTs across runs to see what got tighter / what regressed.

`results-*.json` is gitignored (regenerable, large); `REPORT.md` is committed when Alex chooses (it's the human-readable diff artifact).

## Missing-capability suites

R4 (VRP with time windows) and R5 (project scheduling with precedence) cannot be expressed in current spaces without significant penalty-folding contortions. For each:

- Generator runs normally and emits the instance.
- The suite's *expression layer* attempts the closest fit (R4 folds time-window violations into a penalty term on a regular `TourSpace`; R5 folds precedence into per-step CSP constraints over `ScheduleSpace`).
- The instance runs and records a real quality number — useful as a today-baseline.
- The bench additionally emits a structured `missing_capability` record naming the constraint class, the workaround, and a sketch of what a first-class capability would look like.

This produces both (a) a real quality datapoint today, and (b) prioritized evidence for which space extension unblocks the most coverage. R4 and R5 are the seed inputs for slices 2 and 3.

Partial gaps in R1 / R2 / R7 (soft prefs, multi-objective) are recorded the same way at finer granularity — the constraint class and the workaround are logged even though the suite runs to completion. This is how slice 1 hands the prioritized backlog to slices 2-N.

## Module layout

```
benchmarks/
├── bench.py                      # existing fast-feedback runner — UNCHANGED
├── FINDINGS.md                   # existing audit log — UNCHANGED
├── suites/                       # NEW: one module per suite
│   ├── __init__.py               #   exports SUITE_REGISTRY: {id: SuiteSpec}
│   ├── classical.py              #   C1 / C2 / C3 — imports build_nqueens, build_tsp, build_puzzle8 from bench.py
│   ├── power_grid.py             #   R1
│   ├── rostering.py              #   R2
│   ├── vrp.py                    #   R3 (capacitated) + R4 (time-windows, missing-capability mode)
│   ├── project_scheduling.py     #   R5
│   ├── bin_packing.py            #   R6
│   └── timetabling.py            #   R7
└── realistic/                    # NEW: orchestration + reporting
    ├── __init__.py
    ├── runner.py                 #   CLI entry: python -m benchmarks.realistic
    ├── metrics.py                #   gap-to-oracle, gap-to-LB, feasibility
    ├── report.py                 #   JSON → REPORT.md renderer
    ├── lower_bounds.py           #   1-tree, waste-LB, arc-consistency LB
    └── results-*.json            #   gitignored, generated
```

`benchmarks/suites/` is shared infrastructure: each module defines a `SuiteSpec` dataclass with `generate(size, seed) → instance`, `express(instance) → Space`, and `metadata: {family, constraint_classes, expressibility}`. Both `bench.py` (eventually) and `realistic/runner.py` can consume the registry; for slice 1, only the new runner does.

Classical suites import the existing builders from `bench.py` — no duplication. This is the "fold in existing benchmarks" Alex asked for: classical and realistic share a single registry and a single ladder, even though `bench.py` keeps its own runner.

### CLI

```
python -m benchmarks.realistic                                    # full sweep, default seeds + budgets
python -m benchmarks.realistic --suites R1,R3,R6                  # subset
python -m benchmarks.realistic --tiers S,M --repeat 3              # cheaper iteration
python -m benchmarks.realistic --oracle off                       # auto-pick only, no oracle row
python -m benchmarks.realistic --diff REPORT-A.md REPORT-B.md     # cross-run diff (which suites got tighter / which regressed)
```

## Testing strategy

- Per-suite unit tests in `tests/test_benchmarks_suites_*.py`:
  - Generator is deterministic per seed.
  - Generated instances respect the requested size.
  - `express(instance)` produces a `Space` whose capabilities are sane and whose smallest size doesn't crash a default `solve()`.
- Runner integration test: full S-tier sweep across all suites at 1 seed × 1 s budget, asserts `REPORT.md` generated, no crashes, no unexpected exceptions.
- Metrics unit tests: gap math (against hand-computed examples), feasibility predicate, missing-capability emission shape.
- Report renderer test: known JSON → known REPORT.md fixture (golden file).

The bench does **not** assert specific quality numbers in tests. Locking in current behaviour as ground truth would defeat the bench's purpose (detect when current behaviour is wrong). Quality numbers live in the regression artifact, surfaced for human triage.

## Risks

- **Oracle-row blowup.** Running every compatible algorithm explicitly per (suite, size) is the bench's dominant cost. Mitigation: at L tier, oracle row defaults off; opt-in via `--oracle on`. At M tier, oracle is on by default. At S and classical, oracle always on.
- **Generator realism.** Synthetic uniform-random doesn't match real-job distributions. Mitigation: each generator accepts a `flavor` parameter; slice 1 ships `"uniform"` only. Real-instance replay (Solomon VRPTW, etc.) lands as a separate slice once Alex names datasets to model.
- **Hardware drift makes regression detection noisy.** Different machines / Python builds → different wall-clock. Mitigation: REPORT records CPU / Python / pathos SHA in the `meta` block; diff tooling normalises by reporting *relative* gaps (auto-vs-oracle, not absolute wall-clock). Headline budgets are quoted as targets, not contracts.
- **Missing-capability suites under-measuring.** Penalty-folded workarounds on R4 / R5 may produce numbers so bad that they dominate `gap_to_oracle` and mask other selection bugs. Mitigation: rows marked `missing_capability` display `gap_to_oracle: n/a (penalty-fold workaround)` and are excluded from no-cliff failure counting unless they crash.
- **Long full-sweep wall-clock discourages running.** 1 – 2 hours per full sweep is borderline interactive. Mitigation: a `--quick` flag opts into a trimmed run (M tier only, 3 seeds, oracle off) that finishes in ~10 min — the headline contract still gets exercised; tunability and oracle gaps are checked only on the default full sweep, expected to run weekly or pre-release.

## Out of scope, but enabled

The `missing_capability` records from R4 + R5 (and the partial-gap records from R1 / R2 / R7) become the prioritized backlog for slices 2-N. Each subsequent slice = one spec extending one space to model one constraint class, plus the matching bench row turning from `gap` to a real quality number. Expected sequence (subject to what slice 1's first run actually surfaces as worst gaps):

- **Slice 2:** Time windows in TourSpace → R4 turns green.
- **Slice 3:** Precedence in ScheduleSpace → R5 turns green.
- **Slice 4:** First-class soft constraints / multi-objective scoring → R2 / R7 partial gaps tighten.

The order and shape of those slices are not committed by this spec — they are output, not input.
