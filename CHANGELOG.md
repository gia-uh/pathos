# Changelog

All notable changes to this project are documented here. Format: Keep a Changelog.

## [Unreleased]

## [v0.2.0] - 2026-06-03

Anytime cascade pattern now covers all four classical search families.
Under `mode="auto"` (the default), a family-appropriate meta-algorithm wins
selection and runs a cascade tuned to the problem shape — always returning
the best incumbent within budget instead of `not_found`.

### Features

- **AnytimeCSP** — cascade `[MinConflicts (only if @evaluate), Backtracking]`
  wins selection under `mode="auto"` on CSP-shaped spaces. First phase to
  return a consistent complete assignment wins.
- **AnytimeLocal** — cascade `[HillClimbing, SimulatedAnnealing, TabuSearch]`
  wins selection under `mode="auto"` on pure-optimization spaces
  (`@successors + @evaluate`, no `@goal`). Lowest-cost incumbent wins across
  phases.
- **AnytimeAdversarial** — iterative deepening from depth 1 to `max_depth`
  over `AlphaBeta` (2-player) or `Negamax` (3+ player), threading the
  previous depth's principal variation as `pv_hint` into the next phase
  for move ordering. Wins selection under `mode="auto"` on adversarial
  spaces.
- **Adversarial cancel-token cooperation** — `AlphaBeta`, `Minimax`,
  `Negamax` check the token at the top of their recursion (`(nan, None)`
  sentinel that collapses to `not_found` at the root); `MCTS` checks it
  at the top of its iteration loop and returns best-so-far from the
  partial tree.
- **PV-first move ordering** for `AlphaBeta` and `Negamax` —
  `AnytimeAdversarial` feeds the previous iteration's principal variation
  in as a hint; α-β pruning becomes substantially more effective.
- **Path population** for `Minimax`, `AlphaBeta`, `Negamax` — they now
  return `SearchResult.path` as `[(action, state), ...]` like the other
  search families (contract change documented in the design spec).
- **Mode-contract overrides for MCTS and Negamax** — `MCTS.score_for` bumps
  to 53 under `mode="approximate"` (closing the prior gap where approximate
  did nothing for game spaces); `Negamax.score_for` bumps to 52 when
  `space._players > 2` (closing a bug where `AlphaBeta` silently won
  `"exact"` selection on multi-player games despite not honouring
  `_players`).

### Fixes

- CI: `publish.yml` renamed to `publish.yaml` to match the PyPI trusted
  publisher configuration.
- CI: `tomllib.load(file)` instead of `tomllib.loads(bytes)` for the
  version-consistency check.

### Docs

- Modes & Anytime delivery guide now documents the four-family cascade
  table.
- Cancel-token protocol guide moves the adversarial family from "watchdog
  backstop" to "cooperates"; only `IDAstar` and CSP recursion now rely
  on the watchdog.
- API reference exposes the four meta-algorithms (`AnytimeAStar`,
  `AnytimeLocal`, `AnytimeCSP`, `AnytimeAdversarial`).
- README and `docs/index.md` algorithm-family tables call out the default
  meta-algorithm per family.

## [v0.1.0] - 2026-05-30

Initial PyPI release. Production-ready classical AI search algorithms for
Python — no machine learning. Pure search.

Covers uninformed (BFS, DFS, IDDFS, UCS), informed (A\*, IDA\*, WAStar,
Greedy, BidirectionalA\*), local search (HillClimbing, TabuSearch,
LocalBeamSearch), evolutionary (GA, SA, DE, PSO), adversarial (Minimax,
AlphaBeta, Negamax, MCTS), and CSP (Backtracking, ForwardChecking,
MinConflicts) families.

`AnytimeAStar` cascade meta-algorithm with cooperative cancel-token
protocol; specialized spaces (`GraphSpace`, `CSPSpace`, `TourSpace`,
`GameSpace`).
