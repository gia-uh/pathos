# pathos — tasks

## In-flight

- [ ] **[Handoff 2026-06-10 18:41 — Apply AnytimeLocal vaporize-on-watchdog fix to AnytimeAStar](../../vault/+/agent_drafts/handoffs/handoff-2026-06-10-1841-anytime-astar-vaporize-fix.md)** — next action: mirror the AnytimeLocal pattern at `pathos/algorithms/informed.py:364-382`, with mechanism tests modelled on `tests/test_anytime_local_budget.py`.

## Backlog (bench-driven, in priority order)

- AnytimeAdversarial defensive-pass: same try/except + per-phase deadline plumbing as AnytimeLocal/AnytimeAStar; AB/Negamax already cancel-aware internally so this is belt-and-suspenders.
- Slice-2 TourSpace time-windows extension — R4 baseline is `1.97e8` (penalty-folded); the spec already records `missing_capability: time_windows`. First-class capability is the path to a real improvement.
- Slice-3 ScheduleSpace precedence extension — R5 fairness fix landed (`e6150cf`) but the cascade still lands far from optimum on penalty-folded precedence. First-class capability needed.
- Full default sweep with `--oracle auto` (4–6 h compute) — populates the empty Selection-suboptimal + Tunability columns and gives the audit-quality regression baseline.
