from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


BENCH_SCHEMA: int = 1


@dataclass
class RunRow:
    """One row per (suite, size, seed, mode). `mode` is one of:
    auto_headline | auto_stress | oracle_<AlgName>.
    `missing_capability`, when set, names the constraint class that
    couldn't be expressed; the row is the penalty-folded workaround.
    """
    suite: str
    size: int
    seed: int
    mode: str
    algorithm: str
    cost: float | None
    feasible: bool
    elapsed: float
    nodes_expanded: int
    missing_capability: str | None = None
    epsilon: float | None = None


@dataclass
class SuiteRow:
    """One row per (suite, size) in the regression table — median of
    seeds for cost/elapsed, set-union of algorithm names."""
    suite: str
    size: int
    auto_headline_cost: float | None
    auto_headline_feasible: bool
    auto_headline_algorithm: str
    auto_stress_cost: float | None
    auto_stress_feasible: bool
    oracle_cost: float | None
    oracle_algorithm: str | None
    gap_to_oracle_pct: float | None
    gap_to_lb_pct: float | None
    tunability_gain_pct: float | None
    tunability_algorithm: str | None
    missing_capability: str | None


@dataclass
class NoCliffFailure:
    suite: str
    size: int
    seed: int
    failure_kind: str  # "crash" | "not_found" | "infeasible"
    detail: str  # exception repr or violation count, for triage
    reproduction_args: str  # CLI snippet


@dataclass
class Report:
    bench_schema: int
    meta: dict[str, str]
    suite_rows: list[SuiteRow] = field(default_factory=list)
    raw_rows: list[RunRow] = field(default_factory=list)
    no_cliff_failures: list[NoCliffFailure] = field(default_factory=list)
