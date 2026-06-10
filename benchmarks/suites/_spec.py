from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SuiteSpec:
    """Uniform contract every benchmark suite exposes.

    `generate(size, seed)` produces a problem instance (deterministic
    per seed). `express(instance)` wraps it as a `pathos` space ready
    for `.solver().solve()`. `budgets[tier]` is the auto-headline budget
    in seconds. `lower_bound`, if set, returns a numeric LB on the
    instance (used for absolute-gap reporting).
    """
    id: str
    family: str  # "csp" / "tour" / "graph" / "schedule" / "evaluate" / "mixed"
    constraint_classes: frozenset[str]  # subset of {"i","ii","iii","iv","v"}
    expressibility: str  # "full" | "partial" | "gap"
    sizes: dict[str, tuple[int, ...]]  # {"S": (...), "M": (...), "L": (...)}
    generate: Callable[..., Any]
    express: Callable[[Any], Any]
    budgets: dict[str, float]
    lower_bound: Callable[[Any], float] | None = None
    # If set, the suite is a missing-capability suite. The string names
    # the constraint class that cannot be expressed (e.g. "time_windows").
    missing_capability: str | None = None
    # Per-suite per-tier feasibility tolerance: an answer with
    # violations > tolerance counts as infeasible for the no-cliff gate.
    tolerance: float = 0.0
    notes: str = ""
