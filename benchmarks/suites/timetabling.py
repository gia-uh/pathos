"""R7 — Large CSP timetabling.

Variables: course-section IDs. Domain per variable: a (room, time)
slot pair. Hard constraint: no two variables share the same slot.
Soft preference (partial gap): each variable has a small set of
'preferred' slots; the bench folds preference violations into a
synthetic @evaluate.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
import pathos.algorithms  # noqa: F401 — register algorithms
from pathos import CSPSpace
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


@dataclass(frozen=True)
class TimetableInstance:
    variables: tuple[str, ...]
    slots: tuple[tuple[int, int], ...]   # (room, time)
    preferred: dict[str, frozenset[tuple[int, int]]]


def _generate(size: int, seed: int) -> TimetableInstance:
    rng = random.Random(seed)
    n_rooms = max(3, size // 10)
    n_times = max(5, size // 4)
    slots = tuple((r, t) for r in range(n_rooms) for t in range(n_times))
    variables = tuple(f"course_{i}" for i in range(size))
    preferred = {
        v: frozenset(rng.sample(slots, k=min(3, len(slots))))
        for v in variables
    }
    return TimetableInstance(variables=variables, slots=slots, preferred=preferred)


def _express(inst: TimetableInstance):
    csp = CSPSpace(variables=list(inst.variables))
    slots = inst.slots

    @csp.domain
    def dom(var):
        return list(slots)

    @csp.constraint
    def no_collision(assignment):
        used = set()
        for slot in assignment.values():
            if slot in used:
                return False
            used.add(slot)
        return True

    # Note: @evaluate hookup for soft prefs would require CSPSpace to
    # expose it; in slice 1 we rely on the constraint above and record
    # soft prefs as a partial gap. The instance retains `preferred` so
    # slice-N can wire it once CSPSpace gains soft constraints.

    return csp


register(SuiteSpec(
    id="R7",
    family="csp",
    constraint_classes=frozenset({"iv", "v"}),
    expressibility="partial",
    sizes={"S": (50,), "M": (150,), "L": (300,)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    notes=(
        "CSP timetabling — hard collision constraint, "
        "partial gap: soft_preferences (preferred slots) not first-class."
    ),
))
