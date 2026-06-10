"""R2 — Employee rostering on ScheduleSpace.

Each staff member has a skill set (subset of {"a","b","c"}); each shift
requires one skill. The bench folds skill-mismatch and shift-preference
violations into the penalty term — soft-preference modeling is the
partial gap recorded for slice 2-N follow-up.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
import pathos.algorithms  # noqa: F401 — register algorithms
from pathos import ScheduleSpace  # type: ignore[attr-defined]
from pathos.fairness import weighted_minmax  # type: ignore[attr-defined]
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


SKILLS = ("a", "b", "c")


@dataclass(frozen=True)
class RosteringInstance:
    staff: tuple[str, ...]
    staff_skills: dict[str, frozenset[str]]
    shift_skill: tuple[str, ...]  # skill required per slot
    min_staff_per_shift: int
    preferred_window: dict[str, tuple[int, int]]  # staff -> (start, end)
    n_slots: int


def _generate(size, seed: int) -> RosteringInstance:
    n_staff, n_slots = size
    rng = random.Random(seed)
    staff = tuple(f"emp_{i}" for i in range(n_staff))
    staff_skills = {
        s: frozenset(rng.sample(SKILLS, rng.randint(1, len(SKILLS))))
        for s in staff
    }
    shift_skill = tuple(rng.choice(SKILLS) for _ in range(n_slots))
    preferred_window = {
        s: (
            rng.randint(0, n_slots // 2),
            rng.randint(n_slots // 2 + 1, n_slots),
        )
        for s in staff
    }
    min_staff = max(1, n_staff // 5)
    return RosteringInstance(
        staff=staff,
        staff_skills=dict(staff_skills),
        shift_skill=shift_skill,
        min_staff_per_shift=min_staff,
        preferred_window=preferred_window,
        n_slots=n_slots,
    )


def _express(inst: RosteringInstance):
    space = (
        ScheduleSpace(entities=list(inst.staff), slots=inst.n_slots, penalty=1e3)
        .target(tolerance=0.0)
    )

    @space.demand
    def demand(emp, slot):
        # An employee "demands" 1 unit on a slot they don't have the skill for.
        if inst.shift_skill[slot] not in inst.staff_skills[emp]:
            return 1.0
        return 0.0

    @space.capacity
    def capacity(slot):
        # No skill-mismatch demand may exceed 0 -> capacity 0.
        return 0.0

    @space.fairness
    def fairness(schedule):
        # Equal weight per staff; rewards uptime balance.
        weights = {s: 1.0 for s in inst.staff}
        return weighted_minmax(weights, space)(schedule)

    return space


register(SuiteSpec(
    id="R2",
    family="schedule",
    constraint_classes=frozenset({"iii", "iv", "v"}),
    expressibility="partial",
    sizes={"S": ((20, 72),), "M": ((50, 168),), "L": ((100, 336),)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    notes=(
        "Employee rostering; skill-mismatch penalty-folded. "
        "Partial gap: soft_preferences (preferred shift windows) are "
        "not yet expressed as first-class soft constraints."
    ),
))
