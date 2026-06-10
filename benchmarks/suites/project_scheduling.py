"""R5 — Project scheduling with precedence (missing-capability).

ScheduleSpace doesn't model "task B's earliest start ≥ task A's
completion" as a first-class capability. v1 expresses precedence by
folding violations into the demand/capacity penalty: every slot
where a successor is ON but its predecessor hasn't yet been ON
contributes a fixed cost.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
import pathos.algorithms  # noqa: F401 — register algorithms
from pathos import ScheduleSpace  # type: ignore[attr-defined]
from pathos.fairness import weighted_minmax  # type: ignore[attr-defined]
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


@dataclass(frozen=True)
class ProjectInstance:
    tasks: tuple[str, ...]
    durations: dict[str, int]   # slots
    precedence: tuple[tuple[str, str], ...]  # (predecessor, successor)
    n_slots: int


def _generate(size, seed: int) -> ProjectInstance:
    n_tasks, n_slots = size
    rng = random.Random(seed)
    tasks = tuple(f"task_{i}" for i in range(n_tasks))
    durations = {t: rng.randint(1, max(2, n_slots // 4)) for t in tasks}
    # Build a random DAG by topological-order edges.
    order = list(tasks)
    rng.shuffle(order)
    precedence: list[tuple[str, str]] = []
    for i, t in enumerate(order):
        # Each later task has up to 2 predecessors from earlier ones.
        for _ in range(rng.randint(0, 2)):
            if i == 0:
                break
            p = rng.choice(order[:i])
            precedence.append((p, t))
    return ProjectInstance(
        tasks=tasks,
        durations=durations,
        precedence=tuple(precedence),
        n_slots=n_slots,
    )


def _express(inst: ProjectInstance):
    # Express each task as an entity; demand = task duration spread
    # uniformly across the planning horizon. Capacity per slot is set so
    # at most one task can be "ON" at a time. Precedence violations
    # are folded into the fairness term (since @evaluate is the cascade
    # entry point and ScheduleSpace's penalty path is already in use).
    space = (
        ScheduleSpace(entities=list(inst.tasks), slots=inst.n_slots, penalty=1e3)
        .target(tolerance=0.0)
    )
    preds_by_succ: dict[str, list[str]] = {}
    for u, v in inst.precedence:
        preds_by_succ.setdefault(v, []).append(u)

    @space.demand
    def demand(task, slot):
        return 1.0  # 1 unit per active slot

    @space.capacity
    def capacity(slot):
        return 1.0  # serial execution

    task_to_idx = {t: i for i, t in enumerate(inst.tasks)}
    duration_by_idx = {task_to_idx[t]: d for t, d in inst.durations.items()}

    @space.fairness
    def fairness(schedule):
        # fairness = coverage * precedence_score
        #   coverage = mean over tasks of min(on_count / duration, 1)
        #   precedence_score = 1 - viol / max(1, total_edges)
        # Multiplicative so an empty schedule (perfect precedence, zero
        # coverage) gets fairness=0, not fairness=1. This blocks the
        # degenerate "do nothing" optimum the cascade was finding under
        # the previous additive formulation.
        T = len(schedule)
        N = len(schedule[0]) if T else 0

        # Coverage: how completely is each task scheduled?
        if N == 0:
            return 0.0
        cover_sum = 0.0
        for n in range(N):
            on_count = sum(1 for t in range(T) if schedule[t][n])
            dur = duration_by_idx.get(n, 1)
            cover_sum += min(on_count / dur, 1.0)
        coverage = cover_sum / N

        # Precedence: penalise violations only across edges where the
        # successor has started.
        first_on: dict[int, int | None] = {}
        for n in range(N):
            first_on[n] = next(
                (t for t in range(T) if schedule[t][n]), None,
            )
        viol = 0
        for succ, preds in preds_by_succ.items():
            si = task_to_idx[succ]
            s_first = first_on[si]
            if s_first is None:
                continue
            for p in preds:
                pi = task_to_idx[p]
                p_first = first_on[pi]
                if p_first is None or p_first >= s_first:
                    viol += 1
        precedence_score = max(0.0, 1.0 - viol / max(1, len(inst.precedence)))

        return coverage * precedence_score

    return space


register(SuiteSpec(
    id="R5",
    family="schedule",
    constraint_classes=frozenset({"ii", "iii", "iv"}),
    expressibility="gap",
    sizes={"S": ((20, 72),), "M": ((50, 168),), "L": ((100, 336),)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    missing_capability="precedence",
    notes=(
        "Project scheduling — precedence not first-class on ScheduleSpace; "
        "violations folded into fairness penalty as workaround."
    ),
))
