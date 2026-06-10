from __future__ import annotations
from typing import Callable, Hashable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from pathos.spaces.schedule import ScheduleSpace


def weighted_minmax(
    weights: Mapping[Hashable, float],
    space: "ScheduleSpace",
) -> Callable[[tuple[tuple[bool, ...], ...]], float]:
    """Return a fairness callable bound to `space`.

    The returned function takes a (T, N) tuple-of-tuples bool schedule
    (with the matrix layout produced by `ScheduleSpace._to_matrix`) and
    returns

        min over leaves L of  weights[L] * uptime_fraction(L, schedule)

    where uptime_fraction(L, schedule) is the share of slots in which
    the entity feeding L (per `space._downstream`) was on. Higher =
    fairer.

    Leaves with weight 0 are skipped. Missing leaves (not in `weights`)
    are skipped. If no leaf has positive weight, returns +inf so the
    fairness term does not influence selection.
    """
    # Pre-compute leaf -> entity_index for all entities once.
    entity_index = {e: i for i, e in enumerate(space._entities)}
    leaf_to_idx: list[tuple[Hashable, int, float]] = []
    for entity, idx in entity_index.items():
        for leaf in space._downstream(entity):
            w = weights.get(leaf, 0.0)
            if w > 0.0:
                leaf_to_idx.append((leaf, idx, w))

    if not leaf_to_idx:
        def _empty(schedule: tuple[tuple[bool, ...], ...]) -> float:
            return float("inf")
        return _empty

    def _fairness(schedule: tuple[tuple[bool, ...], ...]) -> float:
        T = len(schedule)
        worst = float("inf")
        for _leaf, idx, w in leaf_to_idx:
            on_count = sum(1 for t in range(T) if schedule[t][idx])
            uptime = on_count / T
            term = w * uptime
            if term < worst:
                worst = term
        return worst

    return _fairness
