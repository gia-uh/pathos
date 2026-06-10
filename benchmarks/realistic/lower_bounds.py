from __future__ import annotations
import math
from typing import Hashable, Mapping, Sequence


def tsp_one_tree_lb(
    nodes: Sequence[Hashable],
    distances: Mapping[tuple[Hashable, Hashable], float],
) -> float:
    """Held-Karp 1-tree lower bound for symmetric TSP.

    For symmetric TSP with n >= 2 nodes:
      LB = weight of MST on nodes minus one + two shortest edges
           incident to the dropped node.

    This is a valid LB on the optimal tour cost.
    """
    n = len(nodes)
    if n < 2:
        return 0.0
    drop = nodes[0]
    rest = [v for v in nodes if v != drop]
    # Prim's MST over `rest` using the distance table.
    in_tree = {rest[0]}
    out_tree = set(rest[1:])
    mst_weight = 0.0
    while out_tree:
        best = math.inf
        best_v = None
        for u in in_tree:
            for v in out_tree:
                d = distances.get((u, v), distances.get((v, u), math.inf))
                if d < best:
                    best = d
                    best_v = v
        if best_v is None:
            return 0.0  # graph disconnected; degenerate LB
        mst_weight += best
        in_tree.add(best_v)
        out_tree.remove(best_v)
    incident = sorted(
        distances.get((drop, v), distances.get((v, drop), math.inf))
        for v in rest
    )
    if len(incident) < 2:
        return mst_weight
    return mst_weight + incident[0] + incident[1]


def bin_packing_waste_lb(items: Sequence[float], capacity: float) -> int:
    """Floor LB on the number of bins required: ceil(sum / capacity)."""
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    return math.ceil(sum(items) / capacity)
