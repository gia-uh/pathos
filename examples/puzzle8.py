"""8-puzzle solved with A* and Manhattan distance heuristic."""
import pathos.algorithms  # ensure all algorithms are registered
from pathos import Space

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def _moves(board):
    i = board.index(0)
    row, col = divmod(i, 3)
    for dr, dc, name in [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            lst = list(board); lst[i], lst[j] = lst[j], lst[i]
            yield name, tuple(lst)


def manhattan(board):
    total = 0
    for i, val in enumerate(board):
        if val == 0:
            continue
        goal_i = GOAL.index(val)
        total += abs(i // 3 - goal_i // 3) + abs(i % 3 - goal_i % 3)
    return total


START = (1, 2, 3, 4, 5, 6, 0, 7, 8)

space = Space().initial(START)


@space.successors
def expand(b): yield from _moves(b)


@space.goal
def solved(b): return b == GOAL


@space.heuristic
def h(b): return manhattan(b)


@space.evaluate
def cost(b): return 1.0


result = space.solver().solve()
print(f"Solved in {len(result.path or [])} moves using {result.algorithm}")
print(f"Nodes expanded: {result.nodes_expanded}")
