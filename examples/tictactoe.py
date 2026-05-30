"""Tic-tac-toe optimal play via Alpha-Beta."""
import pathos.algorithms  # ensure all algorithms are registered
from pathos import GameSpace


def _winner(board):
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]
    for a, b, c in lines:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0


space = GameSpace().initial(tuple([0] * 9))


@space.successors
def moves(board):
    player = 1 + (sum(1 for v in board if v) % 2)
    for i, v in enumerate(board):
        if v == 0:
            new = list(board); new[i] = player
            yield f"p{player}@{i}", tuple(new)


@space.terminal
def is_over(board): return _winner(board) != 0 or 0 not in board


@space.utility
def score(board, player):
    w = _winner(board)
    if w == 0: return 0.0
    return 1.0 if w == player + 1 else -1.0


result = space.solver().solve()
board = result.solution or space._initial
symbols = {0: ".", 1: "X", 2: "O"}
for row in range(3):
    print(" ".join(symbols[board[row * 3 + col]] for col in range(3)))
print(f"\nBest move leads to state with utility {result.cost}")
print(f"Algorithm: {result.algorithm}")
