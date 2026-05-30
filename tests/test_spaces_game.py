import pathos.algorithms  # ensure all algorithms are registered
from pathos.spaces.game import GameSpace

def test_gamespace_tictactoe():
    # Minimal: 1x3 board, first to get 3 in a row wins
    import copy

    def available(board): return [i for i, v in enumerate(board) if v == 0]
    def winner(board):
        if all(v == 1 for v in board): return 1
        if all(v == 2 for v in board): return 2
        return 0

    space = GameSpace().initial(tuple([0, 0, 0]))

    @space.successors
    def moves(board):
        player = 1 + (sum(1 for v in board if v != 0) % 2)
        for i in available(board):
            new = list(board)
            new[i] = player
            yield f"p{player}@{i}", tuple(new)

    @space.terminal
    def is_over(board): return winner(board) != 0 or 0 not in board

    @space.utility
    def score(board, player):
        w = winner(board)
        if w == player + 1: return 1.0
        if w == 0: return 0.0
        return -1.0

    result = space.solver().solve()
    assert result.found
