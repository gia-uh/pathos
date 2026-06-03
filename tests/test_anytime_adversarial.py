"""Selection, cascade behaviour, PV reuse, and cancellation for
AnytimeAdversarial. Mirrors the structure of test_anytime_local.py and
test_anytime_csp.py.
"""
from __future__ import annotations

import math
from typing import Any

import pathos.algorithms  # noqa: F401
from pathos import Space
from pathos.algorithms.adversarial import (
    AlphaBeta, AnytimeAdversarial, MCTS, Minimax, Negamax,
)


def _ttt_space() -> Space:
    """Tic-tac-toe as a Space — board is a 9-tuple of {0, 1, 2} where 0
    is empty, 1 is X (maximizer), 2 is O (minimizer)."""
    space = Space().initial((0,) * 9).adversarial(players=2, maximizing_player=1)

    def _winner(b: tuple[int, ...]) -> int:
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        ]
        for i, j, k in lines:
            if b[i] != 0 and b[i] == b[j] == b[k]:
                return b[i]
        return 0

    def _to_move(b: tuple[int, ...]) -> int:
        # X to move when zero or one more O than X... but simpler:
        return 1 if b.count(1) == b.count(2) else 2

    @space.successors
    def moves(b):
        if _winner(b) or 0 not in b:
            return
        player = _to_move(b)
        for i, cell in enumerate(b):
            if cell == 0:
                new = list(b)
                new[i] = player
                yield i, tuple(new)

    @space.terminal
    def is_terminal(b):
        return _winner(b) != 0 or 0 not in b

    @space.utility
    def util(b, player):
        w = _winner(b)
        if w == 0:
            return 0.0
        return 1.0 if w == player else -1.0

    return space


def _multi_player_space(players: int = 3) -> Space:
    """Trivial 3-player game tree: root → three leaves with player-specific
    utilities. Exists so we can verify _phase_class routes to Negamax when
    _players > 2."""
    space = Space().initial("root").adversarial(players=players, maximizing_player=0)

    @space.successors
    def moves(s):
        if s == "root":
            yield "a", "lA"
            yield "b", "lB"
            yield "c", "lC"

    @space.terminal
    def is_terminal(s): return s != "root"

    @space.utility
    def util(s, player):
        return {"lA": 1.0, "lB": 0.5, "lC": -0.5}.get(s, 0.0)

    return space


def test_anytime_adversarial_wins_under_mode_auto():
    space = _ttt_space().mode("auto")
    result = space.solver().solve()
    assert result.algorithm == "AnytimeAdversarial"


def test_anytime_adversarial_loses_under_mode_exact():
    space = _ttt_space().mode("exact")
    result = space.solver().solve()
    assert result.algorithm != "AnytimeAdversarial"
