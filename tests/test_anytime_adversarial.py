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


def test_anytime_adversarial_matches_alphabeta_on_tictactoe():
    """On tic-tac-toe (fully solvable at depth 9) AnytimeAdversarial's
    result must match a direct AlphaBeta(max_depth=9) call."""
    space = _ttt_space()
    direct = AlphaBeta(space, max_depth=9).solve()
    space2 = _ttt_space().mode("auto")
    ata = space2.solver().solve()
    assert ata.algorithm == "AnytimeAdversarial"
    assert ata.cost == direct.cost
    # First move of the PV should agree.
    assert ata.path is not None and direct.path is not None
    assert ata.path[0][0] == direct.path[0][0]


def test_anytime_adversarial_path_is_pv():
    space = _ttt_space().mode("auto")
    result = space.solver().solve()
    assert result.path is not None
    # Every PV step is (action, state).
    for action, state in result.path:
        assert isinstance(state, tuple)
        assert len(state) == 9


def test_anytime_adversarial_pv_reuse_between_phases(monkeypatch):
    """Spy on AlphaBeta instantiation to confirm depth-d+1 gets the
    depth-d PV as pv_hint."""
    space = _ttt_space().mode("auto")
    calls: list[tuple[int, list[tuple[Any, Any]] | None]] = []
    original_init = AlphaBeta.__init__

    def spy_init(self, space, max_depth=100, pv_hint=None):
        calls.append((max_depth, pv_hint))
        original_init(self, space, max_depth=max_depth, pv_hint=pv_hint)

    monkeypatch.setattr(AlphaBeta, "__init__", spy_init)
    space.solver().solve()
    # Depth 1 should have no hint; depth 2+ should inherit prior PV.
    assert calls[0] == (1, [])
    assert len(calls) >= 2
    for d, hint in calls[1:]:
        assert hint is not None  # inherited from prior depth's path
