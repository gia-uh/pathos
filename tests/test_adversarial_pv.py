"""PV-first move ordering + SearchResult.path population for AB/Minimax/Negamax."""
from __future__ import annotations

import pathos.algorithms  # noqa: F401
from pathos import Space
from pathos.algorithms.adversarial import AlphaBeta, Minimax, Negamax


def _wide_game() -> Space:
    """Branching factor 4 at root, depth 3 — gives PV ordering a measurable
    win because pruning is sensitive to root-child ordering."""
    tree = {
        "root": [("a0", "n0"), ("a1", "n1"), ("a2", "n2"), ("a3", "n3")],
        "n0": [("x", "l0a"), ("y", "l0b")],
        "n1": [("x", "l1a"), ("y", "l1b")],
        "n2": [("x", "l2a"), ("y", "l2b")],
        "n3": [("x", "l3a"), ("y", "l3b")],
    }
    util = {"l0a": 1, "l0b": 4, "l1a": 9, "l1b": 2,
            "l2a": 3, "l2b": 7, "l3a": 5, "l3b": 6}
    terminal = set(util)
    space = Space().initial("root").adversarial(players=2, maximizing_player=0)

    @space.successors
    def moves(s): yield from tree.get(s, [])

    @space.terminal
    def is_terminal(s): return s in terminal

    @space.utility
    def score(s, player):
        v = util.get(s, 0)
        return v if player == 0 else -v

    return space


def test_alphabeta_populates_path_with_pv():
    space = _wide_game()
    result = AlphaBeta(space, max_depth=3).solve()
    assert result.found
    assert result.path is not None
    assert len(result.path) >= 1
    # First step of PV is (action, state) for the root choice.
    action, state = result.path[0]
    assert action in {"a0", "a1", "a2", "a3"}
    assert state in {"n0", "n1", "n2", "n3"}


def test_alphabeta_solution_is_initial_state():
    """Contract change: solution is the position the search was asked
    about (the initial state), not the leaf at PV end."""
    space = _wide_game()
    result = AlphaBeta(space, max_depth=3).solve()
    assert result.solution == "root"


def test_alphabeta_pv_hint_reorders_moves():
    """A correct PV hint puts the right root child first, which AB
    pruning then uses to cut the remaining branches earlier."""
    # First, find optimal first move by running unhinted AB.
    space = _wide_game()
    baseline = AlphaBeta(space, max_depth=3).solve()
    pv = baseline.path
    assert pv is not None and len(pv) >= 1
    # Now re-run with that PV as a hint; result must match.
    hinted = AlphaBeta(space, max_depth=3, pv_hint=pv).solve()
    assert hinted.cost == baseline.cost
    # The hinted PV first step must be the same.
    assert hinted.path is not None
    assert hinted.path[0][0] == pv[0][0]


def test_alphabeta_pv_hint_none_unchanged():
    """pv_hint=None preserves existing behaviour (no reorder)."""
    space = _wide_game()
    a = AlphaBeta(space, max_depth=3, pv_hint=None).solve()
    b = AlphaBeta(space, max_depth=3).solve()
    assert a.cost == b.cost


def test_minimax_populates_path_with_pv():
    space = _wide_game()
    result = Minimax(space, max_depth=3).solve()
    assert result.found
    assert result.path is not None
    assert len(result.path) >= 1
    assert result.solution == "root"


def test_negamax_populates_path_with_pv():
    space = _wide_game()
    result = Negamax(space, max_depth=3).solve()
    assert result.found
    assert result.path is not None
    assert result.solution == "root"


def test_minimax_pv_hint_honoured():
    space = _wide_game()
    baseline = Minimax(space, max_depth=3).solve()
    hinted = Minimax(space, max_depth=3, pv_hint=baseline.path).solve()
    assert hinted.cost == baseline.cost
    assert hinted.path is not None
    assert hinted.path[0][0] == baseline.path[0][0]


def test_negamax_pv_hint_honoured():
    space = _wide_game()
    baseline = Negamax(space, max_depth=3).solve()
    hinted = Negamax(space, max_depth=3, pv_hint=baseline.path).solve()
    assert hinted.cost == baseline.cost
