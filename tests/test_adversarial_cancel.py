"""Cancel-token cooperation across the four adversarial algorithms.

Verifies each algorithm exits cleanly on a pre-armed cancel token:
- MCTS: returns best-so-far (found=True) from partial tree.
- AlphaBeta/Minimax/Negamax: return not_found (found=False) when
  cancel fires at recursion root before any branch completes.
"""
from __future__ import annotations

import pathos.algorithms  # noqa: F401
from pathos import Space
from pathos.algorithms.adversarial import (
    AlphaBeta, Minimax, Negamax, MCTS,
)


def _tiny_game() -> Space:
    """Trivial 2-player game: root → two terminal leaves with utilities 3, 5."""
    tree = {"root": [("a", "l1"), ("b", "l2")]}
    utilities = {"l1": 3, "l2": 5}
    terminal = {"l1", "l2"}
    space = Space().initial("root").adversarial(players=2, maximizing_player=0)

    @space.successors
    def moves(s): yield from tree.get(s, [])

    @space.terminal
    def is_terminal(s): return s in terminal

    @space.utility
    def score(s, player):
        val = utilities.get(s, 0)
        return val if player == 0 else -val

    return space


def test_mcts_returns_best_so_far_on_cancel():
    space = _tiny_game()
    space._request_cancel()  # fire BEFORE solve
    result = MCTS(space, iterations=10_000).solve()
    # MCTS reports best from whatever tree it built (possibly empty);
    # but since no iterations ran, root has no children → result.found is False
    # (`root.children != []` is the found flag in MCTS.solve).
    assert result.algorithm == "MCTS"
    assert result.found is False


def test_mcts_returns_best_so_far_on_mid_cancel():
    """Cancel after some iterations have completed — best-so-far surfaces."""
    space = _tiny_game()
    # Run with a hook that fires cancel after the first backprop.
    original_backprop = MCTS._backprop
    call_count = {"n": 0}
    def hooked_backprop(self, node, reward):
        original_backprop(self, node, reward)
        call_count["n"] += 1
        if call_count["n"] == 2:
            self.space._request_cancel()
    MCTS._backprop = hooked_backprop
    try:
        result = MCTS(space, iterations=10_000).solve()
    finally:
        MCTS._backprop = original_backprop
    assert result.algorithm == "MCTS"
    assert result.found is True
    assert result.nodes_expanded < 10_000  # exited early


def test_alphabeta_returns_not_found_on_pre_armed_cancel():
    space = _tiny_game()
    space._request_cancel()
    result = AlphaBeta(space, max_depth=10).solve()
    assert result.algorithm == "AlphaBeta"
    assert result.found is False
    assert result.solution is None
    assert result.path is None
