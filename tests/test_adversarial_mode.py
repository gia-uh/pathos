"""Mode contract for game spaces — selection matrix across
{auto, exact, approximate} × {2 players, 3+ players}.

After the AnytimeAdversarial + score_for-overrides commits:

| mode          | 2 players          | 3+ players         |
|---------------|--------------------|--------------------|
| auto          | AnytimeAdversarial | AnytimeAdversarial |
| exact         | AlphaBeta          | Negamax            |
| approximate   | MCTS               | MCTS               |
"""
from __future__ import annotations

import pathos.algorithms  # noqa: F401
from pathos import Space
from tests.test_anytime_adversarial import _ttt_space, _multi_player_space


def test_mode_auto_2p_picks_anytime_adversarial():
    space = _ttt_space().mode("auto")
    assert space.solver().solve().algorithm == "AnytimeAdversarial"


def test_mode_auto_3p_picks_anytime_adversarial():
    space = _multi_player_space(players=3).mode("auto")
    assert space.solver().solve().algorithm == "AnytimeAdversarial"


def test_mode_exact_2p_picks_alphabeta():
    space = _ttt_space().mode("exact")
    assert space.solver().solve().algorithm == "AlphaBeta"


def test_mode_exact_3p_picks_negamax():
    """Bug fix: previously AlphaBeta (rank 45) won regardless of _players,
    giving silently-wrong play. Negamax.score_for now bumps to 52 when
    _players > 2."""
    space = _multi_player_space(players=3).mode("exact")
    assert space.solver().solve().algorithm == "Negamax"


def test_mode_approximate_2p_picks_mcts():
    """MCTS.score_for bumps to 53 under approximate; beats AB at 45."""
    space = _ttt_space().mode("approximate")
    assert space.solver().solve().algorithm == "MCTS"


def test_mode_approximate_3p_picks_mcts():
    """MCTS=53 still beats Negamax=52 under approximate (player-count
    bump applies to exact-selection too but is dominated)."""
    space = _multi_player_space(players=3).mode("approximate")
    assert space.solver().solve().algorithm == "MCTS"
