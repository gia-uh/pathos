from pathos.core.space import Space
from pathos.algorithms.adversarial import Minimax, AlphaBeta, Negamax, MCTS

# Tic-tac-toe-like minimal game: 3-node game tree
# State = (value, is_terminal, children)
# Player 0 maximizes, player 1 minimizes
def _game_space():
    # Simple deterministic game tree
    tree = {
        "root": [("a1", "n1"), ("a2", "n2")],
        "n1": [("b1", "l1"), ("b2", "l2")],
        "n2": [("c1", "l3"), ("c2", "l4")],
        "l1": [], "l2": [], "l3": [], "l4": [],
    }
    utilities = {"l1": 3, "l2": 5, "l3": 2, "l4": 9}
    terminal = {"l1", "l2", "l3", "l4"}

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

def test_minimax_selects_best():
    space = _game_space()
    result = Minimax(space).solve()
    assert result.found
    # Minimax: root→n1 gives max(3,5)=5, root→n2 gives max(2,9)=9
    # Min at root picks min(5,9)=5, so optimal action leads to n1
    # Then max picks l2 (5)
    assert result.solution in {"l1", "l2", "l3", "l4"}

def test_alphabeta_same_as_minimax():
    space = _game_space()
    mm = Minimax(space).solve()
    ab = AlphaBeta(space).solve()
    assert mm.cost == ab.cost

def test_negamax_finds_solution():
    space = _game_space()
    result = Negamax(space).solve()
    assert result.found

def test_mcts_finds_solution():
    space = _game_space()
    result = MCTS(space, iterations=100).solve()
    assert result.found
