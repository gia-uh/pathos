from __future__ import annotations
import time
import math
import random
from typing import Any
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register


@register
class Minimax(Algorithm):
    """Minimax search for two-player zero-sum games.

    Requires: successors, terminal, utility.

    Attributes:
        requires: Capability set needed.
        power_rank: 40.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.TERMINAL, Capability.UTILITY})
    power_rank = 40

    def __init__(self, space: Any, max_depth: int = 100) -> None:
        super().__init__(space)
        self.max_depth = max_depth

    def _minimax(self, state: Any, depth: int, is_max: bool) -> tuple[float, Any]:
        if self.space._terminal(state) or depth == 0:
            player = self.space._maximizing_player
            return self.space._utility(state, player), state
        moves = list(self.space._successors(state))
        if not moves:
            return self.space._utility(state, self.space._maximizing_player), state
        if is_max:
            best_val, best_state = -math.inf, None
            for _, child in moves:
                val, leaf = self._minimax(child, depth - 1, False)
                if val > best_val:
                    best_val, best_state = val, leaf
            return best_val, best_state
        else:
            best_val, best_state = math.inf, None
            for _, child in moves:
                val, leaf = self._minimax(child, depth - 1, True)
                if val < best_val:
                    best_val, best_state = val, leaf
            return best_val, best_state

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        val, state = self._minimax(self.space._initial, self.max_depth, True)
        return SearchResult(state, None, val, "Minimax", 0, time.perf_counter() - t0, state is not None)


@register
class AlphaBeta(Algorithm):
    """Alpha-Beta pruning — Minimax with branch pruning for efficiency.

    Requires: successors, terminal, utility.

    Attributes:
        requires: Capability set needed.
        power_rank: 45 (preferred over Minimax when available).
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.TERMINAL, Capability.UTILITY})
    power_rank = 45

    def __init__(
        self,
        space: Any,
        max_depth: int = 100,
        pv_hint: list[tuple[Any, Any]] | None = None,
    ) -> None:
        super().__init__(space)
        self.max_depth = max_depth
        self.pv_hint = pv_hint

    def _ab(
        self,
        state: Any,
        depth: int,
        alpha: float,
        beta: float,
        is_max: bool,
        hint: list[tuple[Any, Any]] | None,
    ) -> tuple[float, list[tuple[Any, Any]] | None]:
        if self.space._cancel_requested():
            return math.nan, None
        if self.space._terminal(state) or depth == 0:
            return self.space._utility(state, self.space._maximizing_player), []
        moves = list(self.space._successors(state))
        if not moves:
            return self.space._utility(state, self.space._maximizing_player), []

        hinted_action = hint[0][0] if hint else None
        if hinted_action is not None:
            moves.sort(key=lambda ac: 0 if ac[0] == hinted_action else 1)
        sub_hint = hint[1:] if hint else None

        best_pv: list[tuple[Any, Any]] = []
        if is_max:
            val = -math.inf
            for action, child in moves:
                child_val, child_pv = self._ab(child, depth - 1, alpha, beta, False, sub_hint)
                if child_pv is None:
                    return math.nan, None
                if child_val > val:
                    val = child_val
                    best_pv = [(action, child), *child_pv]
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
        else:
            val = math.inf
            for action, child in moves:
                child_val, child_pv = self._ab(child, depth - 1, alpha, beta, True, sub_hint)
                if child_pv is None:
                    return math.nan, None
                if child_val < val:
                    val = child_val
                    best_pv = [(action, child), *child_pv]
                beta = min(beta, val)
                if alpha >= beta:
                    break
        return val, best_pv

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        val, pv = self._ab(
            self.space._initial, self.max_depth, -math.inf, math.inf, True, self.pv_hint,
        )
        elapsed = time.perf_counter() - t0
        if pv is None:
            return SearchResult.not_found("AlphaBeta", 0, elapsed)
        return SearchResult(
            solution=self.space._initial,
            path=pv,
            cost=val,
            algorithm="AlphaBeta",
            nodes_expanded=0,
            elapsed=elapsed,
            found=True,
            epsilon=1.0,
        )


@register
class Negamax(Algorithm):
    """Negamax — Minimax variant using score negation for multi-player support.

    Requires: successors, terminal, utility.

    Attributes:
        requires: Capability set needed.
        power_rank: 42.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.TERMINAL, Capability.UTILITY})
    power_rank = 42

    def __init__(self, space: Any, max_depth: int = 100) -> None:
        super().__init__(space)
        self.max_depth = max_depth

    def _negamax(self, state: Any, depth: int, alpha: float, beta: float, player: int) -> tuple[float, Any]:
        if self.space._terminal(state) or depth == 0:
            val = self.space._utility(state, player)
            return val, state
        moves = list(self.space._successors(state))
        if not moves:
            return self.space._utility(state, player), state
        best_val, best_state = -math.inf, moves[0][1]
        next_player = (player + 1) % self.space._players
        for _, child in moves:
            child_val, _ = self._negamax(child, depth - 1, -beta, -alpha, next_player)
            if -child_val > best_val:
                best_val, best_state = -child_val, child
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
        return best_val, best_state

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        val, state = self._negamax(
            self.space._initial, self.max_depth, -math.inf, math.inf,
            self.space._maximizing_player
        )
        return SearchResult(state, None, val, "Negamax", 0, time.perf_counter() - t0, state is not None)


class _MCTSNode:
    __slots__ = ("state", "parent", "children", "visits", "value", "untried")

    def __init__(self, state: Any, parent: _MCTSNode | None, space: Any) -> None:
        self.state = state
        self.parent = parent
        self.children: list[_MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried = list(space._successors(state)) if not space._terminal(state) else []

    def uct_score(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return math.inf
        assert self.parent is not None
        return self.value / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self) -> _MCTSNode:
        return max(self.children, key=lambda n: n.uct_score())

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0


@register
class MCTS(Algorithm):
    """Monte Carlo Tree Search — simulation-based game tree exploration.

    Requires: successors, terminal, utility.

    Attributes:
        requires: Capability set needed.
        power_rank: 43.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.TERMINAL, Capability.UTILITY})
    power_rank = 43

    def __init__(self, space: Any, iterations: int = 1000) -> None:
        super().__init__(space)
        self.iterations = iterations

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        root = _MCTSNode(self.space._initial, None, self.space)

        completed = 0
        for _ in range(self.iterations):
            if self.space._cancel_requested():
                break
            node = self._select(root)
            if not self.space._terminal(node.state):
                node = self._expand(node)
            reward = self._simulate(node.state)
            self._backprop(node, reward)
            completed += 1

        if not root.children:
            return SearchResult.not_found("MCTS", completed, time.perf_counter() - t0)
        best = max(root.children, key=lambda n: n.visits)
        return SearchResult(
            best.state, None,
            self.space._utility(best.state, self.space._maximizing_player),
            "MCTS", completed, time.perf_counter() - t0, True,
        )

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        while not self.space._terminal(node.state) and node.is_fully_expanded():
            node = node.best_child()
        return node

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        action, child_state = node.untried.pop()
        child = _MCTSNode(child_state, node, self.space)
        node.children.append(child)
        return child

    def _simulate(self, state: Any) -> float:
        depth = 0
        while not self.space._terminal(state) and depth < 50:
            moves = list(self.space._successors(state))
            if not moves:
                break
            _, state = random.choice(moves)
            depth += 1
        result: float = self.space._utility(state, self.space._maximizing_player)
        return result

    def _backprop(self, node: _MCTSNode | None, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
