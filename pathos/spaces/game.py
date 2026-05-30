from __future__ import annotations
from pathos.core.space import Space


class GameSpace(Space):
    """
    Space for adversarial games.
    Convenience wrapper — sets adversarial mode by default.
    User provides @successors, @terminal, @utility (and optionally @evaluate).
    """

    def __init__(self) -> None:
        super().__init__()
        self.adversarial(players=2, maximizing_player=0)
