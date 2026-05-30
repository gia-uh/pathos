from enum import Enum, auto


class Capability(Enum):
    SUCCESSORS = auto()
    GOAL = auto()
    HEURISTIC = auto()
    EVALUATE = auto()
    TERMINAL = auto()
    UTILITY = auto()
    REVERSE_SUCCESSORS = auto()
    VARIABLES = auto()
    DOMAINS = auto()
    CONSTRAINTS = auto()
