# PATHOS

Production-ready classical AI search algorithms for Python.

**Define your problem, not your algorithm.**

```python
from pathos import Space

space = Space().initial(start)

@space.successors
def expand(state): ...

@space.heuristic
def h(state): ...

result = space.solver().solve()
```

## Install

```bash
pip install pathos-ai
```

## Quick links

- [Getting Started](getting-started.md)
- [API Reference](api/space.md)
- [GitHub](https://github.com/gia-uh/pathos)
