"""Smoke tests for all examples — verify they run and produce valid output."""
import subprocess
import sys
import os

# Run examples from the repo root so relative paths resolve correctly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script):
    result = subprocess.run(
        [sys.executable, f"examples/{script}"],
        capture_output=True, text=True, timeout=60,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stderr}"
    return result.stdout


def test_route_planning():
    out = _run("route_planning.py")
    assert "Lisboa" in out
    assert "Algorithm" in out


def test_tsp():
    out = _run("tsp.py")
    assert "Cost" in out
    assert "Algorithm" in out


def test_nqueens():
    out = _run("nqueens.py")
    assert "Q" in out


def test_tictactoe():
    out = _run("tictactoe.py")
    assert "Algorithm" in out


def test_puzzle8():
    out = _run("puzzle8.py")
    assert "moves" in out
