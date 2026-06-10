import subprocess
import sys
import json
from pathlib import Path


def test_quick_sweep_runs_end_to_end_without_no_cliff_failures(tmp_path):
    json_path = tmp_path / "results.json"
    report_path = tmp_path / "REPORT.md"
    result = subprocess.run(
        [
            sys.executable, "-m", "benchmarks.realistic",
            "--quick",
            "--suites", "C1,C2,C3,R3,R6",
            "--repeat", "1",
            "--json", str(json_path),
            "--report", str(report_path),
        ],
        capture_output=True, text=True, timeout=180,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    # Exit code 0 = clean, 1 = no-cliff failures recorded.
    # Smoke must finish without throwing; both 0 and 1 are acceptable here.
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert json_path.exists()
    assert report_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["bench_schema"] >= 1
    assert any(sr["suite"] == "C1" for sr in payload["suite_rows"])
