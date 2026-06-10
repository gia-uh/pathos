from benchmarks.realistic.diff import diff_reports, parse_report_md


SAMPLE_A = """# Realistic benchmark report

## Regression table

| suite | size | auto cost | auto feasible | auto alg | oracle cost | gap % | tunability % | tunability alg | missing-cap |
|---|---:|---:|:---:|---|---:|---:|---:|---|---|
| C1 | 8 | 1.000 | Y | Backtracking | 1.000 | 0.000 | 0.000 | — | — |
| R3 | 100 | 500.000 | Y | TabuSearch | 480.000 | 4.167 | 0.000 | — | — |
"""

SAMPLE_B = """# Realistic benchmark report

## Regression table

| suite | size | auto cost | auto feasible | auto alg | oracle cost | gap % | tunability % | tunability alg | missing-cap |
|---|---:|---:|:---:|---|---:|---:|---:|---|---|
| C1 | 8 | 1.000 | Y | Backtracking | 1.000 | 0.000 | 0.000 | — | — |
| R3 | 100 | 470.000 | Y | TabuSearch | 470.000 | 0.000 | 0.000 | — | — |
"""


def test_parse_report_md_extracts_regression_rows():
    rows = parse_report_md(SAMPLE_A)
    assert len(rows) == 2
    assert rows[0]["suite"] == "C1"
    assert rows[1]["suite"] == "R3"


def test_diff_reports_flags_improvements():
    delta = diff_reports(SAMPLE_A, SAMPLE_B)
    r3_delta = [d for d in delta if d["suite"] == "R3" and d["size"] == 100][0]
    assert r3_delta["status"] == "improved"
    assert r3_delta["cost_delta_pct"] < 0


def test_diff_reports_flags_regressions():
    delta = diff_reports(SAMPLE_B, SAMPLE_A)
    r3_delta = [d for d in delta if d["suite"] == "R3" and d["size"] == 100][0]
    assert r3_delta["status"] == "regressed"
    assert r3_delta["cost_delta_pct"] > 0


def test_diff_reports_marks_stable_rows():
    delta = diff_reports(SAMPLE_A, SAMPLE_A)
    assert all(d["status"] == "stable" for d in delta)
