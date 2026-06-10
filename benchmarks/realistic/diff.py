"""Parse two REPORT.md files and emit a delta between their
regression tables. Used by `python -m benchmarks.realistic --diff A B`.
"""
from __future__ import annotations
import re


_ROW_RE = re.compile(
    r"^\|\s*(?P<suite>[^\s|]+)\s*\|\s*(?P<size>\d+)\s*\|\s*"
    r"(?P<cost>[\d.\-—]+)\s*\|\s*(?P<feas>[YN])\s*\|"
)


def parse_report_md(text: str) -> list[dict]:
    rows: list[dict] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("## Regression table"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table:
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        cost_raw = m.group("cost").strip()
        cost = None if cost_raw == "—" else float(cost_raw)
        rows.append({
            "suite": m.group("suite"),
            "size": int(m.group("size")),
            "auto_headline_cost": cost,
            "auto_headline_feasible": m.group("feas") == "Y",
        })
    return rows


def diff_reports(text_a: str, text_b: str) -> list[dict]:
    a_rows = {(r["suite"], r["size"]): r for r in parse_report_md(text_a)}
    b_rows = {(r["suite"], r["size"]): r for r in parse_report_md(text_b)}
    keys = sorted(set(a_rows) | set(b_rows))
    out: list[dict] = []
    for k in keys:
        a = a_rows.get(k)
        b = b_rows.get(k)
        suite, size = k
        if a is None:
            out.append({"suite": suite, "size": size, "status": "new",
                        "cost_delta_pct": None})
            continue
        if b is None:
            out.append({"suite": suite, "size": size, "status": "removed",
                        "cost_delta_pct": None})
            continue
        ca = a["auto_headline_cost"]
        cb = b["auto_headline_cost"]
        if ca is None or cb is None:
            out.append({"suite": suite, "size": size, "status": "unknown",
                        "cost_delta_pct": None})
            continue
        if ca == 0.0 and cb == 0.0:
            delta = 0.0
        elif ca == 0.0:
            delta = float("inf")
        else:
            delta = 100.0 * (cb - ca) / abs(ca)
        if abs(delta) < 1e-6:
            status = "stable"
        elif delta < 0:
            status = "improved"
        else:
            status = "regressed"
        out.append({"suite": suite, "size": size, "status": status,
                    "cost_delta_pct": delta})
    return out
