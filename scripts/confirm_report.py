"""Phase 6: check the CLAUDE.md Section 1 sim criteria per box type from several sim_eval.py runs.

Criteria (game-rule states): placement success >= 90 % at stack height 0-3 and >= 70 % on crooked stacks of height
4+ (max per-level offset >= 10 mm), for every box type; >= 70 % on large onto small (no3_on_no10).
Each bucket pools every run given (general and targeted). Reported: k/n, rate, 95 % Wilson interval;
"met" = rate >= target, "confirmed" = lower 95 % bound >= target.

    python scripts/confirm_report.py --out notes/phase6_confirmation.md <run dirs...>
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

TYPES = ("no3", "no35", "no10")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def crooked(r):
    return r.get("max_offset_mm", 0) >= 10 if "max_offset_mm" in r else r["crooked"] in ("<20mm", ">=20mm")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for d in a.runs:
        rows += json.loads((Path(d) / "episodes.json").read_text())
    lines = [f"Episodes pooled: {len(rows)} from {len(a.runs)} runs", "",
             "| criterion | box type | success | rate | 95 % interval | target | met | confirmed |", "|---|---|---|---|---|---|---|---|"]
    buckets = []
    for t in TYPES:
        buckets.append(("height 0-3", t, [r for r in rows if r["expected_type"] == t and r["height"] <= 3], 0.90))
    for t in TYPES:
        buckets.append(("crooked height 4+", t, [r for r in rows if r["expected_type"] == t and r["height"] >= 4 and crooked(r)], 0.70))
    buckets.append(("large onto small", "no3 on no10", [r for r in rows if r["pairing"] == "no3_on_no10"], 0.70))
    for name, t, b, target in buckets:
        k = sum(r["ok"] for r in b); n = len(b); lo, hi = wilson(k, n)
        rate = k / n if n else float("nan")
        lines.append(f"| {name} | {t} | {k}/{n} | {100 * rate:.1f} % | {100 * lo:.0f}–{100 * hi:.0f} % | ≥ {100 * target:.0f} % | "
                     f"{'yes' if n and rate >= target else 'NO'} | {'yes' if n and lo >= target else 'no'} |")
    fails = Counter(r["reason"] for r in rows if not r["ok"])
    lines += ["", f"Overall: {sum(r['ok'] for r in rows)}/{len(rows)}; failures {dict(fails)}; "
                  f"wrong box {sum(r['wrong_box'] for r in rows)}; backdrop touches {sum(r.get('hit_backdrop', False) for r in rows)}"]
    out = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
    print(out.encode("ascii", "replace").decode() if __import__("sys").stdout.encoding.lower() not in ("utf-8", "utf8") else out)


if __name__ == "__main__":
    main()
