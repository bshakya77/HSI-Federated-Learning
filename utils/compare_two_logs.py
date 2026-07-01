"""
Compare two Flower server log files (same experiment family) and summarize
major evaluation metrics and timing.

What it extracts (when present in log SUMMARY):
- History (loss, distributed)
- History (metrics, distributed, evaluate): val_mse, val_sam (and others if you add them)
- History (metrics, distributed, fit): pre_update_norm, post_update_norm (and others if you add them)

What it extracts (from timestamped lines):
- Round durations: time from "[ROUND k]" to "aggregate_evaluate: received ..."

Usage (PowerShell):
  python utils/compare_two_logs.py --log-a logs/docker-logs/log_a.log --log-b logs/docker-logs/log_b.log

Optional:
  python utils/compare_two_logs.py --log-a ... --log-b ... --out utils/log_compare_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROUND_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        return text[start:]
    return text[start:end]


def parse_distributed_loss(text: str) -> List[Tuple[int, float]]:
    """Parse 'History (loss, distributed): round k: value' block."""
    block = _extract_between(
        text,
        "History (loss, distributed):",
        "History (metrics, distributed, fit):",
    )
    if not block:
        # sometimes evaluate comes next in older logs
        block = _extract_between(text, "History (loss, distributed):", "History (metrics, distributed, evaluate):")
    if not block:
        return []
    return [(int(r), float(v)) for r, v in re.findall(r"round\s+(\d+):\s+([-+eE0-9\.]+)", block)]


PAIR_PAT = re.compile(r"\(\s*(\d+)\s*,\s*([0-9.eE+-]+)\s*\)")


def _extract_list_by_brackets(block_text: str, key_name: str) -> str:
    """Extract list content for 'key_name': [...] using bracket matching."""
    start = block_text.find(f"'{key_name}': [")
    if start < 0:
        return ""
    list_start = block_text.find("[", start) + 1
    depth, i = 1, list_start
    while i < len(block_text) and depth > 0:
        c = block_text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return block_text[list_start:i]
        i += 1
    return ""


def parse_metric_pairs_from_history_block(text: str, history_marker: str, metric_name: str) -> List[Tuple[int, float]]:
    """Parse metric list like 'val_mse': [(1, v), (2, v), ...] from a given History(...) block."""
    start = text.find(history_marker)
    if start < 0:
        return []
    block = text[start:]
    list_content = _extract_list_by_brackets(block, metric_name)
    if not list_content:
        return []
    return [(int(r), float(v)) for r, v in PAIR_PAT.findall(list_content)]


@dataclass(frozen=True)
class SeriesSummary:
    n: int
    first: Optional[float] = None
    last: Optional[float] = None
    min_v: Optional[float] = None
    max_v: Optional[float] = None
    mean: Optional[float] = None
    slope: Optional[float] = None  # simple linear slope vs round index (not time)


def summarize_series(series: Sequence[Tuple[int, float]]) -> SeriesSummary:
    vals = [v for _, v in series]
    n = len(vals)
    if n == 0:
        return SeriesSummary(n=0)

    mean_v = statistics.mean(vals)
    if n > 1:
        x = list(range(1, n + 1))
        xm = sum(x) / n
        ym = mean_v
        slope = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, vals)) / sum((xi - xm) ** 2 for xi in x)
    else:
        slope = 0.0

    return SeriesSummary(
        n=n,
        first=vals[0],
        last=vals[-1],
        min_v=min(vals),
        max_v=max(vals),
        mean=mean_v,
        slope=slope,
    )


def series_diff_by_round(a: Sequence[Tuple[int, float]], b: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Return (round, b-a) for rounds present in both series (by index/ordering)."""
    if not a or not b:
        return []
    n = min(len(a), len(b))
    out: List[Tuple[int, float]] = []
    for i in range(n):
        ra, va = a[i]
        rb, vb = b[i]
        r = rb if ra != rb else ra
        out.append((r, vb - va))
    return out


def parse_round_times(text: str) -> List[Tuple[int, float]]:
    """Compute per-round wall time from '[ROUND k]' to 'aggregate_evaluate: received ...'."""
    lines = text.splitlines()
    starts: Dict[int, datetime] = {}
    ends: Dict[int, datetime] = {}

    for ln in lines:
        m_start = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[ROUND (\d+)\]", ln)
        if m_start:
            r = int(m_start.group(2))
            starts[r] = datetime.strptime(m_start.group(1), ROUND_TS_FMT)
            continue

        m_end = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*aggregate_evaluate: received", ln)
        if m_end and starts:
            ts = datetime.strptime(m_end.group(1), ROUND_TS_FMT)
            # assign this end time to the latest started round without an end
            candidates = [r for r in starts.keys() if r not in ends and starts[r] <= ts]
            if candidates:
                r = max(candidates)
                ends[r] = ts

    durations: List[Tuple[int, float]] = []
    for r in sorted(starts.keys()):
        if r in ends:
            durations.append((r, (ends[r] - starts[r]).total_seconds()))
    return durations


def summarize_round_times(rt: Sequence[Tuple[int, float]]) -> Dict[str, float]:
    if not rt:
        return {}
    vals = [v for _, v in rt]
    return {
        "n": float(len(vals)),
        "mean_sec": float(statistics.mean(vals)),
        "median_sec": float(statistics.median(vals)),
        "min_sec": float(min(vals)),
        "max_sec": float(max(vals)),
        "total_sec": float(sum(vals)),
    }


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.4f}"


def print_series_report(name: str, series: Sequence[Tuple[int, float]]) -> None:
    s = summarize_series(series)
    if s.n == 0:
        print(f"- {name}: n=0 (not found)")
        return
    print(
        f"- {name}: n={s.n}, first={_fmt(s.first)}, last={_fmt(s.last)}, "
        f"min={_fmt(s.min_v)}, max={_fmt(s.max_v)}, mean={_fmt(s.mean)}, slope~{_fmt(s.slope)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-a", type=Path, required=True, help="First log file path (baseline)")
    ap.add_argument("--log-b", type=Path, required=True, help="Second log file path (comparison)")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    text_a = _read_text(args.log_a)
    text_b = _read_text(args.log_b)

    series_keys = {
        "loss_distributed": lambda t: parse_distributed_loss(t),
        "val_mse": lambda t: parse_metric_pairs_from_history_block(t, "History (metrics, distributed, evaluate):", "val_mse"),
        "val_sam": lambda t: parse_metric_pairs_from_history_block(t, "History (metrics, distributed, evaluate):", "val_sam"),
        "pre_update_norm": lambda t: parse_metric_pairs_from_history_block(t, "History (metrics, distributed, fit):", "pre_update_norm"),
        "post_update_norm": lambda t: parse_metric_pairs_from_history_block(t, "History (metrics, distributed, fit):", "post_update_norm"),
    }

    parsed_a: Dict[str, List[Tuple[int, float]]] = {k: fn(text_a) for k, fn in series_keys.items()}
    parsed_b: Dict[str, List[Tuple[int, float]]] = {k: fn(text_b) for k, fn in series_keys.items()}

    rt_a = parse_round_times(text_a)
    rt_b = parse_round_times(text_b)

    print(f"\nA: {args.log_a}")
    for k, series in parsed_a.items():
        print_series_report(k, series)
    if rt_a:
        print("- round_time:", json.dumps(summarize_round_times(rt_a), indent=2))
    else:
        print("- round_time: n/a (could not parse timestamped rounds)")

    print(f"\nB: {args.log_b}")
    for k, series in parsed_b.items():
        print_series_report(k, series)
    if rt_b:
        print("- round_time:", json.dumps(summarize_round_times(rt_b), indent=2))
    else:
        print("- round_time: n/a (could not parse timestamped rounds)")

    print("\nDelta (B - A) at final round (when available):")
    final_deltas: Dict[str, float] = {}
    for k in series_keys.keys():
        a_series = parsed_a[k]
        b_series = parsed_b[k]
        if a_series and b_series:
            d = b_series[min(len(a_series), len(b_series)) - 1][1] - a_series[min(len(a_series), len(b_series)) - 1][1]
            final_deltas[k] = float(d)
            print(f"- {k}: {_fmt(d)}")
        else:
            print(f"- {k}: n/a")

    rt_summary_a = summarize_round_times(rt_a)
    rt_summary_b = summarize_round_times(rt_b)
    if rt_summary_a and rt_summary_b:
        print("\nRound time delta (B - A):")
        print(f"- mean_sec: {_fmt(rt_summary_b['mean_sec'] - rt_summary_a['mean_sec'])}")
        print(f"- total_sec: {_fmt(rt_summary_b['total_sec'] - rt_summary_a['total_sec'])}")

    report = {
        "log_a": str(args.log_a),
        "log_b": str(args.log_b),
        "series_a": {k: parsed_a[k] for k in series_keys.keys()},
        "series_b": {k: parsed_b[k] for k in series_keys.keys()},
        "final_deltas_b_minus_a": final_deltas,
        "round_time_a": rt_summary_a,
        "round_time_b": rt_summary_b,
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

