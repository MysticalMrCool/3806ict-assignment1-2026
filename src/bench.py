"""Benchmark runner: run Baseline and Improved on every formula in every
dataset file under ../datasets, and emit a CSV plus a per-dataset summary.

Usage (from the Assignment 1 folder):
    PowerShell: $env:PYTHONHASHSEED="0"; python -m src.bench
    POSIX sh  : PYTHONHASHSEED=0 python3 -m src.bench

The PYTHONHASHSEED=0 prefix is REQUIRED to get reproducible per-formula step
counts: the prover relies on iteration over Python sets of formula AST nodes,
whose hash (and therefore set-iteration order) varies across processes when
hash randomisation is on.  Without the seed pinned, per-case step counts and
the *fraction* of borderline formulas that finish under STEP_LIMIT can drift
by 1--2 cases between runs; per-dataset PROVED counts on cleanly-decidable
formulas are unaffected.  Wall-clock ms columns are not reproducible regardless.

Outputs:
    results/bench_results.csv
    results/bench_summary.txt
"""

from __future__ import annotations
import os
import sys
import csv
import time
import traceback
from dataclasses import dataclass
from typing import List
from .parser import parse, ParseError
from .baseline import Baseline
from .improved import Improved

# --- Reproducibility check ---------------------------------------------------
# Warn (but do not fail) if PYTHONHASHSEED is not pinned to 0.
if os.environ.get("PYTHONHASHSEED") != "0":
    print("[bench] WARNING: PYTHONHASHSEED is not 0; per-case step counts "
          "may drift across runs. Re-run as:",
          file=sys.stderr)
    print("[bench]    PowerShell: $env:PYTHONHASHSEED=\"0\"; python -m src.bench",
          file=sys.stderr)
    print("[bench]    POSIX sh  : PYTHONHASHSEED=0 python3 -m src.bench",
          file=sys.stderr)


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "datasets")
OUT = os.path.join(ROOT, "results")

STEP_LIMIT = 300
FRESH_CAP = 3


@dataclass
class Row:
    dataset: str
    idx: int
    src: str
    expected: str
    base_status: str
    base_steps: int
    base_ms: float
    impr_status: str
    impr_steps: int
    impr_ms: float


def load(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh.readlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if "\t" not in line:
                continue
            f, tag = line.rsplit("\t", 1)
            items.append((f.strip(), tag.strip()))
    return items


def run_one(prover, formula, limit=STEP_LIMIT):
    t0 = time.perf_counter()
    r = prover.prove(formula)
    return r, (time.perf_counter() - t0) * 1000.0


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(
        [fn for fn in os.listdir(DATA) if fn.endswith(".txt")]
    )
    rows: List[Row] = []
    parse_errors: List[str] = []

    for fn in files:
        ds = os.path.splitext(fn)[0]
        items = load(os.path.join(DATA, fn))
        for idx, (src, expected) in enumerate(items, 1):
            try:
                f = parse(src)
            except ParseError as e:
                parse_errors.append(f"{ds}#{idx}: {e}  :: {src}")
                continue
            try:
                b = Baseline(step_limit=STEP_LIMIT, fresh_cap=FRESH_CAP)
                br, bt = run_one(b, f)
            except Exception as e:
                br, bt = None, 0.0
                parse_errors.append(f"{ds}#{idx} baseline crash: {e}")
            try:
                im = Improved(step_limit=STEP_LIMIT, fresh_cap=FRESH_CAP)
                ir, it = run_one(im, f)
            except Exception as e:
                ir, it = None, 0.0
                parse_errors.append(f"{ds}#{idx} improved crash: {e}")

            rows.append(Row(
                dataset=ds, idx=idx, src=src, expected=expected,
                base_status=br.status if br else "CRASH",
                base_steps=br.steps if br else -1,
                base_ms=bt,
                impr_status=ir.status if ir else "CRASH",
                impr_steps=ir.steps if ir else -1,
                impr_ms=it,
            ))

    # --- write CSV ---
    csv_path = os.path.join(OUT, "bench_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "idx", "formula", "expected",
                    "base_status", "base_steps", "base_ms",
                    "impr_status", "impr_steps", "impr_ms"])
        for r in rows:
            w.writerow([r.dataset, r.idx, r.src, r.expected,
                        r.base_status, r.base_steps, f"{r.base_ms:.3f}",
                        r.impr_status, r.impr_steps, f"{r.impr_ms:.3f}"])

    # --- summary ---
    summary_lines: List[str] = []
    summary_lines.append(f"Benchmark run  (step_limit={STEP_LIMIT}, fresh_cap={FRESH_CAP})")
    summary_lines.append("=" * 78)
    ds_names = sorted({r.dataset for r in rows})
    summary_lines.append(f"{'Dataset':<18}  {'N':>3}  "
                         f"{'Base correct':>12}  {'Impr correct':>12}  "
                         f"{'Base steps sum':>14}  {'Impr steps sum':>14}  "
                         f"{'Base ms sum':>11}  {'Impr ms sum':>11}")
    overall_n = overall_bc = overall_ic = 0
    overall_bs = overall_is = 0
    overall_bt = overall_it = 0.0
    for ds in ds_names:
        rs = [r for r in rows if r.dataset == ds]
        n = len(rs)
        bc = sum(1 for r in rs if r.base_status == r.expected)
        ic = sum(1 for r in rs if r.impr_status == r.expected)
        bs = sum(r.base_steps for r in rs if r.base_steps >= 0)
        is_ = sum(r.impr_steps for r in rs if r.impr_steps >= 0)
        bt = sum(r.base_ms for r in rs)
        it = sum(r.impr_ms for r in rs)
        summary_lines.append(f"{ds:<18}  {n:>3}  "
                             f"{bc:>12}  {ic:>12}  "
                             f"{bs:>12}  {is_:>12}  "
                             f"{bt:>10.1f}  {it:>10.1f}")
        overall_n += n; overall_bc += bc; overall_ic += ic
        overall_bs += bs; overall_is += is_; overall_bt += bt; overall_it += it
    summary_lines.append("-" * 78)
    summary_lines.append(f"{'TOTAL':<18}  {overall_n:>3}  "
                         f"{overall_bc:>12}  {overall_ic:>12}  "
                         f"{overall_bs:>12}  {overall_is:>12}  "
                         f"{overall_bt:>10.1f}  {overall_it:>10.1f}")

    # Case-level divergences: where exactly did the two provers disagree?
    summary_lines.append("")
    summary_lines.append("Divergences (baseline != improved):")
    divergent = [r for r in rows if r.base_status != r.impr_status]
    if not divergent:
        summary_lines.append("  (none)")
    for r in divergent:
        summary_lines.append(
            f"  {r.dataset}#{r.idx:<2}  exp={r.expected:<12}  "
            f"base={r.base_status:<11}({r.base_steps:>4} steps, {r.base_ms:>6.1f}ms)  "
            f"impr={r.impr_status:<11}({r.impr_steps:>4} steps, {r.impr_ms:>6.1f}ms)  "
            f":: {r.src}"
        )

    if parse_errors:
        summary_lines.append("")
        summary_lines.append("Parse/runtime errors:")
        for e in parse_errors:
            summary_lines.append("  " + e)

    summary = "\n".join(summary_lines)
    with open(os.path.join(OUT, "bench_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
