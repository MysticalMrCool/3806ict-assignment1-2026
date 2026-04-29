"""Run baseline + improved on the same formulae, side by side."""
from __future__ import annotations
import time
from .parser import parse
from .baseline import Baseline
from .improved import Improved

CASES = [
    # (src, expected)
    ("P -> P", "proved"),
    ("P & Q -> P", "proved"),
    ("P | ~P", "proved"),
    ("(P -> Q) & (Q -> R) -> (P -> R)", "proved"),
    ("P -> Q -> P", "proved"),
    ("P -> Q", "countermodel"),
    ("P & Q", "countermodel"),
    ("(forall x. P(x)) -> P(c)", "proved"),
    ("(forall x. P(x)) -> (exists x. P(x))", "proved"),
    ("(~(forall x. R(x))) -> (exists x. ~R(x))", "proved"),
    ("(~(exists x. R(x))) -> (forall x. ~R(x))", "proved"),
    ("exists x. (D(x) -> (forall y. D(y)))", "proved"),
    # more challenging
    ("(forall x. (P(x) -> Q(x))) -> ((forall x. P(x)) -> (forall x. Q(x)))", "proved"),
    ("(forall x. (P(x) & Q(x))) -> ((forall x. P(x)) & (forall x. Q(x)))", "proved"),
    ("(exists x. (P(x) | Q(x))) -> ((exists x. P(x)) | (exists x. Q(x)))", "proved"),
    ("(exists x. forall y. R(x, y)) -> (forall y. exists x. R(x, y))", "proved"),
    # classically valid but not usually by intuitionistic methods — still OK in LK
    ("((P -> Q) -> P) -> P", "proved"),
]


def run_one(prover, f, step_limit=2000):
    t0 = time.perf_counter()
    r = prover.prove(f)
    dt = time.perf_counter() - t0
    return r, dt


def main():
    b = Baseline(step_limit=500, fresh_cap=3)
    im = Improved(step_limit=500, fresh_cap=3)
    print(f"{'i':>3} {'src':60} {'want':12}  {'BASE st/step/t(ms)':30}  {'IMPR st/step/t(ms)':30}")
    b_ok = 0; im_ok = 0
    for i, (src, expected) in enumerate(CASES, 1):
        f = parse(src)
        br, bt = run_one(b, f)
        ir, it = run_one(im, f)
        if br.status == expected: b_ok += 1
        if ir.status == expected: im_ok += 1
        base_mark = "OK" if br.status == expected else "FAIL"
        impr_mark = "OK" if ir.status == expected else "FAIL"
        print(f"{i:>3} {src:60} {expected:12}  "
              f"{base_mark} {br.status:<11} {br.steps:>4}/{bt*1000:6.1f}  "
              f"{impr_mark} {ir.status:<11} {ir.steps:>4}/{it*1000:6.1f}")
    print(f"\nBaseline correct: {b_ok}/{len(CASES)}   Improved correct: {im_ok}/{len(CASES)}")


if __name__ == "__main__":
    main()
