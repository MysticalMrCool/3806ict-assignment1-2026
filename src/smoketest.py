"""Smoke test: parse + baseline prove on simple formulae."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse
from src.baseline import Baseline

CASES = [
    # propositional — all provable
    ("P -> P",                                       "proved"),
    ("P & Q -> P",                                   "proved"),
    ("P | ~P",                                       "proved"),
    ("(P -> Q) & (Q -> R) -> (P -> R)",              "proved"),
    ("P -> Q -> P",                                   "proved"),
    # should NOT prove
    ("P -> Q",                                        "countermodel"),
    ("P & Q",                                         "countermodel"),
    # FOL — provable
    ("(forall x. P(x)) -> P(c)",                      "proved"),
    ("(forall x. P(x)) -> (exists x. P(x))",          "proved"),
    ("(~(forall x. R(x))) -> (exists x. ~R(x))",      "proved"),
    ("(~(exists x. R(x))) -> (forall x. ~R(x))",      "proved"),
    # drinker's paradox — tricky classically-valid formula, requires multiple instantiations
    ("exists x. (D(x) -> forall y. D(y))",            "proved"),
]

def main():
    prover = Baseline(step_limit=500, fresh_cap=4)
    ok = 0
    for i, (src, expected) in enumerate(CASES, 1):
        try:
            f = parse(src)
        except Exception as e:
            print(f"[{i}] PARSE FAIL: {src}  -> {e}")
            continue
        result = prover.prove(f)
        mark = "OK" if result.status == expected else "FAIL"
        if mark == "OK":
            ok += 1
        print(f"[{i}] {mark:4} expected={expected:12} got={result.status:12} steps={result.steps:3}  :: {src}")
    print(f"\n{ok}/{len(CASES)} passed.")

if __name__ == "__main__":
    main()
