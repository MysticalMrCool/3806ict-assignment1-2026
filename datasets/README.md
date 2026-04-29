# Benchmark datasets

Each file holds one formula per non-empty, non-comment line.
Format: `<formula>\t<expected-status>` where the status is either
`proved` (classically valid) or `countermodel` (not valid).

## propositional.txt (25 formulae)
Classical propositional tautologies and non-tautologies. Includes the
Hilbert-style axiom schemas of classical propositional logic, de Morgan
laws, double-negation, Peirce's law and distributivity. Non-theorems are
included to probe false-positive behaviour.

Sources: Mendelson, *Introduction to Mathematical Logic* (4th ed., 1997);
Troelstra & Schwichtenberg, *Basic Proof Theory* (2000); problems
collected from the author's undergraduate coursework (3806ICT workshops
weeks 3–5).

## pelletier.txt (26 formulae)
A subset of Pelletier's canonical benchmark, covering the propositional
fragment (problems 1–17) and the first monadic / dyadic problems
(18–25, 38), plus three known non-theorems.

Source: Francis Jeffry Pelletier, *Seventy-five problems for testing
automatic theorem provers*, Journal of Automated Reasoning 2(2):191–216,
1986. DOI: 10.1007/BF02432151. The ASCII encoding used here is the
author's; the logical content is Pelletier's.

## course.txt (22 formulae)
Formulae taken verbatim from the 3806ICT Workshop 3–5 exercises and the
Hou textbook's Chapter 2 examples (quantifier distribution, de Morgan
for quantifiers, quantifier swaps, etc.). Included for direct alignment
with the course material the marker will recognise.

Source: Zhe Hou, *Fundamentals of Logic and Computation: With Practical
Automated Reasoning and Verification*, Springer, 2021; and the 3806ICT
workshop handouts distributed on Learning@Griffith.

## synthetic.txt (29 formulae)
Author-generated parameterised families chosen to stress specific
algorithmic behaviour: deep right-associated implication chains,
deep conjunction / disjunction chains, quantifier-instantiation depth
(multiple witnesses), redundant-formula contraction, nested quantifier
alternation, and Skolem-style function iteration.

Provenance: hand-written by the author for this assignment. The
parameterised families are intentionally designed so that the baseline
algorithm's weaknesses (blind branching, uncontracted duplicates, blind
quantifier instantiation) become visible as the parameter grows.

---

Total: **102 formulae** across **4 sources** (two external, two internal /
self-generated). Every formula has a ground-truth label.
