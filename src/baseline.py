r"""
Baseline: Algorithm 2 from Hou (2021, p. 67) — Naive Backward Proof Search
for first-order logic in the sequent calculus LK.

Reproduces the textbook's rule-ordering faithfully:

    foreach open branch:
        if id / TR / FL applies      -> close branch
        elif an invertible non-branching rule applies:
            /\L, \/R, ->R, ~L, ~R, forall-R, exists-L
        elif an invertible branching rule applies:
            /\R, \/L, ->L
        elif forall-L / exists-R applies with an *unused* term:
            instantiate with that term (keep principal formula)
        elif forall-L / exists-R applies:
            instantiate with a fresh term (keep principal formula)
        else stop.

Because forall-L / exists-R keep the principal formula in the premise, the
algorithm can loop forever on FOL-valid formulae whose witnesses require
many instantiations. A hard cap on total steps is exposed via `step_limit`.

Returns a ProofResult with:
    .status: 'proved' | 'countermodel' | 'aborted'
    .steps:  number of rule applications performed
    .tree:   derivation tree (root SequentNode)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from .formula import (Formula, Term, Var, Func, Top, Bot, Atom,
                      Not, And, Or, Imp, Forall, Exists, is_atomic)


# --- Sequent representation ------------------------------------------------

@dataclass
class Sequent:
    """Γ |- Δ.  Antecedent/succedent as lists (multiset semantics; duplicates OK)."""
    left: List[Formula] = field(default_factory=list)
    right: List[Formula] = field(default_factory=list)
    def __str__(self) -> str:
        l = ", ".join(map(repr, self.left))
        r = ", ".join(map(repr, self.right))
        return f"{l} |- {r}" if self.left else f"|- {r}"
    def copy(self) -> "Sequent":
        return Sequent(list(self.left), list(self.right))

@dataclass
class SequentNode:
    """Node in the derivation tree."""
    seq: Sequent
    rule: Optional[str] = None        # rule applied going DOWN (from premises to conclusion); None if leaf
    children: List["SequentNode"] = field(default_factory=list)
    closed_by: Optional[str] = None   # 'id', 'TR', 'FL' if a leaf axiom; None otherwise

@dataclass
class ProofResult:
    status: str                       # 'proved' | 'countermodel' | 'aborted'
    steps: int
    tree: SequentNode
    # per-branch info for debugging
    open_leaves: int = 0


# --- helpers on terms ------------------------------------------------------

def _all_terms_in_formula(f: Formula) -> Set[Term]:
    out: Set[Term] = set()
    if isinstance(f, (Top, Bot)):
        return out
    if isinstance(f, Atom):
        for a in f.args: out |= _collect_terms(a); out.add(a)
        return out
    if isinstance(f, Not):  return _all_terms_in_formula(f.f)
    if isinstance(f, (And, Or, Imp)): return _all_terms_in_formula(f.l) | _all_terms_in_formula(f.r)
    if isinstance(f, (Forall, Exists)): return _all_terms_in_formula(f.body)
    return out

def _collect_terms(t: Term) -> Set[Term]:
    out: Set[Term] = {t}
    if isinstance(t, Func):
        for a in t.args: out |= _collect_terms(a)
    return out

def _all_terms_in_seq(s: Sequent) -> Set[Term]:
    out: Set[Term] = set()
    for f in s.left + s.right:
        out |= _all_terms_in_formula(f)
    return out


# --- Algorithm 2 -----------------------------------------------------------

class Baseline:
    """Textbook Algorithm 2. Stateful across a single prove() call."""

    INVERTIBLE_NONBRANCHING = ("/\\L", "\\/R", "->R", "~L", "~R", "forall-R", "exists-L")
    INVERTIBLE_BRANCHING    = ("/\\R", "\\/L", "->L")

    def __init__(self, step_limit: int = 2000, fresh_cap: int = 6):
        """
        step_limit  : abort after this many rule applications.
        fresh_cap   : maximum fresh terms introduced per sequent by forall-L/exists-R
                      before the algorithm gives up on that branch. The textbook
                      algorithm has no such bound; we add a minimal safety cap so
                      the baseline can report 'aborted' instead of running forever.
        """
        self.step_limit = step_limit
        self.fresh_cap = fresh_cap
        self._fresh_counter = 0

    # --- public API --------------------------------------------------------

    def prove(self, goal: Formula) -> ProofResult:
        self._fresh_counter = 0
        root_seq = Sequent(left=[], right=[goal])
        root = SequentNode(seq=root_seq)
        steps = 0
        # per-(sequent-id, formula-id) tracking of which terms have been used
        # for quantifier instantiation; keyed by (id(node), id(formula))
        used_terms: Dict[Tuple[int, int], Set[str]] = {}
        fresh_counts: Dict[Tuple[int, int], int] = {}

        open_leaves: List[SequentNode] = [root]
        aborted = False

        while open_leaves:
            if steps >= self.step_limit:
                aborted = True
                break
            node = open_leaves.pop(0)

            # 1. Axiom rules
            closed_by = self._try_axiom(node.seq)
            if closed_by:
                node.rule = closed_by
                node.closed_by = closed_by
                steps += 1
                continue

            # 2. Invertible non-branching rules
            applied = self._try_invertible_nonbranching(node)
            if applied:
                steps += 1
                open_leaves.extend(node.children)
                continue

            # 3. Invertible branching rules
            applied = self._try_invertible_branching(node)
            if applied:
                steps += 1
                open_leaves.extend(node.children)
                continue

            # 4. forall-L / exists-R with an unused term
            applied = self._try_quant_unused(node, used_terms)
            if applied:
                steps += 1
                open_leaves.extend(node.children)
                continue

            # 5. forall-L / exists-R with a fresh term (bounded by fresh_cap)
            applied = self._try_quant_fresh(node, used_terms, fresh_counts)
            if applied:
                steps += 1
                open_leaves.extend(node.children)
                continue

            # 6. nothing applicable -> open counter-model leaf
            node.rule = "stop"
            # leave node with no children and no closed_by

        # Determine status
        open_count = self._count_open_leaves(root)
        if aborted:
            status = "aborted"
        elif open_count == 0:
            status = "proved"
        else:
            status = "countermodel"
        return ProofResult(status=status, steps=steps, tree=root, open_leaves=open_count)

    # --- rule applicators --------------------------------------------------

    def _try_axiom(self, s: Sequent) -> Optional[str]:
        # TR : any Top in succedent
        for f in s.right:
            if isinstance(f, Top): return "TR"
        # FL : any Bot in antecedent
        for f in s.left:
            if isinstance(f, Bot): return "FL"
        # id : atomic formula appears in both sides
        for f in s.left:
            if is_atomic(f) and f in s.right:
                return "id"
        return None

    def _try_invertible_nonbranching(self, node: SequentNode) -> bool:
        s = node.seq

        # /\L : Γ, A, B |- Δ   from Γ, A ∧ B |- Δ
        for i, f in enumerate(s.left):
            if isinstance(f, And):
                new = s.copy()
                del new.left[i]
                new.left.append(f.l); new.left.append(f.r)
                node.rule = "/\\L"
                node.children = [SequentNode(seq=new)]
                return True

        # \/R : Γ |- A, B, Δ   from Γ |- A ∨ B, Δ
        for i, f in enumerate(s.right):
            if isinstance(f, Or):
                new = s.copy()
                del new.right[i]
                new.right.append(f.l); new.right.append(f.r)
                node.rule = "\\/R"
                node.children = [SequentNode(seq=new)]
                return True

        # ->R : Γ, A |- B, Δ   from Γ |- A -> B, Δ
        for i, f in enumerate(s.right):
            if isinstance(f, Imp):
                new = s.copy()
                del new.right[i]
                new.left.append(f.l); new.right.append(f.r)
                node.rule = "->R"
                node.children = [SequentNode(seq=new)]
                return True

        # ~L : Γ |- A, Δ   from Γ, ~A |- Δ
        for i, f in enumerate(s.left):
            if isinstance(f, Not):
                new = s.copy()
                del new.left[i]
                new.right.append(f.f)
                node.rule = "~L"
                node.children = [SequentNode(seq=new)]
                return True

        # ~R : Γ, A |- Δ   from Γ |- ~A, Δ
        for i, f in enumerate(s.right):
            if isinstance(f, Not):
                new = s.copy()
                del new.right[i]
                new.left.append(f.f)
                node.rule = "~R"
                node.children = [SequentNode(seq=new)]
                return True

        # forall-R : Γ |- A[c/x], Δ  (c fresh)   from Γ |- ∀x.A, Δ
        for i, f in enumerate(s.right):
            if isinstance(f, Forall):
                c = self._fresh_term_symbol(s, prefix="c")
                new = s.copy()
                del new.right[i]
                new.right.append(f.body.subst(f.var, Var(c)))
                node.rule = "forall-R"
                node.children = [SequentNode(seq=new)]
                return True

        # exists-L : Γ, A[c/x] |- Δ  (c fresh)   from Γ, ∃x.A |- Δ
        for i, f in enumerate(s.left):
            if isinstance(f, Exists):
                c = self._fresh_term_symbol(s, prefix="c")
                new = s.copy()
                del new.left[i]
                new.left.append(f.body.subst(f.var, Var(c)))
                node.rule = "exists-L"
                node.children = [SequentNode(seq=new)]
                return True

        return False

    def _try_invertible_branching(self, node: SequentNode) -> bool:
        s = node.seq

        # /\R : Γ |- A, Δ  and  Γ |- B, Δ    from Γ |- A ∧ B, Δ
        for i, f in enumerate(s.right):
            if isinstance(f, And):
                left_new = s.copy(); del left_new.right[i]; left_new.right.append(f.l)
                right_new = s.copy(); del right_new.right[i]; right_new.right.append(f.r)
                node.rule = "/\\R"
                node.children = [SequentNode(seq=left_new), SequentNode(seq=right_new)]
                return True

        # \/L : Γ, A |- Δ  and  Γ, B |- Δ    from Γ, A ∨ B |- Δ
        for i, f in enumerate(s.left):
            if isinstance(f, Or):
                left_new = s.copy(); del left_new.left[i]; left_new.left.append(f.l)
                right_new = s.copy(); del right_new.left[i]; right_new.left.append(f.r)
                node.rule = "\\/L"
                node.children = [SequentNode(seq=left_new), SequentNode(seq=right_new)]
                return True

        # ->L : Γ |- A, Δ  and  Γ, B |- Δ    from Γ, A -> B |- Δ
        for i, f in enumerate(s.left):
            if isinstance(f, Imp):
                left_new = s.copy(); del left_new.left[i]; left_new.right.append(f.l)
                right_new = s.copy(); del right_new.left[i]; right_new.left.append(f.r)
                node.rule = "->L"
                node.children = [SequentNode(seq=left_new), SequentNode(seq=right_new)]
                return True

        return False

    def _try_quant_unused(self, node: SequentNode,
                          used_terms: Dict[Tuple[int, int], Set[str]]) -> bool:
        s = node.seq
        # Collect ground-ish terms currently in the sequent. We key "used"
        # status by the term's repr but carry the actual Term object through
        # substitution so Var/Func identity is preserved.
        candidates: List[Term] = []
        seen: Set[str] = set()
        for t in _all_terms_in_seq(s):
            if isinstance(t, (Var, Func)) and not _is_bound_variable(t, s):
                r = repr(t)
                if r not in seen:
                    seen.add(r); candidates.append(t)
        candidates.sort(key=repr)

        # forall-L with unused term
        for i, f in enumerate(s.left):
            if isinstance(f, Forall):
                key = (id(node), id(f))
                used = used_terms.setdefault(key, set())
                for t in candidates:
                    name = repr(t)
                    if name in used:
                        continue
                    # Don't instantiate with the bound variable itself (would be a no-op).
                    if isinstance(t, Var) and t.name == f.var:
                        continue
                    used.add(name)
                    new = s.copy()
                    new.left.append(f.body.subst(f.var, t))
                    child = SequentNode(seq=new)
                    used_terms[(id(child), id(f))] = set(used)
                    node.rule = f"forall-L[{name}]"
                    node.children = [child]
                    return True

        # exists-R with unused term
        for i, f in enumerate(s.right):
            if isinstance(f, Exists):
                key = (id(node), id(f))
                used = used_terms.setdefault(key, set())
                for t in candidates:
                    name = repr(t)
                    if name in used:
                        continue
                    if isinstance(t, Var) and t.name == f.var:
                        continue
                    used.add(name)
                    new = s.copy()
                    new.right.append(f.body.subst(f.var, t))
                    child = SequentNode(seq=new)
                    used_terms[(id(child), id(f))] = set(used)
                    node.rule = f"exists-R[{name}]"
                    node.children = [child]
                    return True
        return False

    def _try_quant_fresh(self, node: SequentNode,
                         used_terms: Dict[Tuple[int, int], Set[str]],
                         fresh_counts: Dict[Tuple[int, int], int]) -> bool:
        s = node.seq
        for i, f in enumerate(s.left):
            if isinstance(f, Forall):
                key = (id(node), id(f))
                if fresh_counts.get(key, 0) >= self.fresh_cap:
                    continue
                fresh_counts[key] = fresh_counts.get(key, 0) + 1
                c = self._fresh_term_symbol(s, prefix="k")
                t = Var(c)
                used_terms.setdefault(key, set()).add(repr(t))
                new = s.copy()
                new.left.append(f.body.subst(f.var, t))
                child = SequentNode(seq=new)
                fresh_counts[(id(child), id(f))] = fresh_counts[key]
                used_terms[(id(child), id(f))] = set(used_terms[key])
                node.rule = f"forall-L[fresh {c}]"
                node.children = [child]
                return True

        for i, f in enumerate(s.right):
            if isinstance(f, Exists):
                key = (id(node), id(f))
                if fresh_counts.get(key, 0) >= self.fresh_cap:
                    continue
                fresh_counts[key] = fresh_counts.get(key, 0) + 1
                c = self._fresh_term_symbol(s, prefix="k")
                t = Var(c)
                used_terms.setdefault(key, set()).add(repr(t))
                new = s.copy()
                new.right.append(f.body.subst(f.var, t))
                child = SequentNode(seq=new)
                fresh_counts[(id(child), id(f))] = fresh_counts[key]
                used_terms[(id(child), id(f))] = set(used_terms[key])
                node.rule = f"exists-R[fresh {c}]"
                node.children = [child]
                return True
        return False

    # --- util --------------------------------------------------------------

    def _fresh_term_symbol(self, s: Sequent, prefix: str = "c") -> str:
        taken: Set[str] = set()
        for t in _all_terms_in_seq(s):
            if isinstance(t, Var): taken.add(t.name)
            elif isinstance(t, Func): taken.add(t.name)
        while True:
            self._fresh_counter += 1
            cand = f"{prefix}{self._fresh_counter}"
            if cand not in taken:
                return cand

    def _count_open_leaves(self, root: SequentNode) -> int:
        n = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if not node.children:
                if node.closed_by is None:
                    n += 1
            else:
                stack.extend(node.children)
        return n


def _is_bound_variable(t: Term, s: Sequent) -> bool:
    """Return True iff t is (or contains) a Var whose name is bound by some
    quantifier in the sequent. We don't want to instantiate a universal with
    its own bound variable (no-op), or with a compound term like f(x) that
    mentions a bound variable — such a substitution is unsound (the bound
    variable escapes its scope) and also useless (we already have that term
    available in its already-substituted forms).
    """
    if isinstance(t, Var):
        for f in s.left + s.right:
            if _name_is_bound_in(t.name, f):
                return True
        return False
    if isinstance(t, Func):
        return any(_is_bound_variable(a, s) for a in t.args)
    return False


def _name_is_bound_in(name: str, f: Formula) -> bool:
    if isinstance(f, (Top, Bot, Atom)): return False
    if isinstance(f, Not): return _name_is_bound_in(name, f.f)
    if isinstance(f, (And, Or, Imp)):
        return _name_is_bound_in(name, f.l) or _name_is_bound_in(name, f.r)
    if isinstance(f, (Forall, Exists)):
        return f.var == name or _name_is_bound_in(name, f.body)
    return False
