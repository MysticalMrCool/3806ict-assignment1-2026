"""
Improved prover — same LK sequent calculus as the baseline, but with four
described improvements layered on top of Algorithm 2:

(I1) Duplicate-formula elimination
        If the new formula produced by a rule application is already present
        on the same side of the sequent, skip the addition (implicit
        contraction). Prevents the list of formulae from growing without bound.

(I2) Branch-local loop detection
        Each branch carries a set of visited sequents (represented as a pair
        of frozenset of repr-strings for left/right). When a rule would
        re-create a sequent already seen on the current branch, we treat
        that rule as inapplicable and try the next one. Guarantees propositional
        termination and curbs FOL quantifier loops.

(I3) Closure-lookahead rule selection
        Before committing to any rule, first check every applicable rule to
        see whether it produces at least one premise that closes immediately
        by id/TR/FL. If such a rule exists, prefer it. Cuts branching work.

(I4) Herbrand-guided quantifier instantiation
        When instantiating forall-L (resp. exists-R) on a formula with head
        predicate P, prefer terms already occurring as arguments of P on the
        succedent (resp. antecedent) — these are the witnesses most likely
        to close by id. Fall back to all other terms, then to a fresh symbol.

The rule ordering from Algorithm 2 is retained:
    axiom >> invertible non-branching >> invertible branching >> quantifier (unused) >> quantifier (fresh)

This module deliberately mirrors baseline.py's structure so the two can be
cleanly compared in the benchmark.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set, FrozenSet
from .formula import (Formula, Term, Var, Func, Top, Bot, Atom,
                      Not, And, Or, Imp, Forall, Exists, is_atomic)
from .baseline import (Sequent, SequentNode, ProofResult,
                       _all_terms_in_seq, _is_bound_variable)


SeqKey = Tuple[FrozenSet[str], FrozenSet[str]]


def _seq_key(s: Sequent) -> SeqKey:
    return (frozenset(repr(f) for f in s.left),
            frozenset(repr(f) for f in s.right))


def _contains_formula(fs: List[Formula], target: Formula) -> bool:
    for f in fs:
        if f == target:
            return True
    return False


def _atoms_on_side(side: List[Formula], pred: Optional[str] = None) -> List[Atom]:
    out: List[Atom] = []
    stack: List[Formula] = list(side)
    while stack:
        f = stack.pop()
        if isinstance(f, Atom):
            if pred is None or f.pred == pred:
                out.append(f)
        elif isinstance(f, Not):
            stack.append(f.f)
        elif isinstance(f, (And, Or, Imp)):
            stack.append(f.l); stack.append(f.r)
        elif isinstance(f, (Forall, Exists)):
            stack.append(f.body)
    return out


def _head_pred(f: Formula) -> Optional[str]:
    """Return the predicate symbol of a quantified formula's core atom, if obvious."""
    cur = f
    while isinstance(cur, (Forall, Exists, Not)):
        cur = cur.body if isinstance(cur, (Forall, Exists)) else cur.f
    if isinstance(cur, Atom):
        return cur.pred
    # if it's a connective we cannot pick a single head predicate
    return None


class Improved:
    """Improved prover (I1–I4). Interface mirrors Baseline."""

    def __init__(self, step_limit: int = 2000, fresh_cap: int = 6):
        self.step_limit = step_limit
        self.fresh_cap = fresh_cap
        self._fresh_counter = 0

    def prove(self, goal: Formula) -> ProofResult:
        self._fresh_counter = 0
        root_seq = Sequent(left=[], right=[goal])
        root = SequentNode(seq=root_seq)
        steps = 0
        used_terms: Dict[Tuple[int, int], Set[str]] = {}
        fresh_counts: Dict[Tuple[int, int], int] = {}
        # branch-local ancestor visits: id(node) -> set of SeqKey observed on
        # the branch ending at node (inclusive of node itself)
        ancestors: Dict[int, Set[SeqKey]] = {id(root): {_seq_key(root.seq)}}

        open_leaves: List[SequentNode] = [root]
        aborted = False

        while open_leaves:
            if steps >= self.step_limit:
                aborted = True
                break
            node = open_leaves.pop(0)

            # 1. axiom
            closed_by = self._try_axiom(node.seq)
            if closed_by:
                node.rule = closed_by
                node.closed_by = closed_by
                steps += 1
                continue

            # 2/3. invertible rules: first look for any rule that produces a
            # premise which immediately closes (closure-lookahead, I3).
            applied = self._try_invertible_with_lookahead(node, ancestors)
            if applied:
                steps += 1
                open_leaves.extend(node.children)
                continue

            # 4. quantifier with unused term (Herbrand-guided, I4)
            applied = self._try_quant_unused(node, used_terms, ancestors)
            if applied:
                steps += 1
                open_leaves.extend(node.children)
                continue

            # 5. quantifier with fresh term
            applied = self._try_quant_fresh(node, used_terms, fresh_counts, ancestors)
            if applied:
                steps += 1
                open_leaves.extend(node.children)
                continue

            node.rule = "stop"

        open_count = self._count_open_leaves(root)
        if aborted:
            status = "aborted"
        elif open_count == 0:
            status = "proved"
        else:
            status = "countermodel"
        return ProofResult(status=status, steps=steps, tree=root, open_leaves=open_count)

    # --- axiom ------------------------------------------------------------

    def _try_axiom(self, s: Sequent) -> Optional[str]:
        for f in s.right:
            if isinstance(f, Top): return "TR"
        for f in s.left:
            if isinstance(f, Bot): return "FL"
        for f in s.left:
            if is_atomic(f) and f in s.right:
                return "id"
        return None

    # --- invertible rules with closure-lookahead --------------------------

    def _try_invertible_with_lookahead(self, node: SequentNode,
                                        ancestors: Dict[int, Set[SeqKey]]) -> bool:
        """
        Gather every invertible (non-branching + branching) rule applicable to
        node.seq. Prefer any rule whose premises all close immediately by
        axiom (I3); otherwise prefer non-branching over branching; otherwise
        first-found in textbook order.
        """
        candidates = self._invertible_candidates(node.seq)
        if not candidates:
            return False

        # Score: 2 = all children close immediately, 1 = some child closes,
        # 0 = no child closes. Break ties: fewer children first (prefer non-branching).
        best = None
        best_score = (-1, 10**9, 10**9)  # (score, -nonbranching_bonus, order_idx)
        for idx, (rule_name, children_seqs, _side_info) in enumerate(candidates):
            closures = sum(1 for cs in children_seqs if self._try_axiom(cs))
            all_close = int(closures == len(children_seqs))
            score = (all_close * 2 + (1 if closures > 0 else 0), -len(children_seqs), -idx)
            if score > best_score:
                best_score = score
                best = (rule_name, children_seqs)

        assert best is not None
        rule_name, children_seqs = best

        # Apply loop-detection filter: if every resulting premise has a
        # sequent key already on the ancestor chain, skip this rule and try
        # the next-best. Easiest implementation: filter candidates.
        branch_seen = ancestors[id(node)]
        new_keys = [_seq_key(cs) for cs in children_seqs]
        if any(k in branch_seen for k in new_keys):
            # degrade this rule and try the next
            remaining = [(r, c, s_) for (r, c, s_), (rn, cs, _) in zip(candidates, candidates)
                         if rn != rule_name]
            # fallback: just take the next candidate whose children don't all repeat
            for rn, cs, _ in candidates:
                nks = [_seq_key(c) for c in cs]
                if rn == rule_name and nks == new_keys:
                    continue
                if not all(k in branch_seen for k in nks):
                    rule_name, children_seqs = rn, cs
                    new_keys = nks
                    break
            else:
                return False

        children_nodes = [SequentNode(seq=cs) for cs in children_seqs]
        node.rule = rule_name
        node.children = children_nodes
        parent_keys = ancestors[id(node)]
        for cn, k in zip(children_nodes, new_keys):
            ancestors[id(cn)] = parent_keys | {k}
        return True

    def _invertible_candidates(self, s: Sequent) -> List[Tuple[str, List[Sequent], object]]:
        """Enumerate every invertible (non-branching + branching) rule applicable.
        Returns list of (rule_name, premises, side_info)."""
        out: List[Tuple[str, List[Sequent], object]] = []

        # Non-branching first (keeps algorithm's preference order)
        for i, f in enumerate(s.left):
            if isinstance(f, And):
                new = s.copy(); del new.left[i]
                self._add_left(new, f.l); self._add_left(new, f.r)
                out.append(("/\\L", [new], None))
                break
        for i, f in enumerate(s.right):
            if isinstance(f, Or):
                new = s.copy(); del new.right[i]
                self._add_right(new, f.l); self._add_right(new, f.r)
                out.append(("\\/R", [new], None))
                break
        for i, f in enumerate(s.right):
            if isinstance(f, Imp):
                new = s.copy(); del new.right[i]
                self._add_left(new, f.l); self._add_right(new, f.r)
                out.append(("->R", [new], None))
                break
        for i, f in enumerate(s.left):
            if isinstance(f, Not):
                new = s.copy(); del new.left[i]
                self._add_right(new, f.f)
                out.append(("~L", [new], None))
                break
        for i, f in enumerate(s.right):
            if isinstance(f, Not):
                new = s.copy(); del new.right[i]
                self._add_left(new, f.f)
                out.append(("~R", [new], None))
                break
        for i, f in enumerate(s.right):
            if isinstance(f, Forall):
                c = self._fresh_term_symbol(s, prefix="c")
                new = s.copy(); del new.right[i]
                self._add_right(new, f.body.subst(f.var, Var(c)))
                out.append(("forall-R", [new], None))
                break
        for i, f in enumerate(s.left):
            if isinstance(f, Exists):
                c = self._fresh_term_symbol(s, prefix="c")
                new = s.copy(); del new.left[i]
                self._add_left(new, f.body.subst(f.var, Var(c)))
                out.append(("exists-L", [new], None))
                break

        # Branching
        for i, f in enumerate(s.right):
            if isinstance(f, And):
                l_ = s.copy(); del l_.right[i]; self._add_right(l_, f.l)
                r_ = s.copy(); del r_.right[i]; self._add_right(r_, f.r)
                out.append(("/\\R", [l_, r_], None))
                break
        for i, f in enumerate(s.left):
            if isinstance(f, Or):
                l_ = s.copy(); del l_.left[i]; self._add_left(l_, f.l)
                r_ = s.copy(); del r_.left[i]; self._add_left(r_, f.r)
                out.append(("\\/L", [l_, r_], None))
                break
        for i, f in enumerate(s.left):
            if isinstance(f, Imp):
                l_ = s.copy(); del l_.left[i]; self._add_right(l_, f.l)
                r_ = s.copy(); del r_.left[i]; self._add_left(r_, f.r)
                out.append(("->L", [l_, r_], None))
                break
        return out

    # --- I1: duplicate-elimination wrappers -------------------------------

    def _add_left(self, s: Sequent, f: Formula) -> None:
        if not _contains_formula(s.left, f):
            s.left.append(f)

    def _add_right(self, s: Sequent, f: Formula) -> None:
        if not _contains_formula(s.right, f):
            s.right.append(f)

    # --- quantifier: unused term with Herbrand priority (I4) --------------

    def _try_quant_unused(self, node: SequentNode,
                          used_terms: Dict[Tuple[int, int], Set[str]],
                          ancestors: Dict[int, Set[SeqKey]]) -> bool:
        s = node.seq
        all_terms: List[Term] = []
        seen: Set[str] = set()
        for t in _all_terms_in_seq(s):
            if isinstance(t, (Var, Func)) and not _is_bound_variable(t, s):
                r = repr(t)
                if r not in seen:
                    seen.add(r); all_terms.append(t)

        # forall-L
        for i, f in enumerate(s.left):
            if isinstance(f, Forall):
                priority = self._herbrand_order(f, s, "right", all_terms)
                key = (id(node), id(f))
                used = used_terms.setdefault(key, set())
                for t in priority:
                    if isinstance(t, Var) and t.name == f.var:
                        continue
                    name = repr(t)
                    if name in used:
                        continue
                    new = s.copy()
                    new_formula = f.body.subst(f.var, t)
                    self._add_left(new, new_formula)
                    k = _seq_key(new)
                    if k in ancestors[id(node)]:
                        # I2: loop — mark term used but don't commit
                        used.add(name)
                        continue
                    used.add(name)
                    child = SequentNode(seq=new)
                    used_terms[(id(child), id(f))] = set(used)
                    ancestors[id(child)] = ancestors[id(node)] | {k}
                    node.rule = f"forall-L[{name}]"
                    node.children = [child]
                    return True

        # exists-R
        for i, f in enumerate(s.right):
            if isinstance(f, Exists):
                priority = self._herbrand_order(f, s, "left", all_terms)
                key = (id(node), id(f))
                used = used_terms.setdefault(key, set())
                for t in priority:
                    if isinstance(t, Var) and t.name == f.var:
                        continue
                    name = repr(t)
                    if name in used:
                        continue
                    new = s.copy()
                    new_formula = f.body.subst(f.var, t)
                    self._add_right(new, new_formula)
                    k = _seq_key(new)
                    if k in ancestors[id(node)]:
                        used.add(name)
                        continue
                    used.add(name)
                    child = SequentNode(seq=new)
                    used_terms[(id(child), id(f))] = set(used)
                    ancestors[id(child)] = ancestors[id(node)] | {k}
                    node.rule = f"exists-R[{name}]"
                    node.children = [child]
                    return True
        return False

    def _herbrand_order(self, quant: Formula, s: Sequent, opposite_side: str,
                        all_terms: List[Term]) -> List[Term]:
        pred = _head_pred(quant)
        if pred is None:
            return all_terms
        opp = s.right if opposite_side == "right" else s.left
        hot_terms: List[Term] = []
        hot_keys: Set[str] = set()
        for at in _atoms_on_side(opp, pred=pred):
            for a in at.args:
                if isinstance(a, (Var, Func)):
                    k = repr(a)
                    if k not in hot_keys and not _is_bound_variable(a, s):
                        hot_keys.add(k); hot_terms.append(a)
        rest = [t for t in all_terms if repr(t) not in hot_keys]
        return hot_terms + rest

    # --- quantifier: fresh term ------------------------------------------

    def _try_quant_fresh(self, node: SequentNode,
                         used_terms: Dict[Tuple[int, int], Set[str]],
                         fresh_counts: Dict[Tuple[int, int], int],
                         ancestors: Dict[int, Set[SeqKey]]) -> bool:
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
                self._add_left(new, f.body.subst(f.var, t))
                k = _seq_key(new)
                if k in ancestors[id(node)]:
                    continue
                child = SequentNode(seq=new)
                fresh_counts[(id(child), id(f))] = fresh_counts[key]
                used_terms[(id(child), id(f))] = set(used_terms[key])
                ancestors[id(child)] = ancestors[id(node)] | {k}
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
                self._add_right(new, f.body.subst(f.var, t))
                k = _seq_key(new)
                if k in ancestors[id(node)]:
                    continue
                child = SequentNode(seq=new)
                fresh_counts[(id(child), id(f))] = fresh_counts[key]
                used_terms[(id(child), id(f))] = set(used_terms[key])
                ancestors[id(child)] = ancestors[id(node)] | {k}
                node.rule = f"exists-R[fresh {c}]"
                node.children = [child]
                return True
        return False

    # --- util ------------------------------------------------------------

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
