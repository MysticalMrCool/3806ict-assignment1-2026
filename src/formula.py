"""
First-order logic formula AST.

Concrete syntax used by the parser (course-aligned):
    T, F                    -- truth constants
    P, P(x), P(x, y, ...)   -- atomic predicates
    ~A                      -- negation
    A & B                   -- conjunction
    A | B                   -- disjunction
    A -> B                  -- implication
    forall x. A             -- universal
    exists x. A             -- existential
Terms:
    x, y, z, ...            -- variables (lowercase identifiers)
    c, d, ...               -- constants (same lexical class; semantics decided by quantifier binding)
    f(t1, ..., tn)          -- function application
Binding precedence (tight to loose):
    ~   >   &   >   |   >   ->   >   forall/exists
Implication is right-associative; &, | are left-associative.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, FrozenSet


# --- Terms -----------------------------------------------------------------

class Term:
    """Base class for terms. Subclasses: Var, Func (includes constants as 0-arity)."""
    __slots__ = ()
    def free_vars(self) -> FrozenSet[str]:
        raise NotImplementedError
    def subst(self, x: str, t: "Term") -> "Term":
        raise NotImplementedError

@dataclass(frozen=True)
class Var(Term):
    name: str
    def __repr__(self) -> str: return self.name
    def free_vars(self) -> FrozenSet[str]: return frozenset({self.name})
    def subst(self, x: str, t: Term) -> Term:
        return t if self.name == x else self

@dataclass(frozen=True)
class Func(Term):
    """Function symbol application. A zero-arity Func is a constant."""
    name: str
    args: Tuple[Term, ...] = field(default_factory=tuple)
    def __repr__(self) -> str:
        if not self.args: return self.name
        return f"{self.name}({', '.join(map(repr, self.args))})"
    def free_vars(self) -> FrozenSet[str]:
        out: FrozenSet[str] = frozenset()
        for a in self.args: out = out | a.free_vars()
        return out
    def subst(self, x: str, t: Term) -> Term:
        return Func(self.name, tuple(a.subst(x, t) for a in self.args))


# --- Formulae --------------------------------------------------------------

class Formula:
    __slots__ = ()
    def free_vars(self) -> FrozenSet[str]:
        raise NotImplementedError
    def subst(self, x: str, t: Term) -> "Formula":
        raise NotImplementedError

@dataclass(frozen=True)
class Top(Formula):
    def __repr__(self) -> str: return "T"
    def free_vars(self) -> FrozenSet[str]: return frozenset()
    def subst(self, x: str, t: Term) -> Formula: return self

@dataclass(frozen=True)
class Bot(Formula):
    def __repr__(self) -> str: return "F"
    def free_vars(self) -> FrozenSet[str]: return frozenset()
    def subst(self, x: str, t: Term) -> Formula: return self

@dataclass(frozen=True)
class Atom(Formula):
    pred: str
    args: Tuple[Term, ...] = field(default_factory=tuple)
    def __repr__(self) -> str:
        if not self.args: return self.pred
        return f"{self.pred}({', '.join(map(repr, self.args))})"
    def free_vars(self) -> FrozenSet[str]:
        out: FrozenSet[str] = frozenset()
        for a in self.args: out = out | a.free_vars()
        return out
    def subst(self, x: str, t: Term) -> Formula:
        return Atom(self.pred, tuple(a.subst(x, t) for a in self.args))

@dataclass(frozen=True)
class Not(Formula):
    f: Formula
    def __repr__(self) -> str: return f"~{_paren(self.f, 4)}"
    def free_vars(self) -> FrozenSet[str]: return self.f.free_vars()
    def subst(self, x: str, t: Term) -> Formula: return Not(self.f.subst(x, t))

@dataclass(frozen=True)
class And(Formula):
    l: Formula; r: Formula
    def __repr__(self) -> str: return f"{_paren(self.l, 3)} & {_paren(self.r, 3)}"
    def free_vars(self) -> FrozenSet[str]: return self.l.free_vars() | self.r.free_vars()
    def subst(self, x: str, t: Term) -> Formula: return And(self.l.subst(x, t), self.r.subst(x, t))

@dataclass(frozen=True)
class Or(Formula):
    l: Formula; r: Formula
    def __repr__(self) -> str: return f"{_paren(self.l, 2)} | {_paren(self.r, 2)}"
    def free_vars(self) -> FrozenSet[str]: return self.l.free_vars() | self.r.free_vars()
    def subst(self, x: str, t: Term) -> Formula: return Or(self.l.subst(x, t), self.r.subst(x, t))

@dataclass(frozen=True)
class Imp(Formula):
    l: Formula; r: Formula
    def __repr__(self) -> str: return f"{_paren(self.l, 2)} -> {_paren(self.r, 1)}"
    def free_vars(self) -> FrozenSet[str]: return self.l.free_vars() | self.r.free_vars()
    def subst(self, x: str, t: Term) -> Formula: return Imp(self.l.subst(x, t), self.r.subst(x, t))

@dataclass(frozen=True)
class Forall(Formula):
    var: str; body: Formula
    def __repr__(self) -> str: return f"forall {self.var}. {self.body!r}"
    def free_vars(self) -> FrozenSet[str]: return self.body.free_vars() - {self.var}
    def subst(self, x: str, t: Term) -> Formula:
        if self.var == x: return self
        # capture-avoiding: if x in free_vars(t) and var in t.free_vars, rename.
        if self.var in t.free_vars():
            fresh = _fresh(self.var, self.body.free_vars() | t.free_vars() | {x})
            renamed_body = self.body.subst(self.var, Var(fresh))
            return Forall(fresh, renamed_body.subst(x, t))
        return Forall(self.var, self.body.subst(x, t))

@dataclass(frozen=True)
class Exists(Formula):
    var: str; body: Formula
    def __repr__(self) -> str: return f"exists {self.var}. {self.body!r}"
    def free_vars(self) -> FrozenSet[str]: return self.body.free_vars() - {self.var}
    def subst(self, x: str, t: Term) -> Formula:
        if self.var == x: return self
        if self.var in t.free_vars():
            fresh = _fresh(self.var, self.body.free_vars() | t.free_vars() | {x})
            renamed_body = self.body.subst(self.var, Var(fresh))
            return Exists(fresh, renamed_body.subst(x, t))
        return Exists(self.var, self.body.subst(x, t))


# --- helpers ---------------------------------------------------------------

_PREC = {Forall: 0, Exists: 0, Imp: 1, Or: 2, And: 3, Not: 4}

def _paren(f: Formula, outer: int) -> str:
    p = _PREC.get(type(f), 5)
    s = repr(f)
    return f"({s})" if p < outer else s

def _fresh(base: str, avoid: set) -> str:
    i = 0
    while True:
        cand = f"{base}{i}" if i else f"{base}_"
        if cand not in avoid:
            return cand
        i += 1


# --- convenience sub-classes for atomic detection --------------------------

def is_atomic(f: Formula) -> bool:
    return isinstance(f, (Atom, Top, Bot))
