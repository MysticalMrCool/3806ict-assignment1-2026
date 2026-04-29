"""
Recursive-descent parser for the course-syntax FOL language.

Grammar (ASCII surface syntax, right-associative ->, left-associative & and |):

    formula  ::= impl
    impl     ::= quant ( '->' impl )?
    quant    ::= ( 'forall' | 'exists' | '!' | '?' ) IDENT '.' quant
               | disj
    disj     ::= conj ( '|' conj )*
    conj     ::= neg  ( '&' neg )*
    neg      ::= '~' neg | atom
    atom     ::= 'T' | 'F'
               | '(' formula ')'
               | IDENT                                # 0-arity predicate or bound var
               | IDENT '(' term (',' term)* ')'       # predicate application

    term     ::= IDENT | IDENT '(' term (',' term)* ')'

Tokens also accepted (Unicode aliases for readability):
    ∀ -> forall, ∃ -> exists, ¬ -> ~, ∧ -> &, ∨ -> |, →/⟶ -> ->, ⊤ -> T, ⊥ -> F
"""

from __future__ import annotations
import re
from typing import List, Tuple
from .formula import (Formula, Term, Var, Func, Top, Bot, Atom,
                      Not, And, Or, Imp, Forall, Exists)

# --- tokeniser -------------------------------------------------------------

_UNICODE_MAP = {
    "∀": "forall", "∃": "exists",
    "¬": "~",
    "∧": "&", "∨": "|",
    "→": "->", "⟶": "->", "⇒": "->",
    "⊤": "T", "⊥": "F",
    "\\forall": "forall", "\\exists": "exists",
    "\\not": "~", "\\neg": "~",
    "\\and": "&", "\\wedge": "&",
    "\\or": "|", "\\vee": "|",
    "\\imp": "->", "\\to": "->", "\\longrightarrow": "->", "\\implies": "->",
    "\\top": "T", "\\bot": "F",
}

_TOKEN_RE = re.compile(r"""
    \s+                                    |   # whitespace
    (?P<arrow>->)                          |
    (?P<ident>[A-Za-z_][A-Za-z_0-9']*)     |
    (?P<punct>[(),.~&|!?])
""", re.VERBOSE)

_KEYWORDS = {"forall", "exists"}
_TRUTH_CONSTS = {"T": "T", "F": "F",
                 "true": "T", "false": "F",
                 "True": "T", "False": "F",
                 "top": "T", "bot": "F"}

class ParseError(Exception):
    pass

def _normalise(src: str) -> str:
    for k, v in _UNICODE_MAP.items():
        src = src.replace(k, v)
    return src

def _tokenise(src: str) -> List[Tuple[str, str]]:
    src = _normalise(src)
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise ParseError(f"Unexpected char {src[i]!r} at pos {i}")
        i = m.end()
        if m.group("arrow"):   out.append(("OP", "->"))
        elif m.group("ident"):
            tok = m.group("ident")
            out.append(("KW" if tok in _KEYWORDS else "ID", tok))
        elif m.group("punct"):
            out.append(("PUNCT", m.group("punct")))
        # whitespace fall-through
    out.append(("EOF", ""))
    return out


# --- parser ----------------------------------------------------------------

class _Parser:
    def __init__(self, toks: List[Tuple[str, str]]):
        self.toks = toks
        self.pos = 0

    def peek(self) -> Tuple[str, str]:
        return self.toks[self.pos]

    def eat(self, kind: str | None = None, text: str | None = None) -> Tuple[str, str]:
        k, t = self.toks[self.pos]
        if kind is not None and k != kind:
            raise ParseError(f"Expected {kind}, got {k}:{t!r}")
        if text is not None and t != text:
            raise ParseError(f"Expected {text!r}, got {t!r}")
        self.pos += 1
        return k, t

    def parse_formula(self) -> Formula:
        f = self.parse_impl()
        if self.peek()[0] != "EOF":
            raise ParseError(f"Trailing input starting at {self.peek()}")
        return f

    def parse_impl(self) -> Formula:
        left = self.parse_quant()
        if self.peek() == ("OP", "->"):
            self.eat("OP", "->")
            right = self.parse_impl()       # right-assoc
            return Imp(left, right)
        return left

    def parse_quant(self) -> Formula:
        k, t = self.peek()
        if (k, t) in (("KW", "forall"), ("PUNCT", "!")):
            self.eat()
            _, x = self.eat("ID")
            self.eat("PUNCT", ".")
            # Quantifier scope extends as far right as possible (Isabelle convention):
            # parse an impl, not just a disj — so `forall x. P(x) -> Q(x)` means
            # `forall x. (P(x) -> Q(x))`.
            body = self.parse_impl()
            return Forall(x, body)
        if (k, t) in (("KW", "exists"), ("PUNCT", "?")):
            self.eat()
            _, x = self.eat("ID")
            self.eat("PUNCT", ".")
            body = self.parse_impl()
            return Exists(x, body)
        return self.parse_disj()

    def parse_disj(self) -> Formula:
        left = self.parse_conj()
        while self.peek() == ("PUNCT", "|"):
            self.eat()
            right = self.parse_conj()
            left = Or(left, right)
        return left

    def parse_conj(self) -> Formula:
        left = self.parse_neg()
        while self.peek() == ("PUNCT", "&"):
            self.eat()
            right = self.parse_neg()
            left = And(left, right)
        return left

    def parse_neg(self) -> Formula:
        if self.peek() == ("PUNCT", "~"):
            self.eat()
            return Not(self.parse_neg())
        return self.parse_atom()

    def parse_atom(self) -> Formula:
        k, t = self.peek()
        # Quantifiers may appear wherever an atom is expected
        # (e.g. in `A | forall x. P(x)`).
        if (k, t) in (("KW", "forall"), ("KW", "exists"),
                      ("PUNCT", "!"), ("PUNCT", "?")):
            return self.parse_quant()
        if (k, t) == ("PUNCT", "("):
            self.eat()
            f = self.parse_impl()
            self.eat("PUNCT", ")")
            return f
        if k == "ID":
            self.eat()
            # Treat T / F (and their aliases) as truth constants only when NOT
            # followed by `(` — otherwise they're ordinary predicate names.
            if t in _TRUTH_CONSTS and self.peek() != ("PUNCT", "("):
                return Top() if _TRUTH_CONSTS[t] == "T" else Bot()
            args: Tuple[Term, ...] = ()
            if self.peek() == ("PUNCT", "("):
                self.eat()
                args = (self.parse_term(),)
                while self.peek() == ("PUNCT", ","):
                    self.eat()
                    args = args + (self.parse_term(),)
                self.eat("PUNCT", ")")
            return Atom(t, args)
        raise ParseError(f"Unexpected token {t!r} while parsing atom")

    def parse_term(self) -> Term:
        k, t = self.peek()
        if k != "ID":
            raise ParseError(f"Expected term, got {t!r}")
        self.eat()
        if self.peek() == ("PUNCT", "("):
            self.eat()
            args = (self.parse_term(),)
            while self.peek() == ("PUNCT", ","):
                self.eat()
                args = args + (self.parse_term(),)
            self.eat("PUNCT", ")")
            return Func(t, args)
        # Bare identifier — whether it's a Var or a 0-arity Func is resolved by scope elsewhere.
        # For our prover we always wrap as Var if it is a quantifier-bound variable;
        # otherwise as Func (constant). Parser conservatively returns Var here and lets
        # the caller interpret; prover treats unbound Vars as free/constants.
        return Var(t)


def parse(src: str) -> Formula:
    """Parse a single formula."""
    return _Parser(_tokenise(src)).parse_formula()

def parse_many(src):
    out = []
    for raw in src.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        out.append(parse(line))
    return out
