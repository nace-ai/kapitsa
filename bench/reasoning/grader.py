#!/usr/bin/env python3
"""Whether a math answer equals the reference. Standard library only (macOS's own python3 runs it).

  extract(text)        the content of the reply's last \\boxed{} (or \\fbox{}); None without one
  equal(pred, gold)    whether two answers, as written inside the box, are the same
  grade(content, gold, reasoning=None)   both, as a record: {"answer", "correct", "source"}

Both answers are normalised first (\\left/\\right, \\dfrac, spacing, degrees, percent, dollars, thousands separators,
unit words in \\text{...}), then compared by shape:
  text            \\text{(C)}, Evelyn: case-insensitive, without parentheses or a leading article
  base numbers    52_8, 4210_{5}: the digits (and the base, when both give one)
  equations       y = 2x + 3: lhs - rhs as expressions, either sign; against a bare value, a single variable's rhs
  tuples, ranges  (3, \\frac{\\pi}{2}), (3,4], unions with \\cup: the same brackets, element by element
  matrices        \\begin{pmatrix}...\\end{pmatrix}: the same shape, element by element (a vector also matches a tuple)
  lists, sets     1,-2 and \\{1\\pm\\sqrt5,-2\\}: the same elements in any order, \\pm giving two
  expressions     everything else: exactly when both sides are rational (numbers, + - * /, integer powers), else as
                  complex numbers to a relative 1e-9, with variables at seeded random points (all must agree, points
                  outside an expression's domain skipped)
A mixed number (1\\frac{4}{5}) is a sum. Anything the parser cannot read is compared as normalised text, so unusual
notation can be marked wrong: `reasoning.py check` grades every MATH-500 reference solution as a regression test.
"""
import cmath
import math
import random
import re
from fractions import Fraction
from typing import List, Optional, Tuple

# ---------------------------------------------------------------- extraction


def extract(text: Optional[str]) -> Optional[str]:
    """The content of the last \\boxed{...} / \\fbox{...} in text (braces balanced), or of `\\boxed 5`; None when there
    is none, it is empty or its braces never close (a reply cut short)."""
    if not text:
        return None
    i = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if i < 0:
        return None
    i += len("\\boxed") if text.startswith("\\boxed", i) else len("\\fbox")
    while i < len(text) and text[i] in " \t":
        i += 1
    if i < len(text) and text[i] == "{":
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j].strip() or None
        return None
    m = re.match(r"[^\s$]+", text[i:])
    return m.group(0) if m else None


def grade(content: Optional[str], gold: str, reasoning: Optional[str] = None) -> dict:
    """The answer of a reply and whether it is right. The answer is the content's last box, else (a reply that answered
    only in its thinking) the reasoning's; source says which, None when neither has one."""
    ans, source = extract(content), "content"
    if ans is None and reasoning:
        ans, source = extract(reasoning), "reasoning"
    if ans is None:
        return {"answer": None, "correct": False, "source": None}
    return {"answer": ans, "correct": equal(ans, gold), "source": source}


# ---------------------------------------------------------------- normalisation

_ROW = "\u0001"  # stands for a matrix row break (\\) while spacing commands are removed
_TEXT_CMDS = r"(?:text|textbf|textit|textrm|textup|textnormal|mathrm|mathbf|mathit|mbox|operatorname)"


def normalize(s: str) -> str:
    """The answer with presentation removed: what remains is compared."""
    s = (s or "").strip()
    s = s.replace("\\$", "")
    s = s.strip().strip("$").strip()
    for a, b in (("\\(", "\\)"), ("\\[", "\\]")):
        if s.startswith(a) and s.endswith(b):
            s = s[len(a):-len(b)].strip()
    s = s.replace("\\\\", _ROW)
    s = re.sub(r"\\(?:left|right)\.", "", s)
    s = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|big|Big|bigg|Bigg)(?![A-Za-z])", "", s)
    s = re.sub(r"\\[dtc]frac(?![A-Za-z])", r"\\frac", s)
    s = re.sub(r"\\[dt]binom(?![A-Za-z])", r"\\binom", s)
    s = s.replace("\\displaystyle", "")
    s = re.sub(r",\\!\s*(?=\d{3}(?!\d))", ",", s)  # 11,\! 111,\! 100: thousands, however spaced
    s = re.sub(r"\\[,;:! ]", "", s)
    s = re.sub(r"\\q?quad(?![A-Za-z])", " ", s)
    s = s.replace("~", " ")
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", s)
    s = re.sub(r"\\(?:circ|degree)(?![A-Za-z])", "", s).replace("°", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("{,}", ",")
    whole = re.fullmatch(r"\\" + _TEXT_CMDS + r"\s*\{([^{}]*)\}", s)
    if whole:
        s = whole.group(1)
    else:
        # unit words (and an area or volume exponent after them): 864 \mbox{ inches}^2, 5.4 \text{ cents}
        s = re.sub(r"\\" + _TEXT_CMDS + r"\s*\{[^{}]*[A-Za-z][^{}]*\}(?:\s*\^\s*\{?\s*[23]\s*\}?)?", " ", s)
        s = re.sub(r"\\" + _TEXT_CMDS + r"\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"^[A-Za-z]\s*\\in(?![A-Za-z])\s*", "", s.strip())
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
    return s.replace(_ROW, "\\\\")


# ---------------------------------------------------------------- top-level structure


def _split_top(s: str, seps: Tuple[str, ...]) -> List[str]:
    """s split at the separators that are outside every bracket ((), [], {}, \\{ \\})."""
    parts, depth, i, start = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] in "{}":
            depth += 1 if s[i + 1] == "{" else -1
            i += 2
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0:
            for sep in seps:
                if s.startswith(sep, i) and not (sep.startswith("\\") and s[i + len(sep):i + len(sep) + 1].isalpha()):
                    parts.append(s[start:i])
                    i += len(sep)
                    start = i
                    break
            else:
                i += 1
                continue
            continue
        i += 1
    parts.append(s[start:])
    return [p.strip() for p in parts]


def _closes_at_end(s: str) -> bool:
    """Whether the bracket opening s closes at its last character (whatever the bracket types)."""
    depth = 0
    for i, c in enumerate(s):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i == len(s) - 1
    return False


def _pm(s: str) -> Optional[List[str]]:
    """a \\pm b as [a + b, a - b] (one \\pm or \\mp only); None without one."""
    parts = re.split(r"\\(?:pm|mp)(?![A-Za-z])", s)
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    lead = a if a else "0"
    return [f"{lead}+({b})", f"{lead}-({b})"]


def _structure(s: str):
    if not s:
        return ("empty",)
    m = re.fullmatch(r"\\begin\{([pbvBV]?matrix|array)\}(?:\{[^{}]*\})?(.*)\\end\{\1\}", s, re.S)
    if m:
        rows = [r for r in (x.strip() for x in m.group(2).split("\\\\")) if r]
        return ("mat", [[c.strip() for c in r.split("&")] for r in rows])
    m = re.fullmatch(r"([0-9A-Za-z]+)_\{?(\d+)\}?", s)
    if m and any(ch.isdigit() for ch in m.group(1)):
        return ("base", m.group(1).upper(), int(m.group(2)))
    parts = _split_top(s, ("\\cup",))
    if len(parts) > 1:
        return ("union", parts)
    if s.startswith("\\{") and s.endswith("\\}"):
        return ("set", _items(s[2:-2]))
    if s[0] in "([" and s[-1] in ")]" and _closes_at_end(s):
        inner = _split_top(s[1:-1], (",",))
        if len(inner) > 1:
            return ("seq", s[0], s[-1], inner)
    eq = _split_top(s, ("=",))
    if len(eq) == 2 and eq[0] and eq[1]:
        return ("eq", eq[0], eq[1])
    items = _items(s)
    if len(items) > 1:
        return ("set", items)
    if re.fullmatch(r"\(?[A-Za-z][A-Za-z ]*\)?", s) and not re.fullmatch(r"[a-z]", s):
        return ("text", _text(s))
    return ("expr", s)


def _items(s: str) -> List[str]:
    out = []
    for p in _split_top(s, (",",)):
        out.extend(_pm(p) or [p])
    return [p for p in out if p]


def _text(s: str) -> str:
    t = s.strip().strip("()").strip().lower()
    return re.sub(r"^(?:a|an|the)\s+", "", t)


# ---------------------------------------------------------------- comparison


def equal(pred: Optional[str], gold: str) -> bool:
    """Whether pred (an extracted answer) is the same as gold."""
    if pred is None:
        return False
    try:
        return _equal(normalize(pred), normalize(gold), 0)
    except RecursionError:
        return False


def _equal(p: str, g: str, depth: int) -> bool:
    if depth > 4:
        return False
    if p == g or p.replace(" ", "") == g.replace(" ", ""):
        return True
    sp, sg = _structure(p), _structure(g)
    kp, kg = sp[0], sg[0]
    if "empty" in (kp, kg):
        return False
    if kg == "eq" and kp != "eq" and _is_var(sg[1]):
        return _equal(p, sg[2], depth + 1)
    if kp == "eq" and kg != "eq" and _is_var(sp[1]):
        return _equal(sp[2], g, depth + 1)
    if kp == "eq" and kg == "eq":
        a = f"({sp[1]})-({sp[2]})"
        return _expr_equal(a, f"({sg[1]})-({sg[2]})") or _expr_equal(a, f"({sg[2]})-({sg[1]})")
    if kg == "text" or kp == "text":
        return _text(p) == _text(g)
    if kg == "base":
        if kp == "base":
            return sp[1] == sg[1] and sp[2] == sg[2]
        return p.upper() == sg[1]
    if kp == "base":
        return False
    if kp == "seq" and kg == "seq":
        return sp[1] == sg[1] and sp[2] == sg[2] and _all_equal(sp[3], sg[3], depth)
    if kp == "union" and kg == "union":
        return _matching(sp[1], sg[1], depth, ordered=False)
    if kp == "mat" or kg == "mat":
        a, b = _cells(sp), _cells(sg)
        if a is None or b is None or len(a) != len(b):
            return False
        return all(len(x) == len(y) and _all_equal(x, y, depth) for x, y in zip(a, b))
    if kp == "set" or kg == "set":
        a = sp[1] if kp == "set" else [p]
        b = sg[1] if kg == "set" else [g]
        return _matching(a, b, depth, ordered=False)
    if kp == "expr" and kg == "expr":
        return _expr_equal(p, g)
    return False


def _is_var(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z](?:_\{?\w+\}?)?|\\[A-Za-z]+", s.strip()))


def _cells(st) -> Optional[List[List[str]]]:
    """A matrix's rows; a column or row vector and a tuple both as one row of cells."""
    if st[0] == "mat":
        rows = st[1]
        if len(rows) > 1 and all(len(r) == 1 for r in rows):
            return [[r[0] for r in rows]]
        return rows
    if st[0] == "seq" and st[1] == "(" and st[2] == ")":
        return [st[3]]
    return None


def _all_equal(a: List[str], b: List[str], depth: int) -> bool:
    return len(a) == len(b) and all(_equal(x, y, depth + 1) for x, y in zip(a, b))


def _matching(a: List[str], b: List[str], depth: int, ordered: bool) -> bool:
    if len(a) != len(b):
        return False
    if ordered:
        return _all_equal(a, b, depth)
    left = list(b)
    for x in a:
        for k, y in enumerate(left):
            if _equal(x, y, depth + 1):
                del left[k]
                break
        else:
            return False
    return True


# ---------------------------------------------------------------- expressions


class ParseError(ValueError):
    pass


_NUM = re.compile(r"\d+(?:\.\d+)?|\.\d+")
_FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "exp",
          "log", "ln", "lg"}
_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota",
          "kappa", "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega",
          "Gamma", "Delta", "Theta", "Lambda", "Xi", "Sigma", "Phi", "Psi", "Omega"}


def _tokens(s: str):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        m = _NUM.match(s, i)
        if m:
            out.append(("num", m.group(0)))
            i = m.end()
            continue
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m:
                out.append(("cmd", m.group(1)))
                i += len(m.group(0))
            else:
                out.append(("sym", s[i:i + 2]))
                i += 2
            continue
        out.append(("var" if c.isalpha() else "sym", c))
        i += 1
    return out


class _Parser:
    def __init__(self, s: str):
        self.t = _tokens(s)
        self.k = 0

    def peek(self, kind=None, val=None):
        tok = self.t[self.k] if self.k < len(self.t) else (None, None)
        if kind is None:
            return tok
        return tok[0] == kind and (val is None or tok[1] == val)

    def take(self):
        if self.k >= len(self.t):
            raise ParseError("unexpected end")
        tok = self.t[self.k]
        self.k += 1
        return tok

    def need(self, val):
        tok = self.take()
        if tok != ("sym", val):
            raise ParseError(f"expected {val}, got {tok}")

    def parse(self):
        e = self.expr()
        if self.k != len(self.t):
            raise ParseError(f"left over: {self.t[self.k:]}")
        return e

    def expr(self):
        node = self.term()
        while self.peek("sym", "+") or self.peek("sym", "-"):
            op = self.take()[1]
            node = (op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while True:
            if self.peek("sym", "*") or self.peek("sym", "/"):
                node = (self.take()[1], node, self.unary())
            elif self.peek("cmd", "cdot") or self.peek("cmd", "times"):
                self.take()
                node = ("*", node, self.unary())
            elif self.peek("cmd", "div"):
                self.take()
                node = ("/", node, self.unary())
            elif self.starts_atom():
                node = ("*", node, self.power())
            else:
                return node

    def unary(self):
        if self.peek("sym", "-"):
            self.take()
            return ("neg", self.unary())
        if self.peek("sym", "+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        base = self.postfix()
        if self.peek("sym", "^"):
            self.take()
            return ("^", base, self.exponent())
        return base

    def exponent(self):
        if self.peek("sym", "{"):
            self.take()
            e = self.expr()
            self.need("}")
            return e
        if self.peek("sym", "-"):
            self.take()
            return ("neg", self.exponent())
        if self.peek("num"):
            return ("n", _number(self.take()[1]))
        return self.postfix()

    def postfix(self):
        node = self.atom()
        while self.peek("sym", "!"):
            self.take()
            node = ("!", node)
        return node

    def starts_atom(self):
        kind, val = self.peek()
        if kind in ("num", "var"):
            return True
        if kind == "sym":
            return val in "([{"
        if kind == "cmd":
            return val in ("frac", "sqrt", "binom", "pi", "infty") or val in _FUNCS or val in _GREEK
        return False

    def group(self, close):
        e = self.expr()
        self.need(close)
        return e

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            node = ("n", _number(val))
            if val.isdigit() and self.peek("cmd", "frac"):  # a mixed number: 1\frac{4}{5}
                save = self.k
                self.take()
                a, b = self.arg(), self.arg()
                if a[0] == "n" and b[0] == "n" and a[1].denominator == 1 and b[1].denominator == 1:
                    return ("+", node, ("/", a, b))
                self.k = save
            return node
        if kind == "var":
            name = val
            if self.peek("sym", "_"):
                self.take()
                name += "_" + self.subscript()
            if name == "i":
                return ("c", "i")
            if name == "e":
                return ("c", "e")
            return ("v", name)
        if kind == "sym":
            if val == "(":
                return self.group(")")
            if val == "[":
                return self.group("]")
            if val == "{":
                return self.group("}")
            if val == "|":
                return ("abs", self.group("|"))
            raise ParseError(f"unexpected {val}")
        if kind == "cmd":
            if val == "frac":
                return ("/", self.arg(), self.arg())
            if val == "sqrt":
                idx = None
                if self.peek("sym", "["):
                    self.take()
                    idx = self.group("]")
                return ("sqrt", self.arg(), idx)
            if val == "binom":
                return ("binom", self.arg(), self.arg())
            if val == "pi":
                return ("c", "pi")
            if val == "infty":
                return ("c", "inf")
            if val in _GREEK:
                return ("v", "\\" + val)
            if val in _FUNCS:
                return self.func(val)
        raise ParseError(f"unexpected {kind} {val}")

    def subscript(self) -> str:
        if self.peek("sym", "{"):
            self.take()
            out = []
            while not self.peek("sym", "}"):
                out.append(self.take()[1])
            self.take()
            return "".join(out)
        return self.take()[1]

    def arg(self):
        """One argument of \\frac, \\sqrt or \\binom: a braced group or a single character (\\frac43, \\sqrt2)."""
        if self.peek("sym", "{"):
            self.take()
            return self.group("}")
        if self.peek("num"):
            v = self.take()[1]
            if len(v) > 1:
                self.t.insert(self.k, ("num", v[1:]))
                v = v[0]
            if v == ".":
                raise ParseError("bad argument")
            return ("n", _number(v))
        if self.peek("var"):
            return self.atom()
        if self.peek("cmd"):
            return self.atom()
        raise ParseError("bad argument")

    def func(self, name):
        base = power = None
        if self.peek("sym", "_"):
            self.take()
            base = self.arg()
        if self.peek("sym", "^"):
            self.take()
            power = self.exponent()
        if self.peek("sym", "(") or self.peek("sym", "{") or self.peek("sym", "["):
            arg = self.atom()
        else:  # \cot x, \sin 2x: the factors up to the next operator or function
            arg = self.power()
            while self.starts_atom() and not (self.peek("cmd") and self.peek()[1] in _FUNCS):
                arg = ("*", arg, self.power())
        node = ("f", name, arg, base)
        if power is not None:
            if power == ("neg", ("n", Fraction(1))):
                return ("finv", name, arg)
            node = ("^", node, power)
        return node


def _number(v: str) -> Fraction:
    try:
        return Fraction(v if not v.startswith(".") else "0" + v)
    except (ValueError, ZeroDivisionError):
        raise ParseError(f"bad number {v}") from None


_INF = complex(math.inf, 0)


def _ev(n, env) -> complex:
    op = n[0]
    if op == "n":
        return complex(float(n[1]))
    if op == "v":
        return env[n[1]]
    if op == "c":
        return {"pi": complex(math.pi), "e": complex(math.e), "i": 1j, "inf": _INF}[n[1]]
    if op == "neg":
        return -_ev(n[1], env)
    if op in "+-*/":
        a, b = _ev(n[1], env), _ev(n[2], env)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        return a / b
    if op == "^":
        a, b = _ev(n[1], env), _ev(n[2], env)
        if abs(b) > 4096 or (abs(a) > 1 and abs(b) * math.log(abs(a)) > 700):
            raise OverflowError("power too large")
        if b.imag == 0 and b.real == int(b.real) and a.imag == 0:
            return complex(a.real ** int(b.real)) if a.real != 0 or b.real >= 0 else complex(math.inf)
        if a == 0:
            return 0j
        return a ** b
    if op == "sqrt":
        a = _ev(n[1], env)
        if n[2] is None:
            return cmath.sqrt(a)
        k = _ev(n[2], env)
        if a.imag == 0 and a.real < 0 and k.imag == 0 and k.real == int(k.real) and int(k.real) % 2 == 1:
            return complex(-((-a.real) ** (1.0 / k.real)))  # the real odd root
        return a ** (1 / k)
    if op == "!":
        a = _ev(n[1], env)
        if a.imag != 0 or a.real != int(a.real) or not 0 <= a.real <= 170:
            raise ValueError("factorial of a non-integer")
        return complex(math.factorial(int(a.real)))
    if op == "binom":
        a, b = _ev(n[1], env), _ev(n[2], env)
        if a.imag or b.imag or a.real != int(a.real) or b.real != int(b.real) or abs(a.real) > 10000:
            raise ValueError("binomial of non-integers or of a huge n")
        return complex(math.comb(int(a.real), int(b.real)) if a.real >= 0 and b.real >= 0 else 0)
    if op == "abs":
        return complex(abs(_ev(n[1], env)))
    if op in ("f", "finv"):
        x = _ev(n[2], env)
        name = n[1]
        if op == "finv":
            name = {"sin": "arcsin", "cos": "arccos", "tan": "arctan"}.get(name)
            if name is None:
                raise ValueError("inverse function")
        if name in ("log", "lg", "ln"):
            if x == 0:
                raise ValueError("log 0")
            if op == "f" and n[3] is not None:
                return cmath.log(x) / cmath.log(_ev(n[3], env))
            return cmath.log(x) if name == "ln" else cmath.log10(x)
        f = {"sin": cmath.sin, "cos": cmath.cos, "tan": cmath.tan, "sinh": cmath.sinh, "cosh": cmath.cosh,
             "tanh": cmath.tanh, "exp": cmath.exp, "arcsin": cmath.asin, "arccos": cmath.acos, "arctan": cmath.atan,
             "cot": lambda v: 1 / cmath.tan(v), "sec": lambda v: 1 / cmath.cos(v), "csc": lambda v: 1 / cmath.sin(v)}
        return f[name](x)
    raise ValueError(f"unknown node {op}")


def _vars(n, out):
    if isinstance(n, tuple):
        if n[0] == "v":
            out.add(n[1])
        for x in n[1:]:
            _vars(x, out)
    return out


def _exact(n) -> Optional[Fraction]:
    """The value of a rational expression (numbers, + - * /, integer powers, factorials), exactly; None otherwise."""
    op = n[0]
    if op == "n":
        return n[1]
    if op == "neg":
        v = _exact(n[1])
        return None if v is None else -v
    if op in ("+", "-", "*", "/", "^"):
        a, b = _exact(n[1]), _exact(n[2])
        if a is None or b is None:
            return None
        if op == "/":
            return None if b == 0 else a / b
        if op == "^":
            if b.denominator != 1 or abs(b) > 256 or (a == 0 and b < 0):
                return None
            return a ** int(b)
        return a + b if op == "+" else a - b if op == "-" else a * b
    if op == "!":
        a = _exact(n[1])
        return Fraction(math.factorial(int(a))) if a is not None and a.denominator == 1 and 0 <= a <= 170 else None
    return None


def _close(a: complex, b: complex) -> bool:
    if cmath.isnan(a) or cmath.isnan(b):
        return False
    if cmath.isinf(a) or cmath.isinf(b):
        return cmath.isinf(a) and cmath.isinf(b) and (a.real > 0) == (b.real > 0) and a.imag == b.imag == 0
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b)) + 1e-12


def _expr_equal(p: str, g: str) -> bool:
    try:
        a, b = _Parser(p).parse(), _Parser(g).parse()
    except (ParseError, RecursionError, IndexError):
        return _plain(p) == _plain(g)
    ea, eb = _exact(a), _exact(b)
    if ea is not None and eb is not None:  # 11111111101 is not 11111111100, whatever the float tolerance
        return ea == eb
    names = sorted(_vars(a, set()) | _vars(b, set()))
    rng = random.Random(20261019)
    points = [{}] if not names else [{v: complex(rng.uniform(0.31, 2.69)) for v in names} for _ in range(8)]
    agreed = 0
    for env in points:
        try:
            x, y = _ev(a, env), _ev(b, env)
        except (ValueError, ZeroDivisionError, OverflowError, KeyError, TypeError):
            continue
        if not _close(x, y):
            return False
        agreed += 1
    return agreed >= (1 if not names else 3)


def _plain(s: str) -> str:
    return re.sub(r"[\s{}]", "", s)
