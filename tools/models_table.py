#!/usr/bin/env python3
"""Writes docs/models.md from the catalog source (src/hub/catalog.cpp): every model's aliases, the Hugging Face repo
and pinned revision each one downloads, its drafter and the download size.

    python3 tools/models_table.py            # rewrite docs/models.md
    python3 tools/models_table.py --check    # exit 1 when docs/models.md differs from the catalog (CI)

Nothing here is typed by hand: the catalog's C++ initializers are parsed as they are, so a catalog change that is not
followed by a rewrite of the table fails CI. Stdlib only, no build.
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "src" / "hub" / "catalog.cpp"
OUT = ROOT / "docs" / "models.md"

DRAFTER_KINDS = {"dflash2", "dflash", "lilicorr", "dspark"}  # Companion::drafter() in src/hub/hub.h
MTP_KINDS = {"mtp", "mtp-sidecar"}


# ---- parsing the catalog's C++ initializers ------------------------------------------------------------------------


def tokenize(src):
    """Strings (adjacent literals are joined by the parser), numbers, identifiers and punctuation; comments dropped."""
    toks, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
        elif src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
        elif src.startswith("/*", i):
            i = src.index("*/", i) + 2
        elif c == '"':
            j = i + 1
            while src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            toks.append(("str", bytes(src[i + 1 : j], "utf-8").decode("unicode_escape")))
            i = j + 1
        elif c.isdigit():
            m = re.compile(r"[0-9][0-9a-zA-Z.']*").match(src, i)
            toks.append(("num", m.group().replace("'", "")))
            i = m.end()
        elif c.isalpha() or c == "_":
            m = re.compile(r"[A-Za-z_][A-Za-z_0-9:]*").match(src, i)
            toks.append(("id", m.group()))
            i = m.end()
        else:
            toks.append(("p", c))
            i += 1
    return toks


class Parser:
    def __init__(self, toks, funcs=None):
        self.t, self.i, self.funcs = toks, 0, funcs or {}

    def peek(self, k=0):
        return self.t[self.i + k] if self.i + k < len(self.t) else ("eof", "")

    def take(self, kind=None, val=None):
        tok = self.peek()
        if (kind and tok[0] != kind) or (val and tok[1] != val):
            raise SystemExit(f"catalog.cpp: expected {kind or ''} {val or ''}, got {tok} at token {self.i}")
        self.i += 1
        return tok

    def value(self):
        tok = self.peek()
        if tok[0] == "str":
            s = ""
            while self.peek()[0] == "str":
                s += self.take()[1]
            return s
        if tok[0] == "num":
            return self.take()[1]
        if tok == ("p", "{"):
            return self.braces()
        if tok[0] == "id" and self.peek(1) == ("p", "("):  # mtp35(): a companion function from the same file
            name = self.take()[1]
            self.take("p", "("), self.take("p", ")")
            if name not in self.funcs:
                raise SystemExit(f"catalog.cpp: unknown function {name}()")
            return self.funcs[name]
        raise SystemExit(f"catalog.cpp: cannot read a value at {tok} (token {self.i})")

    def braces(self):
        """{.a = x, .b = y} -> dict; {x, y} -> list."""
        self.take("p", "{")
        if self.peek() == ("p", "}"):
            self.take()
            return []
        designated = self.peek() == ("p", ".")
        out = {} if designated else []
        while self.peek() != ("p", "}"):
            if designated:
                self.take("p", ".")
                key = self.take("id")[1]
                self.take("p", "=")
                out[key] = self.value()
            else:
                out.append(self.value())
            if self.peek() == ("p", ","):
                self.take()
        self.take("p", "}")
        return out


def parse_catalog(src):
    toks = tokenize(src)
    # the companion helpers: `Companion name() { ... return {...}; }` inside the anonymous namespace
    funcs, i = {}, 0
    while i < len(toks) - 5:
        if toks[i] == ("id", "Companion") and toks[i + 1][0] == "id" and toks[i + 2 : i + 5] == [
            ("p", "("), ("p", ")"), ("p", "{")]:
            name = toks[i + 1][1]
            j = i + 5
            while toks[j] != ("id", "return"):
                j += 1
            p = Parser(toks[j + 1 :], funcs)
            funcs[name] = p.braces()
            i = j
        i += 1
    # the catalog itself: `static const std::vector<CatalogModel> c = { ... };`
    for i in range(len(toks) - 3):
        if toks[i] == ("id", "c") and toks[i + 1] == ("p", "=") and toks[i + 2] == ("p", "{"):
            break
    else:
        raise SystemExit("catalog.cpp: the catalog() table was not found")
    p = Parser(toks[i + 2 :], funcs)
    models = p.braces()
    if not models or not all(isinstance(m, dict) for m in models):
        raise SystemExit("catalog.cpp: the catalog() table did not parse as a list of models")
    check(models, src)
    return models


def check(models, src):
    """Fail loudly when the parse lost something: fewer variants than `.revision =` lines, a short revision, a gap."""
    n_variants = sum(len(m["variants"]) for m in models)
    n_aliased = len(re.findall(r"^\s*\.aliases = ", src, re.M))
    if n_aliased != len(models):
        raise SystemExit(f"catalog.cpp: {n_aliased} `.aliases` but {len(models)} models parsed")
    for m in models:
        names = [v["name"] for v in m["variants"]]
        if m["default_variant"] not in names:
            raise SystemExit(f"catalog.cpp: {m['name']}: default variant {m['default_variant']} is not a variant")
        for v in m["variants"]:
            for r in [v] + [c for c in v["companions"]]:
                if not re.fullmatch(r"[0-9a-f]{40}", r.get("revision", "")) or not r.get("repo"):
                    raise SystemExit(f"catalog.cpp: {m['name']}:{v['name']}: a repo without a pinned revision")
    # every `.revision =` of a variant or a companion function is read: variants + companions of the helper functions
    n_rev = len(re.findall(r"^\s*\.revision = ", src, re.M))
    n_helpers = len(re.findall(r"^Companion \w+\(\) \{", src, re.M))
    if n_rev != n_variants + n_helpers:
        raise SystemExit(f"catalog.cpp: {n_rev} `.revision` lines but {n_variants} variants and {n_helpers} helpers")


# ---- the table -----------------------------------------------------------------------------------------------------


def link(repo):
    return f"[`{repo}`](https://huggingface.co/{repo})"


def rev(repo, revision):
    return f"[`{revision[:7]}`](https://huggingface.co/{repo}/tree/{revision})"


def short_alias(m):
    """The shortest alias, as short_name() in src/hub/resolve.cpp: the first when no alias is shorter."""
    best = m["name"]
    for a in m["aliases"]:
        if len(a) < len(best):
            best = a
    return best


def drafter_cell(v):
    ds = [c for c in v["companions"] if c["kind"] in DRAFTER_KINDS]
    return "<br>".join(f"{link(c['repo'])} @ {rev(c['repo'], c['revision'])}" for c in ds) or "none"


def mtp_cell(v):
    cs = [c for c in v["companions"] if c["kind"] in MTP_KINDS]
    if not cs:  # the 35B's 4bit repo ships its own MTP head
        return "none"
    return "<br>".join(f"{link(c['repo'])} @ {rev(c['repo'], c['revision'])}" + (" (its `mtp.*` tensors)" if c["kind"] == "mtp-sidecar" else "")
                       for c in cs)


def row(m, v, first_col):
    gguf = " (GGUF file, converted on download)" if v.get("gguf") else ""
    size = f"{v['download_gb']} GB"
    return (f"| {first_col} | {link(v['repo'])}{gguf} | {rev(v['repo'], v['revision'])} | {drafter_cell(v)} | {mtp_cell(v)} | {size} |")


def render(models):
    head = ("| `kapitsa serve --model` | Hugging Face repo | Revision | Drafter | MTP head | Download |\n"
            "| --- | --- | --- | --- | --- | ---: |\n")
    default_rows, other_rows = [], []
    for m in models:
        sa = short_alias(m)
        names = ", ".join(f"`{a}`" for a in [sa] + [a for a in m["aliases"] if a != sa])
        for v in m["variants"]:
            is_default = v["name"] == m["default_variant"]
            if is_default:
                col = f"{names}<br><sub>{m['title']}: `{m['name']}`</sub>"
                default_rows.append(row(m, v, col))
            else:
                other_rows.append(row(m, v, f"`{sa}:{v['name']}`"))
    out = [
        "# Models",
        "",
        "Which Hugging Face repo each model name downloads. `kapitsa serve --model 27b` and `kapitsa pull 27b` take any "
        "alias in the first column of the first table; a variant is the alias plus `:variant` (`27b:8bit`). The "
        "repos and revisions are the catalog's ([src/hub/catalog.cpp](../src/hub/catalog.cpp)), so every download is "
        "the same bytes.",
        "",
        "This page is written by [tools/models_table.py](../tools/models_table.py) from the catalog and checked in CI: "
        "after a catalog change run `python3 tools/models_table.py` and commit the result. For what each model is and "
        "which Mac it fits, see [Models and downloads](usage.md#models-and-downloads) and `kapitsa doctor`.",
        "",
        "## Default models",
        "",
        "What `--model <alias>` downloads where it fits the Mac. Where the default variant does not fit, `pull` and "
        "`serve` take the model's smaller variant (`16gb`, `24gb`, `64gb`) listed in the second table.",
        "",
        head + "\n".join(default_rows),
        "",
        "## Other variants",
        "",
        "Another precision of the same model, or a smaller pack for a smaller Mac (`16gb`, `24gb`, `64gb`).",
        "",
        head.replace("`kapitsa serve --model`", "`--model`") + "\n".join(other_rows),
        "",
        "A download is the weights plus the files that speed decoding up. The drafter column names the draft model "
        "that is downloaded with the weights. The MTP head column names the repo its file comes from, with its pinned "
        "revision; `none` means the weights repo has no head to fetch (the 35B's 4-bit repo ships its own) or the model "
        "decodes without one. The download size counts the weights and every companion.",
        "",
    ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 when docs/models.md differs from the catalog")
    ap.add_argument("--catalog", default=str(CATALOG), help=argparse.SUPPRESS)
    ap.add_argument("--out", default=str(OUT), help=argparse.SUPPRESS)
    a = ap.parse_args()
    text = render(parse_catalog(Path(a.catalog).read_text()))
    out = Path(a.out)
    if a.check:
        have = out.read_text() if out.exists() else ""
        if have == text:
            print(f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} matches the catalog")
            return 0
        sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), text.splitlines(True), "docs/models.md",
                                                   "from catalog.cpp"))
        print("\ndocs/models.md differs from src/hub/catalog.cpp: run `python3 tools/models_table.py` and commit it",
              file=sys.stderr)
        return 1
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
