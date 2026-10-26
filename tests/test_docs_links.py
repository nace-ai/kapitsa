#!/usr/bin/env python3
"""The docs' relative links and anchors resolve. CPU only, stdlib only, no network.

Why this exists: docs/benchmarks.md was rewritten in #580 from 2,391 to 417 lines, one section per Mac, and the old
section names survived only because that commit pinned 26 of them as explicit `<a id="m5-max"></a>` anchors so the
README's and docs/kflash.md's links kept resolving. Nothing tested those anchors, so the next section rename would
delete them silently. This runs in its own `docs-links` job rather than inside `bench-readme`: a dead link and a stale
bench table are different failures, and a red X on a required bench check should mean the table.

It checks what GitHub resolves and nothing else: `](#x)` in the same file and `](path#x)` across files, against the
anchors a file defines (its headings, slurgged GitHub's way, plus its `<a id>`/`<a name>` anchors). External URLs,
mail links, fenced code blocks, inline code spans and image links are not links for this purpose. Files are listed
explicitly so a new page has to be added here on purpose, and the scan never leaves the repository.

A stale anchor fails with the shape that makes it fixable: `docs/usage.md:1103 -> #some-anchor (no such heading)`.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Explicit, not globbed: a new page joins the check on purpose.
PAGES = [
    "README.md",
    "CONTRIBUTING.md",
    "docs/architecture.md",
    "docs/benchmarks-m5-max.md",
    "docs/benchmarks.md",
    "docs/kflash-map.md",
    "docs/kflash.md",
    "docs/numerical-history.md",
    "docs/systemone.md",
    "docs/usage.md",
]

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
HTML_ANCHOR = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")
INLINE_CODE = re.compile(r"`[^`\n]*`")
# Not image links: readme_images.py owns those. Optional link title allowed.
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
EXTERNAL = ("http://", "https://", "mailto:", "tel:")


def slug(text):
    """GitHub's anchor for a heading: drop inline code and tags, lowercase, keep letters/digits/`-`/`_`, spaces to `-`."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text).replace("`", "")
    kept = "".join(ch for ch in text.lower() if ch.isalnum() or ch in " -_")
    return kept.replace(" ", "-")


def page_anchors(text):
    """Every anchor the page defines, with GitHub's -1/-2 suffixes for a heading that repeats."""
    seen, anchors, in_fence = {}, set(), False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for a in HTML_ANCHOR.finditer(line):
            anchors.add(a.group(1))
        m = HEADING.match(line)
        if m:
            base = slug(m.group(2))
            n = seen.get(base, 0)
            seen[base] = n + 1
            anchors.add(base if n == 0 else "%s-%d" % (base, n))
    return anchors


def page_links(text):
    """(lineno, target) for each relative link, skipping fenced blocks and inline code spans."""
    out, in_fence = [], False
    for lineno, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK.finditer(INLINE_CODE.sub("", line)):
            target = m.group(1)
            if not target.startswith(EXTERNAL):
                out.append((lineno, target))
    return out


def check(pages, exists=None):
    """Resolve every link in `pages` ({repo-relative path: text}); return [(page, line, target, why)]."""
    exists = exists or (lambda rel: os.path.exists(os.path.join(ROOT, rel)))
    anchors = {rel: page_anchors(text) for rel, text in pages.items()}
    problems, cache = [], {}
    for rel, text in pages.items():
        for lineno, target in page_links(text):
            path, _, frag = target.partition("#")
            if path:
                # ../ resolves from this file's directory, then the path has to stay inside the repository.
                dest = os.path.normpath(os.path.join(os.path.dirname(rel), path)).replace(os.sep, "/")
                if dest == ".." or dest.startswith("../"):
                    problems.append((rel, lineno, target, "escapes the repository"))
                    continue
                if not exists(dest):
                    problems.append((rel, lineno, target, "no such file"))
                    continue
                if not frag or os.path.isdir(os.path.join(ROOT, dest)):
                    continue
                if dest in pages:
                    defined = anchors[dest]
                elif dest.endswith(".md"):
                    if dest not in cache:
                        with open(os.path.join(ROOT, dest), encoding="utf-8", errors="ignore") as fh:
                            cache[dest] = page_anchors(fh.read())
                    defined = cache[dest]
                else:
                    continue
            else:
                defined = anchors[rel]
                if not frag:
                    continue
            if frag not in defined:
                problems.append((rel, lineno, target, "no such heading"))
    return problems


def selftest():
    """The scanner can fail, and only where it should: #580's HTML anchors count, fenced and inline code do not."""
    assert slug("MacBook Pro M5 Max, 128 GB") == "macbook-pro-m5-max-128-gb"
    assert slug("`brew services` and <b>HTML</b> (x): Caf\u00e9") == "brew-services-and-html-x-caf\u00e9"
    assert page_anchors("# A\n# A\n") == {"a", "a-1"}, page_anchors("# A\n# A\n")

    pages = {
        "a.md": ("# One\n"
                 "[x](#one) [y](#two) [z](b.md#sub) [w](b.md#nope) [v](missing.md)\n"
                 "`[inline](#nope)`\n"
                 "```\n[fenced](#nope)\n```\n"),
        "b.md": '# Bee\n<a id="sub"></a>\n',
    }
    want = [("a.md", 2, "#two", "no such heading"),
            ("a.md", 2, "b.md#nope", "no such heading"),
            ("a.md", 2, "missing.md", "no such file")]
    got = check(pages, exists=lambda rel: rel != "missing.md")
    assert got == want, got
    assert check({"a.md": "[up](../../etc/passwd)"}) == [("a.md", 1, "../../etc/passwd", "escapes the repository")]

    # An `<a id>` inside a fenced example is not an anchor GitHub renders, so a link to it is dead.
    assert slug("[Foo](usage.md)") == "foo", slug("[Foo](usage.md)")
    assert page_anchors("## [Foo](usage.md)\n") == {"foo"}, page_anchors("## [Foo](usage.md)\n")
    assert check({"a.md": "# T\n[x](#ghost)\n```html\n<a id=\"ghost\"></a>\n```\n"}) == [
        ("a.md", 2, "#ghost", "no such heading")]
    # A heading that is itself a link slugs to its text, so #foo resolves and #foousagemd does not.
    assert check({"a.md": "## [Foo](usage.md)\n[x](#foo) [y](#foousagemd)\n"},
                 exists=lambda rel: rel == "usage.md") == [("a.md", 2, "#foousagemd", "no such heading")]


def main():
    selftest()
    pages = {}
    for rel in PAGES:
        with open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore") as fh:
            pages[rel] = fh.read()
    links = sum(len(page_links(text)) for text in pages.values())
    problems = check(pages)
    print("test_docs_links: %d files, %d relative links, %d unresolved" % (len(pages), links, len(problems)))
    for rel, lineno, target, why in problems:
        print("  %s:%d -> %s (%s)" % (rel, lineno, target, why))
    if problems:
        print("Add or restore the anchor, or point the link at the heading that exists now.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
