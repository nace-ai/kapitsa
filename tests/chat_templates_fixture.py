#!/usr/bin/env python3
"""Regenerate tests/data/chat_templates.json: every chat template kapitsa can load, as its file writes it.

The fixture is what api_test's `chat_template_source` sweep reads: each catalog variant's template (src/hub/catalog.cpp,
at the variant's pinned revision) and each template in the model folders given on the command line. api_test fails when
a catalog variant has no entry here, so a model added to the catalog cannot skip the sweep: rerun this script.

  python3 tests/chat_templates_fixture.py [--extra REPO@REVISION]... [MODEL_DIR...]

A catalog or --extra entry is fetched from Hugging Face (stdlib only; HF_TOKEN or ~/.cache/huggingface/token for a
gated repo): chat_template.jinja, else tokenizer_config.json's chat_template, as chat_template_source reads a folder;
for a GGUF variant, the file's tokenizer.chat_template, read from its header by range requests (the 10 GB file is not
downloaded), which is what `kapitsa import` writes as the folder's chat_template.jinja. A model folder is read from
disk, keyed by this Mac's host name. Entries already in the fixture are kept, so folders on two Macs are added by
running the script on each.
"""

import argparse
import hashlib
import json
import os
import re
import socket
import struct
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "data", "chat_templates.json")
CATALOG = os.path.join(ROOT, "src", "hub", "catalog.cpp")


def token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else ""


def get(repo, rev, path, rng=None):
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{rev}/{path}")
    if token():
        req.add_header("Authorization", f"Bearer {token()}")
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1] - 1}")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise SystemExit(f"{repo}@{rev}/{path}: HTTP {e.code}")


def folder_template(read):
    # chat_template_source (src/engine/engine.cpp): chat_template.jinja, else tokenizer_config.json's chat_template
    # when it is a string, else nothing
    jinja = read("chat_template.jinja")
    if jinja:
        return jinja.decode("utf-8")
    cfg = read("tokenizer_config.json")
    t = json.loads(cfg).get("chat_template") if cfg else None
    return t if isinstance(t, str) else ""


class GgufHeader:
    """Reads a remote GGUF file's metadata by range requests, as far as the key wanted."""

    def __init__(self, repo, rev, path):
        self.repo, self.rev, self.path = repo, rev, path
        self.buf, self.pos = b"", 0

    def need(self, n):
        while self.pos + n > len(self.buf):
            step = max(8 << 20, n)
            got = get(self.repo, self.rev, self.path, (len(self.buf), len(self.buf) + step))
            if not got:
                raise SystemExit(f"{self.repo}/{self.path}: short read at {len(self.buf)}")
            self.buf += got

    def take(self, fmt):
        n = struct.calcsize(fmt)
        self.need(n)
        v = struct.unpack_from("<" + fmt, self.buf, self.pos)[0]
        self.pos += n
        return v

    def string(self):
        n = self.take("Q")
        self.need(n)
        s = self.buf[self.pos : self.pos + n]
        self.pos += n
        return s

    SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("I"), self.take("Q")
            for _ in range(n):
                self.value(et)
            return None
        return self.take(self.SCALAR[t])

    def chat_template(self):
        if self.take("I") != 0x46554747:
            raise SystemExit(f"{self.repo}/{self.path}: not a GGUF file")
        self.take("I")  # version
        self.take("Q")  # tensors
        for _ in range(self.take("Q")):
            key = self.string().decode()
            t = self.take("I")
            v = self.value(t)
            if key == "tokenizer.chat_template":
                return v.decode("utf-8")
        return ""


def catalog_variants():
    # (repo, revision, gguf path or "") of each variant: the `.repo`/`.revision`/`.gguf = {.path` fields in order.
    # api_test checks the table itself against the fixture, so a variant this misses fails there.
    src = open(CATALOG).read()
    out = []
    for m in re.finditer(r'\{\.name = "[^"]*",(.*?)\.download_gb', src, re.S):
        body = m.group(1)
        repo = re.search(r'\.repo = "([^"]+)"', body)
        rev = re.search(r'\.revision = "([^"]+)"', body)
        gguf = re.search(r'\.gguf = \{\.path = "([^"]+)"', body)
        if repo and rev:
            out.append((repo.group(1), rev.group(1), gguf.group(1) if gguf else ""))
    return out


def hf_key(repo, rev, gguf):
    return f"hf:{repo}@{rev}" + (f"/{gguf}" if gguf else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extra", action="append", default=[], help="REPO@REVISION, a model not in the catalog")
    ap.add_argument("dirs", nargs="*", help="model folders on this Mac")
    a = ap.parse_args()

    fx = json.load(open(FIXTURE)) if os.path.exists(FIXTURE) else {"sources": {}, "templates": {}}

    def add(key, text):
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        fx["templates"][h] = text
        fx["sources"][key] = h
        print(f"{h[:12]} {len(text):6d} {key}", file=sys.stderr)

    remote = [(r, v, g) for r, v, g in catalog_variants()]
    remote += [(e.split("@")[0], e.split("@")[1], "") for e in a.extra]
    for repo, rev, gguf in remote:
        key = hf_key(repo, rev, gguf)
        if key in fx["sources"]:
            continue
        if gguf:
            add(key, GgufHeader(repo, rev, gguf).chat_template())
        else:
            add(key, folder_template(lambda p, repo=repo, rev=rev: get(repo, rev, p)))
    host = socket.gethostname().split(".")[0]
    for d in a.dirs:

        def read(p, d=d):
            f = os.path.join(d, p)
            return open(f, "rb").read() if os.path.exists(f) else None

        add(f"dir:{host}/{os.path.basename(os.path.normpath(d))}", folder_template(read))

    used = set(fx["sources"].values())
    fx["templates"] = {h: t for h, t in sorted(fx["templates"].items()) if h in used}
    fx["sources"] = dict(sorted(fx["sources"].items()))
    with open(FIXTURE, "w") as f:
        json.dump(fx, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(fx['sources'])} sources, {len(fx['templates'])} distinct templates -> {FIXTURE}", file=sys.stderr)


if __name__ == "__main__":
    main()
