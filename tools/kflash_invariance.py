#!/usr/bin/env python3
"""A tree's size and shape must not change a greedy reply.

Runs build/ni_run on one prompt with KFlash's tree forced to different sizes (NI_KFLASH_FORCE: n rows every cycle,
-1 every count in turn, cycle by cycle) and compares the generated ids. Every run decodes the same path of tokens
through trees of other sizes and shapes (the path's rows sit in other slots, jump or follow in sequence, are joined by
other rows), so the ids must be equal; at the first token that differs, a row's logits depended on the tree.

  tools/kflash_invariance.py <model dir> [prompt ids file, default bench/prompts/code_35b.txt] [n_gen=192]
      [--sizes 1,3,5,8,-1,auto] [--env NAME=VALUE ...] [--bin build/ni_run]

Every pair of runs is compared (not just each against the first: the first size is no more right than any other), and the
runs are printed as classes of equal replies. "auto" is the default budget (no forcing: the tree the adaptive budget
picks cycle by cycle, which varies with its learned state). Exit status 1 if any two runs differ.

The switches that undo the invariance fixes of the 35B's tree verify (pass them with --env to bisect a regression;
each is on by default): NI_V8_TREE_ATTN=1 / 0 (the older attention variants: the block as one more chunk after
history chunks cut by the anchor / its keys inside the MMA tiles; 2 = the default, tree_zone),
NI_V8_TREE_PIN=0 (the GDN step's update back to the compiler's contraction), NI_DFLASH_EXACT_MOE=0 (the experts'
numerics follow how many rows share an expert).
"""
import argparse
import os
import subprocess
import sys


def run(binary, model, prompt, n_gen, force, extra):
    env = dict(os.environ)
    env.pop("NI_KFLASH_FORCE", None)
    if str(force) != "auto":
        env["NI_KFLASH_FORCE"] = str(force)
    env.update(extra)
    cmd = [binary, model, "@" + prompt, str(n_gen), "0", "--dflash", "auto"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        sys.exit("ni_run failed (force %s)" % force)
    for line in r.stdout.splitlines():
        if line.startswith("ids:"):
            return [int(x) for x in line[4:].split()]
    sys.exit("no ids line in ni_run's output (force %s)" % force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt", nargs="?", default="bench/prompts/code_35b.txt")
    ap.add_argument("n_gen", nargs="?", type=int, default=192)
    ap.add_argument("--sizes", default="1,3,5,8,-1,auto")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--bin", default="build/ni_run")
    a = ap.parse_args()
    extra = dict(kv.split("=", 1) for kv in a.env)
    sizes = [x if x == "auto" else int(x) for x in a.sizes.split(",")]
    outs = {}
    for n in sizes:
        outs[n] = run(a.bin, a.model, a.prompt, a.n_gen, n, extra)
        print("force %4s: %d tokens" % (n, len(outs[n])), flush=True)

    def first_diff(x, y):
        k = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
        if k is None and len(x) == len(y):
            return None
        return min(len(x), len(y)) if k is None else k

    classes = []  # lists of sizes with equal replies
    for n in sizes:
        for c in classes:
            if first_diff(outs[c[0]], outs[n]) is None:
                c.append(n)
                break
        else:
            classes.append([n])
    bad = 0
    for i, x in enumerate(sizes):
        for y in sizes[i + 1:]:
            k = first_diff(outs[x], outs[y])
            if k is not None:
                bad += 1
                ox, oy = outs[x], outs[y]
                print("force %4s != force %4s: first difference at generated token %d (%s vs %s)" %
                      (x, y, k, ox[k] if k < len(ox) else "end", oy[k] if k < len(oy) else "end"))
    print("classes: " + " ".join("[" + ",".join(str(n) for n in c) + "]" for c in classes))
    print("%d of %d pairs differ" % (bad, len(sizes) * (len(sizes) - 1) // 2))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
