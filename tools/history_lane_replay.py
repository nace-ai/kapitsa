#!/usr/bin/env python3
"""Would a history (n-gram / suffix) candidate lane have beaten the logged drafter cycles? CPU only, stdlib only.

A rowlog (one JSON line per verify cycle, `acc` = accepted drafts) plus its `.drive` file (one JSON line per request,
with `cycles`, `completion` and the output token ids `ids`) and, when it exists, the request's prompt ids, let a cycle
be replayed exactly at the token level: cycle c starts after p emitted tokens, the drafter's accepted drafts were
out[p : p+acc], and the cycle emitted acc+1 tokens (see engine.cpp:3677-3691). At that point a lookup of the last n
tokens of (prompt + out[:p]) in the same text finds an earlier occurrence and proposes the up to K tokens that followed
it (the algorithm of `ngram_drafts`, src/engine/engine.cpp:88: n from nmax down to nmin, most recent occurrence).
m = how many of those proposals equal out[p:], which is what the target would have accepted.

Per cycle it reports, against the logged accepted count a:
  oracle   max(a, m)     the history branch rides in the tree for free and the verify picks the longer path
                         (an upper bound: ignores the row cost of the extra branch)
  replace  m if hit else a   a policy with no information about which side is right (the Flint lesson: a lookup that
                         replaces the drafter loses whenever m < a)
  branch   the oracle with its price: a hit pays min(len(proposals), --max-rows) extra verify rows at --row-ms
           each (on top of the logged rows, so the 8-row cap is ignored) and can win at most that many tokens;
           reported as tokens per ms of (base + row-ms * rows), relative to the logged cycles
  swap     row-neutral: when a lookup hits and the drafter's expected accepted drafts (sum of running products of
           conf * --conf-scale) is below --swap-t, the history chain takes the drafter's rows. Costs nothing (same
           rows), loses when the chain is wrong; --swap-t is swept in sample, so read it as an upper bound
  rescue   at the logged cycle's first rejected draft (acc < K): does an n >= nmin lookup of the text so far
           predict the true token there? A history lane that only re-ranks the drafter's top-16 can rescue
           at most this share of the misses

Caveats, stated because they decide how to read the numbers: only text inside one request is searched (no other
request's output: the benchmark repeats prompts, so a cross-request corpus would find its own answer); the cycle
boundaries are those the drafter produced, not those a history lane would have produced; greedy (T0) logs. Request
boundaries come from the .drive's `cycles`; a request whose rowlog does not sum to its `completion` (0 to 16 tokens too many: the
final cycle overshoots, and the double-buffered log can attribute one boundary cycle to the next request) is skipped and counted.

usage: history_lane_replay.py ROWLOG [ROWLOG ...] [--ids-dir DIR] [--nmax 8] [--nmin 4] [--k 7]
       (ROWLOG's .drive sits beside it; --ids-dir holds <prompt>.ids files, comma-separated token ids)
"""
import argparse
import json
import os


def lookup(text, p, nmax, nmin, k):
    """(n, proposals) for the most recent earlier occurrence of text[p-n:p], n from nmax down to nmin; (0, [])."""
    for n in range(min(nmax, p), nmin - 1, -1):
        key = text[p - n:p]
        for s in range(p - n - 1, -1, -1):
            if text[s:s + n] == key:
                return n, text[s + n:s + n + k]
    return 0, []


def read_ids(path):
    with open(path) as f:
        return [int(x) for x in f.read().replace("\n", ",").split(",") if x.strip()]


def requests(rowlog_path, ids_dir):
    """Yield (name, prompt_ids, out_ids, [acc per cycle]) for the requests whose rowlog blocks add up."""
    drive = rowlog_path[:-len(".rowlog")] + ".drive"
    cyc = [json.loads(l) for l in open(rowlog_path) if l.strip()]
    cyc = [c for c in cyc]
    reqs = [json.loads(l) for l in open(drive) if l.strip()]
    i, skipped = 0, 0
    out = []
    for d in reqs:
        blk = cyc[i:i + d["cycles"]]
        i += d["cycles"]
        if d["prompt"].startswith("warm"):
            continue
        adv = sum(c["acc"] + 1 for c in blk)
        if not 0 <= adv - (d["completion"] - 1) <= 16 or len(blk) != d["cycles"]:
            skipped += 1
            continue
        prompt = []
        if ids_dir and os.path.exists(os.path.join(ids_dir, d["prompt"])):
            prompt = read_ids(os.path.join(ids_dir, d["prompt"]))
        out.append((d["prompt"], prompt, d["ids"], [c["acc"] for c in blk], [c["rows"] for c in blk], [c.get("conf") for c in blk]))
    return out, skipped


KEYS = ("cycles", "logged", "oracle", "replace", "hits", "hit_better", "ms_logged", "ms_branch", "miss",
        "rescued", "branch", "swap")


def replay(prompt, ids, accs, nmax, nmin, k, base_ms=26.8, row_ms=2.57, max_rows=7, rows=None, confs=None, swap_t=0.0, conf_scale=0.7):
    """Per-cycle sums: (cycles, logged tokens, oracle, replace, hits ...)."""
    text = prompt + ids
    p = len(prompt) + 1  # the first output token comes from the prefill; cycle one's anchor is ids[0]
    tot = dict.fromkeys(KEYS, 0)
    for a in accs:
        o = p - len(prompt)
        if o >= len(ids):
            break
        n, prop = lookup(text, p, nmax, nmin, k)
        m = 0
        while m < len(prop) and p + m < len(text) and prop[m] == text[p + m]:
            m += 1
        a = min(a, len(ids) - o - 1) if o + a + 1 > len(ids) else a
        tot["cycles"] += 1
        tot["logged"] += a + 1
        tot["oracle"] += max(a, m) + 1
        tot["replace"] += (m if n else a) + 1
        r = rows[tot["cycles"] - 1] if rows else 8
        tot["branch"] += max(a, min(m, max_rows) if n else 0) + 1
        tot["ms_logged"] += base_ms + row_ms * r
        tot["ms_branch"] += base_ms + row_ms * (r + (min(len(prop), max_rows) if n else 0))
        if a < k and p + a < len(text):
            tot["miss"] += 1
            nn, pr2 = lookup(text, p + a, nmax, nmin, 1)
            tot["rescued"] += 1 if nn and pr2 and pr2[0] == text[p + a] else 0
        if n and confs and confs[tot["cycles"] - 1]:
            e_d, run = 0.0, 1.0
            for cf in confs[tot["cycles"] - 1]:
                run *= cf * conf_scale
                e_d += run
            tot["swap"] += (min(m, max(r - 1, 0)) if e_d < swap_t else a) + 1
        else:
            tot["swap"] += a + 1
        tot["hits"] += 1 if n else 0
        tot["hit_better"] += 1 if n and m > a else 0
        p += a + 1
    return tot


def ledger(path, nmax, nmin, k):
    """Exploratory, tokenizer-free: replay history-only drafting on a recorded agent ledger (bench/harness/agent-traces).
    Text is split into words and punctuation (a proxy: not the model's tokens), each request's completion is replayed
    against its own context. Returns (completion tokens, history-only cycles, cycles with a lookup hit, tokens accepted
    on hits); history-only tokens per cycle = tokens / cycles."""
    import re
    d = json.load(open(path))
    split = lambda t: re.findall(r"\w+|[^\w\s]", t)
    text_of = lambda m: (m.get("content") if isinstance(m.get("content"), str) else "") + " " + json.dumps(m.get("tool_calls", []))
    tot = cyc = hits = acc = 0
    for r in d["requests"]:
        ctx = [t for i in r["messages"] for t in split(text_of(d["messages"][i]))]
        resp = r["response"]
        comp = split((resp.get("content") or "") + " " + json.dumps(resp.get("tool_calls", [])))
        text, p = ctx + comp, len(ctx)
        tot += len(comp)
        while p < len(text):
            n, prop = lookup(text, p, nmax, nmin, k)
            m = 0
            while m < len(prop) and p + m < len(text) and prop[m] == text[p + m]:
                m += 1
            cyc, hits, acc = cyc + 1, hits + (1 if n else 0), acc + (m if n else 0)
            p += m + 1
    return tot, cyc, hits, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rowlogs", nargs="*")
    ap.add_argument("--ledger", help="an agent ledger JSON: the tokenizer-free history-only replay, then exit")
    ap.add_argument("--ids-dir")
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--base-ms", type=float, default=26.8)
    ap.add_argument("--row-ms", type=float, default=2.57)
    ap.add_argument("--max-rows", type=int, default=7)
    ap.add_argument("--swap-t", type=float, default=1.0)
    ap.add_argument("--conf-scale", type=float, default=0.7)
    args = ap.parse_args()
    if args.ledger:
        tot, cyc, hits, acc = ledger(args.ledger, args.nmax, args.nmin, args.k)
        print(f"completion words {tot}; history-only cycles {cyc} ({tot / max(cyc, 1):.3f} tokens/cycle); "
              f"lookup hits {hits} ({100 * hits / max(cyc, 1):.1f}%); mean accepted on a hit {acc / max(hits, 1):.2f}")
        return
    keys = KEYS
    grand = dict.fromkeys(keys, 0)
    by_prompt = {}
    skipped = 0
    for path in args.rowlogs:
        reqs, sk = requests(path, args.ids_dir)
        skipped += sk
        for name, prompt, ids, accs, rows, confs in reqs:
            t = replay(prompt, ids, accs, args.nmax, args.nmin, args.k, args.base_ms, args.row_ms,
                       args.max_rows, rows, confs, args.swap_t, args.conf_scale)
            kind = name.split("_r")[0].rstrip("0123456789.ids")
            for tgt in (grand, by_prompt.setdefault(kind, dict.fromkeys(keys, 0))):
                for key in keys:
                    tgt[key] += t[key]
    print(f"requests skipped (rowlog does not add up): {skipped}")
    print(f"{'prompt':8} {'cycles':>6} {'tok/cyc':>7} {'oracle':>7} {'replace':>7} {'hit%':>5} "
          f"{'hit>drf%':>8} {'branch tok/ms vs logged':>24} {'swap tok/cyc':>12} {'miss rescued%':>14}")
    for name, t in [*sorted(by_prompt.items()), ("ALL", grand)]:
        c = max(t["cycles"], 1)
        net = (t["branch"] / t["ms_branch"]) / (t["logged"] / t["ms_logged"]) if t["ms_branch"] else 0.0
        print(f"{name:8} {t['cycles']:6d} {t['logged'] / c:7.3f} {t['oracle'] / c:7.3f} {t['replace'] / c:7.3f} "
              f"{100 * t['hits'] / c:5.1f} {100 * t['hit_better'] / c:8.1f} "
              f"{net:24.4f} {t['swap'] / c:12.3f} {100 * t['rescued'] / max(t['miss'], 1):14.1f}")


if __name__ == "__main__":
    main()
