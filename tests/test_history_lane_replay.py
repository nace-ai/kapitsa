#!/usr/bin/env python3
"""tools/history_lane_replay.py on a synthetic request: the lookup finds the most recent earlier occurrence at the
longest n, the proposals are scored against the real continuation, the oracle never loses to the logged drafter, a
request whose rowlog does not add up is skipped. CPU only, stdlib only, no model, no GPU, no recorded traces.

usage: test_history_lane_replay.py
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOL)
import history_lane_replay as h  # noqa: E402

FAILED = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


# lookup: the most recent earlier occurrence, longest n first
text = [1, 2, 3, 4, 9, 9, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4]
n, prop = h.lookup(text, len(text), 8, 2, 3)
check(n == 4 and prop == [5, 6, 7], "lookup: longest n, most recent occurrence", f"{n} {prop}")
check(h.lookup([1, 2, 3], 3, 8, 2, 3) == (0, []), "lookup: nothing earlier")
n2, _ = h.lookup([5, 6, 7, 8, 0, 5, 6], 7, 4, 2, 3)
check(n2 == 2, "lookup: falls back to a shorter n", n2)

# replay: out repeats a block; the logged drafter accepted nothing each cycle (acc 0 -> 1 token per cycle)
out = [7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12]
t = h.replay([], out, [0] * 17, nmax=4, nmin=2, k=7)
check(t["logged"] == t["cycles"], "logged tokens = cycles when the drafter accepts nothing", t)
check(t["oracle"] > t["logged"], "oracle gains when the text repeats", t)
check(t["oracle"] >= t["replace"] or True, "replace may lose")
check(t["oracle"] >= t["logged"] and t["oracle"] >= t["replace"], "oracle bounds replace")

# oracle never loses to the logged drafter, even when history mispredicts and the drafter is right
t2 = h.replay([], [1, 2, 3, 4, 1, 2, 3, 9, 9, 9], [3, 3, 3], nmax=4, nmin=2, k=7)
check(t2["oracle"] >= t2["logged"], "oracle >= logged", t2)

# the CLI: a rowlog + drive pair that adds up is replayed; one that does not is skipped
with tempfile.TemporaryDirectory() as d:
    ids = out
    cyc = [{"acc": 0, "rows": 4}] * 17
    with open(os.path.join(d, "a.rowlog"), "w") as f:
        f.write("\n".join(json.dumps(c) for c in cyc) + "\n")
    with open(os.path.join(d, "a.drive"), "w") as f:
        f.write(json.dumps({"prompt": "warm.ids", "cycles": 0, "completion": 1, "ids": []}) + "\n")
        f.write(json.dumps({"prompt": "code_r0.ids", "cycles": 17, "completion": 18, "ids": ids}) + "\n")
    with open(os.path.join(d, "b.rowlog"), "w") as f:
        f.write(json.dumps({"acc": 0, "rows": 4}) + "\n")
    with open(os.path.join(d, "b.drive"), "w") as f:
        f.write(json.dumps({"prompt": "code_r0.ids", "cycles": 1, "completion": 200, "ids": ids}) + "\n")
    r = subprocess.run([sys.executable, os.path.join(TOOL, "history_lane_replay.py"),
                        os.path.join(d, "a.rowlog"), os.path.join(d, "b.rowlog"), "--nmin", "2", "--nmax", "4"],
                       capture_output=True, text=True)
    check(r.returncode == 0, "CLI runs", r.stderr[-300:])
    check("skipped (rowlog does not add up): 1" in r.stdout, "CLI skips the request that does not add up", r.stdout)
    check("ALL" in r.stdout and "code" in r.stdout, "CLI prints the table", r.stdout)

# ledger mode: a request whose completion repeats its context is drafted from history
with tempfile.TemporaryDirectory() as d:
    led = {"messages": [{"role": "user", "content": "alpha beta gamma delta epsilon zeta eta theta iota kappa"}],
           "requests": [{"messages": [0], "response": {"content": "alpha beta gamma delta epsilon zeta eta theta",
                                                       "tool_calls": []}}]}
    lp = os.path.join(d, "ledger.json")
    json.dump(led, open(lp, "w"))
    tot, cyc, hits, acc = h.ledger(lp, 8, 4, 7)
    check(tot > 8 and cyc < tot and hits >= 1 and acc >= 4, "ledger: copied completion is drafted from history",
          (tot, cyc, hits, acc))

if FAILED:
    print(f"{len(FAILED)} FAILED")
    sys.exit(1)
print("all passed")
