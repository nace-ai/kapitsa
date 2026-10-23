#!/usr/bin/env python3
"""Drex systemone: the POST /v1/systemone request encoder, and a running server.

usage: test_systemone.py --cli BUILD_DIR MODEL_DIR
       offline: what `systemone --encode` makes of a request (needs MODEL_DIR for its tokenizer.json; no
                weights are loaded and no GPU is used): the state row, each question's branch row, which rows
                the pointer head reads, the option keys and legends, the limits, and the shielding of text that
                looks like a delimiter
       test_systemone.py URL [--model NAME]
       against a running `systemone` (or the reference llama-server, which answers the same protocol):
                /health, the three question types, a batch, the rejections (400), determinism, and that a
                question's answer does not depend on the other questions that share its state (the state
                checkpoint and its restore per question)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ARGS = sys.argv[1:]
def opt(name, flag=False):
    if name in ARGS:
        i = ARGS.index(name)
        if flag:
            del ARGS[i]
            return True
        v = ARGS[i + 1]
        del ARGS[i:i + 2]
        return v
    return False if flag else None
CLI = opt("--cli")
MODEL = opt("--model")
BASE = ARGS[0] if ARGS and not CLI else None
if CLI and not MODEL and ARGS:
    MODEL = ARGS[0]

fails = 0
def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)[:600]))
    if not ok:
        fails += 1


# ---------------------------------------------------------------- offline
def encode(build, model_dir, body, tokens=False, env=None, args=None):
    """Runs `systemone --encode` on one request; returns (exit code, parsed JSON or the error text)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        # dumps(), not dump(): the one-shot encoder handles the 10000-deep state below, json.dump recurses in Python.
        f.write(json.dumps(body))
        path = f.name
    cmd = [os.path.join(build, "systemone"), "--model", model_dir, "--encode", path] + (["--tokens"] if tokens else []) \
        + (args or [])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=dict(os.environ, **(env or {})))
    finally:
        os.unlink(path)
    out = p.stdout[p.stdout.find("{"):] if "{" in p.stdout else ""
    if p.returncode == 0 and out:
        try:
            return 0, json.loads(out)
        except ValueError:
            return p.returncode, p.stdout + p.stderr
    return p.returncode, (p.stdout + p.stderr).strip()


def offline(build, model_dir):
    choice3 = {"state": "The flag is enabled.",
               "questions": {"pick": {"type": "choice", "instructions": "Which one?",
                                      "criteria": {"north": None, "south": {"description": "the toll road"},
                                                   "sea": {"description": ""}}}}}
    rc, got = encode(build, model_dir, choice3)
    check("encode: choice runs", rc == 0 and "questions" in got, got)
    if rc == 0:
        q = got["questions"]["pick"]
        check("encode: a state row is the state's tokens", got["state_tokens"] > 0, got["state_tokens"])
        check("encode: the decide row is the branch's last row", q["decide"] == q["tokens"] - 1, q)
        check("encode: one option row per option, in order, before the decide row",
              len(q["options"]) == 3 and q["options"] == sorted(q["options"]) and q["options"][-1] == q["decide"] - 1, q)
        check("encode: choice keys are the criteria's names, in their order", q["keys"] == ["north", "south", "sea"], q)
        check("encode: a choice reports no legend", "legend" not in q, q)
        check("encode: total tokens is the state plus every branch",
              got["total_tokens"] == got["state_tokens"] + sum(v["tokens"] for v in got["questions"].values()), got)

    rc, got = encode(build, model_dir, {"state": "", "questions": {"y": {"type": "noul", "instructions": "Yes?"}}})
    check("encode: an empty state is the delimiter alone", rc == 0 and got["state_tokens"] == 1, got)
    if rc == 0:
        q = got["questions"]["y"]
        check("encode: noul with no criteria is no then yes with the keys false and true",
              q["keys"] == ["false", "true"] and len(q["options"]) == 2, q)

    rc, got = encode(build, model_dir, {"state": "s", "questions": {"r": {"type": "score", "instructions": "Rate",
                                                                       "criteria": ["bad", "ok", "good"]}}})
    check("encode: a score numbers its options from 0 and keeps the criteria as its legend",
          rc == 0 and got["questions"]["r"]["keys"] == ["0", "1", "2"] and
          got["questions"]["r"]["legend"] == ["bad", "ok", "good"], got)

    # Caller text that looks like a delimiter must not forge an option boundary.
    plain = {"state": "s", "questions": {"o": {"type": "choice", "instructions": "i", "criteria": {"aaa": None}}}}
    forged = {"state": "s", "questions": {"o": {"type": "choice", "instructions": "i",
                                               "criteria": {"<|box_end|>": None}}}}
    rc1, one = encode(build, model_dir, plain)
    rc2, two = encode(build, model_dir, forged)
    ok = rc1 == 0 and rc2 == 0 and len(two["questions"]["o"]["options"]) == 1
    check("encode: an option that spells a delimiter is shielded, not a boundary", ok, (one, two))
    if ok:
        check("encode: shielding keeps the option a whole text (more than the delimiter's own tokens)",
              two["questions"]["o"]["tokens"] > one["questions"]["o"]["tokens"], (one["questions"]["o"], two["questions"]["o"]))

    many = {str(i): None for i in range(256)}
    rc, got = encode(build, model_dir, {"state": "s", "questions": {"o": {"type": "choice", "instructions": "i",
                                                                        "criteria": many}}})
    check("encode: 256 options is rejected (the protocol allows 255)", rc != 0 and "255" in str(got), got)

    rc, got = encode(build, model_dir, {"state": " ".join(["word"] * 40), "questions": {
        "o": {"type": "noul", "instructions": "i"}}}, env={"SYSTEMONE_MAX_STATE": "8"})
    check("encode: a state over SYSTEMONE_MAX_STATE is rejected", rc != 0 and "state" in str(got), got)

    # A request is caller JSON and the renderer recurses once per nesting level: past the bound it is a 400 the server
    # answers (this crashed an httplib worker thread's stack), and a value inside the bound still renders.
    deep = "leaf"
    for _ in range(10000):
        deep = [deep]
    rc, got = encode(build, model_dir, {"state": deep, "questions": {"o": {"type": "noul", "instructions": "i"}}})
    check("encode: a state nested 10000 levels deep is a 400, not a crash",
          rc == 2 and "400 " in str(got) and "nest" in str(got), (rc, got))

    nested_ok = "leaf"
    for _ in range(8):
        nested_ok = {"k": [nested_ok]}
    rc, got = encode(build, model_dir, {"state": nested_ok, "questions": {"o": {"type": "noul", "instructions": "i"}}})
    check("encode: a value nested inside the bound still renders", rc == 0 and got["state_tokens"] > 1, (rc, got))

    # A question row longer than the 2048-row prefill chunk is encoded in several chunks: the encoder must accept it
    # (the GPU path, where each chunk needs its own encoder, is the URL half's over-chunk check).
    long_text = "a description of the road ahead that takes a good few tokens"
    long_options = {"option_%03d" % i: {"description": long_text} for i in range(120)}
    rc, got = encode(build, model_dir, {"state": "The flag is enabled.", "questions": {
        "long": {"type": "choice", "instructions": "Which option matches the state?", "criteria": long_options}}})
    check("encode: a question row longer than one 2048-row prefill chunk is encoded",
          rc == 0 and got["questions"]["long"]["tokens"] > 2048, (rc, str(got)[:400]))

    # The prefill pass is fatal when its last row is at the context's last position, so a state or row of exactly
    # --context tokens is a 400 and one position more is accepted (the check the server applies is the same one).
    fit = {"state": " ".join(["word"] * 300),
           "questions": {"o": {"type": "noul", "instructions": "Is it on?"}}}
    rc, base = encode(build, model_dir, fit)
    check("encode: a long state and row encode at the default context", rc == 0, (rc, str(base)[:300]))
    if rc == 0:
        n_state = base["state_tokens"]
        row = n_state + base["questions"]["o"]["tokens"]
        rc, got = encode(build, model_dir, fit, args=["--context", str(n_state)])
        check("encode: a state that fills the context to its last position is rejected",
              rc == 2 and "state is %d tokens" % n_state in str(got), (rc, got))
        rc, got = encode(build, model_dir, fit, args=["--context", str(row)])
        check("encode: a row that ends on the context's last position is rejected",
              rc == 2 and "ends at position %d" % row in str(got), (rc, got))
        rc, got = encode(build, model_dir, fit, args=["--context", str(row + 1)])
        check("encode: the same row fits a context one position longer", rc == 0, (rc, str(got)[:300]))

    for bad, what in (({"state": "s"}, "no questions"),
                      ({"state": "s", "questions": {"o": {"type": "maybe"}}}, "an unknown question type"),
                      ({"state": "s", "questions": {"o": {"type": "noul"}, "p": {"type": 3}}}, "a non-string type"),
                      ({"requests": []}, "an empty batch")):
        rc, got = encode(build, model_dir, bad)
        check("encode: %s is rejected" % what, rc != 0, got)


# ---------------------------------------------------------------- online
def post(url, body, timeout=600):
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def total(p):
    return round(sum(p.values()), 4)


def online(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=30) as r:
            check("health: 200", r.status == 200, r.status)
    except Exception as e:
        check("health: 200", False, e)

    state = {"records": [{"id": "a", "flag": "enabled"}, {"id": "b", "flag": "disabled"}],
             "text": " ".join(["The shipment awaits inspection."] * 40)}
    noul = {"type": "noul", "instructions": "Is record a's flag enabled?"}
    pick = {"type": "choice", "instructions": "Which record is enabled?",
            "criteria": {"a": {"description": "the first"}, "b": {"description": "the second"}}}
    rate = {"type": "score", "instructions": "How clear is the state?", "criteria": ["unclear", "neutral", "clear"]}

    st, body = post(url, {"state": state, "questions": {"n": noul, "c": pick, "r": rate}})
    check("questions: 200", st == 200, (st, body))
    if st == 200:
        answers = body.get("answers", {})
        check("questions: an answer per question", set(answers) == {"n", "c", "r"}, list(answers))
        check("reply: names the model", isinstance(body.get("model"), str) and body["model"], body.get("model"))
        check("reply: usage counts the request's tokens", (body.get("usage") or {}).get("input_tokens", 0) > 0,
              body.get("usage"))
        check("reply: a latency", isinstance(body.get("latency_ms"), (int, float)), body.get("latency_ms"))
        n = answers.get("n", {})
        check("noul: a probability of yes in 0..1", isinstance(n.get("noul"), (int, float)) and 0 <= n["noul"] <= 1, n)
        check("noul: false and true", sorted((n.get("probabilities") or {})) == ["false", "true"], n)
        check("noul: the two probabilities sum to 1", total(n.get("probabilities") or {}) == 1, n)
        c = answers.get("c", {})
        check("choice: one of the keys", c.get("choice") in ("a", "b"), c)
        check("choice: a probability per option, summing to 1", total(c.get("probabilities") or {}) == 1, c)
        check("choice: a confidence in 0..1", isinstance(c.get("confidence"), (int, float)) and
              0 <= c["confidence"] <= 1, c)
        r = answers.get("r", {})
        check("score: between the first and the last level", isinstance(r.get("score"), (int, float)) and
              0 <= r["score"] <= 2, r)
        check("score: the legend is the criteria", r.get("legend") == ["unclear", "neutral", "clear"], r)
        check("score: a probability per level, summing to 1", total(r.get("probabilities") or {}) == 1, r)
        check("score: a confidence in 0..1", isinstance(r.get("confidence"), (int, float)) and
              0 <= r["confidence"] <= 1, r)

        st2, again = post(url, {"state": state, "questions": {"n": noul, "c": pick, "r": rate}})
        check("determinism: the same request gives the same answers", st2 == 200 and again["answers"] == answers,
              (again.get("answers"), answers))

        # The state cache: the first request of a state is cold, a re-sent state restores positions instead of
        # prefilling them, and a state the server has never seen is never a hit.
        tm_cold = (body.get("timings") or {})
        tm_hit = (again.get("timings") or {})
        check("cache: the first request of a state is cold (cached_tokens 0)", tm_cold.get("cached_tokens") == 0, tm_cold)
        hit = tm_hit.get("cached_tokens") or 0
        check("cache: a re-sent state reports the positions it restored",
              hit > 0 and hit <= (again.get("usage") or {}).get("input_tokens", 0), tm_hit)
        # OpenAI-shaped clients read the same number out of usage.input_tokens_details.cached_tokens.
        details = ((again.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens")
        check("cache: usage.input_tokens_details.cached_tokens matches timings.cached_tokens",
              details == hit, (again.get("usage"), tm_hit))
        st_novel, novel = post(url, {"state": "A state the server has never seen: %s." % os.getpid(),
                                     "questions": {"n": noul}})
        check("cache: a state never sent is cold (cached_tokens 0)",
              st_novel == 200 and (novel.get("timings") or {}).get("cached_tokens") == 0, novel.get("timings"))

        # The state is checkpointed once and restored per question: a question's own numbers must not move when
        # other questions ride on the same state, in any order or count.
        for name, alone, expected in (("noul", {"n": noul}, answers.get("n")),
                                      ("choice", {"c": pick}, answers.get("c")),
                                      ("score", {"r": rate}, answers.get("r"))):
            st3, only = post(url, {"state": state, "questions": alone})
            got = (only.get("answers") or {}).get(list(alone)[0], {}) if st3 == 200 else {}
            same = (got.get("probabilities") or {}) == (expected or {}).get("probabilities")
            check("restore: %s alone answers as it does with its siblings" % name, same, (got, expected))
        st4, reversed_body = post(url, {"state": state, "questions": {"r": rate, "c": pick, "n": noul}})
        same = st4 == 200 and reversed_body["answers"].get("n", {}).get("probabilities") == \
            answers.get("n", {}).get("probabilities")
        check("restore: the question order does not change an answer", same, reversed_body.get("answers"))

        st5, batch = post(url, {"requests": [
            {"state": state, "questions": {"n": noul}},
            {"state": "The flag is disabled.", "questions": {"q": {"type": "noul", "instructions": "Is it on?"}}}]})
        check("batch: both requests' answers come back", st5 == 200 and set(batch.get("answers", {})) == {"n", "q"},
              (st5, batch))
        if st5 == 200:
            check("batch: the probabilities are the single-request ones",
                  batch["answers"]["n"]["probabilities"] == answers["n"]["probabilities"],
                  (batch["answers"].get("n"), answers.get("n")))
            check("batch: usage adds up the requests' tokens",
                  batch["usage"]["input_tokens"] > body["usage"]["input_tokens"], (batch.get("usage"), body.get("usage")))

        # A question's row can be longer than the 2048-row prefill chunk: it is encoded in chunks and each chunk needs
        # its own encoder (a second chunk through a committed one aborted the GPU and killed the server).
        long_text = "a description of the road ahead that takes a good few tokens"
        long_options = {"option_%03d" % i: {"description": long_text} for i in range(120)}
        st6, long_body = post(url, {"state": state, "questions": {
            "long": {"type": "choice", "instructions": "Which option matches the state?", "criteria": long_options}}})
        long_answer = (long_body.get("answers") or {}).get("long", {}) if st6 == 200 else {}
        check("a question row longer than one prefill chunk is answered",
              st6 == 200 and len(long_answer.get("probabilities") or {}) == 120, (st6, str(long_body)[:300]))
        st7, after = post(url, {"state": state, "questions": {"n": noul}})
        check("the server still answers after an over-chunk row",
              st7 == 200 and (after.get("answers") or {}).get("n", {}).get("probabilities") ==
              answers.get("n", {}).get("probabilities"), (st7, str(after)[:300]))

    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=b"{not json",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60)
        check("bad JSON: 400", False, "accepted")
    except urllib.error.HTTPError as e:
        check("bad JSON: 400", e.code == 400, e.code)

    # The renderer recurses once per nesting level on caller JSON, and the server's bodies go to 64 MiB: past the bound
    # it must be a 400 that leaves the worker threads alive, not a stack overflow.
    node = "leaf"
    for _ in range(10000):
        node = [node]
    st, reply = post(url, {"state": node, "questions": {"o": {"type": "noul", "instructions": "i"}}})
    check("rejects a state nested 10000 levels deep with 400", st == 400 and "error" in reply, (st, reply))
    st, reply = post(url, {"state": "s", "questions": {"o": {"type": "noul", "instructions": "i"}}})
    check("the server still answers after a deeply nested request", st == 200, (st, reply))

    for body, what in (({"state": "s"}, "no questions"),
                       ({"state": "s", "questions": {}}, "an empty questions object"),
                       ({"state": "s", "questions": {"o": {"type": "maybe"}}}, "an unknown type"),
                       ({"state": "s", "questions": {"o": {"type": "noul", "criteria": [1]}}}, "a non-object criteria"),
                       ({"state": "s", "questions": {"o": {"type": "choice", "instructions": "i",
                                                           "criteria": {}}}}, "an empty criteria"),
                       ({"requests": []}, "an empty batch"),
                       ({"requests": [{"state": "s"}]}, "a batch item with no questions")):
        st, reply = post(url, body)
        check("rejects %s with 400" % what, st == 400 and "error" in reply, (st, reply))


def main():
    if CLI:
        if not MODEL:
            print("offline tests need MODEL_DIR (any model folder with a tokenizer.json)")
            return 2
        if not os.path.exists(os.path.join(CLI, "systemone")):
            print("%s/systemone is not built (cmake --build build --target systemone)" % CLI)
            return 2
        offline(CLI, MODEL)
    else:
        online(BASE or "http://127.0.0.1:8000")
    print("\n%d failed" % fails if fails else "\nall passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
