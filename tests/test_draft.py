#!/usr/bin/env python3
"""Reading a user's draft ahead (POST /v1/draft). Needs a model folder and the GPU (the GPU lock: $BENCH_GPU_LOCK,
default /tmp/kapitsa-gpu.lock, unless NACE_GPU_LOCK_HELD=1 or NI_LOCK_HELD=1). Each case is a fresh server (the same
requests to the bit; KAPITSA_HOME temporary, --disk-cache 0, NI_ANE=0, NI_TRACE_TOKENS): a slice of the corpus and a
question, its answer's history prefill, drafts of the next message, then that message, greedy. The follow-up writes
the reply it writes without drafts, token for token; on a MoE model (whose short prefill passes are not row-count
invariant) a reply that takes another token passes only if it moved no more than the one-pass history path
(NI_HISTORY_PREFILL=0) moves it (close()). Exact on every model: an edited draft and one sent before the history
prefill against the one draft of the whole message, a ~2,000-token paste, a paste whose reading a request interrupts.
usage: test_draft.py <kapitsa> <model dir> [--tokens 3000] [--port 8198] [--server-args "--dflash none"] [--keep]
       test_draft.py <kapitsa> <model dir> --measure [--tokens 5000] [--rounds 2]
  --measure: the follow-up's time to its first token (right after a one-token request of another conversation) and
             cached_tokens, a ~40-token question and a ~2,000-token paste, with no draft, a draft of all of it, of all
             but its last word; servers as by default (the Neural Engine split on)"""
import fcntl, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request

ARGS = sys.argv[1:]


def opt(name, default=None, flag=False):
    if name not in ARGS:
        return default
    i = ARGS.index(name)
    v = True if flag else ARGS[i + 1]
    del ARGS[i:i + (1 if flag else 2)]
    return v


SERVER_ARGS = (opt("--server-args", "") or "").split()
MEASURE = opt("--measure", flag=True)
TOKENS = int(opt("--tokens", "5000" if MEASURE else "3000"))
PORT = int(opt("--port", "8198"))
ROUNDS = int(opt("--rounds", "2"))
KEEP = opt("--keep", flag=True)
if len(ARGS) < 2:
    sys.exit(__doc__)
KAPITSA, MODEL = os.path.abspath(ARGS[0]), os.path.abspath(os.path.expanduser(ARGS[1]))
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "bench", "harness", "corpus", "bench_corpus.txt")
BASE = f"http://127.0.0.1:{PORT}"
fails = 0


def check(ok, name, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else "\n     " + str(detail)), flush=True)
    fails += 0 if ok else 1


def gpu_lock():
    if os.environ.get("NACE_GPU_LOCK_HELD") == "1" or os.environ.get("NI_LOCK_HELD") == "1":
        return None
    path = os.environ.get("BENCH_GPU_LOCK") or "/tmp/kapitsa-gpu.lock"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a")
    print(f"waiting for the GPU lock {path} ...", flush=True)
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


class Server:
    def __init__(self, tmp, log, extra=(), env=None):
        self.log_path = log
        self.trace = os.path.join(tmp, f"trace-{time.time_ns()}.jsonl")
        home = os.path.join(tmp, "home")
        e = {k: v for k, v in os.environ.items() if not k.startswith("NI_") and not k.startswith("KAPITSA_")}
        e.update({"KAPITSA_HOME": home, "KAPITSA_NO_AUTO_UPDATE": "1", "KAPITSA_NO_PROMPT": "1"})
        if not MEASURE:  # (measured: the server as it runs by default, the Neural Engine split on where it is)
            e.update({"NI_ANE": "0", "NI_LOG": "2", "NI_TRACE_TOKENS": self.trace})
        e.update(env or {})
        self.log = open(log, "a")
        self.log.write(f"\n==== start {' '.join(extra)}\n")
        self.log.flush()
        self.start_at = self.mark()
        self.p = subprocess.Popen([KAPITSA, "serve", "--model", MODEL, "--port", str(PORT), "--ctx", "32768",
                                   "--disk-cache", "0"] + list(extra) + SERVER_ARGS, env=e, stdout=self.log,
                                  stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.p.poll() is not None:
                raise SystemExit(f"the server exited ({self.p.returncode}); its log: {log}")
            try:
                if urllib.request.urlopen(BASE + "/health", timeout=2).status == 200:
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        else:
            raise SystemExit("the server did not start in 600 s")
        if MEASURE and "compiling the Neural Engine program" in self.text()[self.start_at:]:
            # its program compiles in the background: every prefill measured runs with the split (turn 1 switches it on)
            self.wait_log(r"NI_ANE: (Neural Engine program ready|Neural Engine split unavailable)", 180,
                          since=self.start_at)

    def text(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def mark(self):
        return len(self.text())

    def wait_log(self, pattern, timeout, since=None):
        start = self.mark() if since is None else since
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = re.search(pattern, self.text()[start:])
            if m:
                return m
            time.sleep(0.1)
        return None

    def traced(self):
        """the token ids and margins of the server's requests, in the order they ended"""
        if not os.path.exists(self.trace):
            return []
        with open(self.trace) as f:
            return [json.loads(line) for line in f if line.strip()]

    def stop(self):
        self.p.send_signal(signal.SIGTERM)
        try:
            self.p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.log.close()


def chat(messages, max_tokens=160, on_first=None):
    """(content, usage, time to first token, finish reason) of a streamed greedy chat completion; on_first() is called
    at its first token"""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, text, usage, finish = time.time(), None, "", None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                usage = c["usage"]
            for ch in c.get("choices", []):
                d = ch.get("delta", {})
                if ((d.get("content") or "") + (d.get("reasoning_content") or "")) and first is None:
                    first = time.time() - t0
                    if on_first:
                        on_first()
                text += d.get("content") or ""
                finish = ch.get("finish_reason") or finish
    return text, usage or {}, first or (time.time() - t0), finish


def draft(after, text, **extra):
    """(HTTP status, body) of POST /v1/draft (a None field left out)"""
    body = {k: v for k, v in {"model": "default", "after": after, "text": text, **extra}.items() if v is not None}
    req = urllib.request.Request(BASE + "/v1/draft", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def cached(u):
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


with open(CORPUS, errors="replace") as f:
    CORPUS_TEXT = f.read()


def conversation(image=False):
    """a slice of the corpus of about TOKENS tokens (4.25 characters each) and a question (image: after a PNG)"""
    q = "Here is some source code:\n\n" + CORPUS_TEXT[:int(TOKENS * 4.25)] + "\n\nIn two sentences: what does the first class do?"
    if image:  # a 2x2 gray PNG
        png = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAAAAABX3VL4AAAADklEQVR4nGNoaGBoaAAABgYCASzBUNcAAAAASUVORK5CYII="
        q = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}, {"type": "text", "text": q}]
    return [{"role": "system", "content": "You are a careful code reviewer."}, {"role": "user", "content": q}]


SHORT = ("Thanks. Now name one function in this code that could fail on bad input, explain why in two sentences, "
         "and suggest a one-line fix for it.")  # about 30 tokens
PASTE = ("Here is another file from the same project:\n\n" + CORPUS_TEXT[300000:300000 + 8400] +
         "\n\nDoes it have the same kind of problem? Answer in two sentences.")  # about 2,000 tokens


def less_last_word(text):
    return text[:len(text.rstrip()) - len(text.rstrip().split()[-1])].rstrip()


def first_turn(s, conv, wait=True):
    """turn 1's answer, once the server has prefilled it as the follow-up renders it (wait)"""
    m = s.mark()
    a, u, _, fin = chat(conv, max_tokens=400)
    check(fin == "stop", "turn 1 ends with stop (a conversation drafts can continue)", fin)
    if wait:
        s.wait_log(r"history: prefilled", 5, since=m)
        time.sleep(0.3)
    return a, u


def follow_up(s, conv, answer, message, warm=False):
    """the follow-up: its text, usage, time to first token, and its trace (token ids, margins)"""
    if warm:  # (the GPU awake: a one-token request of another conversation, on another slot)
        chat([{"role": "user", "content": "Say hi."}], max_tokens=1)
    m = s.mark()
    text, u, t, _ = chat(conv + [{"role": "assistant", "content": answer}, {"role": "user", "content": message}])
    tr = s.traced()
    return text, u, t, (tr[-1] if tr else None), s.text()[m:]


def ids(r):
    return [r["first"]] + r["tokens"]


def part(ra, rb):  # the first reply token where two traces differ (their shorter length when one is the other's start)
    return next((k for k, (x, y) in enumerate(zip(ids(ra), ids(rb))) if x != y), min(len(ids(ra)), len(ids(rb))))


def same(a, b):
    """the replies are the same, token for token where traced; else where they part, and the margins there"""
    (ta, _, _, ra, _), (tb, _, _, rb, _) = a, b
    if not (ra and rb):
        return ta == tb, f"\n     A: {ta!r}\n     B: {tb!r}"
    i = part(ra, rb)
    m = [([None] + r["margins"])[i:i + 1] for r in (ra, rb)]
    j = max(0, i - 2)
    return ids(ra) == ids(rb), f"tokens part at {i}: {ids(ra)[j:i + 2]} vs {ids(rb)[j:i + 2]} (margins {m})"


def moe():  # a model that routes experts: its short prefill passes are not row-count invariant (docs/usage.md)
    def has(o):
        return isinstance(o, dict) and (o.get("num_experts", 0) > 0 or any(has(v) for v in o.values()))
    with open(os.path.join(MODEL, "config.json")) as f:
        return has(json.load(f))


SLACK = 0.02  # logits
MOE = moe()


def close(base, x, ctrl):
    """x's reply is base's token for token; or, on a MoE model, a draft moves its logits no more than a request without
    drafts read in one pass does (ctrl: NI_HISTORY_PREFILL=0, the follow-up that comes before the history prefill):
    over the reply positions where base, x and ctrl still decode the same tokens (the same context, teacher-forced
    alike), x's mean change of the top-2 logit gap is at most ctrl's + SLACK, and where x takes another token, base's
    gap there is no wider than ctrl's largest change"""
    ok, why = same(base, x)
    if ok or not MOE:
        return ok, why
    rb, rx, rc = base[3], x[3], ctrl()[3]
    w = min(part(rb, rx), part(rb, rc))
    mb, mx, mc = rb["margins"], rx["margins"], rc["margins"]
    dx = [abs(mb[k] - mx[k]) for k in range(w - 1)]  # reply tokens 1 .. w - 1 (the first has no margin)
    dc = [abs(mb[k] - mc[k]) for k in range(w - 1)]
    if not dx:
        return False, why + "; no shared reply tokens to compare"
    mean_x, mean_c, gap = sum(dx) / len(dx), sum(dc) / len(dc), mb[part(rb, rx) - 1]
    ok = mean_x <= mean_c + SLACK and gap <= max(dc)
    return ok, (f"{why}; over reply tokens 1-{w - 1} the top-2 gap moved {mean_x:.4f} on average, at most "
                f"{max(dx):.4f} (one-pass history: {mean_c:.4f}, at most {max(dc):.4f}); no-draft gap where it "
                f"parts {gap:.4f}")


def run(tmp, log, message, drafts, extra=(), early=False, image=False, env=None):
    """a fresh server: turn 1, then each draft of `drafts` (texts; early: before the history prefill), then the
    follow-up; (follow-up, draft replies, turn 1's answer)"""
    s = Server(tmp, log, extra, env)
    try:
        conv = conversation(image)
        a, _ = first_turn(s, conv, wait=not early)
        replies = [draft(a, d) for d in drafts]
        time.sleep(0.5 if early else 0)  # (the history prefill then finds the conversation read)
        return follow_up(s, conv, a, message), replies, a
    finally:
        s.stop()


def functional(tmp, log):
    words, ctl = SHORT.split(" "), {}
    base, _, a0 = run(tmp, log, SHORT, [])
    # the one-pass history controls (NI_HISTORY_PREFILL=0), run when a MoE model's drafted reply is not the no-draft one
    control = lambda img=False: ctl.get(img) or ctl.setdefault(  # noqa: E731
        img, run(tmp, log, SHORT, [], image=img, env={"NI_HISTORY_PREFILL": "0"})[0])
    cont = lambda res, d: cached(res[1]) == d["cached_tokens"] + d["prefilled_tokens"]  # noqa: E731
    h = cached(base[1])  # the history prefill's state: all of the conversation but its last token
    check(h > TOKENS // 2, "the follow-up continues the history prefill's state", base[1])

    full, ((st, d),), _ = run(tmp, log, SHORT, [SHORT])
    check(st == 200 and d["status"] == "prefilled" and d["cached_tokens"] == h,
          "a draft of the whole message reads it from the history prefill's state", d)
    ok, why = close(base, full, control)
    check(ok, "... and the follow-up writes the reply without drafts (MoE: or moves it no more than the one-pass "
          "history path)", why)
    check(cont(full, d) and "checkpoints: prompt diverges" not in full[4],
          "... continuing the draft's state (cached_tokens = what the draft left, no checkpoint restored)", full[1])
    early, ((_, de),), _ = run(tmp, log, SHORT, [SHORT], early=True)
    ok, why = same(full, early)
    check(ok and de["status"] == "prefilled" and de["cached_tokens"] < h and cont(early, de),
          "a draft before the history prefill ran reads the reply's history too: then the same rows in the same passes "
          "as after it", why or de)
    typed = [" ".join(words[:k]) for k in range(1, len(words) + 1, 3)] + [SHORT]
    for name, drafts in (("a draft less the last word", [less_last_word(SHORT)]), ("drafts as it is typed", typed),
                         ("the draft deleted (back to the answer's end, the user turn's checkpoint) and typed again",
                          [SHORT, "", SHORT])):
        res, rs, _ = run(tmp, log, SHORT, drafts)
        ok, why = close(base, res, control)
        check(ok and cont(res, rs[-1][1]) and all(r[1]["status"] in ("prefilled", "unchanged") for r in rs) and
              (len(rs) != 3 or rs[1][1]["cached_tokens"] == h),
              f"{name}: the reply without drafts, the last draft's state continued", why or [r[1] for r in rs])
    mid = len(words) // 2
    edited = " ".join(words[:mid] + ["completely", "different"] + words[mid + 1:])
    edit, rs, _ = run(tmp, log, SHORT, [SHORT, edited, SHORT])
    ok, why = close(base, edit, control)
    check(ok and rs[1][1]["cached_tokens"] == h and rs[2][1]["cached_tokens"] == h,
          "an edit in the middle and back restores the user turn's checkpoint; the reply without drafts", why or rs)
    ok, why = same(full, edit)
    check(ok and cached(edit[1]) == cached(full[1]), "... then computes the one draft of the whole message's rows in its "
          "passes", why)
    if os.path.exists(os.path.join(MODEL, "preprocessor_config.json")):  # (a model that reads images)
        ibase, _, _ = run(tmp, log, SHORT, [], image=True)
        ifull, ((_, di),), _ = run(tmp, log, SHORT, [SHORT], image=True)
        ok, why = close(ibase, ifull, lambda: control(True))
        check(ok and di["status"] == "prefilled" and cont(ifull, di), "a conversation that starts with an image: read "
              "ahead, the reply without drafts", why or di)

    # 2. a paste of about 2,000 tokens; then 256-row chunks, a request of another conversation sent while it is read
    pbase, _, _ = run(tmp, log, PASTE, [])
    pfull, ((_, pd),), _ = run(tmp, log, PASTE, [PASTE])
    ok, why = same(pbase, pfull)
    check(ok and pd["prefilled_tokens"] > 1500 and cont(pfull, pd), "a pasted draft: read ahead, the same tokens",
          why or pd)
    p256, _, _ = run(tmp, log, PASTE, [], ("--prefill-chunk", "256"))
    s = Server(tmp, log, ("--prefill-chunk", "256"))
    try:
        conv, got = conversation(), {}
        a, _ = first_turn(s, conv)
        th = threading.Thread(target=lambda: got.update(r=draft(a, PASTE)))
        th.start()
        time.sleep(0.1)  # (the draft has the GPU turn: it renders its prompt in a few ms)
        chat([{"role": "user", "content": "Say hi."}], max_tokens=1)
        th.join()
        st, d = got["r"]
        check(st == 200 and d["status"] == "interrupted" and 0 < d["prefilled_tokens"] < 1500,
              "a request arriving while a paste is read stops it at a chunk boundary", d)
        res = follow_up(s, conv, a, PASTE)
        ok, why = same(p256, res)
        check(ok and cont(res, d), "... the follow-up continues what was read, the same tokens", why or res[1])
    finally:
        s.stop()

    # 3. statuses
    s = Server(tmp, log)
    try:
        conv = conversation()
        a, _ = first_turn(s, conv)
        r1, r2 = draft(a, SHORT), draft(a, SHORT)
        check(r1[1]["status"] == "prefilled" and r2[1]["status"] == "unchanged" and r2[1]["prefilled_tokens"] == 0,
              "the same draft again: unchanged", [r1, r2])
        check(draft("an answer this server never wrote", SHORT)[1]["status"] == "no_conversation",
              "another `after`: no_conversation")
        check(draft(a, SHORT, think=True)[0] == 400 and draft(a, SHORT, think=False)[0] == 200 and
              draft(a, None)[0] == 400, "400: \"think\": true (not supported yet), no \"text\"")
        writing = threading.Event()
        th = threading.Thread(target=lambda: chat([{"role": "user", "content": "Count from 1 to 300."}], max_tokens=600,
                                                  on_first=writing.set))
        th.start()
        writing.wait(120)
        busy = draft(a, SHORT + " More.")
        th.join()
        check(busy[1]["status"] == "busy", "a draft while a reply is written: busy", busy)
    finally:
        s.stop()
    check(a0 == a, "every fresh server writes the same first answer")


def measure(tmp, log):
    rows = []
    for rnd in range(ROUNDS):
        for name, msg in (("~40-token question", SHORT), ("~2,000-token paste", PASTE)):
            res = {}
            for cond, drafts in (("no draft", []), ("draft of the whole message", [msg]),
                                 ("draft less its last word", [less_last_word(msg)])):
                s = Server(tmp, log)
                try:
                    conv = conversation()
                    a, u1 = first_turn(s, conv)
                    ds = [draft(a, d)[1] for d in drafts]
                    res[cond] = (follow_up(s, conv, a, msg, warm=True), ds, u1)
                finally:
                    s.stop()
            texts = {c: r[0][0] for c, r in res.items()}
            ane = "Neural Engine busy" in res["no draft"][0][4]
            if len(set(texts.values())) == 1 or not (ane or MOE):
                check(len(set(texts.values())) == 1, f"round {rnd + 1}, {name}: the same reply in all three cases",
                      json.dumps(texts, indent=1))
            else:  # (the functional checks gate how far a MoE model's reply moves)
                print(f"NOTE round {rnd + 1}, {name}: the replies differ ({'the Neural Engine split' if ane else 'MoE'}"
                      f")\n     " + json.dumps(texts, indent=1), flush=True)
            for cond, ((_, u, t, _, _), ds, u1) in res.items():
                d = ds[-1] if ds else None
                rows.append((rnd + 1, name, u1.get("prompt_tokens"), u.get("prompt_tokens"), cond, t, cached(u),
                             f"{d['prefilled_tokens']} in {d['ms']:.0f} ms" if d else "-"))
    print(f"| {os.path.basename(MODEL)} | round | turn 1 tokens | follow-up tokens | before it | first token | cached | "
          "draft read |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |")
    for r in rows:
        print(f"| {r[1]} | {r[0]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.3f} s | {r[6]} | {r[7]} |")


def main():
    lock = gpu_lock()
    tmp = tempfile.mkdtemp(prefix="kapitsa-draft-test-")
    log = os.path.join(tmp, "serve.log")
    print(f"model {MODEL}, conversations of ~{TOKENS} tokens; servers' log: {log}", flush=True)
    try:
        (measure if MEASURE else functional)(tmp, log)
    finally:
        if KEEP:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
        if lock:
            lock.close()
    print("ALL PASSED" if not fails else f"{fails} FAILED", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
