#!/usr/bin/env python3
"""Reading a user's draft ahead (POST /v1/draft, Engine::draft). Needs a model folder and the GPU (the GPU lock:
$BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock, unless NACE_GPU_LOCK_HELD=1 or NI_LOCK_HELD=1). Servers run with
KAPITSA_HOME in a temporary folder and --disk-cache 0, greedy, thinking off; the checks with NI_ANE=0 (the Neural
Engine split moves a prefill's bits by whether its program is ready yet), NI_LOG=2 and NI_TRACE_TOKENS (token ids and
margins). Each case is a fresh server (it computes the same requests to the bit): a slice of the corpus and a question,
the answer's history prefill, drafts of the next message, then that message.
  1. the follow-up without drafts, and after a draft of the whole message (also before the history prefill ran), of
     the message less its last word, as it was typed, edited in the middle and back, deleted and typed again: the
     same tokens, the request continuing the last draft's state (cached_tokens; no "checkpoints: prompt diverges"); an
     edit restores the user turn's checkpoint and then computes what the one draft of the whole message does
  2. a ~2,000-token paste, read ahead; with 256-row chunks, a request of another conversation stops its reading
     ("interrupted") and the follow-up continues what was read; the same tokens as without drafts
  3. unchanged, no_conversation, busy, 400s
usage: test_draft.py <kapitsa> <model dir> [--tokens 3000] [--port 8198] [--server-args "--dflash none"] [--keep]
       test_draft.py <kapitsa> <model dir> --measure [--tokens 5000] [--rounds 2]
  --measure: the follow-up's time to its first token (sent right after a one-token request of another conversation:
             the GPU awake) and cached_tokens, for a ~40-token question and a ~2,000-token paste, without a draft and
             after one of the whole message or of all but its last word; the servers as by default. The replies must be
             the same, but where the Neural Engine split read the follow-up without a draft (a dense model's chunks of
             512 rows or more: the FFN in 8 bits; after a draft the last rows are the GPU's): then a note"""
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
    def __init__(self, tmp, log, extra=()):
        self.log_path = log
        self.trace = os.path.join(tmp, f"trace-{time.time_ns()}.jsonl")
        home = os.path.join(tmp, "home")
        e = {k: v for k, v in os.environ.items() if not k.startswith("NI_") and not k.startswith("KAPITSA_")}
        e.update({"KAPITSA_HOME": home, "KAPITSA_NO_AUTO_UPDATE": "1", "KAPITSA_NO_PROMPT": "1"})
        if not MEASURE:  # (measured: the server as it runs by default, the Neural Engine split on where it is)
            e.update({"NI_ANE": "0", "NI_LOG": "2", "NI_TRACE_TOKENS": self.trace})
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


def same(a, b):
    """the replies are the same, token for token where traced; else where they part, and the margins there"""
    (ta, _, _, ra, _), (tb, _, _, rb, _) = a, b
    ids = lambda r: [r["first"]] + r["tokens"]  # noqa: E731
    if not (ra and rb):
        return ta == tb, f"\n     A: {ta!r}\n     B: {tb!r}"
    i = next((k for k, (x, y) in enumerate(zip(ids(ra), ids(rb))) if x != y), None)
    m = [([None] + r["margins"])[i:i + 1] for r in (ra, rb)] if i is not None else None
    return ids(ra) == ids(rb), f"tokens part at {i} (margins {m})\n     A: {ta!r}\n     B: {tb!r}"


def run(tmp, log, message, drafts, extra=(), early=False, image=False):
    """a fresh server: turn 1, then each draft of `drafts` (texts; early: before the history prefill), then the
    follow-up; (follow-up, draft replies, turn 1's answer)"""
    s = Server(tmp, log, extra)
    try:
        conv = conversation(image)
        a, _ = first_turn(s, conv, wait=not early)
        replies = [draft(a, d) for d in drafts]
        time.sleep(0.5 if early else 0)  # (the history prefill then finds the conversation read)
        return follow_up(s, conv, a, message), replies, a
    finally:
        s.stop()


def functional(tmp, log):
    words = SHORT.split(" ")
    base, _, a0 = run(tmp, log, SHORT, [])
    h = cached(base[1])  # the history prefill's state: all of the conversation but its last token
    print(f"  no draft: prompt {base[1].get('prompt_tokens')}, cached {h}, first token {base[2]:.3f} s", flush=True)
    check(h > TOKENS // 2, "the follow-up continues the history prefill's state", base[1])

    full, (d1,), _ = run(tmp, log, SHORT, [SHORT])
    st, d = d1
    check(st == 200 and d["status"] == "prefilled" and d["cached_tokens"] == h,
          "a draft of the whole message reads it from the history prefill's state", d1)
    ok, why = same(base, full)
    check(ok, "... and the follow-up writes the reply it writes without the draft", why)
    check(cached(full[1]) == d["cached_tokens"] + d["prefilled_tokens"] and
          "checkpoints: prompt diverges" not in full[4],
          "... continuing the draft's state (cached_tokens = what the draft left, no checkpoint restored)",
          f"usage {full[1]}, draft {d}")

    early, (de,), _ = run(tmp, log, SHORT, [SHORT], early=True)
    ok, why = same(full, early)
    check(ok and de[1]["status"] == "prefilled" and de[1]["cached_tokens"] < h and cached(early[1]) == cached(full[1]),
          "a draft before the history prefill ran reads the reply's history too, as the history prefill would: then "
          "the same rows in the same passes", why or [de, early[1]])

    minus, (d2,), _ = run(tmp, log, SHORT, [less_last_word(SHORT)])
    ok, why = same(base, minus)
    check(ok and cached(minus[1]) == d2[1]["cached_tokens"] + d2[1]["prefilled_tokens"],
          "a draft less the last word: the same reply, the draft's state continued", why or minus[1])

    typed = [" ".join(words[:k]) for k in range(1, len(words) + 1, 3)] + [SHORT]
    typing, rs, _ = run(tmp, log, SHORT, typed)
    ok, why = same(base, typing)
    check(ok and all(r[1]["status"] in ("prefilled", "unchanged") for r in rs) and
          cached(typing[1]) == rs[-1][1]["cached_tokens"] + rs[-1][1]["prefilled_tokens"],
          f"drafts as the message is typed ({len(typed)} of them): the same reply, the last draft's state continued",
          why or [r[1] for r in rs])

    mid = len(words) // 2
    edited = " ".join(words[:mid] + ["completely", "different"] + words[mid + 1:])
    edit, rs, _ = run(tmp, log, SHORT, [SHORT, edited, SHORT])
    ok, why = same(base, edit)
    check(ok, "an edit in the middle of the draft and back: the reply without drafts", why)
    check(rs[1][1]["cached_tokens"] == h and rs[2][1]["cached_tokens"] == h,
          "... each edit restores the user turn's checkpoint (the conversation is not read again)", [r[1] for r in rs])
    ok, why = same(full, edit)
    check(ok and cached(edit[1]) == cached(full[1]),
          "... and then computes what the one draft of the whole message does (the same rows in the same passes)",
          why or f"{edit[1]} vs {full[1]}")

    gone, rs, _ = run(tmp, log, SHORT, [SHORT, "", SHORT])
    ok, why = same(base, gone)
    check(ok and rs[1][1]["status"] == "prefilled" and rs[1][1]["cached_tokens"] == h,
          "the draft deleted (back to the answer's end) and typed again: the reply without drafts", why or
          [r[1] for r in rs])

    if os.path.exists(os.path.join(MODEL, "preprocessor_config.json")):  # (a model that reads images)
        ibase, _, _ = run(tmp, log, SHORT, [], image=True)
        ifull, (di,), _ = run(tmp, log, SHORT, [SHORT], image=True)
        ok, why = same(ibase, ifull)
        check(ok and di[1]["status"] == "prefilled" and cached(ifull[1]) == di[1]["cached_tokens"] + di[1]["prefilled_tokens"],
              "a conversation that starts with an image: read ahead, the same reply", why or [di, ifull[1]])

    # 2. a paste of about 2,000 tokens
    pbase, _, _ = run(tmp, log, PASTE, [])
    pfull, (pd,), _ = run(tmp, log, PASTE, [PASTE])
    ok, why = same(pbase, pfull)
    print(f"  paste: prompt {pbase[1].get('prompt_tokens')}, first token {pbase[2]:.3f} s without a draft, "
          f"{pfull[2]:.3f} s after one ({pd[1]})", flush=True)
    check(ok and pd[1]["prefilled_tokens"] > 1500 and cached(pfull[1]) == pd[1]["cached_tokens"] + pd[1]["prefilled_tokens"],
          "a pasted draft: read ahead, then the reply without drafts", why or pd)

    # interrupted: 256-row chunks, a request of another conversation sent while the paste is read
    chunk = ("--prefill-chunk", "256")
    p256, _, _ = run(tmp, log, PASTE, [], chunk)
    s = Server(tmp, log, chunk)
    try:
        conv = conversation()
        a, _ = first_turn(s, conv)
        got = {}
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
        check(ok and cached(res[1]) == d["cached_tokens"] + d["prefilled_tokens"],
              "... and the follow-up continues what was read, with the reply it writes without the draft",
              why or f"{res[1]} vs draft {d}")
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
        check(draft(a, SHORT, think=True)[0] == 400 and draft(a, SHORT, think=False)[0] == 200,
              "\"think\": true is a 400 (not supported yet); false is the default")
        check(draft(a, None)[0] == 400, "a body without \"text\" is a 400")
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
            if len(set(texts.values())) == 1 or not ane:
                check(len(set(texts.values())) == 1, f"round {rnd + 1}, {name}: the same reply in all three cases",
                      json.dumps(texts, indent=1))
            else:
                print(f"NOTE round {rnd + 1}, {name}: the replies differ; the Neural Engine split read the follow-up "
                      "without a draft (rows it computes in 8 bits, after a draft on the GPU)\n     " +
                      json.dumps(texts, indent=1), flush=True)
            for cond, ((_, u, t, _, _), ds, u1) in res.items():
                d = ds[-1] if ds else None
                rows.append((rnd + 1, name, u1.get("prompt_tokens"), u.get("prompt_tokens"), cond, t, cached(u),
                             f"{d['prefilled_tokens']} in {d['ms']:.0f} ms" if d else "-"))
                print(rows[-1], flush=True)
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
