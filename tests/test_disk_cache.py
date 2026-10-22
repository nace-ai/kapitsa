#!/usr/bin/env python3
"""The SSD prefix cache across server restarts (src/engine/engine_disk.cpp). Needs a model folder and the GPU: it takes
the GPU lock ($BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) unless NACE_GPU_LOCK_HELD=1 or NI_LOCK_HELD=1 says the
caller holds it. Servers start here with KAPITSA_HOME in a temporary folder (the cache, logs and KFlash's measurements
stay there; it is removed at the end) and --disk-cache 2G. Greedy, thinking off, NI_ANE=0 (the Neural Engine program
compiles in the background, and a prefill's numbers depend on whether it was ready).
  1. the RAM path: a fresh server: turn 1 (a long prompt from the frozen corpus), then its follow-up, turn 2
  2. a restart: a fresh server, an empty cache: turn 1; the server stops (SIGTERM: it saves what the SSD lacks); a new
     one: turn 2 is cached (usage.prompt_tokens_details.cached_tokens, as many as from RAM) and writes the RAM path's text
  3. write-behind: another conversation's turn 1; the server writes it once idle, then is killed (-9: nothing is saved at
     stop); a new one: its turn 2 is cached, and its save shares turn 1's KV blocks; after another restart, turn 3
     writes the RAM path's text (a server without the cache runs the same three turns)
  4. eviction (--slots 1): a third conversation's turn 1, then a fourth conversation takes the slot (the third is copied
     and written); killed; a new one: the third's turn 2 is cached
  5. models that read images: a conversation that starts with an image, across a restart as from RAM, without encoding
     the image again
  6. a DFlash drafter (when the server loads one): a long system block, then the server stops (SIGTERM): it must save
     without crashing (the prefix checkpoint has no drafter ring), and the follow-up is cached after a restart
usage: test_disk_cache.py <kapitsa> <model dir> [--tokens 4000] [--port 8197] [--server-args "--dflash none"] [--keep]
       test_disk_cache.py <kapitsa> <model dir> --ttft [--tokens N] [--rounds 2]
  --ttft: measures instead, per round: a server (cache on) writes turn 1 and stops; the next one answers turn 2 from the
          SSD (its time to first token; then the bytes its write-behind adds for that turn); a server without the cache
          answers turn 2 cold, then turn 1 and turn 2 again (from RAM). Checks the SSD's turn 2 against RAM's. The
          Neural Engine split is on, as by default (each server waits for its program and a warm-up switches it on).
          Every measured turn 2 follows a short request at once: after a few seconds without requests, a server's
          next request now starts its GPU work 0.3-0.5 s late on an M4 Pro (in every path, cached or not), which
          would hide what the cache itself costs.
  --keep: keep the temporary KAPITSA_HOME (logs and cache)"""
import base64, fcntl, json, os, re, shutil, signal, struct, subprocess, sys, tempfile, time, urllib.error, urllib.request, zlib

ARGS = sys.argv[1:]


def opt(name, default=None, flag=False):
    if name in ARGS:
        i = ARGS.index(name)
        if flag:
            del ARGS[i]
            return True
        v = ARGS[i + 1]
        del ARGS[i:i + 2]
        return v
    return default


SERVER_ARGS = (opt("--server-args", "") or "").split()  # e.g. "--dflash none": the MTP head instead of the drafter
TOKENS = int(opt("--tokens", "4000"))
PORT = int(opt("--port", "8197"))
TTFT = opt("--ttft", flag=True)
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
    path = os.environ.get("BENCH_GPU_LOCK") or os.path.expanduser("/tmp/kapitsa-gpu.lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a")
    print(f"waiting for the GPU lock {path} ...", flush=True)
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


class Server:
    def __init__(self, home, log, slots=2, cache="2G", ane=False):
        self.log_path = log
        e = {k: v for k, v in os.environ.items() if not k.startswith("NI_") and not k.startswith("KAPITSA_")}
        e.update({"KAPITSA_HOME": home, "KAPITSA_NO_AUTO_UPDATE": "1", "KAPITSA_NO_PROMPT": "1",
                  "NI_DISK_CACHE_IDLE_MS": "1500"})
        if not ane:
            e["NI_ANE"] = "0"
        self.log = open(log, "a")
        self.log.write(f"\n==== start: slots {slots}, cache {cache}\n")
        self.log.flush()
        self.start_at = self.mark()
        self.p = subprocess.Popen([KAPITSA, "serve", "--model", MODEL, "--port", str(PORT), "--ctx", "32768",
                                   "--slots", str(slots), "--disk-cache", cache] + SERVER_ARGS, env=e, stdout=self.log,
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
        if cache != "0":  # the cache's files open in the background, right after the model loads
            self.wait_log(r"SSD prefix cache: .* taken back|SSD prefix cache: off", 60, since=self.start_at)
        if ane and "compiling the Neural Engine program" in self.text()[self.start_at:]:
            # its program compiles in the background: every prefill measured runs with the split (the next request
            # switches it on)
            self.wait_log(r"NI_ANE: (Neural Engine program ready|Neural Engine split unavailable)", 180,
                          since=self.start_at)

    def text(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def mark(self):
        return len(self.text())

    def wait_log(self, pattern, timeout, since=None):
        """the first match of `pattern` after offset `since` (default: now), or None after timeout s"""
        start = self.mark() if since is None else since
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = re.search(pattern, self.text()[start:])
            if m:
                return m
            time.sleep(0.25)
        return None

    def stop(self, sig=signal.SIGTERM):
        self.p.send_signal(sig)
        try:
            self.p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.log.close()


def chat(messages, max_tokens=200):
    """(content, usage, time to first token, finish reason) of a streamed greedy chat completion"""
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
                text += d.get("content") or ""
                finish = ch.get("finish_reason") or finish
    return text, usage or {}, first or (time.time() - t0), finish


def cached(u):
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


with open(CORPUS, errors="replace") as f:
    CORPUS_TEXT = f.read()


def conversation(k):
    """conversation k: a slice of the corpus of about TOKENS tokens (4.25 characters each) and a question; its follow-up"""
    n = int(TOKENS * 4.25)
    off = (k * 7919) % max(1, len(CORPUS_TEXT) - n)
    turn1 = [{"role": "system", "content": f"You are a careful code reviewer (session {k})."},
             {"role": "user", "content": "Here is some source code:\n\n" + CORPUS_TEXT[off:off + n] +
                                         "\n\nIn two sentences: what does the first class in it do?"}]
    follow = {"role": "user", "content": "Name one function in it that could fail, and why, in one sentence."}
    return turn1, follow


def turn2(turn1, answer, follow):
    return turn1 + [{"role": "assistant", "content": answer}, follow]


def png_data_url(w=320, h=240):
    """a small PNG (a gradient, made with the standard library) as a data: URL"""
    raw = b"".join(b"\x00" + bytes(v for x in range(w) for v in ((x * 255) // w, (y * 255) // h, 128)) for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def image_conversation():
    """a conversation whose first user message starts with an image, then a slice of the corpus"""
    turn1, follow = conversation(6)
    turn1[1]["content"] = [{"type": "image_url", "image_url": {"url": png_data_url()}},
                           {"type": "text", "text": "Describe the colors of this image in one sentence, then: " +
                                                    turn1[1]["content"]}]
    return turn1, follow


def warm_up():
    """a short request nothing is cached for (the first request after a start pays one-time costs)"""
    chat([{"role": "user", "content": "Say hello in five words."}], max_tokens=8)


def answer_turn1(s, conv):
    """turn 1's answer, once the server has prefilled it as the follow-up renders it (the history prefill)"""
    m = s.mark()
    a, u, t, fin = chat(conv)
    s.wait_log(r"history: prefilled", 3, since=m)  # (none when the live state already matches the follow-up)
    time.sleep(0.5)
    return a, u, t, fin


def functional(tmp, log):
    c1, f1 = conversation(1)
    # 1. the RAM path (no cache: --disk-cache 0)
    s = Server(os.path.join(tmp, "home-ram"), log, cache="0")
    a1, u1, _, fin1 = answer_turn1(s, c1)
    r2, u2, _, _ = chat(turn2(c1, a1, f1))
    print(f"  RAM path: turn 1 {u1.get('prompt_tokens')} tokens ({fin1}), turn 2 {u2.get('prompt_tokens')} tokens, "
          f"{cached(u2)} cached", flush=True)
    s.stop()
    check(cached(u2) > 0, "RAM path: turn 2 reuses turn 1", u2)

    # 2. across a restart (SIGTERM: saved as the server stops)
    home = os.path.join(tmp, "home")
    s = Server(home, log)
    m = s.mark()
    b1, _, _, _ = answer_turn1(s, c1)
    check(b1 == a1, "turn 1 writes the same text on a fresh server")
    s.stop()
    # (by the write-behind when the history prefill took the server's idle time past NI_DISK_CACHE_IDLE_MS)
    check(re.search(r"saved as the server stops|SSD cache: saved slot", s.text()[m:]) is not None,
          "the conversation is on the SSD once the server has stopped", s.text()[m:][-800:])
    s = Server(home, log)
    check(re.search(r"1 conversation taken back", s.text()[s.start_at:]) is not None, "the next server takes it back")
    warm_up()
    m = s.mark()
    d2, w2, _, _ = chat(turn2(c1, b1, f1))
    check(cached(w2) == cached(u2) and cached(w2) > 0, "after the restart turn 2 is cached as from RAM",
          f"cached {cached(w2)} (RAM path {cached(u2)}); log: {s.text()[m:][-600:]}")
    check(d2 == r2, "... and writes the RAM path's text, token for token", f"\n RAM:  {r2!r}\n SSD:  {d2!r}")
    check(re.search(r"SSD cache: read \d+ tokens", s.text()[m:]) is not None, "the log says what was read")

    # 3. write-behind, then killed: nothing saved at stop. Then a third turn after another restart: the second turn's
    #    state shares the first's KV blocks on the SSD, and the third turn writes the RAM path's text
    c3, f3 = conversation(3)
    g3 = {"role": "user", "content": "Now suggest one test for that function, in one sentence."}
    m = s.mark()
    a3, _, _, _ = answer_turn1(s, c3)
    got = s.wait_log(r"SSD cache: saved slot \d+ \(\d+ tokens", 30, since=m)
    check(got is not None, "write-behind: the idle server writes the new conversation", got and got.group(0))
    s.stop(signal.SIGKILL)
    s = Server(home, log)
    m = s.mark()  # (its second turn is saved by the write-behind or as the server stops)
    d3, w3, _, _ = answer_turn1(s, turn2(c3, a3, f3))
    check(cached(w3) > TOKENS // 2, "killed after the write-behind: the conversation is cached after the restart", w3)
    s.stop()
    shared = re.search(r"saved slot \d+ \(\d+ tokens, \d+ restore points?; (\d+) of (\d+) KV blocks", s.text()[m:])
    check(shared is not None and int(shared.group(1)) < int(shared.group(2)),
          "its second turn shares the first turn's KV blocks on the SSD (writes only its new ones)",
          shared and shared.group(0))
    s = Server(home, log)
    t3 = turn2(c3, a3, f3) + [{"role": "assistant", "content": d3}, g3]
    e3, x3, _, _ = chat(t3)
    s.stop()
    s = Server(os.path.join(tmp, "home-ram3"), log, cache="0")  # the RAM path of the same three turns
    ra3, _, _, _ = answer_turn1(s, c3)
    rd3, _, _, _ = answer_turn1(s, turn2(c3, ra3, f3))
    re3, y3, _, _ = chat(turn2(c3, ra3, f3) + [{"role": "assistant", "content": rd3}, g3])
    s.stop()
    check(ra3 == a3 and rd3 == d3 and re3 == e3 and cached(x3) == cached(y3),
          "a third turn after two restarts (blocks shared across turns): the RAM path's text, as many cached tokens",
          f"cached SSD {cached(x3)} RAM {cached(y3)}\n RAM:  {re3!r}\n SSD:  {e3!r}")

    # 4. eviction: one slot, another conversation takes it
    s = Server(home, log, slots=1)
    c4, f4 = conversation(4)
    c5, _ = conversation(5)
    a4, _, _, _ = chat(c4)
    m = s.mark()
    chat(c5, max_tokens=8)  # (right away: before the write-behind)
    got = s.wait_log(r"saved the conversation slot \d+ held|SSD cache: saved slot", 30, since=m)
    check(got is not None and "held" in got.group(0), "a conversation another one replaces is copied and written",
          got and got.group(0))
    s.stop(signal.SIGKILL)
    s = Server(home, log, slots=1)
    _, w4, _, _ = chat(turn2(c4, a4, f4))
    check(cached(w4) > TOKENS // 2, "the replaced conversation is cached after a restart", w4)
    s.stop()

    # 5. a conversation with an image (models that read images): across a restart as from RAM
    if os.path.exists(os.path.join(MODEL, "preprocessor_config.json")):
        c6, f6 = image_conversation()
        s = Server(os.path.join(tmp, "home-ram-image"), log, cache="0")
        a6, _, _, _ = answer_turn1(s, c6)
        r6, u6, _, _ = chat(turn2(c6, a6, f6))
        s.stop()
        s = Server(home, log)
        b6, _, _, _ = answer_turn1(s, c6)
        s.stop()
        s = Server(home, log)
        m = s.mark()
        d6, w6, _, _ = chat(turn2(c6, b6, f6))
        check(b6 == a6 and cached(w6) == cached(u6) and cached(w6) > TOKENS // 2,
              "a conversation with an image: after a restart its follow-up is cached as from RAM",
              f"cached {cached(w6)} (RAM {cached(u6)}); {s.text()[m:][-500:]}")
        check(d6 == r6, "... and writes the RAM path's text", f"\n RAM:  {r6!r}\n SSD:  {d6!r}")
        check(re.search(r"chat: 1 image\(s\), \d+ tokens: 0 encoded .*, 1 in the cached prefix", s.text()[m:]) is not None,
              "... without encoding the image again (its rows come from the SSD)", s.text()[m:][-500:])
        s.stop()

    # 6. a DFlash drafter (the server's own choice, or --server-args "--dflash <drafter>"): the prompt's prefix checkpoint
    #    has no drafter ring, and the save at stop used to dereference its null ring (a segfault: the server died at
    #    SIGTERM). Runs when the server's log says it loads a DFlash2 drafter.
    s = Server(os.path.join(tmp, "home-dflash"), log)
    if "DFlash2 drafter" not in s.text()[s.start_at:]:
        print("SKIP the drafter step: this server loads no DFlash2 drafter", flush=True)
        s.stop()
    else:
        # (a long system block, a short user turn: the prefix checkpoint sits at the end of the system block)
        c7 = [{"role": "system", "content": "You are a coding agent. Repository notes:\n" + CORPUS_TEXT[:TOKENS * 4]},
              {"role": "user", "content": "Summarize the notes in one sentence."}]
        f7 = {"role": "user", "content": "Name one thing in them that could fail, in one sentence."}
        a7, _, _, _ = chat(c7)
        chat([{"role": "user", "content": "Say hello in five words."}], max_tokens=8)  # (a second conversation)
        check(s.p.poll() is None, "with a drafter: the server runs after a long prompt")
        s.stop()
        check(s.p.returncode in (0, -signal.SIGTERM), "with a drafter: the save at stop does not crash", s.p.returncode)
        s = Server(os.path.join(tmp, "home-dflash"), log)
        _, w7, _, _ = chat(turn2(c7, a7, f7))
        check(cached(w7) > TOKENS // 2, "with a drafter: the follow-up is cached after a restart", w7)
        s.stop()


def dir_mb(path):
    """MB of the files under path: what a cache that started empty holds (a save's log line counts only its own pass,
    and a write-behind cut short by the stop leaves the rest to the save at stop)"""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e6


def ttft(tmp, log):
    c1, f1 = conversation(1)
    rows = []
    for rnd in range(ROUNDS):
        home = os.path.join(tmp, f"home-{rnd}")
        # turn 1, written once the server idles (its write-behind) or as it stops
        s = Server(home, log, ane=True)
        warm_up()
        a1, u1, t1, _ = answer_turn1(s, c1)
        s.stop()
        w1 = dir_mb(os.path.join(home, "prefix-cache"))
        # turn 2 from the SSD after the restart, then the write-behind of that turn
        s = Server(home, log, ane=True)
        warm_up()
        m = s.mark()
        d2, w2, t_ssd, _ = chat(turn2(c1, a1, f1))
        rd = s.wait_log(r"SSD cache: read (\d+) tokens .*\(([\d.]+) MB in ([\d.]+) ms\)", 1, since=m)
        wb = s.wait_log(r"SSD cache: saved slot \d+ \(.*: ([\d.]+) MB\)", 20, since=m)
        s.stop(signal.SIGKILL)
        # without the cache: turn 2 cold, then turn 1 and turn 2 from RAM (the warm-up before it takes the other slot)
        s = Server(os.path.join(tmp, f"home-nocache-{rnd}"), log, slots=2, cache="0", ane=True)
        warm_up()
        _, uc, t_cold, _ = chat(turn2(c1, a1, f1))
        b1, _, _, _ = answer_turn1(s, c1)
        warm_up()
        r2, ur, t_ram, _ = chat(turn2(c1, b1, f1))
        s.stop()
        for h in (home, os.path.join(tmp, f"home-nocache-{rnd}")):  # (one round's cache on the disk at a time)
            shutil.rmtree(h, ignore_errors=True)
        check(b1 == a1 and d2 == r2, f"round {rnd + 1}: turn 2 from the SSD writes the RAM path's text",
              f"\n RAM:  {r2!r}\n SSD:  {d2!r}")
        check(cached(w2) == cached(ur) and cached(w2) > 0, f"round {rnd + 1}: as many tokens cached as from RAM",
              f"SSD {cached(w2)}, RAM {cached(ur)}")
        rows.append((u2p := w2.get("prompt_tokens"), t_cold, cached(uc), t_ssd, cached(w2), t_ram, cached(ur),
                     w1, float(wb.group(1)) if wb else -1,
                     f"{rd.group(2)} MB in {rd.group(3)} ms" if rd else "-", t1, u1.get("prompt_tokens")))
        print(f"round {rnd + 1}: turn 2 of {u2p} tokens: cold {t_cold:.3f} s ({cached(uc)} cached), from the SSD after a "
              f"restart {t_ssd:.3f} s ({cached(w2)} cached; read {rows[-1][9]}), from RAM {t_ram:.3f} s "
              f"({cached(ur)} cached); written: turn 1 {rows[-1][7]:.0f} MB, turn 2 {rows[-1][8]:.0f} MB; turn 1 (cold "
              f"too, {u1.get('prompt_tokens')} tokens) {t1:.3f} s", flush=True)
    print("| model | turn 2 tokens | cold TTFT | after a restart (SSD) | from RAM | SSD read | written turn 1 / turn 2 |")
    print("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for r in rows:
        print(f"| {os.path.basename(MODEL)} | {r[0]} | {r[1]:.3f} s | {r[3]:.3f} s ({r[4]} cached) | {r[5]:.3f} s | "
              f"{r[9]} | {r[7]:.0f} / {r[8]:.0f} MB |")


def main():
    lock = gpu_lock()
    tmp = tempfile.mkdtemp(prefix="kapitsa-disk-cache-test-")
    log = os.path.join(tmp, "serve.log")
    print(f"model {MODEL}, prompts of ~{TOKENS} tokens; servers' log: {log}", flush=True)
    try:
        (ttft if TTFT else functional)(tmp, log)
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
