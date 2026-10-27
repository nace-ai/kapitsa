#!/usr/bin/env python3
"""A pass-through stand-in for Splash's engine binary that records what crosses the native pipe, so two front ends
(`kapitsa serve --engine splash` and plain Splash's server) can be compared at the token-id level, not by text.

Used as the engine binary: it runs the real one (SPLASH_TAP_REAL) with the same arguments, copies stdin and stdout
through unchanged and appends one JSON line per finished request to SPLASH_TAP_LOG:
  {"request": id, "prompt_tokens": n, "prompt_md5": ..., "ids": [...], "ids_md5": ..., "finish": ...}
prompt_md5 and ids_md5 are md5 of the little-endian u32 words, so equal values mean equal prompts and equal output ids.
Needs Splash's server/protocol.py (SPLASH_ROOT, default ~/Documents/nace/splash-1.4.0) to decode the engine's frames."""
import hashlib, json, os, struct, subprocess, sys, threading

SPLASH_ROOT = os.environ.get("SPLASH_ROOT", os.path.expanduser("~/Documents/nace/splash-1.4.0"))
sys.path.insert(0, SPLASH_ROOT)
from server import protocol as P  # noqa: E402

HEADER = struct.Struct("<4sHHQ")
PROMPT_COUNT_AT = HEADER.size + 26 + 4  # request fixed part: id u64, priority u8, constraint u8, 2 deadlines u64, logical max u32
PROMPT_AT = HEADER.size + 114
log = open(os.environ["SPLASH_TAP_LOG"], "a", buffering=1)
lock = threading.Lock()
prompts, ids = {}, {}


def md5(words):
    return hashlib.md5(struct.pack("<%dI" % len(words), *words)).hexdigest()


def client_side(child):
    """Server -> engine: forward each frame, noting a request's prompt."""
    src = sys.stdin.buffer
    while True:
        head = src.read(HEADER.size)
        if len(head) < HEADER.size:
            break
        _, _, ftype, size = HEADER.unpack(head)
        body = src.read(size)
        if ftype == P.FrameType.REQUEST:
            frame = head + body
            rid = struct.unpack_from("<Q", frame, HEADER.size)[0]
            n = struct.unpack_from("<I", frame, PROMPT_COUNT_AT)[0]
            words = struct.unpack_from("<%dI" % n, frame, PROMPT_AT)
            with lock:
                prompts[rid] = (n, md5(words))
        try:
            child.stdin.write(head + body)
            child.stdin.flush()
        except (BrokenPipeError, ValueError):
            break
    try:
        child.stdin.close()
    except OSError:
        pass


def engine_side(child):
    parser = P.FrameParser()
    out = sys.stdout.buffer
    while True:
        data = child.stdout.read1(1 << 16)
        if not data:
            break
        out.write(data)
        out.flush()
        view = memoryview(data)
        while len(view):
            step = parser.consume(view)
            view = view[step.consumed_bytes:]
            if step.issue or (step.frame is None and not step.consumed_bytes):
                break
            if step.frame is None:
                continue
            try:
                ev = P.decode_frame(step.frame)
            except P.ProtocolError:
                continue
            if isinstance(ev, P.TokensEvent):
                ids.setdefault(ev.request_id, []).extend(ev.tokens)
            elif isinstance(ev, P.DoneEvent):
                got = ids.pop(ev.request_id, [])
                n, pm = prompts.pop(ev.request_id, (None, None))
                log.write(json.dumps({"request": ev.request_id, "prompt_tokens": n, "prompt_md5": pm, "ids": got,
                                      "ids_md5": md5(got), "finish": getattr(ev.reason, "name", str(ev.reason))}) + "\n")


def main():
    real = os.environ["SPLASH_TAP_REAL"]
    child = subprocess.Popen([real] + sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    threading.Thread(target=client_side, args=(child,), daemon=True).start()
    engine_side(child)
    code = child.wait()
    log.close()
    os._exit(code)  # the stdin thread may be blocked in a read; leaving through sys.exit aborts the interpreter


main()
