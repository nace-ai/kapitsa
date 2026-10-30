#!/usr/bin/env python3
"""A stand-in for `splash serve-native` (protocol v11, stdlib only) for tests/test_splash_engine.py: after the
positionals of serve-native it takes --reply IDS (comma-separated token ids it answers every request with, one frame per
token, honouring max_output_tokens), --prompt-log FILE (each request's prompt ids, sampling, seed and flags, one JSON line), --slow MS
(a pause before every token, for the cancel test), --pid-file FILE (its pid, for the test that kills it) and --mask-words N
(the words per mask row it asks for). A request with the TOKEN_MASK constraint asks for a mask before every token, with
the reply's next two tokens as simulation tokens (row i masks the reply's i-th next token), and sends the reply's token when row 0 allows it, else row 0's lowest
allowed token; each row's allowed count and whether it allows the reply's token go to --mask-log FILE. EOF on stdin ends it."""
import json, os, select, struct, sys, time

args = sys.argv[1:]
assert args[0] == "serve-native"
reply, log, slow, mask_words, mask_log = [], None, 0.0, 0, None
i = 4
while i < len(args):
    if args[i] == "--reply": reply = [int(x) for x in args[i + 1].split(",")]
    elif args[i] == "--prompt-log": log = args[i + 1]
    elif args[i] == "--slow": slow = int(args[i + 1]) / 1000
    elif args[i] == "--mask-words": mask_words = int(args[i + 1])
    elif args[i] == "--mask-log": mask_log = args[i + 1]
    elif args[i] == "--pid-file":
        with open(args[i + 1], "w") as f: f.write(str(os.getpid()))
    i += 2
out, inp = sys.stdout.buffer, sys.stdin.buffer

def frame(t, payload):
    out.write(b"SPLH" + struct.pack("<HHQ", 11, t, len(payload)) + payload)
    out.flush()

def read(n):
    b = b""
    while len(b) < n:
        c = inp.read(n - len(b))
        if not c: sys.exit(0)
        b += c
    return b

def done(rid, reason, n):
    frame(0x0104, struct.pack("<QBBIIQQQIIIII", rid, reason, 0, 0, n, 1000, 2000, 3000, 0, 0, 0, 0, 0))

frame(0x0100, struct.pack("<IIB", 2, 4096, 0))
cancelled = set()
pending = []
mask_id = 0

def allowed(row, t): return t < len(row) * 32 and (row[t // 32] >> (t % 32)) & 1

def first_allowed(row):
    for w, x in enumerate(row):
        if x: return w * 32 + ((x & -x).bit_length() - 1)
    return None

def ask_mask(rid, sim):
    """MASK_REQUEST, then the MASK_RESPONSE's rows; None when the request was cancelled instead."""
    global mask_id
    mask_id += 1
    frame(0x0103, struct.pack("<QQII", rid, mask_id, mask_words, len(sim)) + struct.pack("<%dI" % len(sim), *sim))
    while True:
        h = read(16)
        _, _, t2, n2 = struct.unpack("<4sHHQ", h)
        p2 = read(n2)
        if t2 == 0x0002 and struct.unpack_from("<Q", p2, 0)[0] == rid: return None
        if t2 != 0x0003: continue
        r2, m2, count = struct.unpack_from("<QQI", p2, 0)
        assert (r2, m2) == (rid, mask_id) and count == mask_words * (len(sim) + 1), (r2, m2, count)
        words = struct.unpack_from("<%dI" % count, p2, 20)
        return [words[k * mask_words:(k + 1) * mask_words] for k in range(len(sim) + 1)]
while True:
    h = read(16)
    magic, ver, typ, n = struct.unpack("<4sHHQ", h)
    p = read(n)
    if typ == 0x0001:
        rid = struct.unpack_from("<Q", p, 0)[0]
        constrained = p[9] == 1
        # fixed part: id(8) priority(1) constraint(1) deadlines(16), then u32 logical_max, prompt_count, span_count, ...
        logical_max, count = struct.unpack_from("<II", p, 26)
        ids = list(struct.unpack_from("<%dI" % count, p, 114))
        if log:
            salt = struct.unpack_from("<QQ", p, 86)
            temp, top_p, top_k, pres, freq, rep, min_p = struct.unpack_from("<ffIffff", p, 38)
            seed = struct.unpack_from("<Q", p, 66)[0]
            flags = struct.unpack_from("<I", p, 82)[0]
            sampling = dict(temperature=temp, top_p=top_p, top_k=top_k, presence=pres, frequency=freq, repetition=rep, min_p=min_p)
            with open(log, "a") as f: f.write(json.dumps({"max": logical_max, "prompt": ids, "sampling": sampling, "seed": seed, "flags": flags, "constraint": p[9], "salt": list(salt), "args": args}) + "\n")
        frame(0x0101, struct.pack("<QII", rid, 0, 0))
        sent, off = 0, 0
        for k, t in enumerate(reply[:logical_max]):
            if constrained:
                sim = reply[k:k + 2]  # the draft: this token and the next; row i masks reply[k + i]
                rows = ask_mask(rid, sim)
                if rows is None:
                    cancelled.add(rid)
                    break
                if mask_log:
                    with open(mask_log, "a") as f:
                        f.write(json.dumps({"token": t, "sim": sim, "rows": [[sum(bin(x).count("1") for x in r), bool(allowed(r, u))]
                                                                         for r, u in zip(rows, reply[k:k + 3])]}) + "\n")
                if not allowed(rows[0], t): t = first_allowed(rows[0])
            if slow:
                end = time.time() + slow
                while time.time() < end:
                    time.sleep(0.005)
            frame(0x0102, struct.pack("<QII", rid, off, 1) + struct.pack("<I", t))
            off += 1
            sent += 1
            # a cancel that arrives meanwhile: poll stdin between tokens
            while select.select([inp], [], [], 0)[0]:
                h2 = read(16)
                _, _, t2, n2 = struct.unpack("<4sHHQ", h2)
                p2 = read(n2)
                if t2 == 0x0002 and struct.unpack_from("<Q", p2, 0)[0] == rid: cancelled.add(rid)
            if rid in cancelled: break
        if rid in cancelled: done(rid, 2, sent)
        else: done(rid, 1 if sent >= logical_max and sent < len(reply) else 0, sent)
    elif typ == 0x0004:
        frame(0x0107, struct.pack("<Q", struct.unpack_from("<Q", p, 0)[0]) + b"{}")
