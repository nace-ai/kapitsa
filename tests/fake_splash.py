#!/usr/bin/env python3
"""A stand-in for `splash serve-native` (protocol v11, stdlib only) for tests/test_splash_engine.py: after the
positionals of serve-native it takes --reply IDS (comma-separated token ids it answers every request with, one frame per
token, honouring max_output_tokens), --prompt-log FILE (each request's prompt ids, sampling and flags, one JSON line) and --slow MS (a pause
before every token, for the cancel test). EOF on stdin ends it."""
import json, select, struct, sys, time

args = sys.argv[1:]
assert args[0] == "serve-native"
reply, log, slow = [], None, 0.0
i = 4
while i < len(args):
    if args[i] == "--reply": reply = [int(x) for x in args[i + 1].split(",")]
    elif args[i] == "--prompt-log": log = args[i + 1]
    elif args[i] == "--slow": slow = int(args[i + 1]) / 1000
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
while True:
    h = read(16)
    magic, ver, typ, n = struct.unpack("<4sHHQ", h)
    p = read(n)
    if typ == 0x0001:
        rid = struct.unpack_from("<Q", p, 0)[0]
        # fixed part: id(8) priority(1) constraint(1) deadlines(16), then u32 logical_max, prompt_count, span_count, ...
        logical_max, count = struct.unpack_from("<II", p, 26)
        ids = list(struct.unpack_from("<%dI" % count, p, 114))
        if log:
            temp, top_p, top_k, pres, freq, rep, min_p = struct.unpack_from("<ffIffff", p, 38)
            flags = struct.unpack_from("<I", p, 82)[0]
            sampling = dict(temperature=temp, top_p=top_p, top_k=top_k, presence=pres, frequency=freq, repetition=rep, min_p=min_p)
            with open(log, "a") as f: f.write(json.dumps({"max": logical_max, "prompt": ids, "sampling": sampling, "flags": flags}) + "\n")
        frame(0x0101, struct.pack("<QII", rid, 0, 0))
        sent, off = 0, 0
        for t in reply[:logical_max]:
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
