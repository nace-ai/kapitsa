#!/usr/bin/env python3
"""Startup measurements: the serialized runner with 100 ms telemetry and desktop in the quiet gate."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent


def user_idle_seconds():
    text = subprocess.check_output(['/usr/sbin/ioreg', '-c', 'IOHIDSystem', '-d', '1', '-r'], text=True)
    match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', text)
    if not match:
        raise RuntimeError('Cannot determine user inactivity; no GPU work may start.')
    return int(match.group(1)) / 1e9


def availability(args):
    now = datetime.datetime.now(datetime.timezone.utc)
    first = datetime.datetime.fromisoformat(args.not_before)
    last = datetime.datetime.fromisoformat(args.latest_start)
    idle = user_idle_seconds()
    reason = ('before requested window' if now < first else
              'past latest new-experiment start' if now >= last else
              'user has used this Mac recently' if idle < args.user_idle else None)
    return {'epoch': time.time(), 'user_idle_seconds': idle, 'available': reason is None, 'reason': reason}


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'interrupted by signal {signum}; cleaning up this run')


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def snapshot():
    return json.loads(subprocess.check_output([str(ROOT / "observe"), "1"], text=True))


def delta(a, b, own=()):
    old = {(x["pid"], x["name"]): x["ns"] for x in a["clients"]}
    dt = b["mono"] - a["mono"]
    rows = []
    for x in b["clients"]:
        key = (x["pid"], x["name"])
        # A new client has no interval baseline: do not mistake its lifetime for this interval.
        if key not in old:
            continue
        pct = max(0, x["ns"] - old[key]) / dt / 1e9 * 100
        if pct:
            rows.append({"pid": x["pid"], "name": x["name"], "pct": pct})
    foreign = sum(x["pct"] for x in rows if x["pid"] not in own and x["name"] != "WindowServer")
    desktop = sum(x["pct"] for x in rows if x["name"] == "WindowServer")
    return {"epoch": b["epoch"], "foreign_pct": foreign, "desktop_pct": desktop,
            "thermal": b["thermal"], "swap_mb": b["swap_mb"], "swapouts": b["swapouts"],
            "memory_level": b["memory_level"], "users": sorted(rows, key=lambda x: -x["pct"])}


def descendants(root_pid):
    pairs = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    children = {}
    for row in pairs.splitlines():
        pid, ppid = map(int, row.split())
        children.setdefault(ppid, []).append(pid)
    result, todo = set(), [root_pid]
    while todo:
        pid = todo.pop()
        if pid not in result:
            result.add(pid)
            todo.extend(children.get(pid, []))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--quiet", type=float, default=8)
    p.add_argument("--max-foreign", type=float, default=1)
    p.add_argument("--idle-timeout", type=float, default=180)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--model")
    p.add_argument("--sample-ms", type=int, default=100)
    p.add_argument('--not-before', default='2026-10-28T23:00:00+08:00')
    p.add_argument('--latest-start', default='2026-10-29T07:30:00+08:00')
    p.add_argument('--user-idle', type=float, default=300)
    p.add_argument('--sleep-display', action='store_true', help='authorized for idle performance measurements')
    if "--" not in sys.argv:
        p.error("a command after -- is required")
    split = sys.argv.index("--")
    args = p.parse_args(sys.argv[1:split])
    command = sys.argv[split + 1:]
    if not command:
        p.error("a command after -- is required")
    preflight = availability(args)
    preflight['label'] = args.label
    preflight_dir = ROOT / 'evidence' / 'preflights'
    preflight_dir.mkdir(parents=True, exist_ok=True)
    preflight_file = preflight_dir / f'{time.time_ns()}.json'
    save(preflight_file, preflight)
    if not preflight['available']:
        print(json.dumps(preflight), flush=True)
        return 75
    signal.signal(signal.SIGTERM, interrupted)
    dest = ROOT / "evidence" / "legs" / args.label
    dest.mkdir(parents=True, exist_ok=False)
    overrides = dict(x.split("=", 1) for x in args.env)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "NACE_", "KAPITSA_", "MTL_"))}
    env.update({"KAPITSA_HOME": str(dest / "home"), "KAPITSA_NO_AUTO_UPDATE": "1", "KAPITSA_NO_PROMPT": "1",
                "NI_LOCK_HELD": "1", "NACE_GPU_LOCK_HELD": "1"})
    env.update(overrides)
    Path(env["KAPITSA_HOME"]).mkdir(parents=True, exist_ok=True)
    binary = Path(command[0]).resolve()
    meta = {"command": command, "cwd": os.getcwd(), "env": {k: v for k, v in env.items()
            if k.startswith(("NI_", "NACE_", "KAPITSA_", "MTL_", "BENCH_")) or k in overrides}, "label": args.label,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "quiet_seconds": args.quiet, "max_foreign_pct": args.max_foreign,
            "sample_ms": args.sample_ms, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(dest / "command.json", meta)
    lock_path = os.path.expanduser(os.environ.get("BENCH_GPU_LOCK") or "/tmp/kapitsa-gpu.lock")
    lock_fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o644)
    child = observer = keepawake = None
    deadline = time.monotonic() + args.timeout
    try:
        print(f"{args.label}: waiting for GPU lock", flush=True)
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError("GPU lock wait exceeded deadline")
                time.sleep(1)
        # The user may have returned during a lock wait. Do not start another experiment then.
        if not availability(args)['available']:
            raise RuntimeError('User/window availability changed while waiting for the GPU lock')
        keepawake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        if args.sleep_display:
            subprocess.run(['/usr/bin/pmset', 'displaysleepnow'], check=True)
        initial = snapshot()
        save(dest / "before.json", initial)
        quiet_since = time.monotonic()
        idle_start = quiet_since
        a = initial
        with (dest / "idle.jsonl").open("w") as f:
            while time.monotonic() - quiet_since < args.quiet:
                time.sleep(1)
                b = snapshot()
                row = delta(a, b)
                f.write(json.dumps(row) + "\n")
                f.flush()
                if row["foreign_pct"] + row["desktop_pct"] > args.max_foreign or b["thermal"] != "nominal":
                    quiet_since = time.monotonic()
                a = b
                if not availability(args)['available']:
                    raise RuntimeError('User/window availability changed before the experiment started')
                if time.monotonic() - idle_start > args.idle_timeout:
                    raise TimeoutError("GPU did not become quiet; no experiment launched")
        print(f"{args.label}: GPU quiet, running", flush=True)
        start = time.time()
        child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, errors="replace", start_new_session=True)
        save(dest / "pid.json", {"pid": child.pid, "started_epoch": start, "lock_pid": os.getpid()})
        lines, samples, threads = [], [], []
        events_lock = threading.Lock()
        with (dest / "events.jsonl").open("w") as events:
            def capture(stream, name):
                with (dest / (name + ".txt")).open("w") as f:
                    for line in stream:
                        epoch = time.time()
                        f.write(line)
                        f.flush()
                        with events_lock:
                            events.write(json.dumps({"epoch": epoch, "stream": name, "line": line}) + "\n")
                            events.flush()
                            lines.append((epoch, name, line))

            for stream, name in [(child.stdout, "stdout"), (child.stderr, "stderr")]:
                t = threading.Thread(target=capture, args=(stream, name), daemon=True)
                t.start()
                threads.append(t)
            observer = subprocess.Popen([str(ROOT / "observe"), "0", str(args.sample_ms)], stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True)

            def monitor():
                last = a
                own = {child.pid}
                last_idle = preflight['user_idle_seconds']
                with (dest / "telemetry.jsonl").open("w") as f:
                    for i, line in enumerate(observer.stdout):
                        b = json.loads(line)
                        own |= descendants(child.pid)
                        row = delta(last, b, own)
                        row["owned_pids"] = sorted(own)
                        if i % max(1, 1000 // args.sample_ms) == 0:
                            last_idle = user_idle_seconds()
                        row['user_idle_seconds'] = last_idle
                        samples.append(row)
                        f.write(json.dumps({"raw": b, "delta": row}) + "\n")
                        f.flush()
                        last = b

            monitor_errors = []
            def monitored():
                try:
                    monitor()
                except BaseException as exc:
                    monitor_errors.append(str(exc) or type(exc).__name__)
            mt = threading.Thread(target=monitored, daemon=True)
            mt.start()
            try:
                rc = child.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise TimeoutError("experiment timed out; stopped only its own process group")
            observer.terminate()
            observer.wait(timeout=10)
            mt.join(timeout=10)
            if mt.is_alive() or monitor_errors or not samples:
                raise RuntimeError('Telemetry incomplete: ' + repr(monitor_errors))
            for t in threads:
                t.join(timeout=10)
        ended = time.time()
        after = snapshot()
        save(dest / "after.json", after)
        result = {"label": args.label, "returncode": rc, "wall_seconds": ended - start,
                  "foreign_mean_pct": sum(x["foreign_pct"] for x in samples) / max(1, len(samples)),
                  "foreign_peak_pct": max((x["foreign_pct"] for x in samples), default=None),
                  "thermal_states": sorted({x["thermal"] for x in samples}),
                  "swapouts_delta": after["swapouts"] - initial["swapouts"], "samples": len(samples)}
        result['user_activity_observed'] = any(x['user_idle_seconds'] < args.user_idle for x in samples)
        text = (dest / "stdout.txt").read_text()
        ids_match = re.search(r"^ids:([^\n]*)", text, re.M)
        if ids_match:
            ids = [int(x) for x in ids_match[1].split()]
            eos = [i for i, x in enumerate(ids) if x in (248044, 248046)]
            save(dest / "ids.json", ids)
            result.update({"output_tokens": len(ids), "eos_count": len(eos),
                           "tokens_before_eos": eos[0] if eos else len(ids),
                           "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
            if args.model:
                decoded = subprocess.check_output([str(ROOT / "tokens"), "decode", args.model,
                                                   str(dest / "ids.json")], text=True)
                (dest / "decoded.txt").write_text(decoded)
                result["nonempty_text"] = bool(decoded.strip())
        summaries = [line.strip() for _, name, line in lines if name == "stderr" and
                     (line.startswith("prompt ") or line.startswith("cycle trace") or line.startswith("drafting pass"))]
        result["summaries"] = summaries
        for epoch, _, line in lines:
            if line.startswith("prompt ") and ";" in line:
                match = re.search(r"prompt (\d+) tok in ([\d.]+) ms \(([\d.]+) tok/s\).*decode.*?[ :] (\d+) tok in ([\d.]+) ms \(([\d.]+) tok/s\)", line)
                if match:
                    result.update(dict(zip(["prompt_tokens", "prefill_ms", "prefill_tps", "timed_tokens", "decode_ms", "decode_tps"], map(float, match.groups()))))
                    decode_start = epoch - result["decode_ms"] / 1000
                    ds = [x for x in samples if decode_start <= x["epoch"] <= epoch]
                    result["decode_foreign_mean_pct"] = sum(x["foreign_pct"] for x in ds) / max(1, len(ds))
                    result["decode_foreign_peak_pct"] = max((x["foreign_pct"] for x in ds), default=None)
        save(dest / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return rc
    except BaseException as e:
        save(dest / "error.json", {"type": type(e).__name__, "message": str(e)})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if observer is not None and observer.poll() is None:
            observer.terminate()
            observer.wait(timeout=10)
        if keepawake is not None and keepawake.poll() is None:
            keepawake.terminate()
            keepawake.wait(timeout=10)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
