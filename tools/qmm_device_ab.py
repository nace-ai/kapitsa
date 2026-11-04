#!/usr/bin/env python3
"""Reviewable, guarded A/B for the 4B packed-QMM device default (macOS, stdlib).

Each leg records fresh load/AC/thermal/VM snapshots before and after it, and a
continuous unfiltered foreign-GPU/CPU/environment history. Raw output stays in
the caller's new output directory; summary.json is the compact review artifact.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PAGING = ("pageouts", "swapins", "swapouts")


def remaining(args, limit):
    seconds = min(limit, args.deadline-time.time()) if args.deadline else limit
    if seconds <= 0:
        raise TimeoutError("campaign deadline reached")
    return seconds


def snapshot(probe):
    result = subprocess.run([str(probe)], capture_output=True, text=True, check=True, timeout=2)
    value = json.loads(result.stdout)
    if not value.get("valid"):
        raise RuntimeError("invalid machine snapshot")
    return value


def environment_errors(value, max_load):
    errors = []
    if not value.get("valid"):
        errors.append("invalid_snapshot")
    if value.get("ac") is not True or value.get("power_source") != "AC Power":
        errors.append("not_AC")
    if value.get("low_power") is not False:
        errors.append("low_power_mode")
    if value.get("thermal") != 0:
        errors.append("thermal_not_nominal")
    load = value.get("load1", math.inf)
    if not math.isfinite(load) or not 0 <= load <= max_load:
        errors.append("load1_over_limit")
    if value.get("memory_pressure_level") != 1:
        errors.append("memory_pressure")
    for key in PAGING+("monotonic_ns",):
        if not isinstance(value.get(key), int) or value[key] < 0:
            errors.append("missing_"+key)
    return errors


def validate_leg(before, after, samples, max_load=2.0, min_idle=90.0, max_gap=1.0, min_samples=20):
    errors = ["before:"+x for x in environment_errors(before, max_load)]
    errors += ["after:"+x for x in environment_errors(after, max_load)]
    for key in PAGING:
        if after.get(key, -1) != before.get(key, -1):
            errors.append(key+"_changed")
    if after.get("monotonic_ns", -1) <= before.get("monotonic_ns", -1):
        errors.append("snapshot_time_not_increasing")
    if len(samples) < min_samples:
        errors.append("insufficient_monitor_samples")
    previous_time = -math.inf
    for sample in samples:
        if sample.get("error"):
            errors.append("monitor_error")
            continue
        errors += environment_errors(sample, max_load)
        if sample.get("cpu_idle_pct", 0) < min_idle:
            errors.append("cpu_not_idle")
        if sample.get("foreign_gpu_ns", 0) > 0:
            errors.append("foreign_GPU_activity")
        if "foreign_gpu_ns" not in sample or not sample.get("own_gpu_visible"):
            errors.append("missing_GPU_counters")
        if sample.get("gpu_counter_reset"):
            errors.append("GPU_counter_reset")
        if not 0 < sample.get("interval_s", max_gap+1) <= max_gap:
            errors.append("monitor_gap")
        if sample.get("time", -math.inf) <= previous_time:
            errors.append("monitor_time_not_increasing")
        previous_time = sample.get("time", -math.inf)
        for key in PAGING:
            if sample.get(key, -1) != before.get(key, -1):
                errors.append(key+"_changed")
    return sorted(set(errors))


def gpu_snapshot():
    # Same accumulatedGPUTime source as gpu_hogs.snap(), but fail on a failed or
    # empty query. Do not apply its 0.5% display filter or ambient exemptions.
    out = subprocess.run(["ioreg", "-r", "-c", "AGXDeviceUserClient", "-l", "-w", "0"],
                         capture_output=True, text=True, check=True, timeout=2).stdout
    values = {}
    for block in out.split("+-o AGXDeviceUserClient")[1:]:
        match = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"', block)
        if match:
            key = (int(match[1]), match[2])
            values[key] = values.get(key, 0)+sum(int(x) for x in re.findall(r'"accumulatedGPUTime"=(\d+)', block))
    if not values:
        raise RuntimeError("no AGX process counters returned")
    return values


def cpu_usage(history, item):
    # HOST_CPU_LOAD_INFO may return the same cached ticks to adjacent 250 ms
    # probes. Use a one-second window; zero new ticks is unknown, not 0% idle.
    history.append(item)
    while len(history) > 1 and item["monotonic_ns"]-history[1]["monotonic_ns"] >= 1000000000:
        history.pop(0)
    first = history[0]
    window = (item["monotonic_ns"]-first["monotonic_ns"])/1e9
    ticks = [(x-y) & 0xffffffff for x, y in zip(item["cpu_ticks"], first["cpu_ticks"])]
    return {"cpu_idle_pct": 100*ticks[2]/sum(ticks) if window >= 1 and sum(ticks) else -1,
            "cpu_window_s": window}


class Monitor(threading.Thread):
    def __init__(self, probe, pid, gpu, interval):
        super().__init__(daemon=True)
        self.probe, self.pid, self.gpu, self.interval = probe, pid, gpu, interval
        self.samples, self.lock, self.stopping = [], threading.Lock(), threading.Event()

    def run(self):
        previous = previous_gpu = previous_time = None
        cpu_history = []
        while not self.stopping.is_set():
            started = time.monotonic()
            try:
                item = snapshot(self.probe)
                current_gpu = gpu_snapshot()
                own = self.gpu.proc_tree([self.pid])
                now = time.monotonic()
                item.update(time=now, epoch=time.time())
                item.update(cpu_usage(cpu_history, item))
                if previous is not None:
                    item["interval_s"] = now-previous_time
                    foreign, own_ns, reset = [], 0, False
                    for (pid, name), value in current_gpu.items():
                        old = previous_gpu.get((pid, name), 0)
                        delta = value-old
                        if delta < 0:
                            reset = True
                        if pid in own:
                            own_ns += max(0, delta)
                        elif name != "WindowServer" and delta > 0:
                            foreign.append({"pid": pid, "name": name, "gpu_ns": delta})
                    vanished = [k for k in previous_gpu if k not in current_gpu and k[1] != "WindowServer"]
                    item.update(foreign_gpu=foreign, foreign_gpu_ns=sum(x["gpu_ns"] for x in foreign),
                                own_gpu_ns=own_ns, own_gpu_visible=any(k[0] in own for k in current_gpu),
                                gpu_counter_reset=reset or bool(vanished), gpu_processes_lost=vanished)
                    with self.lock:
                        self.samples.append(item)
                previous, previous_gpu, previous_time = item, current_gpu, now
            except Exception as exc:
                with self.lock:
                    self.samples.append({"time": time.monotonic(), "epoch": time.time(), "error": repr(exc)})
            self.stopping.wait(max(0, self.interval-(time.monotonic()-started)))

    def read(self):
        with self.lock:
            return list(self.samples)

    def stop(self):
        self.stopping.set()
        self.join(timeout=10)


def window_samples(monitor, start, end, args):
    # Retain the intervals straddling both boundaries, including the last GPU
    # counter delta after the HTTP request completes. Never leave an unobserved tail.
    until = time.monotonic()+remaining(args, 2)
    while time.monotonic() < until:
        samples = monitor.read()
        if samples and samples[-1]["time"] >= end:
            return [s for s in samples if s["time"] >= start and s["time"]-s.get("interval_s", 0) <= end]
        time.sleep(.02)
    raise RuntimeError("monitor did not cover the end of the measurement")


def wait_quiet(monitor, args):
    began = time.monotonic()
    while time.monotonic()-began < args.quiet_timeout:
        remaining(args, 1)
        now = time.monotonic()
        samples = [x for x in monitor.read() if x["time"] >= now-args.quiet_seconds]
        if len(samples) >= 2 and samples[-1]["time"]-samples[0]["time"] >= args.quiet_seconds-args.interval-0.05:
            valid = not validate_leg(samples[0], samples[-1], samples, args.max_load, args.min_cpu_idle)
            if valid and all(x.get("own_gpu_ns", 1) == 0 for x in samples):
                return {"waited_s": now-began, "samples": samples}
        time.sleep(0.5)
    raise RuntimeError("machine did not meet the declared quiet conditions")


def power_mode():
    text = subprocess.check_output(["pmset", "-g", "custom"], text=True, timeout=2)
    ac = text.split("AC Power:", 1)[1]
    for line in ac.splitlines():
        parts = line.split()
        if parts[:1] == ["powermode"]:
            return int(parts[1])
    raise RuntimeError("AC power mode unavailable")


@contextlib.contextmanager
def server(binary, model, directory, args):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("NI_")}
    env.update(NI_LOCK_HELD="1", NACE_GPU_LOCK_HELD="1", KAPITSA_NO_AUTO_UPDATE="1", KAPITSA_NO_PROMPT="1",
               KAPITSA_PARENT_PID=str(os.getpid()), KAPITSA_HOME=str(directory/"home"))
    (directory/"home").mkdir()
    command = [str(binary), "serve", "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
               "--slots", "1", "--context", "8192", "--disk-cache", "0", "--private"]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (directory/"server.log").open("x") as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        timer = None
        if args.deadline:
            timer = threading.Timer(max(0, args.deadline-time.time()), process.terminate)
            timer.daemon = True
            timer.start()
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic()+remaining(args, 180)
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("server exited during startup")
                try:
                    with opener.open(base+"/health", timeout=1) as response:
                        healthy = response.status == 200
                    with opener.open(base+"/v1/models", timeout=2) as response:
                        models = json.load(response)
                except OSError:
                    time.sleep(0.25)
                    continue
                if healthy:
                    break
            else:
                raise RuntimeError("server startup timeout")
            yield base+"/v1", process.pid, models["data"][0]["id"]
        finally:
            if timer:
                timer.cancel()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def request_body(messages, model, temperature, count=512):
    return {"model": model, "messages": messages, "max_tokens": count, "temperature": temperature,
            "top_k": 1 if temperature == 0 else 20, "top_p": 1 if temperature == 0 else .95,
            "min_p": 0, "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False}}


def one_leg(label, binary, cases, args, harness, gpu):
    directory = args.out/label
    directory.mkdir()
    record = {"label": label, "binary": str(binary), "requests": [], "invalid": []}
    print(label+": loading and warming", flush=True)
    with server(binary, args.model, directory, args) as (base, pid, model):
        warm = [{"role": "system", "content": "[warm-up] "+harness.DECODE_SYS+"\n"+harness.build_context(2700, offset=100000)},
                {"role": "user", "content": "Summarize the first file."}]
        harness.stream_chat(base, request_body(warm, model, 0, 64), timeout=remaining(args, 300))
        warm = [{"role": "system", "content": "[warm-up coding] "+harness.DECODE_SYS},
                {"role": "user", "content": harness.DECODE_USER}]
        harness.stream_chat(base, request_body(warm, model, 0), timeout=remaining(args, 300))
        monitor = Monitor(args.probe, pid, gpu, args.interval)
        monitor.start()
        try:
            record["quiet"] = wait_quiet(monitor, args)
            start = time.monotonic()
            before = snapshot(args.probe)
            mode_before = power_mode()
            print(label+": measuring", flush=True)
            for repeat in range(args.repeats):
                for name, temperature, messages in cases:
                    evict = [{"role": "user", "content": f"[evict {repeat} {name}] "+"Say ok. "*40}]
                    harness.stream_chat(base, request_body(evict, model, 0, 1), timeout=remaining(args, 120))
                    b = snapshot(args.probe)
                    t0 = time.monotonic()
                    result = harness.stream_chat(base, request_body(messages, model, temperature), timeout=remaining(args, 300), telemetry=True)
                    t1 = time.monotonic()
                    a = snapshot(args.probe)
                    observed = window_samples(monitor, t0, t1, args)
                    count = (result["usage"] or {}).get("completion_tokens")
                    mid, whole, rate = harness.decode_rates(result, count)
                    cached = harness.cached_of(result["usage"], result["timings"])
                    distinct = harness.distinct4(result["content"])
                    invalid = validate_leg(b, a, observed, args.max_load, args.min_cpu_idle, min_samples=8)
                    if count != 512 or result["finish"] != "length" or not result["content"].strip():
                        invalid.append("not_nonempty_512_length")
                    if cached != 0:
                        invalid.append("cached_prompt")
                    if distinct is not None and distinct < .5:
                        invalid.append("repetition")
                    if harness.thinking_leak(result):
                        invalid.append("thinking_leak")
                    if whole is None or not math.isfinite(whole) or whole <= 0:
                        invalid.append("missing_decode_rate")
                    if result["ttft"] is not None:
                        decode = [s for s in observed if s["time"] >= t0+result["ttft"]]
                        if len(decode) < 8:
                            invalid.append("insufficient_decode_samples")
                    item = {"case": name, "repeat": repeat, "temperature": temperature, "start": t0, "end": t1,
                            "before": b, "after": a, "whole_decode_tps": whole, "middle_window_tps": mid,
                            "monitor_samples": len(observed),
                            "ttft_s": result["ttft"], "completion_tokens": count, "cached_tokens": cached,
                            "distinct4": distinct, "text_sha256": hashlib.sha256(result["content"].encode()).hexdigest(),
                            "invalid": invalid, "rate": rate, "stream": result}
                    record["requests"].append(item)
                    time.sleep(1)
            after = snapshot(args.probe)
            mode_after = power_mode()
            end = time.monotonic()
            samples = window_samples(monitor, start, end, args)
            record.update(before=before, after=after, samples=samples, duration_s=end-start,
                          power_mode_before=mode_before, power_mode_after=mode_after)
            record["invalid"] = validate_leg(before, after, samples, args.max_load, args.min_cpu_idle)
            if mode_before != mode_after:
                record["invalid"].append("power_mode_changed")
            if mode_before != 0 or mode_after != 0:
                record["invalid"].append("power_mode_not_automatic")
            if any(r["invalid"] for r in record["requests"]):
                record["invalid"].append("invalid_request")
        except Exception as exc:
            record["invalid"].append(repr(exc))
            record["samples"] = monitor.read()
        finally:
            monitor.stop()
    (directory/"raw.json").write_text(json.dumps(record, indent=2)+"\n")
    print(label+": "+("VALID" if not record["invalid"] else "INVALID "+str(record["invalid"])), flush=True)
    return record


def compact(leg):
    result = {k: leg[k] for k in ("label", "invalid", "duration_s", "before", "after", "power_mode_before", "power_mode_after") if k in leg}
    samples = leg.get("samples", [])
    result["monitor"] = {"samples": len(samples), "foreign_gpu_ns": sum(s.get("foreign_gpu_ns", 0) for s in samples),
                         "max_load1": max((s.get("load1", 0) for s in samples), default=None),
                         "min_cpu_idle_pct": min((s.get("cpu_idle_pct", 0) for s in samples), default=None),
                         "max_interval_s": max((s.get("interval_s", 0) for s in samples), default=None)}
    result["requests"] = [{k: r[k] for k in ("case", "repeat", "temperature", "whole_decode_tps", "middle_window_tps",
                                             "ttft_s", "completion_tokens", "cached_tokens", "distinct4", "text_sha256", "monitor_samples", "invalid")}
                          for r in leg["requests"]]
    return result


def campaign_errors(legs, planned, repeats):
    errors = []
    if len(legs) != planned:
        errors.append("incomplete_campaign")
    if any(leg["invalid"] for leg in legs):
        errors.append("invalid_leg")
    for case in ("short_greedy", "4k_sampled"):
        requests = [r for leg in legs for r in leg["requests"] if r["case"] == case]
        if len(requests) != planned*repeats:
            errors.append(case+":missing_requests")
        if len({r["text_sha256"] for r in requests}) != 1:
            errors.append(case+":outputs_differ")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "new", "model", "probe", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-load", type=float, default=2.0)
    parser.add_argument("--min-cpu-idle", type=float, default=90.0)
    parser.add_argument("--interval", type=float, default=.25)
    parser.add_argument("--quiet-seconds", type=float, default=30)
    parser.add_argument("--quiet-timeout", type=float, default=900)
    parser.add_argument("--deadline-utc")
    args = parser.parse_args()
    if args.pairs < 2 or args.repeats < 1 or not 0 < args.interval <= .25 or args.quiet_seconds < 10:
        parser.error("need >=2 pairs, >=1 repeat, 0 < interval <= .25 and >=10 quiet seconds")
    deadline = datetime.datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")) if args.deadline_utc else None
    args.deadline = deadline.timestamp() if deadline else None
    for name in ("old", "new", "model", "probe", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT/"bench/harness"))
    import bench as harness
    import gpu_hogs as gpu
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    user = "[run m4pro48-fixed] "+harness.DECODE_USER
    short = [{"role": "system", "content": harness.DECODE_SYS}, {"role": "user", "content": user}]
    long = [{"role": "system", "content": harness.DECODE_SYS+"\n\n# Repository context\n"+harness.build_context(4000)},
            {"role": "user", "content": user}]
    cases = [("short_greedy", 0, short), ("4k_sampled", .6, long)]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(corpus_sha256=harness.corpus_sha(), cases=cases,
                  rule="All planned legs must pass; any pageout/swap or foreign GPU activity invalidates a leg. WindowServer only is excluded.")
    config["sha256"] = {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() for name in ("old", "new", "probe")}
    config["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["versions"] = {name: subprocess.check_output([str(getattr(args, name)), "--version"], text=True).strip()
                          for name in ("old", "new")}
    (args.out/"protocol.json").write_text(json.dumps(config, indent=2)+"\n")
    (args.out/"pid").write_text(str(os.getpid())+"\n")
    legs = []
    with contextlib.nullcontext() if gpu.held_by_caller() else gpu.GpuLock(timeout=0):
        for pair in range(1, args.pairs+1):
            for arm, binary in (("A", args.old), ("B", args.new)):
                label = f"{arm}{pair}"
                try:
                    remaining(args, 1)
                    legs.append(one_leg(label, binary, cases, args, harness, gpu))
                except Exception as exc:
                    legs.append({"label": label, "requests": [], "invalid": [repr(exc)]})
                result = {"protocol": config, "planned_legs": args.pairs*2, "legs": [compact(x) for x in legs]}
                result["invalid"] = campaign_errors(legs, args.pairs*2, args.repeats)
                (args.out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
                if args.deadline and time.time() >= args.deadline:
                    raise SystemExit("deadline reached; partial campaign is invalid")
    print("Completed planned series; inspect all validity fields before calculating a verdict.", flush=True)
    raise SystemExit(bool(result["invalid"]))


if __name__ == "__main__":
    main()
