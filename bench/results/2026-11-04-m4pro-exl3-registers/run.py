"""Reproduce the EXL3 register experiment using Kapitsa's GPU lock and contention monitor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "bench/harness"))
from gpu_hogs import ContentionMonitor, held_by_caller, summarize_samples, wait_quiet

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("mode", choices=("small", "check", "real", "bench", "realbench", "resources", "stockreal"))
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--weights", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
if not held_by_caller():
    p.error("run under .dsh/skills/verify-kapitsa/scripts/vk gpu run")
if a.mode in ("real", "realbench", "stockreal") and a.weights is None:
    p.error("this mode requires --weights LAYER0_DIR")
env = dict(os.environ)
for key in ("NI_PROFILE", "NI_DUP", "BENCH_GPU_AMBIENT", "NI_MATH_SAFE", "NI_KV_I8"):
    if env.get(key):
        p.error(f"unexpected override: {key}")
env.update(MTL_SHADER_VALIDATION="1" if a.mode in ("small", "real", "stockreal") else "0",
           MTL_DEBUG_LAYER="1" if a.mode in ("small", "real", "stockreal") else "0")
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=False)
binary = root / "build" / ("exl3_check" if a.mode == "stockreal" else "exl3_register_check")
command = [str(binary)]
if a.mode == "stockreal":
    command += ["--pack", str(a.weights.resolve())]
else:
    command += [str(a.reference.resolve()), a.mode]
    if a.mode in ("real", "realbench"):
        command += [str(a.weights.resolve())]
if a.mode == "resources":
    (out / "archives").mkdir()
    env["NI_ARCHIVE_DIR"] = str(out / "archives")
driver = root / "tools/exl3_register_check.cpp"
if not driver.exists():
    driver = Path(__file__).with_name("exl3_register_check.cpp")
sources = [root / "src/kernels/exl3moe.metal", root / "tools/exl3_check.cpp",
           root / "src/model/moe_exl3.cpp", driver, a.reference.resolve(), binary]
record = {"started": time.time(), "mode": a.mode, "command": command, "timing_eligible": False,
          "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
          "os": subprocess.check_output(["sw_vers"], text=True),
          "power": subprocess.check_output(["pmset", "-g", "batt"], text=True),
          "hashes": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
          "env": {k: v for k, v in env.items() if k.startswith(("NI_", "MTL_", "BENCH_GPU_"))}}
timing = a.mode in ("bench", "realbench")
if timing:
    quiet, waited, busy = wait_quiet(maxpct=2, quiet_s=10, timeout_s=35, exclude_roots=[os.getpid()])
    record.update(quiet=quiet, quiet_wait_s=waited, last_busy=busy)
    if not quiet:
        record.update(returncode=90, reason="GPU not quiet; no timings launched")
        (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        raise SystemExit(90)
owned = [os.getpid()]
monitor = ContentionMonitor(lambda: owned, interval=0.5)
monitor.start()
try:
    with (out / "run.log").open("w") as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        owned.append(child.pid)
        rc = child.wait()
finally:
    monitor.stop()
summary = summarize_samples(monitor.samples)
record.update(returncode=rc, elapsed_s=time.time() - record["started"],
              contention=summary, samples=monitor.samples)
record["timing_eligible"] = (timing and rc == 0 and summary.get("samples", 0) >= 2
    and not summary.get("flagged", True) and summary.get("busy_samples", 1) == 0
    and summary.get("mean_foreign_gpu_pct", 100) < 2 and summary.get("mean_ambient_gpu_pct", 100) < 2)
(out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: record[k] for k in ("mode", "returncode", "contention", "timing_eligible")}))
raise SystemExit(rc)
