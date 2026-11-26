#!/usr/bin/env python3
"""Every check of Inkling-2's image input on this Mac, in one run, for a second setup's evidence (#1419).

usage: ink2_evidence.py MODEL_DIR --python PY [--build build] [--profile precise|fast] [--out FILE]
  MODEL_DIR  an Inkling-2-Mini folder with its vision tower (thinkingmachines/Inkling-2-Mini; `kapitsa pull inkling2`)
  PY         a Python with torch, torchvision, transformers 5.24 or later, Pillow and numpy for the reference:
             python3 -m venv v && v/bin/pip install torch torchvision transformers==5.24.0 pillow numpy
Builds nothing: build kapitsa, vit_check, ink2_check and vision_pixels_test first. Runs, in order:
  1. CPU: build/vision_pixels_test (the size cap, the transparency composite), tests/test_gpu_lock_held.py
  2. GPU, under $BENCH_GPU_LOCK (default /tmp/kapitsa-gpu.lock) held for the whole leg, children with NI_LOCK_HELD=1:
     a. tests/vision/check_ink2.py --gpu on tests/fixtures/vision/ink2_*.png: the processor against Hugging Face's
        bit for bit, the tower against Inkling2VisionModel in float32 (the reference runs on the CPU)
     b. build/ink2_check MODEL_DIR: text-only parity against tests/fixtures/ink2_golden.json (the old path unchanged)
     c. MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 vit_check tower on a 61x45 transparent fixture
     d. `kapitsa serve` on a free port of 127.0.0.1: a greedy answer to "what is in this image" for five fixtures,
        each checked for words the picture must produce
Prints a `Runs:` line (Mac, chip, GPU cores, memory, macOS, commit) and one PASS/FAIL line per check; --out FILE saves
the whole transcript. The serve leg needs the model's memory (about 19 GB resident: a 32 GB Mac or larger).
"""
import base64, json, os, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "bench", "harness"))
import gpu_hogs  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv[:-1] else default


if not args or not opt("--python"):
    sys.exit(__doc__)
MODEL, PY = os.path.abspath(args[0]), opt("--python")
BUILD = os.path.abspath(opt("--build", os.path.join(REPO, "build")))
PROFILE, OUT = opt("--profile", "precise"), opt("--out")
FIX = os.path.join(REPO, "tests", "fixtures", "vision")
log_lines, results = [], []


def say(s=""):
    print(s, flush=True)
    log_lines.append(s)


def sh(cmd, env=None, timeout=3600):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, timeout=timeout, cwd=REPO)
    return r.returncode, (r.stdout + r.stderr).strip()


def record(name, ok, detail=""):
    results.append((name, ok))
    say(("PASS " if ok else "FAIL ") + name + (": " + detail if detail else ""))


def runs_line():
    def q(*c):
        try:
            return subprocess.run(c, capture_output=True, text=True).stdout.strip()
        except OSError:
            return "?"
    hw = q("system_profiler", "SPHardwareDataType")
    model = next((ln.split(":", 1)[1].strip() for ln in hw.splitlines() if "Model Name" in ln), "?")
    cores = next((ln.split(":", 1)[1].strip() for ln in q("system_profiler", "SPDisplaysDataType").splitlines()
                  if "Total Number of Cores" in ln), "?")
    mem = int(q("sysctl", "-n", "hw.memsize") or 0) / 2**30
    return (f"Runs: {model}, {q('sysctl', '-n', 'machdep.cpu.brand_string')}, {cores} GPU cores, {mem:.0f} GB, "
            f"macOS {q('sw_vers', '-productVersion')}, commit {q('git', '-C', REPO, 'rev-parse', '--short', 'HEAD')}, "
            f"profile {PROFILE}")


say(runs_line())
fixtures = sorted(os.path.join(FIX, f) for f in os.listdir(FIX) if f.startswith("ink2_") and f.endswith(".png"))

# ---- 1. CPU ----
rc, out = sh([os.path.join(BUILD, "vision_pixels_test")])
record("vision_pixels_test", rc == 0, out.splitlines()[-1] if out else "")
rc, out = sh([sys.executable, os.path.join(REPO, "tests", "test_gpu_lock_held.py")])
record("test_gpu_lock_held.py", rc == 0, out.splitlines()[-1] if out else "")

# ---- 2. GPU, under the lock ----
lock = None
if not gpu_hogs.held_by_caller():
    gpu_hogs.configure()
    lock = gpu_hogs.GpuLock().acquire()
env = dict(os.environ, NI_LOCK_HELD="1")
server = None
try:
    rc, out = sh([PY, "-I", os.path.join(HERE, "check_ink2.py"), MODEL, os.path.join(BUILD, "vit_check"), *fixtures,
                  "--gpu", "--python", PY], env=env)
    say("\n".join(ln for ln in out.splitlines() if not ln.startswith(("W1", "[kapitsa]"))))
    record("check_ink2.py --gpu", rc == 0, f"{len(fixtures)} images")

    rc, out = sh([os.path.join(BUILD, "ink2_check"), MODEL], env=env)
    for ln in out.splitlines():
        if ln.startswith(("PASS", "FAIL")):
            say("  " + ln)
    record("ink2_check (text-only parity)", rc == 0 and "FAIL" not in out)

    rc, out = sh([os.path.join(BUILD, "vit_check"), "tower", MODEL, os.path.join(FIX, "ink2_rgba.png"),
                  os.path.join(BUILD, "ink2_validation.bin")],
                 env=dict(env, MTL_SHADER_VALIDATION="1", MTL_DEBUG_LAYER="1"))
    record("tower under MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1", rc == 0, "" if rc == 0 else out[-300:])

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([os.path.join(BUILD, "kapitsa"), "serve", "--model", MODEL, "--context", "16k", "--host",
                               "127.0.0.1", "--port", str(port), "--private", "--profile", PROFILE], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO)
    base, t0 = f"http://127.0.0.1:{port}", time.time()
    while time.time() - t0 < 900:
        try:
            if b"ok" in urllib.request.urlopen(base + "/health", timeout=2).read():
                break
        except OSError:
            time.sleep(3)
    want = {"ink2_dialog.png": ("settings", "dark mode", "cancel"), "ink2_scene.png": ("house", "tree", "sun"),
            "ink2_tall.png": ("15", "item"), "ink2_logo_rgba.png": ("red", "circle"), "ink2_rgba.png": ()}
    for name, words in want.items():
        url = "data:image/png;base64," + base64.b64encode(open(os.path.join(FIX, name), "rb").read()).decode()
        body = {"model": "x", "max_tokens": 300, "temperature": 0, "stream": False, "reasoning_effort": "low",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                          {"type": "text", "text": "What is in this image? Describe it briefly."}]}]}
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=900))
            text = r["choices"][0]["message"].get("content") or ""
            missing = [w for w in words if w not in text.lower()]
            record(f"chat {name}", bool(text) and not missing,
                   f"{r['usage']['prompt_tokens']} prompt tokens: {text!r}" + (f" (missing {missing})" if missing else ""))
        except Exception as e:  # noqa: BLE001 (one request's failure is reported, the others still run)
            record(f"chat {name}", False, str(e)[:300])
finally:
    if server:
        server.terminate()
        try:
            out, _ = server.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            out, _ = server.communicate()
        for ln in out.splitlines():
            if "vision:" in ln or "images:" in ln:
                say("  " + ln.strip())
    if lock:
        lock.release()
    try:
        os.remove(os.path.join(BUILD, "ink2_validation.bin"))
    except OSError:
        pass

failed = [n for n, ok in results if not ok]
say(f"\n{len(results) - len(failed)} of {len(results)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
if OUT:
    with open(OUT, "w") as f:
        f.write("\n".join(log_lines) + "\n")
sys.exit(1 if failed else 0)
