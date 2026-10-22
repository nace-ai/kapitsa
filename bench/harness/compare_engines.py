#!/usr/bin/env python3
"""Runs bench.py against several engine configurations, strictly one at a time. Stdlib only.

Per config, inside ONE exclusive flock span on the GPU lock (gpu_hogs.LOCK_PATH, default ~/.cache/nace-gpu.lock,
the lock `lockf -k <file> <cmd>` takes; the wait is logged):
  provenance (versions, git state, OS, power, swap) -> start the server (own process group, <out>/logs)
  -> wait until it answers -> warm-up (nonce'd ~2.9k-token prompt + 64-token decode, one tools request,
  and 3 full decodes for engines with adaptive speculation) -> wait for a quiet GPU (foreign load
  < 5% for 15 s; the server's process tree is never foreign) -> bench.py, with a contention monitor
  sampling foreign GPU load every 2 s (server tree recomputed per sample) -> stop the server and wait
  until its processes are gone, the port is closed and GPU/system memory is back near the baseline
  -> release the lock. Results go to <out>/results/<label>-<stamp>.json with provenance merged in.
When the caller already holds the lock (`lockf -k <lock file> env NACE_GPU_LOCK_HELD=1 compare_engines.py ...`,
e.g. to put a time limit on a run), the harness does not take it again; run one engine's configs per call then, so
the lock is released between engines.

usage: compare_engines.py <configs.json> [more.json ...] [--only label,glob*] [--list] [--dry-run] [--check-flags]
                          [--sizes 4000,16000] [--skip phases] [--agent-trace FILE] [--bench-args "..."] [--reverse]
                          [--quiet-timeout 300] [--out-dir DIR] [--env-file FILE]

bench.py's phases: decode, greedy, prefill, tools and the agent session (a recorded opencode session replayed against
the same server, agent_replay.py); --skip leaves phases out (--skip decode,greedy,prefill,tools: the agent session only).

Config files (configs/*.json) are lists of objects; several files are merged in order:
  label         unique id (log/result file names)          display   engine name shown in tables
  engine        kapitsa|llama.cpp|ollama|lmstudio|mlx-lm|omlx|mtplx (anything else works without version detection)
  version       version string, or "auto" (detected: git state of a Kapitsa checkout, `--version` output,
                venv package versions, LM Studio runtime manifests)
  model_name    table group, e.g. "Qwen3.6-35B-A3B"         model_path weights (recorded; a directory is sized)
  cmd           server argv, or null for an already running server. A bare command name is looked up on PATH.
  extends       label of an earlier config to inherit from; cmd_append adds server args to its cmd
  env           extra environment; env_clear_prefixes drops inherited variables (e.g. ["NI_"])
  base_url, model   OpenAI base URL and the request model id (Ollama: the tag; never /v1/models)
  enabled       false = skipped (placeholders, experiments)
  requires_paths / requires_ollama_tag   skip with a message when missing
  prepare       [{"symlink": {"src", "dst"}}, {"write_json": {"path", "data"}}, {"clean_dir": path inside {state}},
                 {"link_files": {"src", "dst", "exclude": [globs]}}  (per-file symlinks, nothing copied),
                 {"clone_files": {"src", "dst", "files": [names]}}  (per-file APFS clones, nothing copied),
                 {"write_text": {"path", "text", "mode"}}]
  api_key       the bearer token the server wants (rec's key file): bench.py and the warm-up send it
  cleanup       the same steps after the server stopped (e.g. clean_dir of an on-disk KV cache the engine filled)
  stop          extra stop command (e.g. `mtplx stop --port N`); the process tree is killed anyway
  pid_file      file holding the pid of a server that detaches from the launcher (LM Studio's llmster
                daemon): that pid's process tree is the measured server (memory, quiet wait, teardown)
  lms_format    lmstudio: model format whose selected LM Studio runtime is recorded (safetensors|gguf)
  lock          "harness" (default) or "external": cmd is null and the server was started under
                `lockf -k <lock file> ...`, so taking the lock here would wait forever
  adaptive_warm 3 full decodes in the warm-up       flag_check "llama-server": adapt flags to --help
  ready_timeout seconds to wait for /v1/models (default 900)
  sampling_overrides, no_think_style, bench_args    passed to bench.py
  agent_trace, agent_tool_mode   the agent session to replay (a file in agent-traces/); default: the model's entry
                in agent-traces/index.json, none for a model without one
  version_cmd   command whose output is recorded   log_grep  regexes copied from the server log
  gguf_expect   expected blob digest prefix        note      free text, copied to the result

Placeholders in cmd, env, prepare, stop, version_cmd, requires_paths, pid_file, model_path and model ("~" is expanded):
  {repo}      the Kapitsa checkout this harness is in      {harness}  bench/harness
  {models}    $BENCH_MODELS, default ~/models (the weights layout in README.md)
  {out}       the output directory (--out-dir / $BENCH_OUT, default bench/harness/runs)
  {state}     {out}/state: harness-owned engine state (oMLX base dirs, model links); clean_dir works only here
  {home}      $HOME
  {env:NAME} / {env:NAME=DEFAULT}   an environment variable (or local.env entry), else DEFAULT (expanded)
  {first:A|B|...}                   the first alternative that is an existing file or a command on PATH
  {ollama_gguf:TAG|TAG2}            the GGUF weights blob of an Ollama tag, read from the Ollama store's manifests
  {ollama_gguf_draft:TAG|TAG2}      the tag's separate draft GGUF (image.draft layer: e.g. qwen3.5:4b's MTP head)
"""
import argparse, copy, fnmatch, glob, hashlib, json, os, re, shlex, shutil, signal, socket, subprocess, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import bench, gpu_hogs  # noqa: E402

OUT = RESULTS = LOGS = STATE = None


def set_out_dir(path):
    global OUT, RESULTS, LOGS, STATE
    OUT = os.path.abspath(os.path.expanduser(path))
    RESULTS, LOGS, STATE = (os.path.join(OUT, x) for x in ("results", "logs", "state"))


set_out_dir(bench.out_dir())


def models_dir():
    return os.path.abspath(os.path.expanduser(os.environ.get("BENCH_MODELS") or "~/models"))


def ollama_models_dir():
    return os.path.expanduser(os.environ.get("OLLAMA_MODELS") or "~/.ollama/models")


LMS_HOME = os.path.expanduser("~/.lmstudio")
OLLAMA_CANDIDATES = ("ollama", "/Applications/Ollama.app/Contents/Resources/ollama")

RELEVANT_ENV = ("OLLAMA_", "NI_", "KAPITSA_", "MTPLX_", "GGML_", "LLAMA_", "HF_", "MLX_", "OMLX_", "METAL_", "PYTORCH_",
                "LMS_", "BENCH_")
SECRET_ENV = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH", re.I)
DEFAULT_LOG_GREP = {
    "llama.cpp": [r"build|version", r"n_ctx_slot|n_ctx ", r"n_ubatch|n_batch", r"MTP|draft acceptance|speculative",
                  r"load_tensors|model size|file size", r"flash_attn|n_threads"],
    "ollama": [r"starting llama-server|ollama runner|starting runner", r"draft acceptance|speculative decode stats",
               r"Loaded tensors|offload|model weights", r"version|inference compute"],
    "kapitsa": [r"engine ready|loaded |read .* GB|MTP|DFlash2", r"chat:"],
    "mtplx": [r"profile|depth|verify|MTP|mtp", r"warn|WARN|error|Error"],
    "omlx": [r"MTP|mtp", r"memory|guard", r"[Ll]oaded"],
    "mlx-lm": [r"Starting|model"],
    "lmstudio": [r"^###|llmster v|mlx-llm|llama\.cpp-mac|Model loaded|[Ll]oading model|context",
                 r"prompt cache|[Cc]ached|disk", r"tok/s|tokens/s|[Tt]ime to first|Stop reason|stats",
                 r"[Ee]rror|WARN|[Ww]arning|Traceback|speculative|draft|MTP"],
}
ENGINE_ALIASES = {"nace-infer": "kapitsa"}  # Kapitsa's former name, used in older results


class ResolveError(Exception):
    pass


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, timeout=30, env=None):
    """stdout+stderr of cmd (a failure is returned as text, never raised)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return (p.stdout + p.stderr).strip()
    except Exception as e:
        return f"<{cmd[0]} failed: {e!r}>"


def run_ok(cmd, timeout=30):
    """stdout of cmd if it exits 0, else None."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def engine_kind(c):
    e = c.get("engine")
    return ENGINE_ALIASES.get(e, e)


def redact(env):
    return {k: ("<redacted>" if SECRET_ENV.search(k) else v) for k, v in (env or {}).items()}

# ---------------------------------------------------------------- placeholders
def _exe(p):
    """An existing file path, or a command name found on PATH (absolute path), else None."""
    if not p:
        return None
    if "/" in p:
        return p if os.path.exists(p) else None
    return shutil.which(p)


def _split_top(s, sep):
    """s split at `sep` characters that are not inside {...}."""
    out, cur, depth = [], [], 0
    for ch in s:
        depth += (ch == "{") - (ch == "}")
        if ch == sep and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _resolve(inner, ctx):
    name, has_arg, arg = inner.partition(":")
    if name == "repo":
        return REPO
    if name == "harness":
        return HERE
    if name == "models":
        return models_dir()
    if name == "out":
        return OUT
    if name == "state":
        return STATE
    if name == "home":
        return os.path.expanduser("~")
    if name == "env" and has_arg:
        var, eq, default = arg.partition("=")
        v = os.environ.get(var)
        if v:
            return os.path.expanduser(v) if v.startswith("~") else v
        if eq:
            return expand(default, ctx)
        raise ResolveError(f"environment variable {var} is not set (export it or add it to local.env)")
    if name == "first" and has_arg:
        alts = _split_top(arg, "|")
        for alt in alts:
            try:
                p = _exe(expand(alt, ctx))
            except ResolveError:
                p = None
            if p:
                return p
        raise ResolveError(f"none of {alts} exists or is on PATH")
    if name == "ollama_gguf" and has_arg:
        if arg not in ctx.setdefault("gguf", {}):
            ctx["gguf"][arg] = ollama_gguf(arg)
        return ctx["gguf"][arg]
    if name == "ollama_gguf_draft" and has_arg:
        if arg not in ctx.setdefault("gguf_draft", {}):
            ctx["gguf_draft"][arg] = ollama_gguf(arg, "draft")
        return ctx["gguf_draft"][arg]
    raise ResolveError(f"unknown placeholder {{{inner}}}")


def expand(s, ctx):
    """Placeholders ({name} or {name:arg}, nestable) in a string; a leading ~ is expanded. Braces that do not open
    a placeholder (e.g. in a regex) are copied as they are."""
    if not isinstance(s, str):
        return s
    out, i = [], 0
    while i < len(s):
        if s[i] == "{" and re.match(r"[a-z_]+[:}]", s[i + 1:]):
            depth, j = 0, i
            while j < len(s):
                depth += (s[j] == "{") - (s[j] == "}")
                if depth == 0:
                    break
                j += 1
            if j >= len(s):
                raise ResolveError(f"unbalanced braces in {s!r}")
            out.append(_resolve(s[i + 1:j], ctx))
            i = j + 1
        else:
            out.append(s[i])
            i += 1
    r = "".join(out)
    return os.path.expanduser(r) if r.startswith("~") else r


def expand_all(o, ctx):
    if isinstance(o, list):
        return [expand_all(x, ctx) for x in o]
    if isinstance(o, dict):
        return {k: expand_all(v, ctx) for k, v in o.items()}
    return expand(o, ctx)


def resolve_argv(argv, ctx):
    """Expanded argv whose program is an absolute path (a bare name is looked up on PATH)."""
    out = [expand(x, ctx) for x in argv]
    if out and "/" not in out[0]:
        p = shutil.which(out[0])
        if not p:
            raise ResolveError(f"{out[0]} not found on PATH (install it, or point the config's env variable at it)")
        out[0] = p
    return out

# ---------------------------------------------------------------- Ollama manifests
def _split_tag(tag):
    name, _, t = tag.partition(":")
    ns, _, n = name.rpartition("/")
    return (ns or "library"), n, (t or "latest")


def _blob(digest, store=None):
    return os.path.join(store or ollama_models_dir(), "blobs", digest.replace(":", "-"))


def ollama_manifest_paths(tag, store=None):
    ns, n, t = _split_tag(tag)
    out = []
    for top in ("manifests-v2", "manifests"):
        base = os.path.join(store or ollama_models_dir(), top)
        if not os.path.isdir(base):
            continue
        for host in sorted(os.listdir(base)):
            p = os.path.join(base, host, ns, n, t)
            if os.path.isfile(p):
                out.append(p)
    return out


def _load_manifest(text, want_format="gguf", depth=0, store=None):
    """Manifest JSON with layers; a manifest list (runner children); or a pointer to a manifest blob."""
    if depth > 4:
        raise ResolveError("manifest indirection too deep")
    text = text.strip()
    m = re.fullmatch(r"(sha256[:-][0-9a-f]{64})", text)
    if m:
        return _load_manifest(open(_blob(m.group(1).replace("-", ":", 1), store)).read(), want_format, depth + 1, store)
    obj = json.loads(text)
    if isinstance(obj, dict) and obj.get("layers"):
        return obj
    if isinstance(obj, dict) and obj.get("manifests"):
        kids = obj["manifests"]
        pick = [k for k in kids if k.get("format") == want_format or (want_format == "gguf" and k.get("runner") == "llamacpp")] or kids
        return _load_manifest(open(_blob(pick[0]["digest"], store)).read(), want_format, depth + 1, store)
    if isinstance(obj, dict) and isinstance(obj.get("digest"), str):
        return _load_manifest(open(_blob(obj["digest"], store)).read(), want_format, depth + 1, store)
    raise ResolveError("unrecognised manifest format")


def ollama_model_layer(tag, kind="model"):
    """(blob path, digest, size, manifest path) of the model weights layer of an Ollama tag (kind "draft": its
    separate draft model layer, e.g. the MTP head of qwen3.5:4b-mtp-q4_K_M)."""
    errs = []
    for p in ollama_manifest_paths(tag):
        try:
            man = _load_manifest(open(p).read())
            for l in man["layers"]:
                if str(l.get("mediaType", "")).endswith("image." + kind):
                    return _blob(l["digest"]), l["digest"], l.get("size"), p
            errs.append(f"{p}: no image.{kind} layer")
        except Exception as e:
            errs.append(f"{p}: {e}")
    # fallback: ask a running Ollama server for the Modelfile (FROM <blob path>)
    ollama = next(filter(None, (_exe(x) for x in OLLAMA_CANDIDATES)), None)
    for host in ("127.0.0.1:11434",):
        if not ollama or kind != "model":
            break
        out = run([ollama, "show", "--modelfile", tag], timeout=15, env=dict(os.environ, OLLAMA_HOST=host))
        m = re.search(r"^FROM\s+(/\S+sha256-([0-9a-f]{64}))\s*$", out, re.M)
        if m and os.path.isfile(m.group(1)):
            return m.group(1), "sha256:" + m.group(2), os.path.getsize(m.group(1)), f"ollama show @{host}"
    raise ResolveError(f"Ollama tag {tag} not found in {ollama_models_dir()}/manifests*" + (f" ({'; '.join(errs)})" if errs else ""))


def ollama_gguf(tags, kind="model"):
    errs = []
    for tag in tags.split("|"):
        try:
            path, digest, size, man = ollama_model_layer(tag, kind)
            if not os.path.isfile(path):
                raise ResolveError(f"blob {path} missing")
            with open(path, "rb") as fh:
                if fh.read(4) != b"GGUF":
                    raise ResolveError(f"{path} is not a GGUF (MLX tag?)")
            return path
        except ResolveError as e:
            errs.append(str(e))
    raise ResolveError("; ".join(errs))


def ollama_tag_present(tag, store=None):
    return bool(ollama_manifest_paths(tag, store))

# ---------------------------------------------------------------- config handling
def load_configs(paths):
    by, out = {}, []
    for path in paths:
        for c in json.load(open(path)):
            if c.get("extends"):
                if c["extends"] not in by:
                    raise SystemExit(f"{c.get('label')}: extends unknown/later label {c['extends']}")
                m = copy.deepcopy(by[c["extends"]])
                for k, v in c.items():
                    if k in ("extends", "cmd_append"):
                        continue
                    if k == "env" and isinstance(v, dict):
                        m["env"] = dict(m.get("env", {}), **v)
                    else:
                        m[k] = v
                if c.get("cmd_append"):
                    m["cmd"] = list(m.get("cmd") or []) + c["cmd_append"]
                if "enabled" not in c:  # a disabled base is a template; its children run unless they say otherwise
                    m.pop("enabled", None)
                c = m
            if c["label"] in by:
                raise SystemExit(f"duplicate label {c['label']} ({path})")
            c["_file"] = os.path.relpath(os.path.abspath(path), HERE)
            by[c["label"]] = c
            out.append(c)
    return out


def port_of(c):
    u = urllib.parse.urlparse(c.get("base_url") or "")
    return c.get("port") or u.port

def live_slots(cmd):
    """Conversations the server argv keeps resident at once: --slots N (Kapitsa, MLX) or -np / --parallel /
    --max-concurrency N (llama.cpp). 0 = not passed, i.e. whatever the engine's own default is."""
    names = ("--slots", "-np", "--parallel", "--max-concurrency")
    cmd = list(cmd or [])
    for i, tok in enumerate(cmd):
        for n in names:
            if tok == n and i + 1 < len(cmd) and str(cmd[i + 1]).isdigit():
                return int(cmd[i + 1])
            if tok.startswith(n + "=") and tok.split("=", 1)[1].isdigit():
                return int(tok.split("=", 1)[1])
    return 0


def port_open(port, host="127.0.0.1"):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def listener_pid(port):
    out = run(["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], timeout=10)
    pids = [int(x) for x in out.split() if x.isdigit()]
    return pids[0] if pids else None

# ---------------------------------------------------------------- llama-server flag drift
LLAMA_ALIASES = {
    "--spec-draft-n-max": ["--spec-n-max", "--draft-n-max", "--draft-max"],
    "--spec-type": ["--spec-mode"],
    "-lm": ["--load-mode"], "--load-mode": ["-lm"],
    "--no-webui": ["--no-ui"],
    "--ctx-checkpoints": ["-ctxcp", "--swa-checkpoints"],
    "--cache-ram": ["-cram"],
    "--chat-template-file": [],
}


def help_flags(text):
    flags, removed = set(), set()
    for line in text.split("\n"):
        if not line.startswith("-"):
            continue
        m = re.search(r"(?<!,)\s{2,}", line)
        col = line[:m.start()] if m else line
        fs = set(re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", col))
        (removed if "has been removed" in line else flags).update(fs)
    return flags - removed, removed


def adapt_llama_cmd(cmd, help_text):
    """Map flags the installed llama-server no longer knows to their current names (from --help)."""
    flags, removed = help_flags(help_text)
    out, changes, missing = [cmd[0]], [], []
    i = 1
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("-") and not re.fullmatch(r"-?\d+(\.\d+)?", tok) and tok not in flags:
            alt = next((x for x in LLAMA_ALIASES.get(tok, []) if x in flags), None)
            if alt:
                changes.append(f"{tok} -> {alt}")
                tok = alt
            elif tok in ("-lm", "--load-mode") and i + 1 < len(cmd) and cmd[i + 1] == "none" and "--no-mmap" in flags:
                changes.append(f"{tok} none -> --no-mmap")
                out.append("--no-mmap")
                i += 2
                continue
            else:
                missing.append(tok + (" (removed)" if tok in removed else ""))
        out.append(tok)
        i += 1
    return out, changes, missing


_HELP_CACHE = {}


def llama_help(binary):
    if binary not in _HELP_CACHE:
        _HELP_CACHE[binary] = run([binary, "--help"], timeout=60)
    return _HELP_CACHE[binary]

# ---------------------------------------------------------------- provenance
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_state(d, exclude=()):
    """HEAD, branch and local changes of the checkout at d; paths in `exclude` (e.g. this harness) are left out of
    the dirty check, so editing a benchmark config does not mark the engine build dirty."""
    spec = ["--", "."] + [f":(exclude){x}" for x in exclude] if exclude else []
    head = run(["git", "-C", d, "rev-parse", "HEAD"])
    diff = subprocess.run(["git", "-C", d, "diff", "HEAD"] + spec, capture_output=True).stdout
    untracked = run(["git", "-C", d, "ls-files", "--others", "--exclude-standard"] + spec).split("\n")
    return {"head": head, "head_time": run(["git", "-C", d, "log", "-1", "--format=%cI"]),
            "branch": run(["git", "-C", d, "rev-parse", "--abbrev-ref", "HEAD"]),
            "diff_head_sha256": hashlib.sha256(diff).hexdigest(), "diff_bytes": len(diff), "dirty": bool(diff),
            "diff_stat": run(["git", "-C", d, "diff", "HEAD", "--shortstat"] + spec),
            "untracked_files": len([u for u in untracked if u.strip()]), "excluded": list(exclude)}


def checkout_of(binary):
    """Top of the git checkout a binary was built in (e.g. <repo>/build/kapitsa), or None."""
    return run_ok(["git", "-C", os.path.dirname(os.path.realpath(binary)), "rev-parse", "--show-toplevel"])


def kapitsa_version(binary):
    top = checkout_of(binary)
    if top:
        g = git_state(top, exclude=("bench", "docs"))
        return g["head"][:7] + ("+dirty" if g["dirty"] else "")
    m = re.search(r"\b(\d+\.\d+\.\d+[\w.+-]*)", run([binary, "--version"], timeout=30))
    return m.group(1) if m else None


def llama_version(binary):
    m = re.search(r"version: (\S+) \(build (\d+), commit (\w+)\)", run([binary, "--version"], timeout=60))
    return f"{m.group(1)} (b{m.group(2)}, {m.group(3)})" if m else None


def ollama_version(binary):
    out = run([binary, "--version"], timeout=30)
    m = re.search(r"client version is ([0-9][\w.\-]*)", out) or re.search(r"version is ([0-9][\w.\-]*)", out)
    if m:
        return m.group(1)
    app = re.search(r"^(.*?\.app)/", os.path.realpath(binary))
    if app:
        return run(["/usr/libexec/PlistBuddy", "-c", "Print CFBundleShortVersionString", os.path.join(app.group(1), "Contents/Info.plist")])
    return None


PIP_NAMES = {"mlx-lm": ["mlx_lm", "mlx"], "omlx": ["omlx", "mlx_lm", "mlx"], "mtplx": ["mtplx", "mlx_lm", "mlx"]}
LMS_PY_NAMES = ["mlx", "mlx_metal", "mlx_lm", "mlx_vlm", "transformers"]


def pip_versions(cmd0, names):
    """Package versions of the Python environment an entry-point script belongs to: <venv>/bin/<script> ->
    <venv>/lib/python3*/site-packages, else the interpreter in the script's shebang."""
    real = os.path.realpath(cmd0)
    out = {}
    for sp in glob.glob(os.path.join(os.path.dirname(os.path.dirname(real)), "lib", "python3*", "site-packages")):
        for n in names:
            for d in glob.glob(os.path.join(sp, f"{n}-*.dist-info")):
                m = re.match(rf"{re.escape(n)}-(.+)\.dist-info$", os.path.basename(d))
                if m:
                    out[n] = m.group(1)
    if len(out) < len(names):
        try:
            first = open(real, "rb").readline().decode("utf-8", "ignore")
        except Exception:
            first = ""
        if first.startswith("#!") and "python" in first:
            py = first[2:].strip().split()[0]
            code = ("import importlib.metadata as m, json, sys\nv = {}\nfor n in sys.argv[1:]:\n    try: v[n] = m.version(n)\n"
                    "    except Exception: pass\nprint(json.dumps(v))")
            got = run_ok([py, "-c", code] + names, timeout=60)
            try:
                out.update({k: v for k, v in json.loads(got or "{}").items() if k not in out})
            except Exception:
                pass
    return out


def lmstudio_info():
    """LM Studio provenance read from ~/.lmstudio without waking the daemon: llmster (headless LM
    Studio) version, lms CLI commit, the runtime selected per model format and that runtime's own
    versions (MLX: mlx-engine commit + the packages of its vendored venv; llama.cpp: build + commit)."""
    info = {"home": "~/.lmstudio"}
    try:
        loc = json.load(open(os.path.join(LMS_HOME, ".internal", "llmster-install-location.json")))
        m = re.search(r"/llmster/([^/]+)/", loc.get("path", ""))
        info["llmster"] = m.group(1) if m else None
    except Exception as e:
        info["llmster_error"] = repr(e)[:200]
    m = re.search(r"CLI commit: (\w+)", run([os.path.join(LMS_HOME, "bin", "lms"), "version"], timeout=30))
    info["lms_cli_commit"] = m.group(1) if m else None
    try:
        st = json.load(open(os.path.join(LMS_HOME, "settings.json")))
        dev = st.get("developer", {})
        info["settings"] = {"defaultContextLength": st.get("defaultContextLength"),
                            "modelLoadingGuardrails": st.get("modelLoadingGuardrails"),
                            **{k: dev.get(k) for k in ("separateReasoningContentInAPI", "jitModelTTL",
                                                       "unloadPreviousJITModelOnLoad", "backendDownloadChannel")}}
    except Exception as e:
        info["settings_error"] = repr(e)[:200]
    try:
        prefs = json.load(open(os.path.join(LMS_HOME, ".internal", "backend-preferences-v1.json")))
        info["runtime_selection"] = "backend-preferences-v1.json"
    except Exception as e:
        prefs, info["prefs_error"] = [], repr(e)[:200]
    if not prefs:
        # the daemon rewrites that file at shutdown and can leave it empty; without it LM Studio uses the
        # newest installed runtime per model format, so record that
        best = {}
        for d in glob.glob(os.path.join(LMS_HOME, "extensions", "backends", "*", "backend-manifest.json")):
            try:
                man = json.load(open(d))
                key = tuple(int(x) for x in re.findall(r"\d+", str(man.get("version"))))
                for fmt in man.get("supported_model_formats", []):
                    if fmt not in best or key > best[fmt][0]:
                        best[fmt] = (key, {"model_format": fmt, "name": man.get("name") or
                                           os.path.basename(os.path.dirname(d)).rsplit("-", 1)[0], "version": man.get("version")})
            except Exception:
                pass
        prefs = [v for _, v in best.values()]
        info["runtime_selection"] = "newest installed per format (backend-preferences-v1.json empty)"
    info["runtimes"] = {}
    for p in prefs:
        d = os.path.join(LMS_HOME, "extensions", "backends", f"{p.get('name')}-{p.get('version')}")
        r = {"name": p.get("name"), "version": p.get("version")}
        try:
            for _, dd in json.load(open(os.path.join(d, "display-data.json"))):
                for n in dd.get("releaseNotes", []):
                    if n.get("version") == p.get("version"):
                        r["release_notes"] = n.get("releaseNotes", "")[:600]
            m = re.search(r"mlx-engine==(\w+)", r.get("release_notes", ""))
            if m:
                r["mlx_engine"] = m.group(1)
            m = re.search(r"release (b\d+) \(commit (\w+)\)", r.get("release_notes", ""))
            if m:
                r["llama_cpp"] = f"{m.group(1)}, {m.group(2)}"
            man = json.load(open(os.path.join(d, "backend-manifest.json")))
            r["vendor"] = man.get("vendor_lib_package_names") or []
            for v in r["vendor"]:
                for sp in glob.glob(os.path.join(LMS_HOME, "extensions", "backends", "vendor", v, "lib", "python3*", "site-packages")):
                    for n in LMS_PY_NAMES:
                        for di in glob.glob(os.path.join(sp, f"{n}-*.dist-info")):
                            mm = re.match(rf"{n}-(.+)\.dist-info$", os.path.basename(di))
                            if mm:
                                r.setdefault("packages", {})[n] = mm.group(1)
        except Exception as e:
            r["error"] = repr(e)[:200]
        info["runtimes"][p.get("model_format")] = r
    return info


def lmstudio_version(c, info=None):
    info = info or lmstudio_info()
    r = info.get("runtimes", {}).get(c.get("lms_format") or "safetensors") or {}
    if not r:
        return f"llmster {info.get('llmster')}" if info.get("llmster") else None
    pk = r.get("packages", {})
    if "mlx" in (r.get("name") or ""):
        inner = ", ".join(x for x in [f"mlx-engine {r['mlx_engine']}" if r.get("mlx_engine") else None] +
                          [f"{k.replace('_', '-')} {pk[k]}" for k in ("mlx", "mlx_lm", "mlx_vlm") if k in pk] if x)
        rt = f"MLX runtime {r.get('version')}"
    else:
        inner = r.get("llama_cpp") or ""
        rt = f"llama.cpp runtime {r.get('version')}"
    return f"llmster {info.get('llmster')}, {rt}" + (f" ({inner})" if inner else "")


_VERSION_CACHE = {}


def detect_version(c, cmd, ctx):
    key = (engine_kind(c), cmd[0] if cmd else None, c.get("lms_format"))
    if key not in _VERSION_CACHE:
        _VERSION_CACHE[key] = _detect_version(c, cmd, ctx)
    return _VERSION_CACHE[key]


def _detect_version(c, cmd, ctx):
    e = engine_kind(c)
    try:
        if e == "lmstudio":
            return lmstudio_version(c)
        if not cmd or not os.path.isfile(cmd[0]):
            return None
        if e == "kapitsa":
            return kapitsa_version(cmd[0])
        if e == "llama.cpp":
            return llama_version(cmd[0])
        if e == "ollama":
            return ollama_version(cmd[0])
        if e in PIP_NAMES:
            v = pip_versions(cmd[0], PIP_NAMES[e])
            return " ".join(f"{k} {v[k]}" for k in PIP_NAMES[e] if k in v) or None
    except Exception:
        return None
    return None


def model_weights(c, ctx, env=None):
    """Size of the weights the engine reads (dir of safetensors, GGUF file, or Ollama manifest)."""
    info = {"model_path": c.get("model_path")}
    try:
        if engine_kind(c) == "ollama" and c.get("model"):
            store = (env or {}).get("OLLAMA_MODELS") or ollama_models_dir()
            for p in ollama_manifest_paths(c["model"], store):
                info["ollama_manifest"] = os.path.relpath(p, store)
                man = _load_manifest(open(p).read(), "safetensors" if "nvfp4" in c["model"] or "mlx" in c["model"] else "gguf",
                                     store=store)
                info["ollama_layers_bytes"] = sum(l.get("size", 0) for l in man["layers"])
                info["ollama_model_layers"] = [l["digest"] for l in man["layers"] if str(l.get("mediaType", "")).endswith("image.model")][:3]
                break
        for k, v in ctx.get("gguf", {}).items():
            info["gguf"] = {"tags": k, "bytes": os.path.getsize(v), "digest": os.path.basename(v)}
        for k, v in ctx.get("gguf_draft", {}).items():
            info["gguf_draft"] = {"tags": k, "bytes": os.path.getsize(v), "digest": os.path.basename(v)}
        mp = expand(c.get("model_path") or "", ctx)
        if mp and os.path.isdir(mp):
            fs = [f for f in glob.glob(os.path.join(mp, "*.safetensors"))]
            info["safetensors_bytes"] = sum(os.path.getsize(os.path.realpath(f)) for f in fs)
            info["safetensors_files"] = sorted(os.path.basename(f) for f in fs)
    except Exception as e:
        info["error"] = repr(e)[:200]
    return info


def system_state():
    d = {"sw_vers": dict(re.findall(r"^(\w+):\s*(.+)$", run(["sw_vers"]), re.M)),
         "hw_model": run(["sysctl", "-n", "hw.model"]), "cpu": run(["sysctl", "-n", "machdep.cpu.brand_string"]),
         "memsize": run(["sysctl", "-n", "hw.memsize"]), "power": run(["pmset", "-g", "batt"]).split("\n")[0],
         "powermode": (re.search(r"powermode\s+(\d+)", run(["pmset", "-g"])) or [None, None])[1],
         "vm": bench.vm_snapshot(), "gpu_mem": gpu_mem(),
         # free disk: LM Studio's MLX engine and MTPLX keep follow-up checkpoints on disk only above 10 GiB free
         "disk_free_bytes": disk_free()}
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=1) as r:
            d["ollama_app_loaded"] = [m.get("name") for m in json.load(r).get("models", [])]
    except Exception:
        d["ollama_app_loaded"] = "not running"
    return d


def disk_free(path=None):
    """Free bytes on the volume of the home folder (LM Studio's and MTPLX's on-disk prompt caches write nothing when
    under 10 GiB are free)."""
    try:
        return shutil.disk_usage(path or os.path.expanduser("~")).free
    except Exception:
        return None


DISK_CACHE_FLOOR = 10 * 2**30
DISK_CACHE_ENGINES = ("lmstudio", "mtplx")


def gpu_mem():
    out = run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"], timeout=10)
    g = lambda k: int(m.group(1)) if (m := re.search(rf'"{re.escape(k)}"=(\d+)', out)) else None
    return {"in_use": g("In use system memory"), "alloc": g("Alloc system memory"), "util_pct": g("Device Utilization %")}

# ---------------------------------------------------------------- lifecycle
def do_prepare(actions, ctx, dry):
    lines = []
    for act in actions or []:
        act = expand_all(act, ctx)
        if "symlink" in act:
            src, dst = act["symlink"]["src"], act["symlink"]["dst"]
            lines.append(f"symlink {dst} -> {src}")
            if not dry:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.islink(dst):
                    if os.readlink(dst) != src:
                        os.remove(dst); os.symlink(src, dst)
                elif os.path.exists(dst):
                    raise ResolveError(f"{dst} exists and is not a symlink")
                else:
                    os.symlink(src, dst)
        elif "write_json" in act:
            p = act["write_json"]["path"]
            lines.append(f"write {p}: {json.dumps(act['write_json']['data'])}")
            if not dry:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(act["write_json"]["data"], open(p, "w"), indent=2)
        elif "link_files" in act:
            # per-file symlinks of src's top-level files into dst (resolved targets), for engines that want a real
            # model directory of their own; nothing is copied. A regular file of the same size already in dst (the
            # engine downloaded it itself) is left alone.
            src, dst = act["link_files"]["src"], act["link_files"]["dst"]
            excl = act["link_files"].get("exclude", [])
            names = sorted(f for f in os.listdir(src) if not f.startswith(".") and not any(
                fnmatch.fnmatch(f, x) for x in excl)) if os.path.isdir(src) else []
            if not names:
                raise ResolveError(f"link_files: nothing to link in {src}")
            lines.append(f"link {len(names)} files of {src} into {dst} (excluded: {excl or '-'})")
            if not dry:
                os.makedirs(dst, exist_ok=True)
                for f in names:
                    s, d = os.path.realpath(os.path.join(src, f)), os.path.join(dst, f)
                    if os.path.islink(d):
                        if os.readlink(d) == s:
                            continue
                        os.remove(d)
                    elif os.path.exists(d):
                        if os.path.isfile(d) and os.path.getsize(d) == os.path.getsize(s):
                            continue
                        raise ResolveError(f"{d} exists, is not a symlink and differs in size from {s}")
                    os.symlink(s, d)
                for f in os.listdir(dst):  # links that are no longer wanted (e.g. a newly excluded file)
                    if f not in names and os.path.islink(os.path.join(dst, f)):
                        os.remove(os.path.join(dst, f))
        elif "clone_files" in act:
            # per-file APFS clones (cp -c: no bytes copied, the clone's blocks shared with the original until either
            # changes) of src's files into dst, for engines that refuse symlinks (rec checks that its model files are
            # regular files of the manifest's sizes); "files" names them (default: src's top-level files). A regular
            # file of the same size already in dst is left alone.
            src, dst = act["clone_files"]["src"], act["clone_files"]["dst"]
            names = act["clone_files"].get("files") or (sorted(
                f for f in os.listdir(src) if not f.startswith(".") and os.path.isfile(os.path.join(src, f)))
                if os.path.isdir(src) else [])
            missing = [f for f in names if not os.path.isfile(os.path.join(src, f))]
            if not names or missing:
                raise ResolveError(f"clone_files: {', '.join(missing) or 'nothing'} to clone in {src}")
            lines.append(f"clone {len(names)} files of {src} into {dst}")
            if not dry:
                os.makedirs(dst, exist_ok=True)
                for f in names:
                    s_, d = os.path.realpath(os.path.join(src, f)), os.path.join(dst, f)
                    if os.path.islink(d):
                        os.remove(d)
                    elif os.path.isfile(d) and os.path.getsize(d) == os.path.getsize(s_):
                        continue
                    if subprocess.run(["cp", "-c", s_, d]).returncode != 0:
                        raise ResolveError(f"clone_files: cp -c {s_} {d} failed (not on the same APFS volume?)")
        elif "write_text" in act:
            p = act["write_text"]["path"]
            lines.append(f"write {p} ({len(act['write_text']['text'])} characters)")
            if not dry:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    f.write(act["write_text"]["text"])
                os.chmod(p, int(str(act["write_text"].get("mode", "600")), 8))
        elif "clean_dir" in act:
            p = os.path.realpath(act["clean_dir"])
            if not p.startswith(os.path.realpath(STATE) + os.sep):
                raise ResolveError(f"clean_dir outside {STATE}: {p}")
            lines.append(f"empty {p} (harness-owned cache dir)")
            if not dry and os.path.isdir(p):
                for f in os.listdir(p):
                    q = os.path.join(p, f)
                    if os.path.isdir(q) and not os.path.islink(q):
                        shutil.rmtree(q)
                    else:
                        os.remove(q)
    return lines


def wait_ready(base, proc, port, timeout=900):
    t0 = time.time()
    # with the config's key (run_config sets BENCH_API_KEY): a server that wants one (rec) answers 401 without it
    req = urllib.request.Request(base.rstrip("/") + "/models",
                                 headers={"Authorization": "Bearer " + os.environ.get("BENCH_API_KEY", "local")})
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True, time.time() - t0
        except Exception:
            pass
        # a launcher that exits non-zero without a listener died; exit 0 may be a daemonizing launcher
        # (MTPLX) whose server is still loading, so keep waiting for the port then
        if proc and proc.poll() not in (None, 0) and not port_open(port):
            return False, time.time() - t0
        time.sleep(2)
    return False, time.time() - t0


def stop_server(c, proc, roots, port, ctx, base_mem, env=None):
    info = {}
    pids = gpu_hogs.proc_tree(roots)
    if c.get("stop"):
        try:
            info["stop_cmd"] = run(resolve_argv(c["stop"], ctx), timeout=60, env=dict(os.environ, **(env or {})))[-300:]
        except ResolveError as e:
            info["stop_cmd"] = f"<{e}>"
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc:
            try:
                os.killpg(proc.pid, sig)
            except Exception:
                pass
        for p in pids:
            try:
                os.kill(p, sig)
            except Exception:
                pass
        t0 = time.time()
        while time.time() - t0 < (60 if sig == signal.SIGTERM else 15):
            if proc:
                proc.poll()
            alive = [p for p in pids if _alive(p)]
            if not alive:
                break
            time.sleep(0.5)
        info["killed_with"] = signal.Signals(sig).name
        if not [p for p in pids if _alive(p)]:
            break
    info["survivors"] = [p for p in pids if _alive(p)]
    t0 = time.time()
    while port and port_open(port) and time.time() - t0 < 30:
        time.sleep(0.5)
    info["port_closed"] = not (port and port_open(port))
    # memory release: GPU in-use memory back within 1 GB of the pre-start baseline and the system's
    # available-memory level within 5 points of it (or 120 s)
    t0 = time.time()
    time.sleep(5)
    while time.time() - t0 < 120:
        g, v = gpu_mem(), bench.vm_snapshot()
        ok_g = base_mem["gpu"].get("in_use") is None or g.get("in_use") is None or g["in_use"] <= base_mem["gpu"]["in_use"] + 2**30
        ok_l = base_mem["vm"].get("memorystatus_level") is None or v.get("memorystatus_level", 0) >= base_mem["vm"]["memorystatus_level"] - 5
        if ok_g and ok_l:
            break
        time.sleep(2)
    info["memory_release_s"] = round(time.time() - t0, 1)
    info["gpu_mem_after"] = gpu_mem()
    info["vm_after"] = bench.vm_snapshot()
    return info


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # zombies of other parents still answer kill(0); treat state Z as dead
    st = run(["ps", "-o", "stat=", "-p", str(pid)], timeout=5)
    return bool(st) and not st.startswith("Z")


def grep_log(path, pats, maxlines=200):
    out = []
    try:
        rx = [re.compile(p) for p in pats]
        for line in open(path, errors="ignore"):
            if any(r.search(line) for r in rx):
                out.append(line.rstrip()[:300])
    except Exception as e:
        return [f"<log read failed: {e}>"]
    if len(out) > maxlines:
        out = out[:maxlines // 2] + [f"... {len(out) - maxlines} lines omitted ..."] + out[-(maxlines // 2):]
    return out


def write_json(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, path)

# ---------------------------------------------------------------- one config
def plan(c, args, stamp):
    """Resolve everything a run needs; returns (plan dict, ctx). Never starts anything."""
    ctx = {}
    p = {"label": c["label"], "skip": None, "errors": []}
    if c.get("enabled") is False:
        p["skip"] = "disabled" + (f" ({c['note']})" if c.get("note") else "")
    if not str(c.get("base_url", "")).startswith("http"):
        p["skip"] = p["skip"] or "no base_url (placeholder)"
    p["cmd"] = None
    if c.get("cmd"):
        try:
            p["cmd"] = resolve_argv(c["cmd"], ctx)
        except ResolveError as e:
            p["errors"].append(str(e))
            p["cmd"] = []
            for x in c["cmd"]:
                try:
                    p["cmd"].append(expand(x, ctx))
                except ResolveError:
                    p["cmd"].append(f"<unresolved {x}>")
    try:
        p["env"] = expand_all(c.get("env", {}), ctx)
    except ResolveError as e:
        p["errors"].append(str(e))
        p["env"] = c.get("env", {})
    p["api_key"] = c.get("api_key")
    for rp in c.get("requires_paths", []):
        try:
            if not os.path.exists(expand(rp, ctx)):
                p["errors"].append(f"missing {expand(rp, ctx)}")
        except ResolveError as e:
            p["errors"].append(str(e))
    if c.get("requires_ollama_tag"):
        store = p["env"].get("OLLAMA_MODELS") if isinstance(p.get("env"), dict) else None
        if not ollama_tag_present(c["requires_ollama_tag"], store):
            p["errors"].append(f"Ollama tag {c['requires_ollama_tag']} not in {store or ollama_models_dir()} (pull or import it first)")
    if c.get("gguf_expect") and ctx.get("gguf"):
        for v in ctx["gguf"].values():
            if c["gguf_expect"] not in os.path.basename(v):
                p["errors"].append(f"GGUF {os.path.basename(v)} is not the expected blob {c['gguf_expect']}*")
    p["version"] = c.get("version")
    if c.get("version") in (None, "auto"):
        p["version"] = detect_version(c, p["cmd"], ctx) or "unknown"
    p["port"] = port_of(c)
    p["log"] = os.path.join(LOGS, f"{c['label']}-{stamp}.log")
    p["out"] = os.path.join(RESULTS, f"{c['label']}-{stamp}.json")
    try:
        model_id = expand(c.get("model", ""), ctx)
    except ResolveError as e:
        p["errors"].append(str(e))
        model_id = c.get("model", "")
    bargs = ["--base-url", c.get("base_url", ""), "--model", model_id, "--label", c["label"],
             "--sizes", args.sizes, "--out", p["out"], "--warmup", "none"]
    p["server_slots"] = live_slots(p.get("cmd"))
    bargs += ["--server-slots", str(p["server_slots"])]
    if c.get("no_think_style"):
        bargs += ["--no-think-style", c["no_think_style"]]
    if c.get("sampling_overrides"):
        bargs += ["--sampling-overrides", json.dumps(c["sampling_overrides"])]
    if args.skip:
        bargs += ["--skip", args.skip]
    if args.agentic:
        bargs += ["--agentic"]
    trace, mode = agent_session_of(c, args)
    bargs += ["--agent-trace", trace or "none"]
    if trace and mode:
        bargs += ["--agent-tool-mode", mode]
    bargs += shlex.split(args.bench_args or "") + [str(x) for x in c.get("bench_args", [])]
    p["bench_args"] = bargs
    return p, ctx


AGENT_INDEX = os.path.join(HERE, "agent-traces", "index.json")


def agent_session_of(c, args):
    """(trace path or None, tool mode or None) of a config's agent phase: --agent-trace (for every config), else the
    config's own "agent_trace", else the entry for its model in agent-traces/index.json (no entry: no agent phase)."""
    if args.agent_trace:
        return (None, None) if args.agent_trace.lower() == "none" else (os.path.abspath(args.agent_trace), None)
    if c.get("agent_trace"):
        return os.path.join(HERE, "agent-traces", c["agent_trace"]), c.get("agent_tool_mode")
    try:
        e = json.load(open(AGENT_INDEX)).get(c.get("model_name") or "")
    except Exception:
        e = None
    if not e or not e.get("trace"):
        return None, None
    return os.path.join(HERE, "agent-traces", e["trace"]), e.get("tool_mode")


class NoLock:
    """For "lock": "external" (the GPU lock is held by whoever started the (cmd null) server) and for
    NACE_GPU_LOCK_HELD=1 (the caller of this script holds it, e.g. `lockf -k <lock file> ...`)."""

    def __init__(self, log, why="config lock=external"):
        self.log, self.wait_s, self.why, self.t0 = log, 0.0, why, None

    def acquire(self):
        self.log(f"GPU lock held externally ({self.why}); holders now: {gpu_hogs.lock_holders()}")
        self.t0 = time.time()
        return self

    def release(self):  # how long this config ran inside the externally held lock
        return time.time() - self.t0 if self.t0 else 0.0


def run_config(c, args, stamp):
    # the bearer token the server wants (config "api_key"; bench.post sends $BENCH_API_KEY, else "local"): for this
    # config's warm-up in this process and bench.py's requests in its child
    key = c.get("api_key")
    old_key = os.environ.get("BENCH_API_KEY")
    if key:
        os.environ["BENCH_API_KEY"] = key
    else:
        os.environ.pop("BENCH_API_KEY", None)
    try:
        return run_config_keyed(c, args, stamp)
    finally:
        if old_key is None:
            os.environ.pop("BENCH_API_KEY", None)
        else:
            os.environ["BENCH_API_KEY"] = old_key


def run_config_keyed(c, args, stamp):
    p, ctx = plan(c, args, stamp)
    label = c["label"]
    print(f"\n===== {label}  [{c.get('engine')}] {c.get('display')}  version {p['version']}", flush=True)
    if p["skip"] or p["errors"]:
        log(f"{label}: SKIP: {p['skip'] or '; '.join(p['errors'])}")
        return
    port = p["port"]
    if c.get("lock") == "external":  # cmd null and the server's own launcher holds the lock (lockf -k ... serve)
        lock = NoLock(lambda m: log(f"{label}: {m}"))
    elif os.environ.get("NACE_GPU_LOCK_HELD"):  # our caller holds it (lockf -k <lock file> ... compare_engines.py)
        lock = NoLock(lambda m: log(f"{label}: {m}"), "by the caller, NACE_GPU_LOCK_HELD=1")
    else:
        lock = gpu_hogs.GpuLock(timeout=args.lock_timeout, log=lambda m: log(f"{label}: {m}"))
    try:
        lock.acquire()
    except TimeoutError as e:
        log(f"{label}: SKIP: {e}")
        return
    proc = None
    kill_roots = []   # processes we started (stopped at the end)
    mon_roots = []    # never foreign for the quiet wait / contention monitor
    result = {"label": label}
    prov = {}
    base_mem = {"gpu": {}, "vm": {}}
    teardown = {}
    held = 0.0
    try:
        prov.update({"system_start": system_state(), "cmd": p["cmd"], "cmd_shell": shlex.join(p["cmd"]) if p["cmd"] else None,
                     "env": redact(p["env"]), "env_cleared": [], "weights": model_weights(c, ctx, p["env"]),
                     "harness": bench.harness_info(), "config_file": c.get("_file"), "env_file_keys": args.env_keys,
                     "gpu_lock_path": gpu_hogs.LOCK_PATH, "gpu_ambient": sorted(gpu_hogs.AMBIENT)})
        base_mem = {"gpu": prov["system_start"]["gpu_mem"], "vm": prov["system_start"]["vm"]}
        if engine_kind(c) == "kapitsa" and p["cmd"] and os.path.isfile(p["cmd"][0]):
            top = checkout_of(p["cmd"][0])
            if top:
                prov["kapitsa_git"] = git_state(top, exclude=("bench", "docs"))
            prov["binary"] = {"sha256": sha256_file(p["cmd"][0]),
                              "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p["cmd"][0])))}
        if engine_kind(c) == "lmstudio":
            prov["lmstudio"] = lmstudio_info()
        if c.get("version_cmd"):
            prov["version_cmd_output"] = run(resolve_argv(c["version_cmd"], ctx), timeout=60, env=dict(os.environ, **p["env"]))[-600:]
        if p["cmd"] and c.get("flag_check") == "llama-server":
            new, changes, missing = adapt_llama_cmd(p["cmd"], llama_help(p["cmd"][0]))
            prov["flag_changes"] = changes
            if missing:
                raise ResolveError(f"llama-server does not know {missing} (see `{p['cmd'][0]} --help`)")
            if changes:
                log(f"{label}: flags adapted to this llama-server: {changes}")
            p["cmd"] = new
            prov["cmd"], prov["cmd_shell"] = new, shlex.join(new)
        prov["prepare"] = do_prepare(c.get("prepare"), ctx, False)
        if p["cmd"]:
            if port and port_open(port):
                raise ResolveError(f"port {port} is already in use (pid {listener_pid(port)}); stop that server first")
            env = {k: v for k, v in os.environ.items() if not any(k.startswith(x) for x in c.get("env_clear_prefixes", []))}
            prov["env_cleared"] = sorted(set(os.environ) - set(env))
            env.update(p["env"])
            env.pop("NACE_GPU_LOCK_HELD", None)
            prov["env_relevant"] = redact({k: v for k, v in env.items() if k.startswith(RELEVANT_ENV)})
            os.makedirs(LOGS, exist_ok=True)
            os.makedirs(STATE, exist_ok=True)
            logf = open(p["log"], "w")
            log(f"{label}: starting {shlex.join(p['cmd'])}")
            proc = subprocess.Popen(p["cmd"], env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
                                    cwd=STATE)
            kill_roots.append(proc.pid)
            mon_roots.append(proc.pid)
        ok, waited = wait_ready(c["base_url"], proc, port, c.get("ready_timeout", 900))
        prov["ready_s"] = round(waited, 1)
        lp = listener_pid(port) if port else None
        if lp and lp not in mon_roots:
            mon_roots.append(lp)
            if proc:  # a daemon our launcher left behind (MTPLX); an external server (cmd null) is never killed
                kill_roots.append(lp)
        if not ok:
            raise ResolveError(f"server never became ready ({waited:.0f}s); see {p['log']}")
        srv_pid = proc.pid if proc and (not lp or lp in gpu_hogs.proc_tree([proc.pid])) else (lp or 0)
        prov["server_pid"], prov["listener_pid"] = srv_pid, lp
        if c.get("pid_file"):  # a server that detaches from its launcher (LM Studio's llmster daemon)
            try:
                rp = int(open(expand(c["pid_file"], ctx)).read().split()[0])
            except Exception as e:
                raise ResolveError(f"pid_file {c['pid_file']}: {e!r}")
            if not _alive(rp):
                raise ResolveError(f"pid_file {c['pid_file']}: pid {rp} is not running")
            if rp not in mon_roots:
                mon_roots.append(rp)
                if proc:
                    kill_roots.append(rp)
            srv_pid = rp  # its whole tree (daemon + engine worker) is the measured server
            prov["server_pid"], prov["pid_file_pid"] = rp, rp
            prov["server_tree"] = run(["ps", "-o", "pid=,ppid=,rss=,comm=", "-p",
                                       ",".join(map(str, sorted(gpu_hogs.proc_tree([rp]))))], timeout=10)[:2000]
        log(f"{label}: ready after {waited:.0f}s (pid {srv_pid}, listener {lp})")
        # warm-up before the GPU-quiet wait, with the benchmark's own sampling/thinking settings
        ba = bench.parse_args(p["bench_args"])
        cold = None
        if args.agentic:  # first requests right after ready, before any warm-up (agentic.py cold_start)
            import agentic
            bench.load_corpus()
            cold = agentic.cold_start(ba)
            cold["spawn_to_ready_s"] = prov["ready_s"] if proc else None  # external server: not ours to time
            cold["spawn_to_first_token_s"] = (round(prov["ready_s"] + cold["first"]["ttft_s"], 2)
                                              if proc and cold["first"].get("ttft_s") is not None else None)
            log(f"{label}: cold start: ready {cold['spawn_to_ready_s']}s, first TTFT {cold['first'].get('ttft_s')}s, "
                f"repeat median {cold['repeat_median_ttft_s']}s")
        result["warmup"] = bench.warmup(ba, adaptive=bool(c.get("adaptive_warm")), log=lambda m: log(f"{label}:{m}"))
        if not any(s["ok"] for s in result["warmup"]["steps"]):
            raise ResolveError(f"warm-up failed: {result['warmup']['steps'][0].get('error')}")
        leaks = [s["name"] for s in result["warmup"]["steps"] if s.get("thinking_leak")]
        if leaks and not getattr(ba, "always_reasons", False):  # (a model that always reasons: expected)
            log(f"{label}: WARNING: thinking was not switched off in warm-up {leaks}; runs will be INVALID")
        qok, qwait, busy = gpu_hogs.wait_quiet(args.quiet_max, args.quiet_s, args.quiet_timeout, exclude_roots=lambda: mon_roots,
                                               log=lambda m: log(f"{label}: {m}"), own_max=args.quiet_max)
        prov["gpu_quiet_wait_s"] = round(qwait, 1)
        if not qok:
            # We hold the GPU lock, so whatever is still busy is outside our control (another
            # session, a system daemon). Measure anyway and let the contention monitor flag it,
            # rather than throwing away the slot.
            prov["gpu_not_quiet"] = [f"{n}[{pp}] {pct:.0f}%" for pct, (pp, n) in busy]
            log(f"{label}: GPU not quiet after {qwait:.0f}s ({prov['gpu_not_quiet']}); benchmarking anyway, contention will be flagged")
        else:
            log(f"{label}: GPU quiet after {qwait:.0f}s; benchmarking")
        mon = gpu_hogs.ContentionMonitor(lambda: mon_roots)
        mon.start()
        t_bench0 = time.time()
        try:
            rc = subprocess.run([sys.executable, "-u", os.path.join(HERE, "bench.py")] + p["bench_args"] +
                                (["--pid", str(srv_pid)] if srv_pid else []),
                                env=dict(os.environ, NACE_GPU_LOCK_HELD="1")).returncode
        finally:
            t_bench1 = time.time()
            mon.stop()
        prov["bench_rc"] = rc
        try:
            result = dict(json.load(open(p["out"])), warmup=result["warmup"])
            if cold:
                result["cold_start"] = cold
                result["invalid"] = list(result.get("invalid") or []) + cold.get("invalid", [])
        except Exception as e:
            result.update({"schema": bench.SCHEMA, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "invalid": [f"bench produced no result (rc={rc}): {e!r}"[:300]], "flags": []})
        phases = result.get("phases", {})
        cont = {"overall": gpu_hogs.summarize_samples(mon.samples, t_bench0, t_bench1),
                "phases": {k: gpu_hogs.summarize_samples(mon.samples, v[0], v[1]) for k, v in phases.items() if k != "warmup"},
                "samples": mon.samples, "threshold_pct": gpu_hogs.FOREIGN_MAX_PCT,
                "ambient_names": sorted(gpu_hogs.AMBIENT)}
        result["contention"] = cont
        bad = [k for k, v in cont["phases"].items() if v.get("flagged")]
        if bad or cont["overall"].get("flagged"):
            result.setdefault("flags", []).append(
                f"CONTENTION: sustained foreign GPU load > {gpu_hogs.FOREIGN_MAX_PCT:.0f}% in {bad or ['bench']} "
                f"(mean {cont['overall'].get('mean_foreign_gpu_pct')}%, top {cont['overall'].get('top')})")
    except (ResolveError, TimeoutError, OSError) as e:
        log(f"{label}: FAILED: {e}")
        result.update({"schema": bench.SCHEMA, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "invalid": [f"harness: {e}"[:400]], "flags": []})
    finally:
        try:
            if kill_roots:
                teardown = stop_server(c, proc, kill_roots, port, ctx, base_mem, p.get("env"))
                log(f"{label}: stopped ({teardown.get('killed_with')}, survivors {teardown.get('survivors')}, "
                    f"memory released after {teardown.get('memory_release_s')}s)")
            if c.get("cleanup"):
                try:
                    teardown["cleanup"] = do_prepare(c["cleanup"], ctx, False)
                except (ResolveError, OSError) as e:
                    teardown["cleanup"] = [f"<failed: {e}>"]
            teardown["disk_free_bytes"] = disk_free()
        finally:
            held = lock.release()
    # merge config + provenance into the result
    cfg = {k: v for k, v in c.items() if not k.startswith("_")}
    result.update({"config_label": label, "display": c.get("display") or label, "engine": engine_kind(c),
                   "version": p["version"], "version_config": c.get("version"), "model_name": c.get("model_name"),
                   "model_path": c.get("model_path"), "note": c.get("note"), "config": cfg, "provenance": prov,
                   "gpu_lock": dict(result.get("gpu_lock") or {}, harness_wait_s=round(lock.wait_s or 0, 2),
                                    harness_held_s=round(held, 1),
                                    harness_lock=f"held {lock.why}" if isinstance(lock, NoLock) else "flock by the harness"),
                   "teardown": teardown})
    free0 = ((prov.get("system_start") or {}).get("disk_free_bytes"))
    if engine_kind(c) in DISK_CACHE_ENGINES and free0 is not None and free0 < DISK_CACHE_FLOOR:
        result.setdefault("flags", []).append(
            f"disk: {free0 / 2**30:.1f} GiB free at start, under the 10 GiB below which {engine_kind(c)} keeps no "
            "prompt cache on disk")
    if c.get("version") not in (None, "auto"):
        det = detect_version(c, p["cmd"], ctx)
        result["version_detected"] = det
        if det and c["version"] not in det:
            result.setdefault("flags", []).append(f"version: config says {c['version']}, installed is {det}")
    if p["cmd"] and os.path.exists(p["log"]):
        result["server_log"] = grep_log(p["log"], c.get("log_grep") or DEFAULT_LOG_GREP.get(engine_kind(c), []))
        result["server_log_path"] = os.path.relpath(p["log"], OUT)
    os.makedirs(RESULTS, exist_ok=True)
    write_json(p["out"], result)
    log(f"{label}: -> {p['out']}" + (f"  INVALID: {result['invalid']}" if result.get("invalid") else "")
        + (f"  flags: {len(result.get('flags', []))}" if result.get("flags") else ""))


def dry_run(c, args, stamp):
    p, ctx = plan(c, args, stamp)
    status = "SKIP: " + (p["skip"] or "; ".join(p["errors"])) if (p["skip"] or p["errors"]) else "ready"
    ver = p["version"]
    if c.get("version") not in (None, "auto", "TBD"):
        ver += f" (installed: {detect_version(c, p['cmd'], ctx) or '?'})"
    print(f"\n===== {c['label']}  [{c.get('engine')}] \"{c.get('display')}\"  model {c.get('model_name')}  version {ver}")
    print(f"  config   : {c.get('_file')}")
    print(f"  status   : {status}")
    if c.get("note"):
        print(f"  note     : {c['note']}")
    if not str(c.get("base_url", "")).startswith("http"):
        return
    if p["cmd"] is None:
        print("  server   : (none: already-running server)")
    else:
        cmd = p["cmd"]
        if args.check_flags and c.get("flag_check") == "llama-server" and not p["errors"]:
            cmd, changes, missing = adapt_llama_cmd(cmd, llama_help(cmd[0]))
            print(f"  flags    : checked against `{cmd[0]} --help`: changes {changes or 'none'}, unknown {missing or 'none'}")
        env_s = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in redact(p["env"]).items())
        clr = c.get("env_clear_prefixes")
        print(f"  env      : {env_s or '-'}" + (f"   (inherited {', '.join(x + '*' for x in clr)} cleared)" if clr else ""))
        try:
            for line in do_prepare(c.get("prepare"), ctx, True):
                print(f"  prepare  : {line}")
        except (ResolveError, OSError) as e:
            print(f"  prepare  : <{e}>")
        print(f"  server   : {shlex.join(cmd)}")
        print(f"  log      : {os.path.relpath(p['log'], OUT)}")
    print(f"  ready    : GET {c.get('base_url')}/models; request model id {p['bench_args'][p['bench_args'].index('--model') + 1]!r}")
    if c.get("pid_file"):
        print(f"  server   : measured as the process tree of the pid in {expand(c['pid_file'], ctx)} (detached daemon)")
    print(f"  lock     : " + ("held externally by the server's launcher (lock=external)" if c.get("lock") == "external"
                              else f"flock {gpu_hogs.LOCK_PATH} from server start to memory release"))
    print(f"  warm-up  : ~{bench.WARM_TOKENS}-token nonce'd prompt + 64-token decode, 1 tools request"
          + (", 3 x full decode (adaptive speculation)" if c.get("adaptive_warm") else ""))
    print(f"  quiet    : every other process < {args.quiet_max:.0f}% GPU and the idle server tree < {args.quiet_max:.0f}% for 15 s, "
          f"timeout {args.quiet_timeout:.0f}s; ambient: {', '.join(sorted(gpu_hogs.AMBIENT))}")
    ba = bench.parse_args(p["bench_args"])
    skipped = set(filter(None, ba.skip.split(",")))
    phases = [x for x in ("decode", "greedy", "prefill", "tools") if x not in skipped]
    if ba.agent_trace and "agent" not in skipped:
        phases.append(f"agent ({os.path.relpath(ba.agent_trace, HERE)}" + ("" if os.path.exists(ba.agent_trace) else ", MISSING")
                      + f", tool mode {ba.agent_tool_mode})")
    print(f"  phases   : {', '.join(phases) or '(none)'}")
    print(f"  bench    : NACE_GPU_LOCK_HELD=1 python3 bench.py {shlex.join(p['bench_args'])} --pid <server pid>")
    if c.get("stop"):
        try:
            print(f"  stop     : {shlex.join(resolve_argv(c['stop'], ctx))} + kill process tree")
        except ResolveError as e:
            print(f"  stop     : <{e}> + kill process tree")
    print(f"  result   : {os.path.relpath(p['out'], OUT)} (in {OUT})")


_STOPPING = []


def _exit_on_signal(signum, frame):
    """SIGINT, SIGTERM and SIGHUP end the run, and the running config's finally block still stops its server (a
    detached daemon such as LM Studio's would otherwise keep its model loaded) and releases the GPU lock. Signals
    that arrive while that teardown runs are ignored, so a second Ctrl+C or a job runner's follow-up SIGTERM cannot
    cut it short."""
    if _STOPPING:
        return
    _STOPPING.append(signum)
    raise SystemExit(128 + signum)


def main():
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)
    ap = argparse.ArgumentParser(description="Run bench.py over engine configs, one at a time, under the GPU lock.")
    ap.add_argument("configs", nargs="+", help="config files (configs/*.json); merged in order")
    ap.add_argument("--only", default="", help="comma list of labels or globs, e.g. q36-*,q38-27b-kapitsa")
    ap.add_argument("--list", action="store_true", help="list the configs and exit")
    ap.add_argument("--sizes", default="4000,16000")
    ap.add_argument("--bench-args", default="", help="extra bench.py args for every config (shell-quoted string)")
    ap.add_argument("--skip", default="", help="bench.py phases to leave out: decode,greedy,prefill,tools,agent "
                    "(e.g. --skip decode,greedy,prefill,tools runs only the agent session)")
    ap.add_argument("--agent-trace", default="", help="agent session to replay for every config (default: the "
                    "model's entry in agent-traces/index.json; none = no agent phase)")
    ap.add_argument("--agentic", action="store_true", help="also measure the agent-harness metrics (agentic.py): cold start, "
                    "context sweep, cache behaviour, sustained decode/thermal; tune with --bench-args '--sustain-s 60 --sweep-sizes 1000,16000'")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan per config; start nothing")
    ap.add_argument("--check-flags", action="store_true", help="with --dry-run: run `llama-server --help` and check flags")
    ap.add_argument("--reverse", action="store_true", help="run configs in reverse order (interleave across sessions)")
    ap.add_argument("--quiet-max", type=float, default=gpu_hogs.FOREIGN_MAX_PCT)
    ap.add_argument("--quiet-timeout", type=float, default=300)
    ap.add_argument("--quiet-s", type=float, default=15, help="seconds the GPU must stay quiet before a run (default 15)")
    ap.add_argument("--lock-timeout", type=float, default=None)
    ap.add_argument("--out-dir", default=None, help="results, logs and engine state (default $BENCH_OUT or bench/harness/runs)")
    ap.add_argument("--env-file", default=None, help=f"KEY=VALUE file of paths for this Mac (default {os.path.relpath(bench.LOCAL_ENV)} if present)")
    args = ap.parse_args()
    if args.env_file and not os.path.exists(args.env_file):
        raise SystemExit(f"no env file {args.env_file}")
    env_file = args.env_file or (bench.LOCAL_ENV if os.path.exists(bench.LOCAL_ENV) else None)
    args.env_keys = sorted(bench.load_env_file(env_file)) if env_file else []
    set_out_dir(args.out_dir or bench.out_dir())
    os.environ["BENCH_OUT"] = OUT  # bench.py (a child) writes where we read
    cfgs = load_configs(args.configs)
    pats = [x for x in args.only.split(",") if x]
    for pat in pats:
        if not any(fnmatch.fnmatch(c["label"], pat) for c in cfgs):
            raise SystemExit(f"--only {pat}: no such label (labels: {', '.join(c['label'] for c in cfgs)})")
    sel = [c for c in cfgs if not pats or any(fnmatch.fnmatch(c["label"], x) for x in pats)]
    if args.reverse:
        sel = sel[::-1]
    if args.list:
        for c in sel:
            print(f"{c['label']:28} {c.get('model_name') or '-':17} {c.get('engine') or '-':10} {c.get('display') or ''}"
                  + ("  [disabled]" if c.get("enabled") is False else ""))
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.dry_run:
        print(f"dry run of {', '.join(args.configs)}: {len(sel)} configs; corpus {bench.CORPUS_FILE} "
              f"({'frozen corpus, sha256 ok' if bench.corpus_record()['frozen'] else 'NOT the frozen corpus'}); "
              f"models {models_dir()}; out {OUT}" + (f"; env file {env_file} ({len(args.env_keys)} variables set)" if env_file else ""))
        for c in sel:
            dry_run(c, args, stamp)
        return
    for c in sel:
        run_config(c, args, time.strftime("%Y%m%d-%H%M%S"))


if __name__ == "__main__":
    main()
