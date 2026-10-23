#!/usr/bin/env python3
"""`kapitsa update` and the automatic update, on the CPU, against a release server in this process
(KAPITSA_UPDATE_URL): nothing is fetched from GitHub, and only temporary installs are touched.

usage: test_update.py [BUILD_DIR]   (default: build; needs BUILD_DIR/kapitsa and BUILD_DIR/update_test, the full build)

The releases are real ones: this build's kapitsa signed as packaging/release/build.sh signs a release (ad hoc, hardened
runtime, identifier ai.nace.kapitsa) and packed as it packs one, with its sha256 and its bench/harness. A newer (or
older) release is the same binary with its version string changed, signed again: a real kapitsa that is the version it
says (checked first).
Installs are made by install.sh --from-file into temporary folders.

1. update_test: version order, release names, kinds of install by path.
2. --check: up to date, newer, older, prereleases, numeric order, no release, unreachable, a bad KAPITSA_UPDATE_URL.
3. kapitsa update on an install.sh install: the binary switched in one rename (it runs throughout), the old one kept as
   the rollback copy (its inode: a process running it keeps it), the license files and the benchmark harness updated,
   the state; an install from before the harness shipped gets it.
4. Refused, with nothing changed: a wrong sha256, an unsigned binary, a broken signature, another identifier, another
   version, a tarball that is not a release, a symlink, a redirect away from GitHub, a missing release.
5. --rollback (twice), --version (an older release, asked for), and the errors around them.
6. Homebrew, app bundle and source build: detected, never replaced; brew upgrade runs in a terminal only.
7. Automatic updates: a command starts a detached low-priority child once a day, which does not hold the command up or
   change its output; the line on the next run in a terminal; the throttle; the off switches and the commands left
   alone; the time limit; kapitsa bench holding the binary; a binary replaced meanwhile; two installs, one home; a
   symlink; refusals and network failures; a folder that can't be written; a rolled-back release skipped.
8. What the requests carry: a fixed User-Agent and language, no cookies, nothing about the Mac; GETs and HEADs only.
"""
import errno, fcntl, hashlib, http.server, io, json, os, pty, re, select, shutil, signal, subprocess, sys, tarfile
import tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
BUILD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "build"))
KAPITSA, UPDATE_TEST = os.path.join(BUILD, "kapitsa"), os.path.join(BUILD, "update_test")
IDENTIFIER = "ai.nace.kapitsa"  # packaging/release/build.sh's
DAY = 24 * 3600

FAILS = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else f"  ({str(detail)[-600:]})"), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def run(args, e, timeout=60, **kw):
    return subprocess.run(args, env=e, capture_output=True, text=True, timeout=timeout, **kw)


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x04\x08]")


def tty(args, e, timeout=30):
    """Runs args on a pseudo-terminal (stdin, stdout and stderr): (exit code, output without escape codes)."""
    pid, fd = pty.fork()
    if pid == 0:
        try:
            os.execve(args[0], args, e)
        finally:
            os._exit(127)
    out, t0 = b"", time.time()
    while time.time() - t0 < timeout:
        rl, _, _ = select.select([fd], [], [], 0.2)
        if rl:
            try:
                b = os.read(fd, 65536)
            except OSError:
                break
            if not b:
                break
            out += b
    else:
        os.kill(pid, signal.SIGKILL)
    _, st = os.waitpid(pid, 0)
    os.close(fd)
    # one space between words: a long line may wrap
    return os.waitstatus_to_exitcode(st), " ".join(ANSI.sub("", out.decode("utf-8", "replace")).split())


def has(out, text):
    """`text` is in a terminal's output, whatever wraps it (whitespace aside)"""
    return "".join(text.split()) in "".join(out.split())


def wait_for(pred, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def version_of(path, e=None):
    r = subprocess.run([path, "--version"], capture_output=True, text=True, env=e or BASE_ENV, timeout=30)
    m = re.match(r"kapitsa (\S+)", r.stdout)
    return m.group(1) if m else ""


TMP = os.path.realpath(tempfile.mkdtemp(prefix="kapitsa-update-test-"))
USER_HOME = os.path.join(TMP, "user")
os.makedirs(USER_HOME)
# nothing of this Mac's own: a home of its own, no Hugging Face, no question asked, a GPU lock of its own (never the real one)
BASE_ENV = {k: v for k, v in os.environ.items()
            if k not in ("CI", "KAPITSA_NO_AUTO_UPDATE", "KAPITSA_UPDATE_URL", "KAPITSA_UPDATE_TIMEOUT_S", "KAPITSA_HOME",
                         "NO_COLOR", "FORCE_COLOR", "CLICOLOR_FORCE", "NI_LOG", "TERM_PROGRAM", "TMUX", "STY", "COLORFGBG")}
BASE_ENV.update(HOME=USER_HOME, HF_ENDPOINT="http://127.0.0.1:9", KAPITSA_NO_PROMPT="1", TERM="xterm-256color",
                COLUMNS="200", BENCH_GPU_LOCK=os.path.join(TMP, "gpu.lock"))


# ---------------------------------------------------------------- the release server


class Releases(http.server.BaseHTTPRequestHandler):
    """GitHub's release pages as the updater uses them: /releases/latest redirects to /releases/tag/vX.Y.Z (to /releases
    when there is no release), and /releases/download/vX.Y.Z/<file> serves the files. Every request is logged."""
    files = {}       # URL path -> bytes
    latest = None    # the latest release's version
    delay = 0.0      # seconds before /releases/latest answers
    hang = False     # /releases/latest never answers
    redirect = {}    # URL path -> Location
    log = []         # (method, path, headers)
    mu = threading.Lock()

    def log_message(self, *a):
        pass

    def answer(self):
        with Releases.mu:
            Releases.log.append((self.command, self.path, dict(self.headers)))
        p = self.path
        if p.startswith("/hang"):  # a Hugging Face that never answers (a kapitsa kept running, waiting on it)
            time.sleep(90)
            return
        if p == "/releases/latest":
            if Releases.hang:
                time.sleep(90)
                return
            time.sleep(Releases.delay)
            return self.reply(302, b"", {"Location": "/releases/tag/v" + Releases.latest if Releases.latest else "/releases"})
        if p in Releases.redirect:
            return self.reply(302, b"", {"Location": Releases.redirect[p]})
        if p == "/releases" or p.startswith("/releases/tag/"):
            return self.reply(200, b"<html>a release page</html>")
        if p in Releases.files:
            return self.reply(200, Releases.files[p])
        self.reply(404, b"Not Found")

    def reply(self, code, body, headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_GET = do_HEAD = answer


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Releases)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
PORT = server.server_address[1]
URL = f"http://127.0.0.1:{PORT}/releases"


def logged():
    with Releases.mu:
        return len(Releases.log)


def since(n, method=None):
    with Releases.mu:
        return [x for x in Releases.log[n:] if method is None or x[0] == method]


def env(home, **kw):
    e = dict(BASE_ENV, KAPITSA_HOME=home, KAPITSA_UPDATE_URL=URL)
    for k, v in kw.items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    return e


# ---------------------------------------------------------------- releases and installs

V = version_of(KAPITSA)
if not re.fullmatch(r"\d+\.\d+\.\d+", V):
    sys.exit(f"test_update.py: {KAPITSA} --version gives no X.Y.Z version ({V!r})")


def bump(v, by=1):
    """A version `by` releases newer (older for a negative `by`) with as many characters, so that a binary can say it"""
    maj, mi, pa = map(int, v.split("."))
    for c in ((maj, mi, pa + by), (maj, mi + by, pa), (maj + by, mi, pa)):
        s = "%d.%d.%d" % c
        if min(c) >= 0 and len(s) == len(v):
            return s
    sys.exit(f"test_update.py: no version {by:+d} from {v} with as many characters")


N, N2, O = bump(V), bump(V, 2), bump(V, -1)  # the next release, the one after, the one before


def sign(path, identifier=IDENTIFIER):
    subprocess.run(["codesign", "--force", "--sign", "-", "--identifier", identifier, "--options", "runtime", path],
                   check=True, capture_output=True)


_bins = {}


def binary(version, how="signed", identifier=IDENTIFIER):
    """A kapitsa that reports `version`: this build's binary, its version string changed (same length), then signed as a
    release (how="signed"), or left with a signature the change broke ("broken"), or without one ("unsigned")."""
    key = (version, how, identifier)
    if key in _bins:
        return _bins[key]
    d = tempfile.mkdtemp(prefix="bin-", dir=TMP)
    path = os.path.join(d, "kapitsa")
    shutil.copy(KAPITSA, path)
    sign(path, identifier)  # first as a release, so that "broken" breaks a valid release signature
    if version != V:
        b = open(path, "rb").read()
        old, new = (V + "\0").encode(), (version + "\0").encode()
        hits = [m.start() for m in re.finditer(re.escape(old), b) if b[m.start() - 1:m.start()] not in b"0123456789."]
        for i in hits:
            b = b[:i] + new + b[i + len(new):]
        open(path, "wb").write(b)
    if how == "signed":
        sign(path, identifier)
    elif how == "unsigned":
        subprocess.run(["codesign", "--remove-signature", path], check=True, capture_output=True)
    _bins[key] = path
    return path


def tar_release(version, binary_path=None, sha256=None, files=("kapitsa", "LICENSE", "THIRD_PARTY_NOTICES"), symlink=False,
                harness=True, companions=None):
    """Serves release `version` as packaging/release/build.sh packs it: kapitsa, LICENSE, THIRD_PARTY_NOTICES and
    bench/harness in a folder of the release's name (ustar, gzip), and its .sha256 (shasum's format) or `sha256` in
    its place. Also written to TMP/dl for install.sh --from-file. Returns the tarball's path there.
    companions: the path of a kapitsa-systemone to pack, with a kapitsa-drex-pointer of `version` beside it."""
    name = f"kapitsa-{version}-darwin-arm64"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.USTAR_FORMAT) as t:
        def entry(arc, kind=tarfile.REGTYPE, mode=0o644, data=b""):
            ti = tarfile.TarInfo(arc)
            ti.type, ti.mode, ti.size, ti.mtime, ti.uname, ti.gname = kind, mode, len(data), 0, "root", "wheel"
            t.addfile(ti, io.BytesIO(data) if data else None)
        entry(name, tarfile.DIRTYPE, 0o755)
        if symlink:  # kapitsa as a symlink to something else on this Mac
            ti = tarfile.TarInfo(f"{name}/kapitsa")
            ti.type, ti.linkname = tarfile.SYMTYPE, "/bin/ls"
            t.addfile(ti)
        elif "kapitsa" in files:
            entry(f"{name}/kapitsa", mode=0o755, data=open(binary_path or binary(version), "rb").read())
        for f in ("LICENSE", "THIRD_PARTY_NOTICES"):
            if f in files:
                entry(f"{name}/{f}", data=f"{f} of {version}\n".encode())
        if harness:  # bench/harness, which `kapitsa bench compare` runs (a nested file too, as the real one has)
            entry(f"{name}/bench/harness/kbench.py", mode=0o755, data=f"# kbench.py of {version}\n".encode())
            entry(f"{name}/bench/harness/configs/q35-4b.json", data=f'{{"release": "{version}"}}\n'.encode())
        if companions:
            entry(f"{name}/kapitsa-systemone", mode=0o755, data=open(companions, "rb").read())
            entry(f"{name}/kapitsa-drex-pointer", mode=0o755, data=f"#!/bin/sh\necho drex_pointer {version}\n".encode())
    data = buf.getvalue()
    sums = f"{sha256 or hashlib.sha256(data).hexdigest()}  {name}.tar.gz\n".encode()
    url = f"/releases/download/v{version}/{name}.tar.gz"
    Releases.files[url], Releases.files[url + ".sha256"] = data, sums
    os.makedirs(os.path.join(TMP, "dl"), exist_ok=True)
    path = os.path.join(TMP, "dl", f"{name}.tar.gz")
    open(path, "wb").write(data)
    open(path + ".sha256", "wb").write(sums)
    return path


def standalone(tag, version=None):
    """An install.sh install of release `version` (default: this build's) into TMP/tag/bin: its kapitsa"""
    version = version or V
    prefix = os.path.join(TMP, tag, "bin")
    r = run([os.path.join(REPO, "install.sh"), "--from-file", tar_release(version), "--prefix", prefix], BASE_ENV, timeout=120)
    if r.returncode != 0:
        sys.exit(f"test_update.py: install.sh failed: {r.stderr[-500:]}")
    tar_release(version)  # served again as it was (an earlier case may have served something else under its name)
    return os.path.join(prefix, "kapitsa")


def home(tag):
    h = os.path.join(TMP, tag, "home")
    os.makedirs(h, exist_ok=True)
    return h


def state(h):
    try:
        return json.load(open(os.path.join(h, "update.json")))
    except (OSError, ValueError):
        return {}


def rec(h, path):
    return state(h).get("binaries", {}).get(path, {})


def set_rec(h, path, **kw):
    """Changes a binary's record in the state file, as time passing would (checked) or as a check left it"""
    s = state(h) or {"automatic": True, "binaries": {}}
    s.setdefault("binaries", {}).setdefault(path, {}).update(kw)
    for k, v in list(s["binaries"][path].items()):
        if v is None:
            del s["binaries"][path][k]
    json.dump(s, open(os.path.join(h, "update.json"), "w"))


def children(path):
    """The background checks of the binary at `path` still running: [(pid, ppid, pgid, pri, tty)]"""
    r = subprocess.run(["ps", "-axww", "-o", "pid=,ppid=,pgid=,pri=,tty=,command="], capture_output=True, text=True)
    out = []
    for l in r.stdout.splitlines():
        f = l.split(None, 5)
        if len(f) == 6 and f[5] == f"{path} update --background":
            out.append(tuple(f[:5]))
    return out


def quiet(path, timeout=20):
    return wait_for(lambda: not children(path), timeout)


def leftovers(binpath):
    return [n for n in os.listdir(os.path.dirname(binpath)) if n.startswith(".kapitsa-update-")]


def notices_of(binpath):
    return os.path.join(os.path.dirname(os.path.dirname(binpath)), "share", "kapitsa")


# ---------------------------------------------------------------- 1. the pieces

r = subprocess.run([UPDATE_TEST], capture_output=True, text=True, timeout=60)
check(r.returncode == 0 and "0 failed" in r.stdout, "update_test: versions, release names, kinds of install", r.stdout + r.stderr)

# the fixtures: the next release must be a coherent kapitsa of that version (its update code sees it too)
Releases.latest = N
bn = binary(N)
r = run([bn, "update", "--check"], env(home("fixture")))
if not check(version_of(bn) == N and r.returncode == 0 and "up to date" in r.stdout,
             f"fixture: the {N} binary is a kapitsa {N} through and through", r.stdout + r.stderr):
    sys.exit(1)
for v in (N2, O):
    check(version_of(binary(v)) == v, f"fixture: a kapitsa {v}")
tar_release(N), tar_release(N2), tar_release(O)

# ---------------------------------------------------------------- 2. --check

S1, H1 = standalone("s1"), home("s1")
sha_s1 = sha(S1)
maj, mi, pa = map(int, V.split("."))
for latest, code, says in [(V, 0, "is up to date (the latest release)"),
                           (N, 100, f"kapitsa {N} is available (this is {V}). Update: kapitsa update"),
                           (O, 0, f"newer than the latest release, {O}"), (V + "-rc.1", 0, "newer than the latest release"),
                           (N + "-rc.1", 100, "is available"), (f"{maj}.{mi}.1{pa}", 100, "is available")]:
    Releases.latest = latest
    n0 = logged()
    r = run([S1, "update", "--check"], env(H1))
    check(r.returncode == code and says in r.stdout and not since(n0, "GET"),
          f"--check, latest {latest}: exit {code}, nothing downloaded", f"{r.returncode} {r.stdout} {r.stderr}")
Releases.latest = None
r = run([S1, "update", "--check"], env(H1))
check(r.returncode == 1 and "found no release" in r.stderr, "--check with no release: exit 1", r.stderr)
r = run([S1, "update", "--check"], env(H1, KAPITSA_UPDATE_URL="http://127.0.0.1:9/releases"))
check(r.returncode == 1 and "Can't check for a new release" in r.stderr, "--check, nothing answering: exit 1", r.stderr)
r = run([S1, "update", "--check"], env(H1, KAPITSA_UPDATE_URL="http://example.com/releases"))
check(r.returncode == 1 and "KAPITSA_UPDATE_URL must be an https URL" in r.stderr, "plain http is refused but to this Mac", r.stderr)
check(sha(S1) == sha_s1 and not os.path.exists(os.path.join(H1, "update.json")), "--check changes nothing, not even the state")

# ---------------------------------------------------------------- 3. kapitsa update on an install.sh install

Releases.latest = N
ino0, size0 = os.stat(S1).st_ino, os.stat(S1).st_size
held = open(S1, "rb")  # as a running process holds its binary
# a kapitsa of the old binary that keeps running meanwhile (waiting on a Hugging Face that never answers)
runner = subprocess.Popen([S1, "pull", "4b"], env=env(H1, KAPITSA_NO_AUTO_UPDATE="1", HF_ENDPOINT=f"http://127.0.0.1:{PORT}/hang"),
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.5)
# the binary runs throughout the update: never missing, never half written
seen, bad, stop = set(), [], threading.Event()


def watch():
    while not stop.is_set():
        try:
            v = version_of(S1, BASE_ENV)
            seen.add(v)
            if v not in (V, N):
                bad.append(v)
        except Exception as ex:  # noqa: BLE001
            bad.append(repr(ex))


w = threading.Thread(target=watch)
w.start()
time.sleep(0.3)
r = run([S1, "update"], env(H1))
time.sleep(0.3)
stop.set()
w.join()
prev = os.path.join(notices_of(S1), "kapitsa.previous")
check(r.returncode == 0 and f"Updated kapitsa {V} -> {N}: {S1}" in r.stdout and version_of(S1) == N,
      f"kapitsa update installs {N}", r.stdout + r.stderr)
check(os.stat(S1).st_ino != ino0 and os.path.exists(prev) and os.stat(prev).st_ino == ino0 and sha(prev) == sha_s1,
      "a new file renamed into place; the old binary kept, the same inode, as the rollback copy")
check(not bad and seen == {V, N}, "the binary ran throughout: the old version, then the new one, nothing in between",
      f"seen {seen}, bad {bad[:3]}")
check(os.fstat(held.fileno()).st_ino == ino0 and hashlib.sha256(held.read()).hexdigest() == sha_s1,
      "a process holding the old binary keeps it unchanged")
held.close()
check(runner.poll() is None, "a kapitsa running the old binary keeps running")
runner.kill(), runner.wait()
check(open(os.path.join(notices_of(S1), "LICENSE")).read() == f"LICENSE of {N}\n" and
      open(os.path.join(notices_of(S1), "THIRD_PARTY_NOTICES")).read() == f"THIRD_PARTY_NOTICES of {N}\n",
      "the license files updated where install.sh put them")
kb1 = os.path.join(notices_of(S1), "bench", "harness", "kbench.py")
check(open(kb1).read() == f"# kbench.py of {N}\n" and os.access(kb1, os.X_OK) and
      open(os.path.join(notices_of(S1), "bench", "harness", "configs", "q35-4b.json")).read() == f'{{"release": "{N}"}}\n',
      "the benchmark harness updated beside the notices, where bench compare looks for it", open(kb1).read())
check(not leftovers(S1), "no temporary folder left beside the binary", leftovers(S1))
r1 = rec(H1, S1)
check(r1.get("result") == "updated" and r1.get("from") == V and r1.get("to") == N and not r1.get("notice") and
      time.time() - r1.get("checked", 0) < 60, "the state: updated, from and to, checked now, nothing left to say", r1)
n0 = logged()
r = run([S1, "update"], env(H1))
check(r.returncode == 0 and f"kapitsa {N} is up to date" in r.stdout and not since(n0, "GET"),
      "run again: up to date, nothing downloaded", r.stdout + r.stderr)

# a user's local.env in the harness survives an update
SH1, HH1 = standalone("sh1"), home("sh1")
lenv = os.path.join(notices_of(SH1), "bench", "harness", "local.env")
os.makedirs(os.path.dirname(lenv), exist_ok=True)
open(lenv, "w").write("LLAMA_SERVER=/x/llama-server\n")
tar_release(N)
r = run([SH1, "update"], env(HH1))
kb1b = os.path.join(notices_of(SH1), "bench", "harness", "kbench.py")
check(r.returncode == 0 and open(kb1b).read() == f"# kbench.py of {N}\n" and os.path.isfile(lenv)
      and open(lenv).read() == "LLAMA_SERVER=/x/llama-server\n",
      "the harness updated and the user's local.env kept", r.stdout + r.stderr)

# an install made before releases shipped the harness (v0.2.5 and older) gets it from the next update
SH2, HH2 = standalone("sh2"), home("sh2")
shutil.rmtree(os.path.join(notices_of(SH2), "bench"), ignore_errors=True)
tar_release(N)
r = run([SH2, "update"], env(HH2))
hk2 = os.path.join(notices_of(SH2), "bench", "harness", "kbench.py")
check(r.returncode == 0 and os.path.isfile(hk2) and open(hk2).read() == f"# kbench.py of {N}\n",
      "an install without a harness gets it on update", r.stdout + r.stderr)

# kapitsa-systemone and kapitsa-drex-pointer (Drex decision models) come with a release that carries them, beside the
# binary; a server signed as something else is left out with a warning, and kapitsa itself still updates
SYSTEMONE = os.path.join(BUILD, "systemone")
if not os.path.isfile(SYSTEMONE):
    print(f"SKIP the Drex companions on update: {SYSTEMONE} is not built (cmake --build build --target systemone; "
          f"nace-ai/kapitsa#422 adds it to CI's build targets)")
else:
    def companion(identifier):
        d = tempfile.mkdtemp(prefix="sys-", dir=TMP)
        path = os.path.join(d, "kapitsa-systemone")
        shutil.copy(SYSTEMONE, path)
        sign(path, identifier)
        return path
    SC1, HC1 = standalone("sc1"), home("sc1")
    tar_release(N, companions=companion(IDENTIFIER + "-systemone"))
    r = run([SC1, "update"], env(HC1))
    sys1, ptr1 = os.path.join(os.path.dirname(SC1), "kapitsa-systemone"), os.path.join(os.path.dirname(SC1), "kapitsa-drex-pointer")
    h = run([sys1, "--help"], BASE_ENV) if os.path.isfile(sys1) else None
    check(r.returncode == 0 and version_of(SC1) == N and h is not None and h.returncode == 0 and
          "usage: systemone" in h.stderr and os.access(ptr1, os.X_OK) and
          run([ptr1], BASE_ENV).stdout == f"drex_pointer {N}\n",
          "an update installs kapitsa-systemone and kapitsa-drex-pointer beside the binary", r.stdout + r.stderr)
    SC2, HC2 = standalone("sc2"), home("sc2")
    tar_release(N, companions=companion("ai.example.other"))
    r = run([SC2, "update"], env(HC2))
    check(r.returncode == 0 and version_of(SC2) == N and
          not os.path.exists(os.path.join(os.path.dirname(SC2), "kapitsa-systemone")) and
          "kapitsa-systemone" in r.stdout + r.stderr,
          "a kapitsa-systemone signed as another program is not installed, with a warning; kapitsa updates", r.stdout + r.stderr)
    tar_release(N)

# ---------------------------------------------------------------- 4. refused releases change nothing

S2, H2 = standalone("s2"), home("s2")
sha2, ino2 = sha(S2), os.stat(S2).st_ino
lic2 = open(os.path.join(notices_of(S2), "LICENSE")).read()
Releases.latest = N
cases = [
    ("a wrong sha256", dict(sha256="0" * 64), "its sha256 is"),
    ("an unsigned binary", dict(binary_path=binary(N, "unsigned")), "it is not signed"),
    ("a signature broken by a change", dict(binary_path=binary(N, "broken")), "its code signature is not valid"),
    ("another identifier", dict(binary_path=binary(N, identifier="com.example.kapitsa")), 'signed as "com.example.kapitsa"'),
    ("another version inside", dict(binary_path=binary(V)), f"is version {V}"),
    ("no license files", dict(files=("kapitsa",)), "is not a Kapitsa release"),
    ("kapitsa a symlink", dict(symlink=True), "is not a Kapitsa release (no kapitsa file)"),
]
for what, kw, says in cases:
    tar_release(N, **kw)
    r = run([S2, "update"], env(H2))
    check(r.returncode == 1 and says in r.stderr and "Nothing was changed" in r.stderr, f"refused: {what}", r.stderr)
tar_release(N)
url = f"/releases/download/v{N}/kapitsa-{N}-darwin-arm64.tar.gz"
Releases.redirect[url] = "https://example.com/kapitsa.tar.gz"
r = run([S2, "update"], env(H2))
check(r.returncode == 1 and "redirects to https://example.com/kapitsa.tar.gz, which is not GitHub" in r.stderr,
      "refused: a download redirected away from GitHub (not followed)", r.stderr)
del Releases.redirect[url]
saved = Releases.files.pop(url)
r = run([S2, "update"], env(H2))
check(r.returncode == 1 and "there is no" in r.stderr, "a release without its tarball: exit 1", r.stderr)
Releases.files[url] = saved
check(sha(S2) == sha2 and os.stat(S2).st_ino == ino2 and not os.path.exists(os.path.join(notices_of(S2), "kapitsa.previous"))
      and open(os.path.join(notices_of(S2), "LICENSE")).read() == lic2 and not leftovers(S2),
      "after all of them: the same binary, no rollback copy, the same license files, nothing left over")
check(rec(H2, S2).get("result") in ("refused", "network") and not rec(H2, S2).get("notice"),
      "the state says why, with nothing left to say (it was said)", rec(H2, S2))

# ---------------------------------------------------------------- 5. --rollback and --version

r = run([S1, "update", "--rollback"], env(H1))
check(r.returncode == 0 and f"Rolled kapitsa back {N} -> {V}" in r.stdout and version_of(S1) == V and version_of(prev) == N,
      f"--rollback: back to {V}, {N} kept in its place", r.stdout + r.stderr)
check(rec(H1, S1).get("skip") == N, f"automatic updates skip {N} from now on", rec(H1, S1))
r = run([S1, "update", "--rollback"], env(H1))
check(r.returncode == 0 and version_of(S1) == N and version_of(prev) == V and "skip" not in rec(H1, S1),
      "--rollback again: forward to the newer one, and nothing skipped", r.stdout + r.stderr)
run([S1, "update", "--rollback"], env(H1))
set_rec(H1, S1, checked=0)
run([S1, "list"], env(H1))
check(wait_for(lambda: rec(H1, S1).get("result") == "skipped") and version_of(S1) == V,
      f"rolled back from {N}: the automatic update leaves {N} alone", rec(H1, S1))
quiet(S1)
Releases.latest = N2
set_rec(H1, S1, checked=0)
run([S1, "list"], env(H1))
check(wait_for(lambda: rec(H1, S1).get("result") == "updated") and version_of(S1) == N2,
      f"and installs the next one, {N2}", rec(H1, S1))
quiet(S1)
Releases.latest = N

S3, H3 = standalone("s3"), home("s3")
r = run([S3, "update", "--version", O], env(H3))
check(r.returncode == 0 and version_of(S3) == O and rec(H3, S3).get("skip") == V,
      f"--version {O}: an older release when asked for; automatic updates skip {V}", r.stdout + r.stderr)
r = run([S3, "update", "--check"], env(H3, KAPITSA_NO_AUTO_UPDATE="1"))
check(r.returncode == 100, f"the {O} binary knows it is {O}", r.stdout)
Releases.latest = V
set_rec(H3, S3, checked=0)
run([S3, "list"], env(H3))
check(wait_for(lambda: rec(H3, S3).get("result") == "skipped") and version_of(S3) == O,
      f"latest {V}, the version moved away from: left alone", rec(H3, S3))
quiet(S3)
Releases.latest = N
r = run([S3, "update", "--version", "9.9.9"], env(H3))
check(r.returncode == 1 and "there is no" in r.stderr and version_of(S3) == O, "--version of no release: exit 1, nothing changed",
      r.stderr)
r = run([S3, "update", "--version", O], env(H3))
check(r.returncode == 0 and "already installed" in r.stdout, "--version of the version installed: nothing to do", r.stdout)
for bad in (["--version", "x.y"], ["--version", "1.2.3/../../x"], ["--auto", "maybe"], ["--check", "--rollback"], ["--nope"]):
    r = run([S3, "update"] + bad, env(H3))
    check(r.returncode == 2, "usage error: " + " ".join(bad), r.stderr)
r = run([standalone("s3b"), "update", "--rollback"], env(home("s3b")))
check(r.returncode == 1 and "No previous version is kept" in r.stderr, "--rollback with nothing kept: exit 1", r.stderr)

# ---------------------------------------------------------------- 6. Homebrew, app bundle, source build

rel = binary(V)
HB = os.path.join(TMP, "hb")
cellar = os.path.join(HB, "Cellar", "kapitsa", V, "bin", "kapitsa")
os.makedirs(os.path.dirname(cellar))
shutil.copy(rel, cellar)
os.makedirs(os.path.join(HB, "bin"))
os.symlink(f"../Cellar/kapitsa/{V}/bin/kapitsa", os.path.join(HB, "bin", "kapitsa"))
brew_log = os.path.join(HB, "brew.log")
open(os.path.join(HB, "bin", "brew"), "w").write(f"#!/bin/sh\necho \"$@\" >> '{brew_log}'\n")
os.chmod(os.path.join(HB, "bin", "brew"), 0o755)
HBK, HHB = os.path.join(HB, "bin", "kapitsa"), home("hbh")
sha_hb = sha(cellar)
r = run([HBK, "update", "--check"], env(HHB))
check(r.returncode == 100 and "Update: brew upgrade nace-ai/tap/kapitsa" in r.stdout, "Homebrew --check names brew upgrade",
      r.stdout + r.stderr)
n0 = logged()
r = run([HBK, "update"], env(HHB))
check(r.returncode == 1 and "brew upgrade nace-ai/tap/kapitsa" in r.stderr and not os.path.exists(brew_log) and
      not since(n0, "GET"), "Homebrew, not in a terminal: says brew upgrade, runs nothing, downloads nothing", r.stderr)
code, out = tty([HBK, "update"], env(HHB))
check(code == 0 and open(brew_log).read() == "upgrade nace-ai/tap/kapitsa\n", "Homebrew in a terminal: runs brew upgrade",
      out[-300:])
for args in (["--version", N], ["--rollback"]):
    r = run([HBK, "update"] + args, env(HHB))
    check(r.returncode == 1, "Homebrew: " + " ".join(args) + " refused", r.stderr)
check(sha(cellar) == sha_hb and os.listdir(os.path.dirname(cellar)) == ["kapitsa"], "Homebrew's files untouched")
head_bin = os.path.join(HB, "Cellar", "kapitsa", "HEAD-abc1234", "bin", "kapitsa")
os.makedirs(os.path.dirname(head_bin))
shutil.copy(rel, head_bin)
os.remove(brew_log)
code, out = tty([head_bin, "update"], env(HHB))
check(open(brew_log).read() == "upgrade --fetch-HEAD nace-ai/tap/kapitsa\n", "a --HEAD install: brew upgrade --fetch-HEAD", out)

APPK = os.path.join(TMP, "app", "Nace Agent.app", "Contents", "Resources", "kapitsa", "kapitsa")
os.makedirs(os.path.dirname(APPK))
shutil.copy(rel, APPK)
HAPP = home("apph")
n0 = logged()
r = run([APPK, "update"], env(HAPP))
check(r.returncode == 1 and "Nace Agent.app" in r.stderr and "update the app" in r.stderr, "in an app: the app updates it",
      r.stderr)
r = run([APPK, "update", "--check"], env(HAPP))
check(r.returncode == 1, "in an app: --check says the same", r.stderr)
run([APPK, "list"], env(HAPP))
tty([APPK, "list"], env(HAPP))
time.sleep(1)
check(logged() == n0 and not os.path.exists(os.path.join(HAPP, "update.json")) and sha(APPK) == sha(rel),
      "in an app: nothing asked of the network, no state, the binary untouched")

SRC = os.path.join(TMP, "my src")
os.makedirs(os.path.join(SRC, ".git"))
os.makedirs(os.path.join(SRC, "build"))
open(os.path.join(SRC, "build", "CMakeCache.txt"), "w").write(f"CMAKE_HOME_DIRECTORY:INTERNAL={SRC}\n")
SRCK, HSRC = os.path.join(SRC, "build", "kapitsa"), home("srch")
shutil.copy(KAPITSA, SRCK)
how_src = f"git -C '{SRC}' pull && cmake --build '{SRC}/build'"
r = run([SRCK, "update"], env(HSRC))
check(r.returncode == 1 and how_src in r.stderr and sha(SRCK) == sha(KAPITSA), "a source build: how to rebuild it, never replaced",
      r.stderr)
r = run([SRCK, "update", "--check"], env(HSRC))
check(r.returncode == 100 and how_src in r.stdout, "a source build's --check", r.stdout)
Releases.latest = V
r = run([SRCK, "update"], env(HSRC))
check(r.returncode == 0 and "up to date" in r.stdout and "follows main" in r.stdout, "a source build at the latest: up to date",
      r.stdout)
Releases.latest = N

# ---------------------------------------------------------------- 7. automatic updates

# a. from a command: a child that does not hold the command up or change its output
S4, H4 = standalone("s4"), home("s4")
ref = run([S4, "list"], env(H4, KAPITSA_NO_AUTO_UPDATE="1"))
Releases.delay = 2.0
t0 = time.time()
r = run([S4, "list"], env(H4))
dt = time.time() - t0
check(r.returncode == 0 and dt < 1.0 and (r.stdout, r.stderr) == (ref.stdout, ref.stderr),
      f"the command returns at once ({dt:.2f} s; the release page answers in 2 s), its output unchanged", r.stderr)
time.sleep(0.5)
kids = children(S4)
check(len(kids) == 1 and kids[0][1] == "1" and kids[0][0] == kids[0][2] and kids[0][3] == "4" and kids[0][4] == "??",
      "one child, detached: launchd's, a session of its own, background priority, no terminal", kids)
check(wait_for(lambda: rec(H4, S4).get("result") == "updated") and version_of(S4) == N and version_of(
    os.path.join(notices_of(S4), "kapitsa.previous")) == V, f"the child installs {N}, keeping {V}", rec(H4, S4))
Releases.delay = 0.0
quiet(S4)
# b. the line on the next run in a terminal, once
r = run([S4, "list"], env(H4))
check("Updated" not in r.stdout + r.stderr and rec(H4, S4).get("notice"), "a run that is not in a terminal says nothing")
code, out = tty([S4, "list"], env(H4))
line = f"Updated Kapitsa to {N} (from {V}); restart running servers to use it"
check(out.count(line) == 1 and out.index(line) < out.index("No models downloaded"), "the next run in a terminal: one line, first",
      out[:400])
code, out = tty([S4, "list"], env(H4))
check("Updated" not in out and not rec(H4, S4).get("notice"), "and only once", out[:300])
# c. once a day
n0 = logged()
for _ in range(3):
    run([S4, "list"], env(H4))
time.sleep(1)
check(logged() == n0 and not children(S4), "checked today: no other check")
now = int(time.time())
set_rec(H4, S4, checked=now - DAY + 3600)
run([S4, "pull", "--dry-run", "4b"], env(H4))
time.sleep(1)
check(logged() == n0, "23 hours later: still none")
for when, what in ((now - DAY - 3600, "25 hours later"), (now + 2 * DAY, "a clock set back")):
    set_rec(H4, S4, checked=when, result=None)
    n0 = logged()
    run([S4, "list", "--json"], env(H4))
    check(wait_for(lambda: rec(H4, S4).get("result") == "up to date") and since(n0),
          f"{what}: a check, which finds it up to date", rec(H4, S4))
    quiet(S4)
# d. the off switches, and what is left alone
for desc, e in (("KAPITSA_NO_AUTO_UPDATE=1", env(H4, KAPITSA_NO_AUTO_UPDATE="1")), ("CI=true", env(H4, CI="true")),
                ("CI=1", env(H4, CI="1"))):
    set_rec(H4, S4, checked=0)
    n0 = logged()
    run([S4, "list"], e)
    tty([S4, "list"], e)
    time.sleep(1)
    check(logged() == n0 and rec(H4, S4).get("checked") == 0, f"{desc}: no check")
set_rec(H4, S4, checked=0)
r = run([S4, "update", "--auto", "off"], env(H4))
check(r.returncode == 0 and state(H4).get("automatic") is False, "--auto off: kept in update.json", r.stdout + r.stderr)
n0 = logged()
run([S4, "list"], env(H4))
time.sleep(1)
check(logged() == n0, "--auto off: no check")
r = run([S4, "update", "--auto", "on"], env(H4))
check(r.returncode == 0 and state(H4).get("automatic") is True, "--auto on", r.stdout)
n0 = logged()
for args in ([], ["--version"], ["-v"], ["--help"], ["help"], ["list", "--help"], ["serve", "-h"], ["bench", "--help"],
             ["update", "--help"], ["nonsense"]):
    run([S4] + args, env(H4))
time.sleep(1)
check(logged() == n0 and rec(H4, S4).get("checked") == 0, "help, the version, bench, update and unknown commands start no check")
set_rec(H4, S4, checked=0)
n0 = logged()
run([S4, "list"], env(H4, CI="false"))
check(wait_for(lambda: len(since(n0)) > 0), "CI=false is not CI: a check")
quiet(S4)
# e. the time limit
S5, H5 = standalone("s5"), home("s5")
Releases.hang = True
t0 = time.time()
run([S5, "list"], env(H5, KAPITSA_UPDATE_TIMEOUT_S="2"))
check(wait_for(lambda: "time limit" in rec(H5, S5).get("why", ""), 15) and quiet(S5, 5) and time.time() - t0 < 10 and
      sha(S5) == sha(rel) and not leftovers(S5) and not rec(H5, S5).get("notice"),
      "a check that hangs ends at its time limit, silently, nothing changed", rec(H5, S5))
Releases.hang = False
# f. kapitsa bench holds the binary
S6, H6 = standalone("s6"), home("s6")
lock = os.open(S6, os.O_RDONLY)
fcntl.flock(lock, fcntl.LOCK_SH)  # what kapitsa bench takes for its run
n0 = logged()
r = run([S6, "update"], env(H6))
check(r.returncode == 1 and "kapitsa bench is running" in r.stderr and sha(S6) == sha(rel) and not since(n0, "GET"),
      "while kapitsa bench holds the binary, kapitsa update leaves it (and downloads nothing)", r.stderr)
set_rec(H6, S6, checked=0)
run([S6, "list"], env(H6))
check(wait_for(lambda: rec(H6, S6).get("result") == "busy") and sha(S6) == sha(rel) and
      DAY - 3600 - 60 < time.time() - rec(H6, S6).get("checked", 0) < DAY - 3600 + 60,
      "and so does the automatic update, which tries again in an hour", rec(H6, S6))
quiet(S6)
fcntl.flock(lock, fcntl.LOCK_UN)
os.close(lock)
bench = subprocess.Popen([S6, "bench", "--model", "4b", "--yes"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=env(H6, HF_ENDPOINT=f"http://127.0.0.1:{PORT}/hang"))
time.sleep(1.5)
fd = os.open(S6, os.O_RDONLY)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    held_by_bench = False
except OSError as ex:
    held_by_bench = ex.errno == errno.EWOULDBLOCK
os.close(fd)
check(bench.poll() is None and held_by_bench, "a running kapitsa bench holds its binary")
bench.terminate()
try:
    bench.wait(10)
except subprocess.TimeoutExpired:
    bench.kill(), bench.wait()
r = run([S6, "update"], env(H6))
check(r.returncode == 0 and version_of(S6) == N, "after it: updated", r.stdout + r.stderr)
# g. the binary replaced while the check runs: never touched
S7, H7 = standalone("s7"), home("s7")
Releases.delay = 1.5
n0 = logged()
run([S7, "list"], env(H7))
wait_for(lambda: [x for x in since(n0) if x[:2] == ("HEAD", "/releases/latest")])  # the child asks, then waits
other = os.path.join(TMP, "s7", "other")
shutil.copy(KAPITSA, other)
os.rename(other, S7)  # as an installer would put another binary there
check(wait_for(lambda: rec(H7, S7).get("result") == "error") and "changed since this kapitsa started" in rec(H7, S7).get("why", "")
      and sha(S7) == sha(KAPITSA) and not os.path.exists(os.path.join(notices_of(S7), "kapitsa.previous")),
      "a binary replaced meanwhile is not this kapitsa: left as it is", rec(H7, S7))
Releases.delay = 0.0
quiet(S7)
# h. two installs, one home: only the binary that runs is updated
S8a, S8b, H8 = standalone("s8a"), standalone("s8b"), home("s8")
run([S8a, "list"], env(H8))
check(wait_for(lambda: rec(H8, S8a).get("result") == "updated") and version_of(S8a) == N and version_of(S8b) == V and
      not rec(H8, S8b), "two installs, one home: only the one that ran is updated")
quiet(S8a)
run([S8b, "list"], env(H8))
check(wait_for(lambda: rec(H8, S8b).get("result") == "updated") and version_of(S8b) == N, "then the other, when it runs")
quiet(S8b)
# i. through a symlink: the file it points to is updated, the symlink stays
real = standalone("s9/opt")
link = os.path.join(TMP, "s9", "kapitsa")
os.symlink(real, link)
H9 = home("s9")
run([link, "list"], env(H9))
check(wait_for(lambda: rec(H9, real).get("result") == "updated") and os.path.islink(link) and os.readlink(link) == real and
      version_of(link) == N and os.path.exists(os.path.join(notices_of(real), "kapitsa.previous")),
      "run through a symlink: the file it points to is updated, the symlink kept", rec(H9, real))
quiet(real)
# j. a refused release, in the background: said on the next run in a terminal; the binary unchanged
S10, H10 = standalone("s10"), home("s10")
tar_release(N, sha256="1" * 64)
run([S10, "list"], env(H10))
check(wait_for(lambda: rec(H10, S10).get("result") == "refused") and sha(S10) == sha(rel), "a bad sha256 in the background: refused",
      rec(H10, S10))
quiet(S10)
code, out = tty([S10, "list"], env(H10))
check(has(out, f"Kapitsa {N} was not installed: its sha256 is") and has(out, "kapitsa update tries again"),
      "the next run in a terminal says why", out[:500])
tar_release(N)
# k. a network failure in the background: silent, tried again tomorrow
S11, H11 = standalone("s11"), home("s11")
run([S11, "list"], env(H11, KAPITSA_UPDATE_URL="http://127.0.0.1:9/releases"))
check(wait_for(lambda: rec(H11, S11).get("result") == "network") and not rec(H11, S11).get("notice"), "nothing answering: recorded",
      rec(H11, S11))
quiet(S11)
code, out = tty([S11, "list"], env(H11))
check("Kapitsa" not in out.split("No models downloaded")[0], "and never said", out[:300])
# l. a folder this user can't write: told in a terminal, with the command that works
S12, H12 = standalone("s12"), home("s12")
os.chmod(os.path.dirname(S12), 0o555)
try:
    r = run([S12, "update"], env(H12))
    check(r.returncode == 1 and f"sudo {S12} update" in r.stderr and sha(S12) == sha(rel), "kapitsa update: the sudo command",
          r.stderr)
    n0 = logged()
    run([S12, "list"], env(H12))
    time.sleep(1)
    check(logged() == n0, "not in a terminal: no check (it could only tell)")
    tty([S12, "list"], env(H12))
    check(wait_for(lambda: rec(H12, S12).get("result") == "available"), "in a terminal: a check, which only tells", rec(H12, S12))
    quiet(S12)
    code, out = tty([S12, "list"], env(H12))
    check(has(out, f"Kapitsa {N} is available: sudo {S12} update"), "the next run says the command", out[:400])
    check(sha(S12) == sha(rel), "and the binary is untouched")
finally:
    os.chmod(os.path.dirname(S12), 0o755)
# m. Homebrew, a source build and a binary built from source only tell, in a terminal
SB = os.path.join(TMP, "sb", "bin", "kapitsa")
os.makedirs(os.path.dirname(SB))
os.makedirs(os.path.join(TMP, "sb", "share", "kapitsa"))
shutil.copy(KAPITSA, SB)  # ./install.sh in a clone puts its own build there
for path, h, how in ((HBK, HHB, "brew upgrade nace-ai/tap/kapitsa"), (SRCK, HSRC, how_src), (SB, home("sbh"), "kapitsa update")):
    real_path = os.path.realpath(path)
    n0 = logged()
    run([path, "list"], env(h))
    time.sleep(1)
    check(logged() == n0, f"{how}: not in a terminal, no check")
    tty([path, "list"], env(h))
    check(wait_for(lambda: rec(h, real_path).get("result") == "available") and sha(real_path) == sha(os.path.realpath(path)),
          f"{how}: in a terminal, a check that only records the release", rec(h, real_path))
    quiet(real_path)
    code, out = tty([path, "list"], env(h))
    check(has(out, f"Kapitsa {N} is available: {how}"), f"{how}: the next run in a terminal says so", out[:400])
check(sha(cellar) == sha_hb and sha(SRCK) == sha(KAPITSA) and sha(SB) == sha(KAPITSA), "none of them replaced")
r = run([SB, "update"], env(home("sbh")))
check(r.returncode == 0 and version_of(SB) == N, "kapitsa update replaces a binary built from source when asked",
      r.stdout + r.stderr)

# ---------------------------------------------------------------- 8. what the requests carry

ALLOWED = {"host", "accept", "accept-language", "accept-encoding", "connection", "user-agent"}
mine = [x for x in Releases.log if x[1].startswith("/releases")]
bad = [x for x in mine if x[0] not in ("GET", "HEAD") or "?" in x[1] or x[2].get("User-Agent") != "kapitsa" or
       x[2].get("Accept-Language") != "en" or not set(k.lower() for k in x[2]) <= ALLOWED]
check(mine and not bad, f"{len(mine)} requests: GET or HEAD, no query, User-Agent kapitsa, Accept-Language en, no cookies or "
      "anything else", bad[:2])

# ---------------------------------------------------------------- done

for p in [S1, S2, S3, S4, S5, S6, S7, S8a, S8b, real, S10, S11, S12, SB, os.path.realpath(HBK), SRCK, APPK]:
    quiet(p, 10)
if FAILS:
    print(f"\nFAILED: {len(FAILS)} (the files are in {TMP})")
    sys.exit(1)
shutil.rmtree(TMP, ignore_errors=True)
print("\nok: 0 failed")
