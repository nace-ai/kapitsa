#!/usr/bin/env python3
"""Which Mac's GPU is free soonest, and a queue entry for a job that waits for one (stdlib only).

  gpu.py [--model NAME] [--chip M5] [--min-macos 26] [--json]
      Probe this Mac and every Mac in the hosts file (read-only: sysctl, vm_stat, lsof, ps over ssh), print each one's
      GPU lock holder and waiters (start time, declared ETA, detach.py job and its log's last line), free memory and
      swap, and name the Mac expected to be free first. This estimate does not reserve a GPU or authorize a run.
      Coordinate remote runs with the machine's owner.
  gpu.py run --eta MIN [--label TEXT] [--cwd DIR] -- CMD...
      Queue CMD behind this Mac's GPU lock: record "started, ETA, label" in /tmp/kapitsa-gpu.queue/<pid>, then exec
      `lockf -k /tmp/kapitsa-gpu.lock CMD` with NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1, so tools that take the lock
      themselves (logit_dump, kapitsa bench) do not wait on their own hold. It never reorders, kills or preempts
      another holder or waiter, and never launches anything on another Mac.

Hosts file: $VK_HOSTS, default ~/.config/verify-kapitsa/hosts, one Mac per line: `name user@host [socks=HOST:PORT]`
Use socks=HOST:PORT when SSH requires a SOCKS proxy, for example with userspace Tailscale.
"""
import argparse, json, os, re, shlex, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

LOCK = "/tmp/kapitsa-gpu.lock"
QUEUE = "/tmp/kapitsa-gpu.queue"
DEFAULT_ETA = float(os.environ.get("VK_GPU_DEFAULT_ETA", 20))  # minutes, for a job that declared none
OVERDUE = 5  # minutes still assumed for a job past its declared ETA
MODELISH = re.compile(r"kapitsa|ollama|llama|mlx|lms|LM Studio|logit_dump|ni_run|ni_eval", re.I)

# Runs under sh on every Mac, local or remote; prints "key value..." lines that parse() reads.
PROBE = r"""
L=${BENCH_GPU_LOCK:-/tmp/kapitsa-gpu.lock}; Q=/tmp/kapitsa-gpu.queue
echo "chip $(sysctl -n machdep.cpu.brand_string)"; echo "memsize $(sysctl -n hw.memsize)"
echo "macos $(sw_vers -productVersion)"; echo "now $(date +%s)"; echo "swap $(sysctl -n vm.swapusage)"
vm_stat | awk '/page size/{print "pagesize",$8} /Pages free/{print "free",$3} /Pages inactive/{print "inactive",$3}
  /Pages speculative/{print "speculative",$3} /Pages purgeable/{print "purgeable",$3}'
R=""  # every pid that uses the lock, and its ancestors
for p in $(lsof -t "$L" 2>/dev/null); do echo "open $p $(ps -o ppid=,etime=,command= -p $p | cut -c1-200)"
  a=""; q=$p; while q=$(ps -o ppid= -p "$q" | tr -d ' '); [ -n "$q" ] && [ "$q" -gt 1 ]; do a="$a $q"; done; echo "anc $p$a"; R="$R $p$a"; done
for p in $(pgrep -f "^(/usr/bin/)?lockf .*$L"); do echo "lockf $p $(ps -o ppid=,etime=,command= -p $p | cut -c1-200)"
  a=""; q=$p; while q=$(ps -o ppid= -p "$q" | tr -d ' '); [ -n "$q" ] && [ "$q" -gt 1 ]; do a="$a $q"; done; echo "anc $p$a"; R="$R $p$a"; done
for f in "$Q"/*; do [ -f "$f" ] || continue; p=${f##*/}; kill -0 "$p" 2>/dev/null && echo "eta $p $(head -1 "$f") " && sed -n 's/^held /held '"$p"' /p' "$f"; done
# the detach.py job among those pids (pid and start time both match), named with its log's last line: one awk pass
# over the job files, which can number in the thousands
[ -n "$R" ] && set -- ~/.agents/jobs/*.json && [ -f "$1" ] && awk -v R="$R " '
  function out() { if (p != "" && index(R, " " p " ")) print p "\t" n "\t" s "\t" g }
  function val(  v) { v = $0; sub(/^ *"[a-z]+": *"?/, "", v); sub(/"?,? *$/, "", v); return v }
  FNR == 1 { out(); p = n = s = g = "" }
  /^ *"pid":/ { p = val() } /^ *"name":/ { n = val() } /^ *"lstart":/ { s = val() } /^ *"log":/ { g = val() }
  END { out() }' "$@" | while IFS="$(printf '\t')" read -r p n s g; do
  [ "$(ps -o lstart= -p "$p" 2>/dev/null | sed 's/^ *//;s/ *$//')" = "$s" ] || continue
  echo "job $p $n"; echo "jobtail $p $(tail -c 4000 "$g" 2>/dev/null | tr '\r' '\n' | grep -v '^ *$' | tail -1 | cut -c1-160)"; done
ps -axo rss=,pid=,etime=,command= | awk '$1>3000000' | cut -c1-200 | sed 's/^ */big /'
echo "models $( (ls ~/.cache/kapitsa/models; [ -n "$NACE_MODELS" ] && ls "$NACE_MODELS"; [ -d "$HOME/Documents/nace/models" ] && ls "$HOME/Documents/nace/models") 2>/dev/null | sort -u | tr '\n' ' ')"
"""


def hosts_file():
    return os.path.expanduser(os.environ.get("VK_HOSTS", "~/.config/verify-kapitsa/hosts"))


def read_hosts():
    hosts = [{"name": "local", "target": None, "socks": None}]
    if os.path.exists(hosts_file()):
        for line in open(hosts_file()):
            f = line.split("#", 1)[0].split()
            if len(f) >= 2:
                opts = dict(x.split("=", 1) for x in f[2:] if "=" in x)
                hosts.append({"name": f[0], "target": f[1], "socks": opts.get("socks")})
    return hosts


def ssh_argv(h, remote_cmd):
    a = ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes"]
    if h["socks"]:
        a += ["-o", f"ProxyCommand=nc -x {h['socks']} %h %p"]
    return a + [h["target"], remote_cmd]


def etime_s(s):  # ps etime: [[dd-]hh:]mm:ss
    d, _, rest = s.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse(text):
    d = {"open": [], "lockf": [], "eta": {}, "big": [], "models": [], "anc": {}, "jobs": {}}
    vm = {}
    for line in text.splitlines():
        key, _, rest = line.partition(" ")
        if key in ("open", "lockf"):
            f = rest.split(None, 3)
            if len(f) >= 3:
                d[key].append({"pid": int(f[0]), "ppid": int(f[1]), "age_s": etime_s(f[2]), "cmd": f[3] if len(f) > 3 else ""})
        elif key == "eta":
            f = rest.split(None, 3)
            if len(f) >= 3:
                d["eta"][int(f[0])] = {"start": int(f[1]), "eta_min": float(f[2]), "label": f[3].strip() if len(f) > 3 else ""}
        elif key == "anc":
            f = [int(x) for x in rest.split()]
            if f:
                d["anc"][f[0]] = set(f[1:])
        elif key in ("job", "jobtail"):
            f = rest.split(None, 1)
            if f and f[0].isdigit():
                d["jobs"].setdefault(int(f[0]), {})["name" if key == "job" else "tail"] = f[1].strip() if len(f) > 1 else ""
        elif key == "swap":
            m = re.search(r"used = ([\d.]+)M", rest)
            d["swap_gb"] = float(m.group(1)) / 1024 if m else 0.0
        elif key == "held":
            f = rest.split()
            if len(f) == 2 and int(f[0]) in d["eta"]:
                d["eta"][int(f[0])]["held"] = int(f[1])
        elif key == "big":
            f = rest.split(None, 3)
            if len(f) == 4:
                d["big"].append({"rss_gb": int(f[0]) / 1048576, "pid": int(f[1]), "cmd": f[3]})
        elif key == "models":
            d["models"] = rest.split()
        elif key in ("pagesize", "free", "inactive", "speculative", "purgeable"):
            vm[key] = int(rest.strip().rstrip("."))
        elif key in ("chip", "macos"):
            d[key] = rest.strip()
        elif key in ("memsize", "now"):
            d[key] = int(rest)
    if vm.get("pagesize"):
        d["avail_gb"] = sum(vm.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable")) * vm["pagesize"] / 2**30
    return d


def estimate(d):
    """Minutes until the lock is expected free, the holder and the waiters, each with its ETA and how it is known."""
    now = d.get("now", int(time.time()))

    def job(p, holding=False):
        # the detach.py job this process runs under, if any: itself or an ancestor
        dj = next((d["jobs"][a] for a in [p["pid"], *sorted(d["anc"].get(p["pid"], ()), reverse=True)] if a in d["jobs"]), {})
        p = {**p, "start": now - p["age_s"], "job": dj.get("name", ""), "log_tail": dj.get("tail", "")}
        e = d["eta"].get(p["pid"])
        if e:  # an ETA is minutes of hold: a waiter has all of it ahead, the holder counts down from acquiring
            left = (e.get("held", e["start"]) + e["eta_min"] * 60 - now) / 60 if holding else e["eta_min"]
            return {**p, "label": e["label"], "left_min": left if left > 0 else OVERDUE, "known": left > 0 or "overdue"}
        return {**p, "label": "", "left_min": DEFAULT_ETA, "known": False}

    # A waiting lockf blocks inside open(O_EXLOCK) and holds no fd, so a lockf in lsof is the holder; flock waiters
    # (kapitsa bench, logit_dump) do hold an fd, so without a lockf the holder is the oldest open.
    lockf_pids = {p["pid"] for p in d["lockf"]}
    opens = sorted(d["open"], key=lambda p: -p["age_s"])
    holder = next((p for p in opens if p["pid"] in lockf_pids), opens[0] if opens else None)
    # every descendant of the holder inherits its fd (lockf -> bash leg.sh -> kapitsa serve all show in lsof):
    # they are the holder, not waiters
    def under_holder(p):
        return holder and holder["pid"] in {p["pid"], p["ppid"]} | d["anc"].get(p["pid"], set())

    waiters = [p for p in d["lockf"] + d["open"] if not under_holder(p)]
    seen, q = set(), []
    for p in sorted(waiters, key=lambda p: -p["age_s"]):  # flock does not promise FIFO; oldest first is a guess
        if p["pid"] not in seen:
            seen.add(p["pid"])
            q.append(job(p))
    h = job(holder, holding=True) if holder else None
    wait = (h["left_min"] if h else 0) + sum(w["left_min"] for w in q)
    outside = [] if holder else [b for b in d["big"] if MODELISH.search(b["cmd"])]
    if outside:  # a model loaded without the lock: nothing says when it ends
        wait = max(wait, DEFAULT_ETA)
    return {"holder": h, "waiters": q, "wait_min": wait, "outside": outside}


def probe(h):
    try:
        if h["target"]:
            r = subprocess.run(ssh_argv(h, "sh -s"), input=PROBE, capture_output=True, text=True, timeout=25)
        else:
            r = subprocess.run(["sh", "-s"], input=PROBE, capture_output=True, text=True, timeout=25)
    except subprocess.TimeoutExpired:
        return {**h, "error": "probe timed out"}
    if r.returncode != 0 and "chip" not in r.stdout:
        return {**h, "error": (r.stderr.strip().splitlines() or ["ssh failed"])[-1][:120]}
    d = parse(r.stdout)
    return {**h, **d, **estimate(d)}


def macos_tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "0")[:3])


def has_model(d, name):
    return any(m == name or m.startswith(name + "-") for m in d.get("models", []))


def tailnet_peers():
    sock = os.path.expanduser("~/.tailscale/tailscaled.sock")
    argv = ["tailscale"] + (["--socket", sock] if os.path.exists(sock) else []) + ["status", "--json"]
    try:
        st = json.loads(subprocess.run(argv, capture_output=True, text=True, timeout=10).stdout)
    except Exception:
        return []
    out = []
    for p in (st.get("Peer") or {}).values():
        dns = (p.get("DNSName") or "").rstrip(".")
        ips = p.get("TailscaleIPs") or ["?"]
        # every name a hosts file may use for it: the IPs, the MagicDNS name and its short form
        out.append((p.get("HostName", "?"), ips[0], p.get("Online"), set(ips) | {dns, dns.split(".")[0]} - {""}))
    return out


def describe(j):
    if not j:
        return "free"
    lab = j["label"] or j["cmd"][:60]
    eta = f"~{j['left_min']:.0f}m left" if j["known"] is True else (
        "past its ETA" if j["known"] == "overdue" else f"no ETA, {DEFAULT_ETA:.0f}m assumed")
    job = f" [detach job {j['job']}]" if j.get("job") else ""
    tail = f"\n          log: {j['log_tail'][:110]}" if j.get("log_tail") else ""
    return f"pid {j['pid']} since {time.strftime('%H:%M', time.localtime(j['start']))} ({j['age_s'] // 60}m), {eta}: {lab}{job}{tail}"


def cmd_status(a):
    hosts = read_hosts()
    with ThreadPoolExecutor(len(hosts)) as ex:
        res = list(ex.map(probe, hosts))
    for d in res:
        why = []
        if "error" in d:
            why.append("unreachable")
        else:
            if a.model and not has_model(d, a.model):
                why.append(f"no {a.model} folder")
            if a.chip and a.chip.lower() not in d.get("chip", "").lower():
                why.append(f"not an {a.chip}")
            if a.min_macos and macos_tuple(d.get("macos")) < macos_tuple(a.min_macos):
                why.append(f"macOS below {a.min_macos}")
        d["ineligible"] = why
    ok = sorted((d for d in res if not d["ineligible"]), key=lambda d: (round(d["wait_min"]), d["name"] != "local"))
    pick = ok[0] if ok else None
    if a.json:
        print(json.dumps({"pick": pick and pick["name"], "hosts": res}, indent=1, default=str))
        return 0 if pick else 1
    for d in res:
        head = f"{d['name']} ({d['target'] or 'this Mac'})"
        if "error" in d:
            print(f"{head}: {d['error']}")
            continue
        print(f"{head}: {d.get('chip')}, {d.get('memsize', 0) / 2**30:.0f} GB, macOS {d.get('macos')}, "
              f"{d.get('avail_gb', 0):.0f} GB available, swap {d.get('swap_gb', 0):.1f} GB used")
        print(f"  lock:   {describe(d['holder'])}")
        for i, w in enumerate(d["waiters"], 1):
            print(f"  wait {i}: {describe(w)}")
        for b in d["outside"]:
            print(f"  loaded outside the lock: pid {b['pid']} {b['rss_gb']:.1f} GB: {b['cmd'][:80]}")
        print(f"  free in: {'now' if d['wait_min'] == 0 else '~%.0f min' % d['wait_min']}"
              + (f"   (skipped: {', '.join(d['ineligible'])})" if d["ineligible"] else ""))
    # a Mac commented out in the hosts file was looked at and left out on purpose: do not offer it again
    known = set(re.findall(r"@([\w.:-]+)", open(hosts_file()).read())) if os.path.exists(hosts_file()) else set()
    extra = [p for p in tailnet_peers() if not p[3] & known]
    if extra:
        print("on the tailnet, not in " + hosts_file() + ": "
              + ", ".join(f"{n} ({ip}{'' if on else ', offline'})" for n, ip, on, _ in extra))
    if not pick:
        print("pick: none (every Mac is unreachable or skipped)")
        return 1
    print(f"pick: {pick['name']} ({'free now' if pick['wait_min'] == 0 else 'free in ~%.0f min' % pick['wait_min']})"
          + ("" if pick["name"] == "local" else "; advice only: build and launch it there yourself, and clean it up"))
    return 0


def queue_script(eta, label, cmd, lock):
    """sh that records this shell's pid in the queue, then becomes lockf: the queue file names lockf's pid."""
    label = re.sub(r"\s+", " ", label).strip()[:100]
    return (f"mkdir -p {QUEUE}; for f in {QUEUE}/*; do [ -f \"$f\" ] && ! kill -0 \"${{f##*/}}\" 2>/dev/null"
            f" && rm -f \"$f\"; done; echo \"$(date +%s) {eta:g} {label}\" > {QUEUE}/$$; "
            + f"NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 exec lockf -k {shlex.quote(lock)} "
            # once the lock is ours, note when (the holder's ETA counts from here), then become CMD
            + "sh -c " + shlex.quote(f'echo "held $(date +%s)" >> {QUEUE}/' + "$1; shift; exec \"$@\"")
            + " vk-held $$ " + " ".join(shlex.quote(c) for c in cmd))


def cmd_run(a):
    if not a.cmd:
        sys.exit("gpu.py run: nothing to run (put the command after --)")
    label = a.label or os.path.basename(a.cmd[0])
    if "'" in label or '"' in label or "$" in label or "`" in label:
        sys.exit("gpu.py run: keep quotes, $ and backticks out of --label")
    if a.cwd:
        try:  # before the queue entry, so a bad directory leaves none
            os.chdir(os.path.expanduser(a.cwd))
        except OSError as e:
            sys.exit(f"gpu.py run: --cwd {a.cwd}: {e.strerror}")
    os.execvp("sh", ["sh", "-c", queue_script(a.eta, label, a.cmd, os.environ.get("BENCH_GPU_LOCK", LOCK))])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        p = argparse.ArgumentParser(prog="gpu.py run")
        p.add_argument("--eta", type=float, required=True, help="minutes the job expects to hold the GPU")
        p.add_argument("--label")
        p.add_argument("--cwd", help="directory to run in")
        argv = sys.argv[2:]
        cmd = argv[argv.index("--") + 1:] if "--" in argv else []
        a = p.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
        a.cmd = cmd
        return cmd_run(a)
    p = argparse.ArgumentParser(prog="gpu.py")
    p.add_argument("--model", help="only Macs with this model folder (a floor stem matches NAME-*)")
    p.add_argument("--chip", help="only Macs whose chip names this, e.g. M5 (the tensor-unit kernels)")
    p.add_argument("--min-macos", help="only Macs on at least this macOS, e.g. 26")
    p.add_argument("--json", action="store_true")
    return cmd_status(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
