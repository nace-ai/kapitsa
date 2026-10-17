#!/usr/bin/env python3
"""Downloads the text-model part of a Hugging Face model repo: config/tokenizer files and every safetensors
tensor except the vision tower (and optionally the MTP head), written as a local safetensors file per shard.

  fetch_text.py <repo> <out dir> [--revision R] [--keep-mtp] [--conns N] [--max-gb G]

Parallel HTTP range requests (one connection here tops out at ~6 MB/s; 16 reach ~30-38 MB/s), each tensor fetched
in <= 64 MB pieces. Stdlib only. Refuses to start when the download would leave less than 6 GB free on the disk.
"""
import json, os, shutil, struct, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

args = sys.argv[1:]
def opt(name, default=None, flag=False):
    if name in args:
        i = args.index(name)
        if flag:
            del args[i]
            return True
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default
rev = opt("--revision", "main")
keep_mtp = opt("--keep-mtp", flag=True)
conns = int(opt("--conns", "16"))
max_gb = float(opt("--max-gb", "1e9"))
if len(args) != 2:
    sys.exit(__doc__)
repo, out = args
os.makedirs(out, exist_ok=True)
UA = {"User-Agent": "kapitsa-quant-eval"}


def url(f):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{f}"


def get(f, a=None, b=None, tries=8):
    h = dict(UA)
    if a is not None:
        h["Range"] = f"bytes={a}-{b - 1}"
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url(f), headers=h), timeout=300) as r:
                data = r.read()
            if a is not None and len(data) != b - a:
                raise IOError(f"{f}: short read {len(data)} of {b - a}")
            return data
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


meta = json.loads(urllib.request.urlopen(urllib.request.Request(
    f"https://huggingface.co/api/models/{repo}/revision/{rev}", headers=UA), timeout=60).read())
files = [s["rfilename"] for s in meta["siblings"]]
small = [f for f in files if f.endswith((".json", ".jinja", ".txt", ".model", ".tiktoken")) and "/" not in f]
shards = [f for f in files if f.endswith(".safetensors") and "/" not in f]
print(f"{repo}@{meta.get('sha', rev)[:12]}: {len(shards)} shards", flush=True)


def keep(name):
    if "vision_tower" in name or name.startswith("model.visual") or ".visual." in name or name.startswith("visual."):
        return False
    return keep_mtp or not name.startswith("mtp.")


plan = []  # (shard, header json, data offset, kept names)
total = 0
for s in shards:
    n = struct.unpack("<Q", get(s, 0, 8))[0]
    hdr = json.loads(get(s, 8, 8 + n))
    names = [k for k in hdr if k != "__metadata__" and keep(k)]
    total += sum(hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0] for k in names)
    plan.append((s, hdr, 8 + n, names))
free = shutil.disk_usage(out).free
print(f"text tensors: {total / 1e9:.2f} GB; disk free {free / 1e9:.1f} GB", flush=True)
if total / 1e9 > max_gb:
    sys.exit(f"{total / 1e9:.2f} GB is more than --max-gb {max_gb}")
if free - total < 6e9:
    sys.exit("refusing: the download would leave less than 6 GB free")

for f in small:
    open(os.path.join(out, f), "wb").write(get(f))

done = [0]
lock = threading.Lock()
t0 = time.time()
for s, hdr, data0, names in plan:
    if not names:
        continue
    names.sort(key=lambda k: hdr[k]["data_offsets"][0])
    new = {"__metadata__": hdr.get("__metadata__", {"format": "pt"})}
    off = 0
    for k in names:
        a, b = hdr[k]["data_offsets"]
        new[k] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"], "data_offsets": [off, off + b - a]}
        off += b - a
    hj = json.dumps(new, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    path = os.path.join(out, s)
    tmp = path + ".part"
    with open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hj)))
        fo.write(hj)
        fo.truncate(8 + len(hj) + off)
    pieces = []
    for k in names:
        a, b = hdr[k]["data_offsets"]
        dst = 8 + len(hj) + new[k]["data_offsets"][0]
        for p in range(a, b, 64 << 20):
            q = min(b, p + (64 << 20))
            pieces.append((data0 + p, data0 + q, dst + (p - a)))

    def work(pc):
        a, b, dst = pc
        data = get(s, a, b)
        with open(tmp, "r+b") as fo:
            fo.seek(dst)
            fo.write(data)
        with lock:
            done[0] += b - a
            el = time.time() - t0
            print(f"\r{done[0] / 1e9:.2f} / {total / 1e9:.2f} GB, {done[0] / 1e6 / max(el, 1e-3):.1f} MB/s   ", end="",
                  flush=True)

    with ThreadPoolExecutor(conns) as ex:
        list(ex.map(work, pieces))
    os.replace(tmp, path)
# index for the kept tensors
wm = {}
for s, hdr, data0, names in plan:
    for k in names:
        wm[k] = s
json.dump({"metadata": {"total_size": total}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
print(f"\ndone: {total / 1e9:.2f} GB in {time.time() - t0:.0f} s -> {out}")
