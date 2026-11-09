#!/usr/bin/env python3
"""Link every rival cell in readme_images.json to the result files behind its number (`src`), so readme_images.py shows
the version those files record instead of one typed by hand. A cell is linked to the files of the same engine, model and
Mac whose decode (median, end-to-end median or mean) rounds to the cell's number; where that leaves two versions, the
session its note names, then the version the cell already showed, picks one; a cell still ambiguous or without a file is
listed and left for a hand-written `src` (or `version` + `version_from`). Usage: link_sources.py [--write]"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readme_images as ri  # noqa: E402

MAC = {"M5 Max": ("Apple M5 Max", 128), "M5 Pro": ("Apple M5 Pro", 48), "M4 Pro": ("Apple M4 Pro", 48),
       "M3 Pro": ("Apple M3 Pro", 36), "M1 Pro": ("Apple M1 Pro", 16)}
VERSION_TOKEN = re.compile(r"(?:mlx-lm [\d.]+, )?mlx-vlm [\d.]+|mlx-lm [\d.]+|MLX runtime [\d.]+|llama\.cpp runtime [\d.]+"
                           r"|\bb\d{4,5}\b|\d+\.\d+[\w.]*|\b[0-9a-f]{7,8}\b")


def index():
    out = []
    for f in glob.glob(str(ri.RESULTS / "**" / "*.json"), recursive=True):
        if os.path.getsize(f) > 20e6:
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "decode_summary" not in d or d.get("engine") in (None, "kapitsa"):
            continue
        ss = (d.get("provenance") or {}).get("system_start") or {}
        ds = d["decode_summary"]
        out.append(dict(path=os.path.relpath(f, ri.RESULTS), engine=d["engine"], version=d.get("version"),
                        model=d.get("model_name") or "", cpu=ss.get("cpu"),
                        mem=round(int(ss["memsize"]) / 2 ** 30) if ss.get("memsize") else None,
                        vals=[ds.get(k) for k in ("median_tps", "median_tps_e2e", "mean_tps") if ds.get(k) is not None]))
    return out


def model_ok(row, mn):
    a, b = (re.sub(r"[^a-z0-9]", "", s.lower()) for s in (row, mn))
    if row.startswith("Flint") and ("2-bit" in mn) != ("2-bit" in row):
        return False
    return (a in b or a.replace("flashnext180b", "flashnext") in b
            or any(row.startswith(p) and p.lower() in b for p in ("Lightning", "Super", "Bolt", "Dory", "Flint", "Flash-Next".replace("-", ""))))


def candidates(files, mac, row, eng, val):
    cpu, mem = next(v for k, v in MAC.items() if k in mac)
    return [x for x in files if x["engine"] == ri.RESULT_ENGINE[eng] and x["cpu"] == cpu and x["mem"] == mem
            and model_ok(row, x["model"]) and any(abs(v - val) <= 0.55 for v in x["vals"])]


def template(old, eng):
    if not old:
        return None
    m = VERSION_TOKEN.search(old)
    return (old[:m.start()] + "{v}" + old[m.end():]) if m else old + ", {v}"


NOT_RECORDED = "version not recorded"
# cells the matching can't settle: (Mac, row, engine) -> the files behind the number, or the engine version a record states
MANUAL = {
    ("M5 Pro", "Qwen3.8 27B", "mlx-serve"): {"src": ["2026-10-22-m5pro-48gb/results/q38-27b-mlxserve-dflash2-20261022-100013.json",
                                                    "2026-10-22-m5pro-48gb/results/q38-27b-mlxserve-dflash2-20261022-095547.json"]},  # the mean of two runs
    ("M5 Pro", "Flash-Next 180B Q2", "Splash"): {"version": "1.5.0", "version_from": "2026-10-30-m5-pro-flash-next-kflash/README.md"},
    ("M5 Pro", "Flash-Next 180B Q2", "llama.cpp"): {"version": "c7474afc", "version_from": "2026-10-28-m5-pro-flash-next-engines/README.md"},
    ("M1 Pro", "Qwen3.6 35B-A3B 2-bit", "llama.cpp"): {"version": "b11669", "version_from": "2026-10-22-m1pro-16gb/README.md"},
    ("M1 Pro", "Nano 4B", "llama.cpp"): {"version": "b11669", "version_from": "2026-10-22-m1pro-16gb/README.md"},
    ("M1 Pro", "Nano 9B v2", "MLX"): {"version": "mlx-lm 0.32.2", "version_from": "2026-10-22-m1pro-16gb/README.md"},
    ("M1 Pro", "Nano 4B", "MLX"): {"version": NOT_RECORDED},
    ("M1 Pro", "Nano 9B v2", "llama.cpp"): {"version": NOT_RECORDED},
    ("M5 Max", "Super-VL 120B", "MLX"): {"src": ["2026-10-18-m5-max/super-mlxvlm-mtp-20261018-050315.json"], "version": "mlx-vlm 0.7.8",
                                         "version_from": "2026-10-18-m5-max/README.md"},   # the file's version field is empty
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--data", type=Path, default=HERE / "readme_images.json")
    ap.add_argument("--head", type=Path, help="readme_images.json as it is now: the changes list compares with what its cells show")
    ap.add_argument("--main", type=Path, help="readme_images.json as it was before versions were filled: the old details")
    a = ap.parse_args()
    data = json.loads(a.data.read_text(encoding="utf-8"))
    old = json.loads(a.main.read_text(encoding="utf-8")) if a.main else data
    head = json.loads(a.head.read_text(encoding="utf-8")) if a.head else old
    files = index()
    stat, changed, left = collections.Counter(), [], []
    for kind in ("poster", "tables"):
        for mi, mac in enumerate(data[kind]["macs"]):
            for ri_, row in enumerate(mac["rows"]):
                for eng, c in row["cells"].items():
                    if eng == "Kapitsa" or not isinstance(c, dict) or not ri.is_num(c.get("value")):
                        continue
                    oc = old[kind]["macs"][mi]["rows"][ri_]["cells"][eng]
                    olddetail = oc.get("detail") if isinstance(oc, dict) else None
                    man = MANUAL.get((next(k for k in MAC if k in mac["name"]), row["model"], eng))
                    if man:
                        stat["manual"] += 1
                        if a.write:
                            for k in ("src", "version", "version_from"):
                                c.pop(k, None)
                            c.update(man)
                            t = template(olddetail, eng)
                            if t and VERSION_TOKEN.search(olddetail or "") is None and t != "{v}":
                                c["detail"] = t
                            elif t and t != "{v}":
                                c["detail"] = t
                            else:
                                c.pop("detail", None)
                        continue
                    cs = candidates(files, mac["name"], row["model"], eng, c["value"])
                    by = collections.defaultdict(list)
                    for x in cs:
                        by[ri.version_label(eng, x["version"])].append(x)
                    if len(by) > 1:
                        session = [x for x in cs if x["path"].split("/")[0] in (c.get("note") or "")]
                        keep = [x for x in session if True] or [x for x in cs if olddetail and (ri.version_label(eng, x["version"]) or "\0") in olddetail]
                        by = collections.defaultdict(list)
                        for x in keep:
                            by[ri.version_label(eng, x["version"])].append(x)
                    where = f"{kind} {mac['name'][-9:]} / {row['model']} / {eng} {c['value']} (was {olddetail!r})"
                    if len(by) != 1:
                        left.append((where, sorted(map(str, by)), len(cs)))
                        stat["left"] += 1
                        continue
                    (lab, xs), = by.items()
                    if lab is None:
                        stat["not recorded"] += 1
                        if a.write:
                            c["src"] = sorted(x["path"] for x in xs)
                            c["version"] = NOT_RECORDED
                            t = template(olddetail, eng)
                            c.pop("detail", None) if not t or t == "{v}" else c.__setitem__("detail", t)
                        continue
                    stat["linked"] += 1
                    t = template(olddetail, eng)
                    shown = (t or "{v}").replace("{v}", lab)
                    hc = head[kind]["macs"][mi]["rows"][ri_]["cells"][eng]
                    if shown != (hc.get("detail") if isinstance(hc, dict) else None):
                        changed.append(f"{where.split(' (was')[0]} (shows {hc.get('detail')!r}) -> {shown!r}")
                    if a.write:
                        c["src"] = sorted(x["path"] for x in xs)
                        if t and t != "{v}":
                            c["detail"] = t
                        else:
                            c.pop("detail", None)
    print(stat)
    print("CHANGED vs the version the cell showed:")
    print("\n".join("  " + x for x in changed))
    print("LEFT for a hand-written src:")
    for w, v, n in left:
        print("  ", w, v, n)
    if a.write:
        write_rows(a.data, data)
    return data


def write_rows(path, data):
    """rewrite only the cells rows of the data file, in place, keeping the file's layout"""
    out, kind, mac = [], "poster", None
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip().startswith('"tables"'):
            kind = "tables"
        m = re.match(r'^ *"name": "(MacBook[^"]*)",$', line)
        if m:
            mac = m.group(1)
        st = line.strip()
        if st.startswith('{"model":') and '"cells"' in st:
            comma = st.endswith(",")
            row = json.loads(st[:-1] if comma else st)
            new = next(r for mc in data[kind]["macs"] if mc["name"] == mac for r in mc["rows"] if r["model"] == row["model"])
            line = re.match(r" *", line).group(0) + json.dumps(new, ensure_ascii=False) + ("," if comma else "")
        out.append(line)
    text = "\n".join(out)
    json.loads(text)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
