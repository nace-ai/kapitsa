#!/usr/bin/env python3
"""Markdown tables from the quantization measurements.

  table.py stream <eval_stream result.json> [bytes json]     quality of every candidate against bf16
  table.py v4 <dir with one subdir per variant (measure.sh)>  4B: Kapitsa / MLX quality and speed
"""
import json, os, re, sys


def f(x, d=4):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def stream(path):
    r = json.load(open(path))
    print(f"base {r['base']}, {r['positions']} positions ({', '.join(r['sequences'])}), {r['layers']} layers")
    print("| variant | recipe | KL mean | KL p95 | KL p99 | top-1 agree | PPL (bf16 " + f(r["candidates"]["bf16"]["ppl"], 3) + ") |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for name, e in sorted(r["candidates"].items(), key=lambda kv: kv[1].get("kl_mean", 0)):
        if name == "bf16":
            continue
        print(f"| {name} | `{e['spec']}` | {e['kl_mean']:.5f} | {e['kl_p95']:.4f} | {e['kl_p99']:.4f} | "
              f"{100 * e['top1_agree']:.2f}% | {e['ppl']:.3f} |")


def last_json(p):
    try:
        lines = [l for l in open(p) if l.strip().startswith("{")]
        return json.loads(lines[-1]) if lines else None
    except FileNotFoundError:
        return None


def rate(p, pat):
    try:
        v = [float(m) for m in re.findall(pat, open(p).read())]
        return sorted(v)[len(v) // 2] if v else None
    except FileNotFoundError:
        return None


def v4(d):
    print("| variant | Kapitsa KL | top-1 | PPL | MLX KL | prefill 4k (K / MLX) | decode (K / MLX) | MTP decode | GB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isdir(p):
            continue
        k = last_json(os.path.join(p, "k_quality.jsonl")) or {}
        m = last_json(os.path.join(p, "m_quality.jsonl")) or {}
        ms = last_json(os.path.join(p, "m_speed.json")) or {}
        pre = rate(os.path.join(p, "k_prefill.txt"), r"\(([0-9.]+) tok/s\)")
        dec = rate(os.path.join(p, "k_decode.txt"), r"\(([0-9.]+) tok/s\)")
        spec = rate(os.path.join(p, "k_spec.txt"), r"\(([0-9.]+) tok/s\)")
        print(f"| {name} | {f(k.get('kl_mean'), 5)} | {f(k.get('top1_agree') and 100 * k['top1_agree'], 2)}% | "
              f"{f(k.get('ppl'), 3)} | {f(m.get('kl_mean'), 5)} | {f(pre, 0)} / {f(ms.get('prefill_tok_s'), 0)} | "
              f"{f(dec, 1)} / {f(ms.get('decode_tok_s'), 1)} | {f(spec, 1)} | {f(ms.get('peak_gb'), 2)} |")


if __name__ == "__main__":
    {"stream": stream, "v4": v4}[sys.argv[1]](sys.argv[2])
