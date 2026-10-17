#!/usr/bin/env python3
"""`kapitsa mtp-sidecar` writes the same bytes as tools/mtp_sidecar.py (offline, stdlib only).

usage: test_sidecar.py <path to kapitsa> [<path to tools/mtp_sidecar.py>]

Builds a small checkpoint like Qwen/Qwen3.5-4B (two shards, the MTP head's tensors spread over both and mixed with
the trunk's, an index), with every finite bf16 value in a norm tensor (the + 1 rounding), and compares both tools'
output byte for byte, for both hidden-state variants. The real 4B sidecar is checked by `kapitsa pull 4b` itself:
its sha256 is pinned in the catalog (4f491dd2...).
"""
import json, os, random, struct, subprocess, sys, tempfile

kapitsa = sys.argv[1] if len(sys.argv) > 1 else "build/kapitsa"
tool = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "mtp_sidecar.py")
if not os.path.exists(tool):
    print(f"SKIP: {tool} not found")
    sys.exit(0)

rng = random.Random(7)
finite = [h for h in range(0x10000) if (h >> 7) & 0xFF != 0xFF]  # 65,280 finite bf16 values
assert len(finite) == 65280

def bf16(values):
    return struct.pack("<%dH" % len(values), *values)

def rand_bf16(n):
    return bf16([rng.choice(finite) for _ in range(n)])

def safetensors(tensors):
    header, data, off = {}, b"", 0
    for name, (shape, raw) in tensors.items():
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + len(raw)]}
        data += raw
        off += len(raw)
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + data

shards = {
    "model-00001-of-00002.safetensors": {
        "model.embed_tokens.weight": ([8, 4], rand_bf16(32)),
        "mtp.layers.0.mlp.down_proj.weight": ([4, 6], rand_bf16(24)),
        "mtp.pre_fc_norm_embedding.weight": ([4], rand_bf16(4)),
        "mtp.norm.weight": ([len(finite)], bf16(finite)),
    },
    "model-00002-of-00002.safetensors": {
        "model.norm.weight": ([4], rand_bf16(4)),
        "mtp.fc.weight": ([4, 8], rand_bf16(32)),
        "mtp.layers.0.input_layernorm.weight": ([4], rand_bf16(4)),
        "mtp.layers.0.self_attn.q_norm.weight": ([2], rand_bf16(2)),
        "mtp.layers.0.self_attn.k_norm.weight": ([2], rand_bf16(2)),
        "mtp.layers.0.post_attention_layernorm.weight": ([4], rand_bf16(4)),
        "mtp.pre_fc_norm_hidden.weight": ([4], rand_bf16(4)),
    },
}
fails = 0
with tempfile.TemporaryDirectory(prefix="kapitsa-sidecar-") as d:
    src = os.path.join(d, "ckpt")
    os.makedirs(src)
    wm = {}
    for fn, t in shards.items():
        with open(os.path.join(src, fn), "wb") as f:
            f.write(safetensors(t))
        wm.update({k: fn for k in t})
    with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    for hidden in ("post_norm", "pre_norm"):
        a, b = os.path.join(d, f"py-{hidden}.safetensors"), os.path.join(d, f"cpp-{hidden}.safetensors")
        subprocess.run([sys.executable, "-I", tool, src, a, "--hidden", hidden], check=True, capture_output=True)
        subprocess.run([kapitsa, "mtp-sidecar", src, b, "--hidden", hidden], check=True, capture_output=True)
        same = open(a, "rb").read() == open(b, "rb").read()
        print(("PASS" if same else "FAIL") + f" mtp-sidecar == tools/mtp_sidecar.py ({hidden}, {os.path.getsize(a)} bytes, "
              f"every finite bf16 value + 1)")
        fails += not same
print("FAILED: %d" % fails if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
