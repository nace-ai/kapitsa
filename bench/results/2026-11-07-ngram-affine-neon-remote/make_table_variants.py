"""Convert fn_synth.py's valid affine table into the two independent .bin test layouts."""
import argparse
import json
import struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("out", type=Path)
args = parser.parse_args()
model = args.model.resolve()
args.out.mkdir(parents=True, exist_ok=False)
data = (model / "ple.safetensors").read_bytes()
length = struct.unpack("<Q", data[:8])[0]
header = json.loads(data[8:8 + length])
payload = data[8 + length:]
entries = {name.rsplit(".", 1)[-1]: value for name, value in header.items() if name != "__metadata__"}
assert set(entries) == {"weight", "scales", "biases"}

def write(path, tensors, body):
    h = json.dumps(tensors, separators=(",", ":")).encode()
    h += b" " * ((-(8 + len(h))) % 16384)
    path.write_bytes(struct.pack("<Q", len(h)) + h + body)

for variant in ["affine", "bf16"]:
    folder = args.out / ("synthetic-bin-" + variant)
    folder.mkdir()
    for f in model.iterdir():
        if f.name != "ple.safetensors":
            (folder / f.name).symlink_to(f)
    if variant == "affine":
        write(folder / "ngram_table.bin", entries, payload)
    else:
        rows = entries["weight"]["shape"][0]
        dim = 512 // 8  # fn_synth.py's default hidden 512 / 8 n-gram heads
        words = [0x3c00 + ((i * 17 + 3) % 128) for i in range(rows * dim)]
        body = struct.pack("<" + "H" * len(words), *words)
        write(folder / "ngram_table.bin", {"weight": {"dtype": "BF16", "shape": [rows, dim],
              "data_offsets": [0, len(body)]}}, body)
