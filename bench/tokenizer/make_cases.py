"""Write the fixed tokenizer stress/control cases to stdout; does not run a model."""

import base64
import json
from pathlib import Path
import random

root = Path(__file__).resolve().parents[2]
rng = random.Random(72519)
cases = []
# Fixed from a reference-only timing probe; identical for every arm and device.
iterations = [177, 987, 20000, 20000, 20000, 20000, 2950, 20000, 3904,
              237, 20000, 1037, 9, 3588, 68, 3, 990, 13, 201]


def add(name, text):
    cases.append({"name": name, "text": text, "iterations": iterations[len(cases)]})


add("corpus-64k", (root / "bench/corpus.txt").read_text()[:65536])
add("source-32k", (root / "src/server/service_config.cpp").read_text()[:32768])
add("chat-short", "<|im_start|>user\nPlease explain how HTTP caching works.\n<|im_end|>\n<|im_start|>assistant\n")
for n in [64, 256, 1024, 8192, 32768]:
    add(f"letters-{n}", "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n)))
    add(f"repeat-{n}", "a" * n)
    add(f"cjk-{n}", "".join(rng.choice("天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈")
                          for _ in range(n // 3)))
add("base64-64k", base64.b64encode(rng.randbytes(49152)).decode())
print(json.dumps(cases, ensure_ascii=False))
