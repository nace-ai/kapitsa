"""ctx_sweep.py BASE_URL KEY OUT.json TOKENIZER REPO SIZES: prompt reading (time to the first token) and writing (the
middle of a 256-token reply) at each prompt size (tokens, Flint's tokenizer). The prompt is a coding agent's context:
the repo's benchmark corpus, then its source files in path order, cut to the size, with a random nonce at its very
start so no engine reuses an earlier request's cache. Sampling as Flint's card: T 1.0, top_p 0.95, top_k 20."""
import glob, json, os, random, sys, time, urllib.request
from tokenizers import Tokenizer

base, key, out, tok_path, repo = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
sizes = [int(s) for s in sys.argv[6].split(",")]
tok = Tokenizer.from_file(tok_path)
parts = [open(os.path.join(repo, "bench/corpus.txt")).read()]
for pat in ("src/**/*.cpp", "src/**/*.h", "src/**/*.mm", "src/**/*.metal", "tools/*.py", "docs/*.md"):
    for f in sorted(glob.glob(os.path.join(repo, pat), recursive=True)):
        parts.append(f"\n### FILE: {os.path.relpath(f, repo)}\n" + open(f, errors="replace").read())
ids = tok.encode("".join(parts), add_special_tokens=False).ids
print(f"context source: {len(ids)} tokens", flush=True)
QUESTION = "Walk through this code base file by file and explain in detail what each part does and how they connect."
results = []
for size in sizes:
    body_ids = ids[: max(0, size - 60)]
    nonce = f"session {random.getrandbits(64):016x}"
    msgs = [{"role": "system", "content": f"[{nonce}] You are a coding assistant working in this repository.\n\n"
                                          + tok.decode(body_ids)},
            {"role": "user", "content": QUESTION}]
    body = {"model": "flint", "stream": True, "max_tokens": 256, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "messages": msgs, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    times, usage = [], None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    times.append(time.time())
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens") or len(times)
    ttft = times[0] - t0 if times else None
    ev = len(times)
    i0, i1 = min(16, ev - 2), max(ev - 4, 1)
    dec = (ct * (i1 - i0) / ev) / (times[i1 - 1] - times[i0]) if ev > 20 else None
    rec = {"size": size, "prompt_tokens": pt, "ttft_s": ttft, "prefill_tps": pt / ttft if pt and ttft else None,
           "completion_tokens": ct, "events": ev, "decode_tps": dec}
    results.append(rec)
    print(json.dumps(rec), flush=True)
    json.dump(results, open(out, "w"), indent=1)
