# Decode speed per task on any OpenAI-compatible server, streamed: tokens after the first one / time after the first one.
# usage: stream.py BASE_URL MODEL REPO LABEL
import json, sys, time, urllib.request
base, model, repo, label = sys.argv[1:5]
src = open(f"{repo}/tools/readme_images/reddit_grid.py").read().splitlines()[:120]
code = "\n".join(src)
work = {
    "write code": "Write a Python function that merges two sorted lists into one sorted list, with a docstring and three "
                  "unit tests using unittest.",
    "edit code": "Add a short docstring to every function in this file and return the whole file, unchanged otherwise, "
                 "in one code block.\n\n```python\n" + code + "\n```",
    "explain code": "Explain what this code does, function by function.\n\n```python\n" + code + "\n```",
}
def run(p, temp):
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 512, "temperature": temp,
            "top_p": 0.95, "top_k": 20, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; n = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            for c in d.get("choices") or []:
                if (c.get("delta") or {}).get("content"):
                    last = time.time(); first = first or last
    return n, (n - 1) / (last - first), first - t0
run("Say hi.", 0.0)  # warm-up
for name, p in work.items():
    for temp in (0.0, 0.6):
        for i in range(2):
            n, tps, ttft = run(p, temp)
            print(json.dumps({"engine": label, "task": name, "T": temp, "tokens": n, "tps": round(tps, 2), "ttft": round(ttft, 2)}), flush=True)
