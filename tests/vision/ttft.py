#!/usr/bin/env python3
"""Time to first token of image prompts through a running `kapitsa serve` (stdlib only).

usage: ttft.py BASE_URL WARMUP.png IMAGE.png [IMAGE.png ...]
A warm-up request with WARMUP.png first (the encoder's first load), then per image a cold request (a new image: the
encoder and the whole prompt run) and the same request again (cached: the prompt is reused). Thinking off, 1 token.
"""
import base64, json, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")


def ask(url, text):
    body = {"model": "x", "max_tokens": 1, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": text}]}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, usage = None, {}
    for line in urllib.request.urlopen(req, timeout=900):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        c = json.loads(line[6:])
        d = (c.get("choices") or [{}])[0].get("delta", {}) if c.get("choices") else {}
        if first is None and (d.get("content") or d.get("reasoning_content")):
            first = time.time() - t0
        usage = c.get("usage") or usage
    return first if first is not None else time.time() - t0, usage


def data_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


ask(data_url(sys.argv[2]), "warm-up")
for path in sys.argv[3:]:
    u = data_url(path)
    cold, us = ask(u, "Describe this screenshot.")
    warm, uw = ask(u, "Describe this screenshot.")
    print(json.dumps({"image": path.rsplit("/", 1)[-1], "prompt_tokens": us.get("prompt_tokens"),
                      "ttft_cold_s": round(cold, 3), "ttft_cached_s": round(warm, 3),
                      "cached_tokens": (uw.get("prompt_tokens_details") or {}).get("cached_tokens")}), flush=True)
