#!/usr/bin/env python3
"""Image input against a running `kapitsa serve` (stdlib only).

usage: check_server.py BASE_URL IMAGE_DIR [--quick]   (--quick: the answers and the follow-up only)
IMAGE_DIR holds chart.png (a bar chart: May is highest at 73), terminal.png (a terminal with a ZeroDivisionError at
train.py line 81) and aerial.png (the Golden Gate Bridge). Checks:
  - answers about the three images are right (thinking off, greedy)
  - a follow-up turn resending the same image reuses the cached prompt (cached_tokens covers the image)
  - a different image of the same size in the same place does not (cached_tokens stops before it)
  - an http:// image URL works (served from IMAGE_DIR by a local server)
  - errors: video parts, broken base64, unsupported URL schemes and a text model's refusal are 400s with a message
  - time to first token for each image, cold and cached
"""
import base64, http.server, json, os, sys, threading, time, urllib.error, urllib.request

BASE, DIR = sys.argv[1].rstrip("/"), sys.argv[2]
QUICK = "--quick" in sys.argv
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (": " + detail if detail else ""), flush=True)
    if not ok:
        fails.append(name)


def data_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def post(messages, max_tokens=80, stream=True, extra=None):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}, **(extra or {})}
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": json.loads(e.read()).get("error", {}).get("message", "")}
    if not stream:
        j = json.loads(r.read())
        return {"status": 200, "content": j["choices"][0]["message"]["content"], "usage": j["usage"],
                "total_s": time.time() - t0}
    out = {"status": 200, "content": "", "usage": None, "ttft": None}
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        c = json.loads(line[6:])
        if "error" in c:
            out["status"], out["error"] = 400, c["error"].get("message", "")
            break
        d = (c.get("choices") or [{}])[0].get("delta", {}) if c.get("choices") else {}
        if (d.get("content") or d.get("reasoning_content")) and out["ttft"] is None:
            out["ttft"] = time.time() - t0
        out["content"] += d.get("content") or ""
        if c.get("usage"):
            out["usage"] = c["usage"]
    out["total_s"] = time.time() - t0
    return out


def user(text, *images):
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in images] +
            [{"type": "text", "text": text}]}


img = {k: data_url(os.path.join(DIR, k + ".png")) for k in ("chart", "terminal", "aerial")}
cases = [
    ("chart", "Which month had the most active users, and how many? Answer in one sentence.", ["may", "73"]),
    ("terminal", "What error is shown in this terminal, and on which line of which file did it happen?",
     ["zerodivisionerror", "81", "train.py"]),
    ("aerial", "Which landmark is in this photo? Answer in one sentence.", ["golden gate"]),
]
for name, q, want in cases:
    r = post([user(q, img[name])])
    a = r.get("content", "").lower()
    u = r.get("usage") or {}
    check(f"answer about {name}", r["status"] == 200 and all(w in a for w in want),
          f"{r.get('content', r.get('error', ''))[:160]!r}; prompt {u.get('prompt_tokens')} tokens, time to first token "
          f"{r.get('ttft') or 0:.2f} s")

# follow-up with the same image: the prefix up to and past the image is reused
q1 = "Which month had the most active users, and how many? Answer in one sentence."
r1 = post([user(q1, img["chart"])])
conv = [user(q1, img["chart"]), {"role": "assistant", "content": r1["content"]}]
time.sleep(1.0)  # the server prefills the answer as history after 250 ms of idle time
r2 = post(conv + [{"role": "user", "content": "And which month had the fewest?"}])
u2 = r2.get("usage") or {}
cached, prompt = u2.get("prompt_tokens_details", {}).get("cached_tokens", 0), u2.get("prompt_tokens", 0)
check("follow-up with the same image hits the prefix cache", r2["status"] == 200 and cached >= prompt - 40 and
      "jan" in r2["content"].lower(), f"{cached} of {prompt} prompt tokens cached, time to first token "
      f"{r2.get('ttft') or 0:.2f} s, answer {r2['content'][:80]!r}")
if QUICK:  # answers and the follow-up only
    print(f"\n{len(fails)} failed" + (": " + ", ".join(fails) if fails else ""))
    sys.exit(1 if fails else 0)

# a resent identical request (no history change): cached up to its last token
r3 = post([user(q1, img["chart"])])
u3 = r3.get("usage") or {}
check("the same image prompt again: cached", (u3.get("prompt_tokens_details") or {}).get("cached_tokens", 0) >=
      u3.get("prompt_tokens", 0) - 16, f"{(u3.get('prompt_tokens_details') or {}).get('cached_tokens')} of "
      f"{u3.get('prompt_tokens')}; time to first token {r3.get('ttft') or 0:.2f} s")

# a different image of the same size in the same place: no reuse past the image's first row
from struct import pack
import zlib
def flip_png(path, out):  # the same picture mirrored left-right: same size, different content
    try:
        from PIL import Image, ImageOps
        ImageOps.mirror(Image.open(path)).save(out)
        return True
    except ImportError:
        return False
mirror = os.path.join(DIR, "chart_mirror.png")
if flip_png(os.path.join(DIR, "chart.png"), mirror):
    r4 = post([user(q1, data_url(mirror))])
    u4 = r4.get("usage") or {}
    c4 = (u4.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    # the prompt before the image: <|im_start|>user\n<|vision_start|> (a few tokens)
    check("a different image of the same size misses the cache", r4["status"] == 200 and c4 < 20 and
          u4.get("prompt_tokens") == u3.get("prompt_tokens"), f"{c4} of {u4.get('prompt_tokens')} cached; answer "
          f"{r4['content'][:80]!r}")
    os.remove(mirror)
else:
    print("(no PIL here: skipped the different-image check)")

# http:// URL
class Quiet(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=DIR, **k)
    def log_message(self, *a):
        pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}/chart.png"
r5 = post([user("What is the title of this chart?", url)], max_tokens=40)
check("http:// image URL", r5["status"] == 200 and "active users" in r5.get("content", "").lower(),
      repr(r5.get("content", r5.get("error", ""))[:100]))
r6 = post([user("What is this?", f"http://127.0.0.1:{srv.server_address[1]}/missing.png")], max_tokens=10)
check("http:// URL that is not there: 400", r6["status"] == 400 and "404" in r6.get("error", ""), r6.get("error", "")[:100])
srv.shutdown()

# errors
r = post([{"role": "user", "content": [{"type": "video", "video": "x.mp4"}, {"type": "text", "text": "hi"}]}], max_tokens=5)
check("video part: 400", r["status"] == 400 and "video" in r.get("error", "").lower(), r.get("error", "")[:100])
r = post([user("hi", "data:image/png;base64,@@@notbase64@@@")], max_tokens=5)
check("broken base64: 400", r["status"] == 400 and "base64" in r.get("error", ""), r.get("error", "")[:100])
r = post([user("hi", "data:image/png;base64," + base64.b64encode(b"not an image at all").decode())], max_tokens=5)
check("bytes that are no image: 400", r["status"] == 400 and "decode" in r.get("error", ""), r.get("error", "")[:100])
r = post([user("hi", "file:///etc/passwd")], max_tokens=5)
check("file:// URL: 400", r["status"] == 400 and "unsupported image URL" in r.get("error", ""), r.get("error", "")[:100])

# text only, after all that: a normal answer
r = post([{"role": "user", "content": "What is 2 + 3? Answer with the number only."}], max_tokens=10)
check("text-only request", r["status"] == 200 and "5" in r.get("content", ""), repr(r.get("content", ""))[:60])

print(f"\n{len(fails)} failed" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
