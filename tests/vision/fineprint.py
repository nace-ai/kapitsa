#!/usr/bin/env python3
"""Small text in Retina screenshots against a running `kapitsa serve`: does the image size cap
(--image-max-pixels) keep it readable?

usage: fineprint.py BASE_URL [IMAGE_DIR]   (needs Pillow; IMAGE_DIR: where to write the two images)
Two 2880x1800 screenshots as macOS takes them on a Retina display (2x): a code editor in 11 pt Menlo (22 px) with a
10 pt status bar, and a table in 10 pt Helvetica (20 px). The answers are read off them (thinking off, greedy); the
script prints each answer and whether it is right, and the prompt's size in tokens.
"""
import base64, json, os, random, sys, tempfile, time, urllib.request

from PIL import Image, ImageDraw, ImageFont

BASE = sys.argv[1].rstrip("/")
DIR = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp()
MENLO, HELV = "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Helvetica.ttc"


def editor(path):
    rng = random.Random(7)
    W, H = 2880, 1800
    im = Image.new("RGB", (W, H), (30, 31, 34))
    d = ImageDraw.Draw(im)
    code, ui = ImageFont.truetype(MENLO, 22), ImageFont.truetype(HELV, 20)
    d.rectangle((0, 0, 480, H), fill=(37, 38, 42))
    for i, n in enumerate(["client.py", "retry.py", "config.py", "server.py", "models.py", "cache.py", "auth.py"]):
        d.text((28, 70 + i * 34), n, fill=(190, 190, 190), font=ui)
    idents = ["request", "response", "attempt", "delay", "timeout", "session", "payload", "headers", "status", "logger"]
    lines = []
    for _ in range(63):
        a, b, c = rng.choice(idents), rng.choice(idents), rng.choice(idents)
        lines.append("    " * rng.randint(0, 2) + rng.choice(
            [f"{a} = {b}.get('{c}')", f"if {a} is None:", f"return {a}", f"{a}.update({b}={c})", f"for {a} in {b}:",
             f"logger.debug('{a} %s', {b})", f"{a} += {rng.randint(2, 900)}"]))
    lines[30], lines[31] = "RETRY_BACKOFF_MS = 4375", "MAX_ATTEMPTS = 6"
    for i, t in enumerate(lines):
        d.text((500, 40 + i * 27), f"{i + 186:4d}", fill=(110, 110, 110), font=code)
        d.text((580, 40 + i * 27), t, fill=(212, 212, 212), font=code)
    d.rectangle((0, H - 34, W, H), fill=(0, 122, 204))
    d.text((W - 640, H - 28), "Ln 217, Col 33    Spaces: 4    UTF-8    Python 3.12", fill=(255, 255, 255), font=ui)
    im.save(path)


def table(path):
    rng = random.Random(11)
    W, H = 2880, 1800
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f, fb = ImageFont.truetype(HELV, 20), ImageFont.truetype(HELV, 20, index=1)
    xs = [60, 520, 900, 1200, 1500, 1800]
    for j, c in enumerate(["Company", "Region", "Q1", "Q2", "Q3", "Q4"]):
        d.text((xs[j], 60), c, fill="black", font=fb)
    names = ["Acme Robotics", "Bluefin Data", "Cedar Health", "Delta Freight", "Ember Studio", "Falcon Grid",
             "Garnet Bio", "Harbor Labs", "Iris Optics", "Juniper AI", "Kestrel Labs", "Lumen Works", "Maple Finance",
             "Nova Retail", "Onyx Mining", "Pine Logistics", "Quartz Media", "Raven Security", "Sable Foods",
             "Tidal Energy", "Umber Textiles", "Vega Motors", "Willow Care", "Xeno Pharma", "Yarrow Farms",
             "Zephyr Air"]
    rows = {}
    for i, name in enumerate(names):
        y = 110 + i * 40
        if i % 2:
            d.rectangle((40, y - 6, 2100, y + 30), fill=(244, 246, 250))
        row = [name, rng.choice(["EMEA", "APAC", "NA", "LATAM"])] + [f"{rng.randint(10000, 99999):,}" for _ in range(4)]
        rows[name] = row
        for j, t in enumerate(row):
            d.text((xs[j], y), t, fill=(30, 30, 30), font=f)
    im.save(path)
    return rows


def ask(path, question, max_tokens=40):
    url = "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": question}]}]}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}),
        timeout=900).read())
    return r["choices"][0]["message"].get("content") or "", r["usage"]["prompt_tokens"], time.time() - t0


ed, tb = os.path.join(DIR, "retina_editor.png"), os.path.join(DIR, "retina_table.png")
editor(ed)
rows = table(tb)
cases = [
    (ed, "What value is RETRY_BACKOFF_MS set to, and on which line number? Answer as: value, line.", ["4375", "216"]),
    (ed, "What line and column does the status bar at the bottom right show? Answer as: Ln X, Col Y.", ["217", "33"]),
    (tb, "What is the Q3 value for Kestrel Labs? Answer with the number only.", [rows["Kestrel Labs"][4]]),
    (tb, "What is the Q1 value for Garnet Bio? Answer with the number only.", [rows["Garnet Bio"][2]]),
    (tb, "Which region is Raven Security in? One word.", [rows["Raven Security"][1]]),
]
right = 0
for path, q, want in cases:
    a, n, dt = ask(path, q)
    ok = all(w.replace(",", "") in a.replace(",", "") for w in want)
    right += ok
    print(("ok   " if ok else "MISS ") + f"{os.path.basename(path)} ({n} prompt tokens, {dt:.1f} s): {q[:48]!r} -> {a.strip()[:60]!r}"
          f" (want {', '.join(want)})", flush=True)
print(f"{right} of {len(cases)} read right")
