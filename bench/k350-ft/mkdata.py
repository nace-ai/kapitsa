#!/usr/bin/env python3
"""#350 drafter fine-tune: English coding conversations for self-distillation (CPU only).

Source: Magicoder-OSS-Instruct-75K problems and solutions (never the 19 held-out agent requests). Each item is an
OpenAI-style request (messages, optional tools) the 35B answers through `kapitsa serve`; gen.py sends them.
Styles: qa (plain coding question), implement / fix / explain / tests (agent turns with tools, thinking off,
a read tool call and its numbered file listing in the history).
Writes <out>/items.jsonl."""
import json, random, re, sys

SRC, OUT, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = random.Random(350)

SYSTEM_AGENT = (
    "You are a coding agent working in the user's project through a terminal harness. Help the user with software "
    "engineering tasks using the tools available. Keep answers short and concrete. Read a file before you edit it, "
    "make the smallest change that fixes the problem, and run the tests after editing when the project has them.\n\n"
    "<env>\n  Working directory: /work/task\n  Platform: darwin\n</env>")

def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req, "additionalProperties": False}}}

S = {"type": "string"}
TOOLS = [
    fn("shell", "Run a shell command in the working directory and return its output.",
       {"command": S, "workdir": S, "description": S}, ["command"]),
    fn("read", "Read a file. Lines come back numbered as `N: text`; use offset and limit for long files.",
       {"filePath": S, "offset": {"type": "number"}, "limit": {"type": "number"}}, ["filePath"]),
    fn("edit", "Replace exact text in a file. oldString must match the file exactly, without line-number prefixes.",
       {"filePath": S, "oldString": S, "newString": S, "replaceAll": {"type": "boolean"}},
       ["filePath", "oldString", "newString"]),
    fn("write", "Write a whole file, replacing it if it exists.", {"filePath": S, "content": S}, ["filePath", "content"]),
    fn("glob", "Find files by glob pattern.", {"pattern": S, "path": S}, ["pattern"]),
    fn("grep", "Search file contents with a regular expression.", {"pattern": S, "path": S, "include": S}, ["pattern"]),
]

EXT = {"python": "py", "cpp": "cpp", "java": "java", "typescript": "ts", "javascript": "js", "rust": "rs", "go": "go",
       "csharp": "cs", "php": "php", "swift": "swift", "shell": "sh", "ruby": "rb", "kotlin": "kt", "c": "c"}
MUT = [("<=", "<"), (">=", ">"), (" < ", " <= "), (" > ", " >= "), ("==", "!="), ("+ 1", "- 1"), ("- 1", "+ 1"),
       (" and ", " or "), ("True", "False"), ("range(1, ", "range(0, "), ("[0]", "[1]"), ("+=", "-="), (" * ", " + ")]

def code_of(sol):
    m = re.findall(r"```[a-zA-Z0-9_+#-]*\n(.*?)```", sol, re.S)
    if not m:
        return None
    c = max(m, key=len).rstrip("\n")
    n = c.count("\n") + 1
    return c if 6 <= n <= 160 else None

def mutate(code):
    opts = [(a, b) for a, b in MUT if a in code]
    if not opts:
        return None
    a, b = rng.choice(opts)
    idx = [m.start() for m in re.finditer(re.escape(a), code)]
    i = rng.choice(idx)
    return code[:i] + b + code[i + len(a):]

def numbered(c):
    return "\n".join(f"{i + 1}: {l}" for i, l in enumerate(c.split("\n")))

def fname(lang, problem):
    m = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", problem)
    base = m.group(1) if m else "solution"
    base = re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()[:30] or "solution"
    d = rng.choice(["", "src/", "lib/", "app/"])
    return f"{d}{base}.{EXT.get(lang, 'txt')}"

def read_turns(path, code, intro):
    return [
        {"role": "assistant", "content": intro, "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "read", "arguments": {"filePath": "/work/task/" + path}}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": numbered(code)},
    ]

rows = []
with open(SRC) as f:
    for line in f:
        r = json.loads(line)
        if r.get("lang") in EXT:
            rows.append(r)
py = [r for r in rows if r["lang"] == "python"]
other = [r for r in rows if r["lang"] != "python"]
rng.shuffle(py); rng.shuffle(other)
pool = py[: int(N * 0.75) * 2] + other[: int(N * 0.25) * 2]
rng.shuffle(pool)
items = []
for r in pool:
    if len(items) >= N:
        break
    prob, sol, lang = r["problem"].strip(), r["solution"], r["lang"]
    code = code_of(sol)
    u = rng.random()
    if code is None or u < 0.30:
        items.append({"style": "qa", "messages": [{"role": "user", "content": prob}]})
        continue
    path = fname(lang, prob)
    msgs = [{"role": "system", "content": SYSTEM_AGENT}]
    if u < 0.42:
        msgs.append({"role": "user", "content": f"{prob}\n\nWrite it to `{path}`."})
        style = "implement"
    elif u < 0.80:
        bad = mutate(code)
        if bad is None:
            continue
        summary = prob.split("\n\n")[0][:600]
        msgs.append({"role": "user", "content": rng.choice([
            f"`{path}` is not doing what it should. It is supposed to do this: {summary}\nFind the bug and fix it.",
            f"Something is wrong in {path}; the tests for it fail. What it should do: {summary}\nPlease fix it.",
            f"Fix the bug in `{path}`. Spec: {summary}"])})
        msgs += read_turns(path, bad, rng.choice(["I'll read the file first.", "Let me look at the code.", ""]))
        style = "fix"
    elif u < 0.90:
        msgs.append({"role": "user", "content": f"Explain what `{path}` does and point out anything that could go wrong."})
        msgs += read_turns(path, code, "")
        style = "explain"
    else:
        msgs.append({"role": "user", "content": f"Add unit tests for `{path}`."})
        msgs += read_turns(path, code, "I'll read the module first.")
        style = "tests"
    items.append({"style": style, "messages": msgs, "tools": TOOLS})
with open(OUT, "w") as f:
    for i, it in enumerate(items):
        it["id"] = i
        f.write(json.dumps(it) + "\n")
from collections import Counter
print(len(items), Counter(i["style"] for i in items))
