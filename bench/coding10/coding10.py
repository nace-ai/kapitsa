#!/usr/bin/env python3
"""coding10: ten simple Python coding prompts against an OpenAI-compatible server (kapitsa serve).

Each prompt asks for one function; the answer passes when the hidden asserts in prompts.json run clean.
Time is wall clock from the first request to the finished answer (every turn, tool call and retry
included). An arm is a JSON object of levers (see ARM_DEFAULTS); arms run interleaved per prompt.

  python3 bench/coding10/coding10.py --url http://127.0.0.1:8350 --arms arms/base.json arms/opt.json \
      --out results.json [--reps 1] [--only p01_fizzbuzz]

Stdlib only. Runs model-written code in a subprocess with a timeout, in a temporary directory.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "littlecoder"))
import tools_impl  # noqa: E402  little-coder's tool runtime, ported

HERE = os.path.dirname(os.path.abspath(__file__))

ARM_DEFAULTS = {
    "name": "base",
    "system": None,            # system prompt text, or "@file" relative to this folder
    "thinking": None,          # None: server default; False/True: chat_template_kwargs.enable_thinking
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "presence_penalty": None,
    "max_tokens": 8192,
    "show_examples": True,     # put the visible example assert in the prompt
    "suffix": "",             # appended to the user prompt
    "self_test_retries": 0,    # run the visible examples locally; on failure send the error back, up to N times
    "tools": None,             # None, or "@tools.json": OpenAI tool definitions; loop runs tool calls
    "max_turns": 6,            # tool loop cap
    "extra": {},               # merged into the request body as-is
    "task_template": None,     # "@file": user prompt template with {entry_point} and {problem statement / docstring}
    "tool_runtime": "builtin", # "builtin" or "littlecoder" (littlecoder/tools_impl.py, a workdir per answer)
    "write_guard": True,       # littlecoder runtime: write_file refuses existing files
    "text_tool_fallback": False,  # parse tool calls the model printed as text (littlecoder parser)
    "autorun": False,          # after write_file/edit_file of solution.py, append the examples' test result (saves a run turn)
    "thinking_budget": None,   # N: thinking turn capped at N tokens; if cut, re-ask with thinking off (little-coder)
    "retry_style": "plain",    # "plain" or "lc" (little-coder's "The tests failed. Output: ..." turn)
}


def load_arm(path):
    with open(path) as f:
        arm = json.load(f)
    out = dict(ARM_DEFAULTS)
    out.update(arm)
    for key in ("system", "tools", "task_template"):
        v = out.get(key)
        if isinstance(v, str) and v.startswith("@"):
            p = os.path.join(os.path.dirname(os.path.abspath(path)), v[1:])
            with open(p) as f:
                out[key] = json.load(f) if key == "tools" else f.read()
    return out


def chat(url, body, timeout=900):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract_code(text, entry):
    """Last fenced block that defines the entry function; else any last block; else the raw text if it defines it."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = CODE_RE.findall(text)
    for b in reversed(blocks):
        if re.search(r"^\s*def\s+%s\s*\(" % re.escape(entry), b, re.M):
            return b
    if blocks:
        return blocks[-1]
    if re.search(r"^def\s+%s\s*\(" % re.escape(entry), text, re.M):
        return text
    return None


def run_python(code, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "main.py")
        with open(p, "w") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, p], cwd=d, capture_output=True, text=True, timeout=timeout)
            return r.returncode, (r.stdout + r.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            return 124, "timeout after %ds" % timeout


def check(code, asserts):
    if code is None:
        return False, "no code"
    rc, out = run_python(code + "\n\n" + "\n".join(asserts) + "\nprint('ALL_OK')\n")
    return rc == 0 and "ALL_OK" in out, out.strip()[-400:]


def tool_exec(name, args, state):
    """Built-in tool runner: run_python / write_file / run_tests / read_file. Unknown names are reported back."""
    if name in ("run_python", "python", "execute_python"):
        code = args.get("code", "")
        state["last_tool_code"] = code
        rc, out = run_python(code)
        return "exit %d\n%s" % (rc, out)
    if name == "write_file":
        state.setdefault("files", {})[args.get("path", "solution.py")] = args.get("content", "")
        state["last_tool_code"] = args.get("content", "")
        return "wrote %s (%d bytes)" % (args.get("path", "solution.py"), len(args.get("content", "")))
    if name == "read_file":
        return state.get("files", {}).get(args.get("path", ""), "error: no such file")
    if name == "run_tests":
        code = args.get("code") or state.get("last_tool_code") or ""
        ok, out = check(code, state["examples"])
        return ("PASS" if ok else "FAIL") + "\n" + out
    return "error: unknown tool %s" % name


def build_body(arm, messages):
    body = {"model": "local", "messages": messages, "max_tokens": arm["max_tokens"], "stream": False}
    for k in ("temperature", "top_p", "top_k", "presence_penalty"):
        if arm[k] is not None:
            body[k] = arm[k]
    if arm["thinking"] is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(arm["thinking"])}
    if arm["tools"]:
        body["tools"] = arm["tools"]
    body.update(arm.get("extra") or {})
    return body


def solve(url, arm, item):
    prompt = item["prompt"]
    if arm["show_examples"] and item.get("examples"):
        prompt += "\n\nExample:\n" + "\n".join(item["examples"])
    if arm["task_template"]:
        prompt = arm["task_template"].replace("{entry_point}", item["entry"]).replace(
            "{problem statement / docstring}", prompt)
    if arm["suffix"]:
        prompt += "\n\n" + arm["suffix"]
    messages = []
    if arm["system"]:
        messages.append({"role": "system", "content": arm["system"]})
    messages.append({"role": "user", "content": prompt})
    state = {"examples": item.get("examples", [])}
    workdir = tempfile.mkdtemp(prefix="c10_")
    budget_used = False
    t0 = time.time()
    turns, ctoks, retries, tool_calls = 0, 0, 0, 0
    code, finish, err = None, None, None
    while True:
        turns += 1
        body = build_body(arm, messages)
        budgeted = arm["thinking_budget"] and not budget_used and arm["thinking"] is not False
        if budgeted:
            body["max_tokens"] = arm["thinking_budget"]
        try:
            resp = chat(url, body)
        except Exception as e:  # noqa: BLE001 - a failed request is a failed answer, logged
            err = repr(e)[:300]
            break
        msg = resp["choices"][0]["message"]
        finish = resp["choices"][0].get("finish_reason")
        ctoks += (resp.get("usage") or {}).get("completion_tokens", 0)
        calls = msg.get("tool_calls") or []
        if not calls and arm["text_tool_fallback"] and arm["tools"]:
            parsed = tools_impl.parse_text_tool_calls(msg.get("content") or "")
            calls = [{"id": "t%d" % i, "type": "function",
                      "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", c.get("input", {})))}}
                     for i, c in enumerate(parsed) if c.get("name")]
        if budgeted and finish == "length" and not calls and not extract_code(msg.get("content") or "", item["entry"]):
            budget_used = True
            reasoning = msg.get("reasoning_content") or msg.get("content") or ""
            messages.append({"role": "assistant", "content": "Here is my analysis so far:\n" + reasoning[-2000:]})
            messages.append({"role": "user", "content":
                             "Good analysis. Now produce the final answer directly \u2014 no more deliberation needed."})
            arm = dict(arm, thinking=False)
            continue
        if calls and arm["tools"] and turns < arm["max_turns"]:
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                tool_calls += 1
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    args, out = {}, "error: arguments are not valid JSON: %s" % e
                else:
                    if arm["tool_runtime"] == "littlecoder":
                        out = tools_impl.run_tool(c["function"]["name"], args, workdir, write_guard=arm["write_guard"])
                        sol = os.path.join(workdir, "solution.py")
                        if (arm["autorun"] and c["function"]["name"] in ("write_file", "edit_file")
                                and item.get("examples") and os.path.exists(sol)):
                            with open(sol) as f:
                                ok_ex, out_ex = check(f.read(), item["examples"])
                            out += "\nexamples run automatically: " + ("ALL PASS" if ok_ex else "FAILED\n" + out_ex)
                    else:
                        out = tool_exec(c["function"]["name"], args, state)
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": out})
            continue
        content = msg.get("content") or ""
        code = extract_code(content, item["entry"])
        if code is None and os.path.exists(os.path.join(workdir, "solution.py")):
            with open(os.path.join(workdir, "solution.py")) as f:
                code = f.read()
        if code is None and state.get("last_tool_code"):
            code = state["last_tool_code"]
        if retries < arm["self_test_retries"] and item.get("examples"):
            ok, out = check(code, item["examples"])
            if not ok:
                retries += 1
                messages.append({"role": "assistant", "content": content})
                if arm["retry_style"] == "lc":
                    messages.append({"role": "user", "content": "The tests failed. Output:\n\n```\n%s\n```\n\n"
                                     "Fix the implementation and try again." % (out if code else "no ```python code block found")[-4000:]})
                    continue
                messages.append({"role": "user", "content":
                                 "Your code fails the example:\n%s\nFix it and reply with the complete corrected "
                                 "function in one ```python block." % (out if code else "no ```python code block found")})
                continue
        break
    wall = time.time() - t0
    shutil.rmtree(workdir, ignore_errors=True)
    ok, out = check(code, item.get("examples", []) + item["tests"])
    return {"id": item["id"], "pass": ok, "wall_s": round(wall, 2), "turns": turns, "retries": retries,
            "tool_calls": tool_calls, "budget_cut": budget_used, "completion_tokens": ctoks, "finish": finish, "error": err,
            "check_out": out[-300:], "code": code}


def sysinfo():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"load1": sh("sysctl -n vm.loadavg").split()[1:2], "ac": "AC Power" in sh("pmset -g batt")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8350")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--label", default="")
    ap.add_argument("--require-ac", action="store_true", help="wait for AC before each prompt, redo one that ends on battery, keep it out of the summary")
    a = ap.parse_args()
    with open(os.path.join(HERE, "prompts.json")) as f:
        items = json.load(f)
    if a.only:
        items = [i for i in items if i["id"] in a.only]
    arms = [load_arm(p) for p in a.arms]
    try:
        models = json.loads(urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=10).read())
    except Exception as e:  # noqa: BLE001
        models = {"error": repr(e)}
    # warmup: one short request so the first prompt does not pay for loading
    try:
        chat(a.url, {"model": "local", "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 8,
                     "chat_template_kwargs": {"enable_thinking": False}})
    except Exception as e:  # noqa: BLE001
        print("warmup failed:", e, file=sys.stderr)
    rows = []
    for rep in range(a.reps):
        for item in items:
            order = arms if (rep + items.index(item)) % 2 == 0 else list(reversed(arms))
            for arm in order:
                while True:  # a timed prompt counts only if it starts and ends on AC; one that ends on battery is redone
                    while a.require_ac and not sysinfo()["ac"]:
                        print("waiting for AC", time.strftime("%H:%M:%S"), flush=True)
                        time.sleep(30)
                    before = sysinfo()
                    r = solve(a.url, arm, item)
                    after = sysinfo()
                    r.update(arm=arm["name"], rep=rep, sys=after, sys_before=before,
                             on_battery=not (before["ac"] and after["ac"]))
                    rows.append(r)
                    print("%-10s %-22s rep%d %s %7.1fs tok=%d turns=%d%s" % (arm["name"], item["id"], rep,
                          "PASS" if r["pass"] else "fail", r["wall_s"], r["completion_tokens"], r["turns"],
                          " ON_BATTERY" if r["on_battery"] else ""), flush=True)
                    if not (a.require_ac and r["on_battery"]):
                        break
    summary = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm["name"] and not (a.require_ac and r.get("on_battery"))]
        summary[arm["name"]] = {"pass": sum(r["pass"] for r in rs), "n": len(rs),
                                "wall_s_total": round(sum(r["wall_s"] for r in rs), 1),
                                "wall_s_median": round(sorted(r["wall_s"] for r in rs)[len(rs) // 2], 1),
                                "completion_tokens": sum(r["completion_tokens"] for r in rs)}
    for k, v in summary.items():
        print("SUMMARY %s pass=%d/%d wall_total=%.1fs median=%.1fs tok=%d" % (k, v["pass"], v["n"],
              v["wall_s_total"], v["wall_s_median"], v["completion_tokens"]))
    commit = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"label": a.label, "argv": sys.argv, "bench_commit": commit, "url": a.url, "models": models,
           "server_argv": os.environ.get("CODING10_SERVER_ARGV", ""),
           "kapitsa_version": os.environ.get("CODING10_KAPITSA_VERSION", ""),
           "arms": arms, "summary": summary, "rows": rows,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
