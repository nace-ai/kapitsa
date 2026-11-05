"""Pure-stdlib tool runtime ported from little-coder (github.com/itayinbarr/little-coder).

run_tool(name, args, workdir) -> str executes one of the four tools in tools.json.
Also ports little-coder's text tool-call fallback (repair_json, parse_text_tool_calls)
for models that print a call instead of emitting a native one.

Sources:
  write guard ............ .pi/extensions/write-guard/index.ts (editRecipe), v0.0.5 tools.py::_write
  edit semantics/errors .. skills/tools/edit.md (oldText exact + unique, edits vs original)
  read format ............ skills/tools/read.md ("N\\tline", offset 0-indexed, limit)
  output truncation ...... v0.0.5 tool_registry.execute_tool (first half + last quarter)
  repair_json/parse ...... .pi/extensions/output-parser/parser.ts, v0.0.5 local/output_parser.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

RUN_TIMEOUT_S = 10
MAX_OUTPUT = 8000  # little-coder uses 32000; small models do better with less


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    # tool_registry.execute_tool: first_half + [... N chars truncated ...] + last_quarter
    if len(text) <= limit:
        return text
    half, quarter = limit // 2, limit // 4
    cut = len(text) - half - quarter
    return f"{text[:half]}\n[... {cut} chars truncated ...]\n{text[-quarter:]}"


def _resolve(workdir: str, path: str) -> str:
    # Relative paths, and absolute ones outside workdir, land inside workdir
    # (write-guard's rewrite of "/foo.py" to cwd/foo.py, made strict here).
    root = os.path.realpath(workdir)
    p = os.path.realpath(os.path.join(root, path.lstrip("/")) if not path.startswith(root) else path)
    if p != root and not p.startswith(root + os.sep):
        raise ValueError(f"path escapes workdir: {path}")
    return p


def _write_file(args: dict, workdir: str, write_guard: bool) -> str:
    path, content = args.get("path"), args.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        return "Error: write_file requires string 'path' and 'content'."
    p = _resolve(workdir, path)
    if write_guard and os.path.exists(p):
        return (  # write-guard/index.ts editRecipe, tool names adapted
            f"Write refused — {path} already exists.\n\n"
            "write_file is for creating NEW files only. To change an existing file, use edit_file:\n"
            f'  {{"name": "edit_file", "arguments": {{"path": "{path}", '
            '"edits": [{"oldText": "<exact text currently in the file>", "newText": "<replacement text>"}]}}\n\n'
            "If you do not already know the file's current content, read it first with read_file to get the "
            "exact text for oldText (whitespace and indentation must match). Include enough surrounding "
            "context (2-3 lines) to make oldText unique in the file.\n\n"
            "For multiple changes, pass multiple entries in edits[] — one per location. Do NOT retry "
            "write_file; it will be refused again."
        )
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}."


def _edit_file(args: dict, workdir: str) -> str:
    path, edits = args.get("path"), args.get("edits")
    if isinstance(edits, dict):
        edits = [edits]
    if edits is None and "oldText" in args:  # common small-model flattening
        edits = [{"oldText": args["oldText"], "newText": args.get("newText", "")}]
    if not isinstance(path, str) or not isinstance(edits, list) or not edits:
        return "Error: edit_file requires 'path' and a non-empty 'edits' array of {oldText, newText}."
    p = _resolve(workdir, path)
    if not os.path.exists(p):
        return f"Error: {path} does not exist. Use write_file to create it."
    with open(p, encoding="utf-8") as f:
        original = f.read()
    spans = []
    for i, e in enumerate(edits):
        old, new = e.get("oldText"), e.get("newText", "")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return f"Error: edits[{i}] needs a non-empty string oldText and a string newText."
        n = original.count(old)
        if n == 0:  # recovery text: skills/tools/edit.md
            return (f"Error: edits[{i}] String not found in {path}. Use read_file to get the exact current "
                    "content (whitespace often differs), then retry edit_file with the exact string. "
                    "Do NOT fall back to write_file.")
        if n > 1:
            return (f"Error: edits[{i}] oldText Found multiple times ({n}) in {path}. Include more surrounding "
                    "context so oldText is unique, then retry edit_file.")
        start = original.index(old)
        spans.append((start, start + len(old), new))
    spans.sort()
    for a, b in zip(spans, spans[1:]):
        if a[1] > b[0]:
            return "Error: edits overlap. edits are matched against the ORIGINAL file; do not overlap or nest them."
    out, pos = [], 0
    for s, e, new in spans:
        out.append(original[pos:s]); out.append(new); pos = e
    out.append(original[pos:])
    with open(p, "w", encoding="utf-8") as f:
        f.write("".join(out))
    return f"Applied {len(spans)} edit(s) to {path}."


def _read_file(args: dict, workdir: str) -> str:
    path = args.get("path")
    if not isinstance(path, str):
        return "Error: read_file requires 'path'."
    p = _resolve(workdir, path)
    if not os.path.isfile(p):
        names = sorted(os.listdir(workdir))
        return f"Error: {path} not found. Files in workdir: {', '.join(names) or '(none)'}"
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    offset = int(args.get("offset") or 0)
    limit = int(args.get("limit") or len(lines))
    chunk = lines[offset:offset + limit]
    return "\n".join(f"{offset + i + 1}\t{l}" for i, l in enumerate(chunk)) or "(empty file)"


def _run_python(args: dict, workdir: str) -> str:
    code, path = args.get("code"), args.get("path")
    if not code and not path:
        return "Error: run_python needs 'code' (a snippet) or 'path' (a .py file in the workdir)."
    with tempfile.TemporaryDirectory(prefix="lc_run_") as tmp:
        # Copy the workspace so the snippet can `from solution import f` but cannot mutate it.
        for name in os.listdir(workdir):
            src = os.path.join(workdir, name)
            if os.path.isfile(src) and os.path.getsize(src) < 5_000_000:
                shutil.copy2(src, tmp)
        if code:
            target = os.path.join(tmp, "_snippet.py")
            with open(target, "w", encoding="utf-8") as f:
                f.write(code)
        else:
            target = os.path.join(tmp, os.path.basename(_resolve(workdir, path)))
            if not os.path.isfile(target):
                return f"Error: {path} not found in workdir."
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "HOME": tmp}
        try:
            r = subprocess.run([sys.executable, "-E", "-s", "-B", os.path.basename(target)], cwd=tmp,
                               capture_output=True, text=True, timeout=RUN_TIMEOUT_S, env=env,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return f"exit=timeout: killed after {RUN_TIMEOUT_S}s (infinite loop or too slow?)"
        out = (r.stdout + ("\n" if r.stdout and r.stderr else "") + r.stderr).strip()
        return _truncate(f"exit={r.returncode}\n{out or '(no output)'}")


def run_tool(name: str, args, workdir: str, write_guard: bool = True) -> str:
    """Execute a tool call. Never raises: errors come back as text the model can act on."""
    try:
        if isinstance(args, str):  # OpenAI sends arguments as a JSON string
            args = repair_json(args)
        if not isinstance(args, dict) or "_raw" in args:
            return f"Error: the arguments for tool '{name}' were malformed (not valid JSON). " \
                   "Please provide the arguments as a proper JSON object."
        n = name.strip().lower().replace("-", "_")
        n = {"write": "write_file", "edit": "edit_file", "read": "read_file",
             "run": "run_python", "python": "run_python", "run_tests": "run_python"}.get(n, n)
        if n == "write_file":
            return _truncate(_write_file(args, workdir, write_guard))
        if n == "edit_file":
            return _truncate(_edit_file(args, workdir))
        if n == "read_file":
            return _truncate(_read_file(args, workdir))
        if n == "run_python":
            return _run_python(args, workdir)
        # quality.ts buildCorrectionMessage("unknown_tool:")
        return f"Tool '{name}' does not exist. Available tools are: write_file, edit_file, read_file, " \
               "run_python. Please use one of these."
    except Exception as e:  # noqa: BLE001 - surface every failure to the model
        return f"Error: {type(e).__name__}: {e}"


# ── Text tool-call fallback (output-parser) ────────────────────────────────

def _escape_newlines_in_json_strings(text: str) -> str:
    out, in_str, i = [], False, 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_str and i + 1 < len(text):
            out.append(text[i:i + 2]); i += 2; continue
        if ch == '"':
            in_str = not in_str; out.append(ch)
        elif in_str and ch in "\n\t\r":
            out.append({"\n": "\\n", "\t": "\\t", "\r": "\\r"}[ch])
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _balanced_objects(s: str) -> list[str]:
    out, i = [], 0
    while True:
        start = s.find("{", i)
        if start < 0:
            return out
        depth, quote, esc, closed = 0, None, False, -1
        for j in range(start, len(s)):
            c = s[j]
            if quote:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == quote: quote = None
                continue
            if c in "'\"": quote = c
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0: closed = j; break
        if closed < 0:
            return out
        out.append(s[start:closed + 1]); i = closed + 1


def repair_json(raw: str) -> dict:
    """parser.ts repairJson: direct → escape newlines → trailing commas → quotes → keys → close braces."""
    t = (raw or "").strip()
    if not t:
        return {}
    for cand in (t, _escape_newlines_in_json_strings(t)):
        try:
            v = json.loads(cand)
            return v if isinstance(v, dict) else {"_raw": raw}
        except ValueError:
            pass
    fixed = re.sub(r",\s*]", "]", re.sub(r",\s*}", "}", _escape_newlines_in_json_strings(t)))
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')
    if '": ' not in fixed and '":"' not in fixed:
        fixed = re.sub(r"(?<=[{,\s])(\w+)\s*:", r'"\1":', fixed)
    fixed += "}" * max(0, fixed.count("{") - fixed.count("}"))
    fixed += "]" * max(0, fixed.count("[") - fixed.count("]"))
    for cand in [fixed] + _balanced_objects(fixed)[:1]:
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    return {"_raw": raw}


def _call_input(d: dict) -> dict:
    raw = next((d[k] for k in ("input", "parameters", "arguments", "args") if k in d), {})
    if isinstance(raw, str):
        raw = repair_json(raw)
    return raw if isinstance(raw, dict) else {}


def parse_text_tool_calls(text: str) -> list[dict]:
    """```tool / ```json fences, <tool_call> tags (Qwen), then bare {"name":..,"arguments":..} objects."""
    calls = []
    blocks = re.findall(r"```(?:tool|json)\s*\n(.*?)\n```", text, re.S)
    blocks += re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    for b in blocks:
        d = repair_json(b)
        if isinstance(d.get("name"), str) and d["name"]:
            calls.append({"name": d["name"], "arguments": _call_input(d)})
    if not calls:
        for chunk in _balanced_objects(text):
            if not re.search(r'"name"\s*:\s*"', chunk):
                continue
            d = repair_json(chunk)
            nested = bool(re.search(r"[{\[]", chunk[1:-1]))
            has_args = any(k in d for k in ("input", "parameters", "arguments", "args"))
            if nested and not has_args:
                continue  # prose about a config blob with a "name" key is not a call (#96)
            if isinstance(d.get("name"), str) and d["name"]:
                calls.append({"name": d["name"], "arguments": _call_input(d)})
    return calls


def extract_final_code(text: str) -> str | None:
    """Last ```python fenced block in the reply (harness glue, not from little-coder)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    return blocks[-1].strip() + "\n" if blocks else None
