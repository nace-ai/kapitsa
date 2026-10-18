#!/usr/bin/env python3
"""Reference prompt lengths for an agent trace, with a model's own tokenizer and chat template. Needs transformers.

  agent_trace_tokens.py agent-traces/opencode-ledger.json --model-dir ~/models/qwen3.6-35b-a3b-4bit [--write]

Renders every request of the trace the way agent_replay.py sends it, thinking off with the generation prompt:
"rendered" (the tools block in the system message, no tools field) and "api" (the tools as a request field). It
checks that both give the same text (they must: the rendered block is the template's own), counts the tokens, and
with --write stores them in the trace as each request's ref_prompt_tokens {"rendered": n, "api": n}, which the replay
compares with the prompt_tokens every engine reports (a conversation an engine cut short is INVALID). Tool-call
arguments are parsed from their JSON strings before rendering, as servers do for this template.
"""
import argparse, copy, json, os, sys, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import agent_replay  # noqa: E402


def for_template(msgs):
    out = copy.deepcopy(msgs)
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.model_dir))
    t = agent_replay.load_trace(a.trace)
    n = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
    mismatches = 0
    for rq in t["requests"]:
        counts = {}
        texts = {}
        for mode in ("rendered", "api"):
            msgs, extra = agent_replay.request_parts(t, rq, mode)
            texts[mode] = tok.apply_chat_template(for_template(msgs), tools=extra.get("tools"), tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
            counts[mode] = n(texts[mode])
        same = texts["rendered"] == texts["api"]
        mismatches += not same
        rec = (rq.get("response") or {}).get("prompt_tokens")
        print(f"request {rq['i']:2d} {rq['kind']:4s}: {counts['rendered']} tokens rendered, {counts['api']} api"
              f"{'' if same else ' (TEXT DIFFERS)'}; recorded by the server: {rec}")
        rq["ref_prompt_tokens"] = counts
    t["ref_prompt_tokens_from"] = {"tokenizer": os.path.basename(os.path.normpath(a.model_dir)),
                                   "chat_template": "the model's chat_template.jinja, enable_thinking=false",
                                   "rendered_equals_api": mismatches == 0}
    if mismatches:
        print(f"{mismatches} requests render differently with the tools block in the system message", file=sys.stderr)
    if a.write:
        for k in ("_path", "_sha256"):
            t.pop(k, None)
        with open(a.trace, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(t, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote ref_prompt_tokens into {a.trace}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
