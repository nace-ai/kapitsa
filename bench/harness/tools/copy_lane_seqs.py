#!/usr/bin/env python3
"""Token sequences of an agent trace for tools/copy_lane_check (the copy lane's calibration). Needs transformers.

  copy_lane_seqs.py agent-traces/opencode-ledger.json --model-dir ~/models/qwen3.6-35b-a3b-4bit --out seqs.txt

One line per request: its prompt as agent_replay.py sends it (the "rendered" tool mode, thinking off, with the
generation prompt), then "|", then the recorded answer as the model writes it (the chat template's rendering of the
answer after that prompt, without the closing <|im_end|>), both as comma-separated token ids.
"""
import argparse, os, sys, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import agent_replay  # noqa: E402
from agent_trace_tokens import for_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.model_dir), local_files_only=True)
    t = agent_replay.load_trace(a.trace)
    n_out = 0
    with open(a.out, "w") as f:
        for rq in t["requests"]:
            msgs, _ = agent_replay.request_parts(t, rq, "rendered")
            resp = rq["response"]
            answer = {"role": "assistant", "content": resp.get("content") or ""}
            if resp.get("tool_calls"):
                answer["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for c in resp["tool_calls"]]
            kw = dict(tokenize=False, enable_thinking=False)
            prompt = tok.apply_chat_template(for_template(msgs), add_generation_prompt=True, **kw)
            full = tok.apply_chat_template(for_template(msgs + [answer]), add_generation_prompt=False, **kw)
            if not full.startswith(prompt):
                raise SystemExit(f"request {rq['i']}: the answer's rendering does not continue the prompt")
            out = full[len(prompt):]
            out = out[:out.rindex("<|im_end|>")] if "<|im_end|>" in out else out
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            o_ids = tok(out, add_special_tokens=False)["input_ids"]
            n_out += len(o_ids)
            f.write(",".join(map(str, p_ids)) + "|" + ",".join(map(str, o_ids)) + "\n")
    print(f"{len(t['requests'])} requests, {n_out} answer tokens -> {a.out}")


if __name__ == "__main__":
    main()
