"""Stop-string regressions against an already running, GPU-locked server.

Start it with verify-kapitsa's vk serve, then run NI_LOCK_HELD=1 python3
 tests/test_stop_sequences.py http://127.0.0.1:8090. The JSON report retains
requests and responses, including the control stream used to choose a split stop.
"""
import argparse
import json
import os
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()
    if os.environ.get("NI_LOCK_HELD") != "1" and os.environ.get("NACE_GPU_LOCK_HELD") != "1":
        parser.error("the server must hold the GPU lock; set NI_LOCK_HELD=1 after starting it with vk serve")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    body = {"model": model, "messages": [{"role": "user", "content":
            "Write the numbers one through ten as English words, separated by commas."}],
            "temperature": 0, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False}}
    result = {"model": model, "cases": [], "failures": []}

    def request(route, stream, stop=None, max_tokens=96):
        b = dict(body, stream=stream, max_tokens=max_tokens)
        if stop is not None:
            b["stop_sequences" if route == "/v1/messages" else "stop"] = [stop]
        req = urllib.request.Request(args.url.rstrip("/") + route, data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=180) as r:
            if not stream:
                d = json.load(r)
                if route == "/v1/messages":
                    content = "".join(x.get("text", "") for x in d["content"] if x["type"] == "text")
                else:
                    content = d["choices"][0]["message"].get("content") or ""
                return {"request": b, "response": d, "content": content}
            chunks, events = [], []
            for line in r:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                d = json.loads(line[6:])
                events.append(d)
                if route == "/v1/messages":
                    if d.get("type") == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta":
                        chunks.append(d["delta"]["text"])
                else:
                    chunks.extend(ch["delta"]["content"] for ch in d.get("choices", [])
                                  if ch.get("delta", {}).get("content"))
            return {"request": b, "events": events, "chunks": chunks, "content": "".join(chunks)}

    def check(name, actual, expected):
        if actual != expected:
            result["failures"].append({"name": name, "actual": actual, "expected": expected})

    for route in ["/v1/chat/completions", "/v1/messages"]:
        control = request(route, True)
        chunks = [x for x in control["chunks"] if x]
        if len(chunks) < 2:
            result["failures"].append({"route": route, "reason": "control needs at least two content chunks"})
            result["cases"].append({"route": route, "control": control})
            continue
        stop = "".join(chunks[:2])
        entry = {"route": route, "control": control, "chosen_stop": stop}
        for stream in [True, False]:
            key = "streamed" if stream else "nonstreamed"
            entry[key] = request(route, stream, stop)
            check(route + " " + key + " split stop", entry[key]["content"], "")
            if route == "/v1/messages":
                if stream:
                    end = next((x["delta"] for x in entry[key]["events"] if x.get("type") == "message_delta"), {})
                else:
                    end = entry[key]["response"]
                check(route + " " + key + " stop reason", end.get("stop_reason"), "stop_sequence")
                check(route + " " + key + " matched stop", end.get("stop_sequence"), stop)
            else:
                choices = entry[key]["events"][-1].get("choices", []) if stream else entry[key]["response"]["choices"]
                # A usage-only final event may follow the finish event.
                if stream:
                    choices = [c for event in entry[key]["events"] for c in event.get("choices", []) if c.get("finish_reason")]
                check(route + " " + key + " stop reason", choices[-1].get("finish_reason") if choices else None, "stop")
        # A stop that contains the entire reply as its prefix must be flushed unchanged at EOS and at max_tokens.
        absent = control["content"] + " [stop string absent from reply]"
        entry["unmatched_eos"] = request(route, True, absent)
        check(route + " unmatched EOS prefix", entry["unmatched_eos"]["content"], control["content"])
        entry["length_control"] = request(route, True, max_tokens=1)
        entry["unmatched_length"] = request(route, True, absent, max_tokens=1)
        check(route + " unmatched length prefix", entry["unmatched_length"]["content"], entry["length_control"]["content"])
        result["cases"].append(entry)
    result["pass"] = not result["failures"]
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
