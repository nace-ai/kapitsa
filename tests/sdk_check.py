#!/usr/bin/env python3
"""The Anthropic and OpenAI Python SDKs read Kapitsa's Messages and Responses replies and its Chat Completions streams
(no model, no GPU).

    python -m venv venv && venv/bin/pip install anthropic openai
    venv/bin/python tests/sdk_check.py build/api_test

Runs `api_test --dump` for the replies of each scenario (as the server writes them: text, reasoning + text, reasoning
+ two tool calls, a custom tool call, a length cut, the context running out, a stop sequence, errors; streams with
keepalives), serves them from a fake server, and checks that each SDK parses them: every event and response validates
against the SDK's own types (pydantic, required fields and literals), the stream helpers assemble the same message or
response as the non-streamed reply, and the errors come out as the SDKs' error classes with the messages the agents
key on."""
import json, os, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import openai
from pydantic import TypeAdapter

API_TEST = sys.argv[1] if len(sys.argv) > 1 else "build/api_test"
CASES = tempfile.mkdtemp(prefix="kapitsa-sdk-")
subprocess.run([API_TEST, "--dump", CASES], check=True, capture_output=True)

fails = 0
def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "\n     " + str(detail)[:600]))
    fails += 0 if ok else 1


class Fake(BaseHTTPRequestHandler):
    """POST /v1/messages, /v1/messages/count_tokens, /v1/responses, /v1/chat/completions (streams only): the recorded
    reply of the scenario the request's model names (a stream when it asks for one)"""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        api = "responses" if self.path.startswith("/v1/responses") else \
            "chat" if self.path.startswith("/v1/chat/completions") else "anthropic"
        name = "count" if self.path.startswith("/v1/messages/count_tokens") else body.get("model", "")
        base = os.path.join(CASES, f"{api}-{name}")
        status = int(open(base + ".status").read()) if os.path.exists(base + ".status") else 200
        stream = body.get("stream") and status == 200
        data = open(base + (".sse" if stream else ".json"), "rb").read()
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{server.server_address[1]}"


def plain(items):
    """Output items as JSON, without the stream helper's parsed and parsed_arguments fields"""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if not k.startswith("parsed")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return [strip(json.loads(o.to_json())) for o in items]


def raw_events(path):
    """(event name, data) of a recorded stream (a Chat Completions stream's closing [DONE] left out)"""
    out = []
    for frame in open(path).read().split("\n\n"):
        name = data = None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
        if data is not None:
            out.append((name, data))
    return out


# ---------------------------------------------------------------- Anthropic Messages

ac = anthropic.Anthropic(base_url=BASE, api_key="k", max_retries=0)
MSG = [{"role": "user", "content": "hi"}]
EVENT = TypeAdapter(anthropic.types.RawMessageStreamEvent)
expect = {"text": ("end_turn", ["text"]), "reasoning-text": ("end_turn", ["thinking", "text"]),
          "reasoning-tools": ("tool_use", ["thinking", "tool_use", "tool_use"]),
          "custom-tool": ("tool_use", ["thinking", "tool_use"]), "length": ("max_tokens", ["thinking", "text"]),
          "context-full": ("model_context_window_exceeded", ["text"]), "stop-sequence": ("stop_sequence", ["text"])}
for name, (stop, kinds) in expect.items():
    for omitted in (False, True):
        case = name + ("-omitted" if omitted else "")
        body = json.load(open(os.path.join(CASES, f"anthropic-{case}.json")))
        try:
            anthropic.types.Message.model_validate(body)
            valid = True
        except Exception as e:
            valid = e
        check(f"anthropic {case}: the response validates as anthropic.types.Message", valid is True, valid)
        m = ac.messages.create(model=case, max_tokens=100, messages=MSG)
        check(f"anthropic {case}: messages.create parses it (stop_reason {stop}, blocks {kinds})",
              m.stop_reason == stop and [b.type for b in m.content][:len(kinds)] == kinds and m.usage.cache_read_input_tokens == 60
              and m.usage.input_tokens == 40, m)
        if name == "stop-sequence":
            check("anthropic: stop_sequence names the string", m.stop_sequence == "STOP", m.stop_sequence)
        bad = []
        for ev_name, data in raw_events(os.path.join(CASES, f"anthropic-{case}.sse")):
            if ev_name == "ping":  # the SDK skips pings before parsing events
                if data != {"type": "ping"}:
                    bad.append((ev_name, data))
                continue
            try:
                ev = EVENT.validate_python(data)
                if ev.type != ev_name:
                    bad.append((ev_name, ev.type))
            except Exception as e:
                bad.append((ev_name, str(e)[:300]))
        check(f"anthropic {case}: every stream event validates as a RawMessageStreamEvent of its name", not bad, bad)
        with ac.messages.stream(model=case, max_tokens=100, messages=MSG) as s:
            for _ in s:
                pass
            final = s.get_final_message()
        same = final.stop_reason == m.stop_reason and [b.model_dump() for b in final.content] == [b.model_dump() for b in m.content[:len(final.content)]]
        check(f"anthropic {case}: messages.stream assembles the same message", same,
              (final.model_dump(), m.model_dump()))
        thinking = [b for b in final.content if b.type == "thinking"]
        if thinking:
            check(f"anthropic {case}: the thinking block's signature is set" + (", its text omitted" if omitted else ""),
                  thinking[0].signature.startswith("kapitsa.r1.") and (thinking[0].thinking == "") == omitted, thinking[0])
        tools = [b for b in final.content if b.type == "tool_use"]
        if name == "reasoning-tools":
            check("anthropic: tool_use inputs parsed from input_json_delta",
                  [t.input for t in tools] == [{"file_path": "/a"}, {"file_path": "/b", "limit": 5}] and tools[0].id.startswith("toolu_"), tools)

try:
    ac.messages.create(model="too-long", max_tokens=100, messages=MSG)
    check("anthropic: a too-long prompt raises BadRequestError", False, "no error")
except anthropic.BadRequestError as e:
    check("anthropic: a too-long prompt raises BadRequestError \"prompt is too long: N tokens > M maximum\"",
          e.status_code == 400 and e.body["error"]["type"] == "invalid_request_error" and
          "prompt is too long: 140000 tokens > 131070 maximum" in str(e), e)
try:
    with ac.messages.stream(model="error", max_tokens=100, messages=MSG) as s:
        for _ in s:
            pass
    check("anthropic: an error event in the stream raises", False, "no error")
except anthropic.APIError as e:
    check("anthropic: an error event in the stream raises APIError with its message", "the engine failed" in str(e), e)
n = ac.messages.count_tokens(model="count", messages=MSG)
check("anthropic: messages.count_tokens", n.input_tokens == 1234, n)

# ---------------------------------------------------------------- OpenAI Responses

oc = openai.OpenAI(base_url=BASE + "/v1", api_key="k", max_retries=0)
EV = TypeAdapter(openai.types.responses.ResponseStreamEvent)
want = {"text": ("completed", ["message"]), "reasoning-text": ("completed", ["reasoning", "message"]),
        "reasoning-tools": ("completed", ["reasoning", "function_call", "function_call"]),
        "custom-tool": ("completed", ["reasoning", "custom_tool_call"]), "length": ("incomplete", ["reasoning", "message"]),
        "context-full": ("incomplete", ["message"]), "stop-sequence": ("completed", ["message"])}
for name, (status, kinds) in want.items():
    body = json.load(open(os.path.join(CASES, f"responses-{name}.json")))
    try:
        openai.types.responses.Response.model_validate(body)
        valid = True
    except Exception as e:
        valid = e
    check(f"responses {name}: the response validates as openai.types.responses.Response", valid is True, valid)
    r = oc.responses.create(model=name, input="hi")
    check(f"responses {name}: responses.create parses it (status {status}, items {kinds})",
          r.status == status and [o.type for o in r.output] == kinds and r.usage.input_tokens_details.cached_tokens == 60 and
          (status != "incomplete" or r.incomplete_details.reason == "max_output_tokens"), r)
    bad, seqs, events = [], [], raw_events(os.path.join(CASES, f"responses-{name}.sse"))
    for ev_name, data in events:
        try:
            ev = EV.validate_python(data)
            if ev.type != ev_name:
                bad.append((ev_name, ev.type))
            seqs.append(ev.sequence_number)
        except Exception as e:
            bad.append((ev_name, str(e)[:300]))
    check(f"responses {name}: every stream event validates as a ResponseStreamEvent of its name", not bad, bad)
    check(f"responses {name}: sequence_number 0, 1, 2, ...", seqs == list(range(len(seqs))), seqs)
    got, final = [], None
    with oc.responses.stream(model=name, input="hi") as s:
        for ev in s:
            got.append(ev.type)
        try:
            final = s.get_final_response()
        except RuntimeError as e:
            final = e
    if status == "completed":
        same = not isinstance(final, Exception) and plain(final.output) == plain(r.output)
        check(f"responses {name}: responses.stream assembles the same response", same, final)
    else:
        check(f"responses {name}: the stream ends with response.incomplete", got[-1] == "response.incomplete", got[-3:])
    if name == "reasoning-text":
        rs = r.output[0]
        check("responses: the reasoning item carries reasoning_text and encrypted_content",
              rs.content[0].type == "reasoning_text" and rs.content[0].text == "Think." and rs.encrypted_content.startswith("kapitsa.r1."), rs)
    if name == "custom-tool":
        c = r.output[1]
        check("responses: a custom tool call carries its input", c.input == "*** Begin Patch\n*** End Patch" and c.name == "apply_patch", c)
    if name == "reasoning-tools":
        check("responses: function calls carry their arguments", [json.loads(o.arguments) for o in r.output[1:]] ==
              [{"file_path": "/a"}, {"file_path": "/b", "limit": 5}], r.output)

try:
    oc.responses.create(model="too-long", input="hi")
    check("responses: a too-long prompt raises BadRequestError", False, "no error")
except openai.BadRequestError as e:
    check("responses: a too-long prompt raises BadRequestError with code context_length_exceeded",
          e.status_code == 400 and e.code == "context_length_exceeded", e)
types = []
with oc.responses.stream(model="error", input="hi") as s:
    for ev in s:
        types.append(ev.type)
check("responses: an error after the start is a response.failed event", types[-1] == "response.failed", types)

# ---------------------------------------------------------------- OpenAI Chat Completions (streams, a keepalive chunk
# after the role chunk and after every delta; a prompt over the context is a plain 400)

CHUNK = openai.types.chat.ChatCompletionChunk
chat_want = {"text": ("stop", "", "Hello", 0), "reasoning-text": ("stop", "Think.", "Hi", 0),
             "reasoning-tools": ("tool_calls", "Two reads.", "", 2), "custom-tool": ("tool_calls", "Patch.", "", 1),
             "length": ("length", "long", "cut", 0), "context-full": ("length", "", "cut", 0),
             "stop-sequence": ("stop", "", "abc", 0)}
for name, (finish, think, text, ncalls) in chat_want.items():
    events, bad = raw_events(os.path.join(CASES, f"chat-{name}.sse")), []
    for _, data in events:
        try:
            CHUNK.model_validate(data)
        except Exception as e:
            bad.append(str(e)[:300])
    check(f"chat {name}: every chunk, keepalives included, validates as a ChatCompletionChunk", not bad, bad)
    chunks = list(oc.chat.completions.create(model=name, messages=MSG, stream=True))
    content = "".join(c.choices[0].delta.content or "" for c in chunks)
    reasoning = "".join((c.choices[0].delta.model_extra or {}).get("reasoning_content") or "" for c in chunks)
    check(f"chat {name}: chat.completions.create(stream=True) yields every chunk, text {text!r}, reasoning {think!r}",
          len(chunks) == len(events) and content == text and reasoning == think and
          chunks[-1].choices[0].finish_reason == finish and chunks[-1].usage.prompt_tokens_details.cached_tokens == 60,
          [c.model_dump(exclude_unset=True) for c in chunks])
    keep = [c for c in chunks if c.choices[0].delta.model_dump(exclude_unset=True) == {} and c.choices[0].finish_reason is None]
    check(f"chat {name}: each keepalive comes out as a chunk with an empty delta and no finish_reason",
          len(keep) == sum(d["choices"][0]["delta"] == {} and d["choices"][0]["finish_reason"] is None for _, d in events) >= 2,
          [c.model_dump(exclude_unset=True) for c in chunks])
    with oc.chat.completions.stream(model=name, messages=MSG) as s:
        for _ in s:
            pass
        try:
            final = s.get_final_completion()
        except openai.LengthFinishReasonError as e:  # the helper's parsing refuses a reply cut by length
            final = e.completion
    msg = final.choices[0].message
    check(f"chat {name}: chat.completions.stream assembles the reply (finish_reason {finish}, {ncalls} tool calls)",
          final.choices[0].finish_reason == finish and (msg.content or "") == text and len(msg.tool_calls or []) == ncalls,
          final)
    if name == "reasoning-tools":
        check("chat: tool calls carry their arguments and ids, indexed 0, 1",
              [json.loads(t.function.arguments) for t in msg.tool_calls] == [{"file_path": "/a"}, {"file_path": "/b", "limit": 5}]
              and [t.id for t in msg.tool_calls] == ["call_A", "call_B"], msg.tool_calls)
try:
    for _ in oc.chat.completions.create(model="error", messages=MSG, stream=True):
        pass
    check("chat: an error event in the stream raises", False, "no error")
except openai.APIError as e:
    check("chat: an error event in the stream raises APIError with its message", "the engine failed" in str(e), e)
try:
    oc.chat.completions.create(model="too-long", messages=MSG, stream=True)
    check("chat: a too-long prompt raises BadRequestError", False, "no error")
except openai.BadRequestError as e:
    check("chat: a too-long prompt, streamed, raises BadRequestError with code context_length_exceeded",
          e.status_code == 400 and e.code == "context_length_exceeded" and "maximum context length" in str(e), e)

server.shutdown()
print("FAILED: %d" % fails if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
