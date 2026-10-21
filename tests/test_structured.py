#!/usr/bin/env python3
"""Structured output (grammar-constrained decoding: src/engine/grammar.h).
usage: test_structured.py --cli BUILD_DIR [MODEL_DIR]   offline: the grammar and mask engine's own checks (build/grammar_test;
                                                        with a model folder, masks against brute force on its tokenizer)
                                                        and how the three APIs read response_format, text.format,
                                                        output_format, tool_choice and strict tools (build/structured_check)
       test_structured.py URL [--api-key KEY] [--quick]  against a running `kapitsa serve`: replies to several prompts and
                                                        schemas, greedy and sampled, each valid against its schema
                                                        (jsonschema); json_object; reasoning on; Responses and Messages;
                                                        tool_choice required / named / any / tool; strict tools"""
import json, os, subprocess, sys, time, urllib.error, urllib.request

ARGS = sys.argv[1:]
def opt(name, flag=False):
    if name in ARGS:
        i = ARGS.index(name)
        if flag:
            del ARGS[i]
            return True
        v = ARGS[i + 1]
        del ARGS[i:i + 2]
        return v
    return False if flag else None
CLI = opt("--cli")
API_KEY = opt("--api-key")
QUICK = opt("--quick", flag=True)
BASE = ARGS[0] if ARGS and not CLI else "http://127.0.0.1:8090"

fails = 0
def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)[:600]))
    if not ok:
        fails += 1

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather and a forecast for a place.",
    "parameters": {"type": "object", "properties": {
        "location": {"type": "string", "description": "City and country"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer", "minimum": 1, "maximum": 7, "description": "Days of forecast"}},
        "required": ["location"], "additionalProperties": False}}}
SHELL = {"type": "function", "function": {
    "name": "run_shell", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"], "additionalProperties": False}


# ---------------------------------------------------------------- offline

def cli_checks(build, model):
    exe = os.path.join(build, "grammar_test")
    args = [exe, "--quick"] + ([model] if model else [])
    p = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=1800)
    lines = p.stdout.splitlines()
    summary = [l for l in lines if l.startswith("mask time") or l.endswith("failed")]
    check("grammar_test: grammars, refusals, masks against brute force" + (" (" + "; ".join(summary) + ")" if summary else ""),
          p.returncode == 0, "\n".join(l for l in lines if l.startswith("FAIL"))[:2000] or p.stderr[-2000:])

    tools = [WEATHER, SHELL]
    schema_rf = {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA, "strict": True}}
    cases = [
        # (name, api, body, expected grammar summary prefix (None: no grammar) or ("error", param))
        ("chat: json_object", "chat", {"response_format": {"type": "json_object"}}, "JSON"),
        ("chat: json_schema, strict", "chat", {"response_format": schema_rf}, "JSON"),
        ("chat: text", "chat", {"response_format": {"type": "text"}}, None),
        ("chat: no format, tools, auto", "chat", {"tools": tools}, None),
        ("chat: json_schema without a schema: any object", "chat",
         {"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}}, "JSON"),
        ("chat: unknown format type", "chat", {"response_format": {"type": "xml"}}, ("error", "response_format.type")),
        ("chat: format not an object", "chat", {"response_format": "json"}, ("error", "response_format")),
        ("chat: json_schema missing", "chat", {"response_format": {"type": "json_schema"}},
         ("error", "response_format.json_schema")),
        ("chat: schema not a schema", "chat", {"response_format": {"type": "json_schema", "json_schema": {"schema": 5}}},
         ("error", "response_format.json_schema.schema")),
        ("chat: strict not a boolean", "chat",
         {"response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA, "strict": "yes"}}},
         ("error", "response_format.json_schema.strict")),
        ("chat: unknown type in the schema", "chat",
         {"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "widget"}}}},
         ("error", "response_format")),
        ("chat: remote $ref", "chat",
         {"response_format": {"type": "json_schema", "json_schema": {"schema": {"$ref": "https://example.com/s.json"}}}},
         ("error", "response_format")),
        ("chat: a schema no value satisfies", "chat",
         {"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "integer", "minimum": 3, "maximum": 1}}}},
         ("error", "response_format")),
        ("chat: tool_choice required", "chat", {"tools": tools, "tool_choice": "required"}, "tool calls (get_weather, run_shell)"),
        ("chat: tool_choice named", "chat",
         {"tools": tools, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}, "tool calls (get_weather)"),
        ("chat: one call (parallel_tool_calls false)", "chat",
         {"tools": tools, "tool_choice": "required", "parallel_tool_calls": False}, "tool calls (get_weather, run_shell; one call)"),
        ("chat: a named tool the request lacks", "chat",
         {"tools": tools, "tool_choice": {"type": "function", "function": {"name": "send_mail"}}}, ("error", "tool_choice")),
        ("chat: required without tools", "chat", {"tool_choice": "required"}, ("error", "tool_choice")),
        ("chat: unknown tool_choice", "chat", {"tools": tools, "tool_choice": "maybe"}, ("error", "tool_choice")),
        ("chat: tool_choice object without a name", "chat", {"tools": tools, "tool_choice": {"type": "function"}},
         ("error", "tool_choice")),
        ("chat: tool_choice none with a format: the format", "chat",
         {"tools": tools, "tool_choice": "none", "response_format": {"type": "json_object"}}, "JSON,"),
        ("chat: a strict tool, auto", "chat",
         {"tools": [dict(WEATHER, function=dict(WEATHER["function"], strict=True)), SHELL]},
         "text or tool calls (get_weather, run_shell free)"),
        ("chat: a format beside tools", "chat", {"tools": tools, "response_format": {"type": "json_object"}},
         "JSON or tool calls"),
        ("chat: a strict tool with a schema it cannot hold", "chat",
         {"tools": [{"type": "function", "function": {"name": "f", "strict": True,
                                                      "parameters": {"type": "object", "properties": {"a": {"$ref": "#/nope"}}}}}]},
         ("error", "tools")),
        ("responses: text.format json_schema", "responses",
         {"input": "hi", "text": {"format": {"type": "json_schema", "name": "person", "schema": SCHEMA, "strict": True}}}, "JSON"),
        ("responses: text.format json_object", "responses", {"input": "hi", "text": {"format": {"type": "json_object"}}}, "JSON"),
        ("responses: text.format text", "responses", {"input": "hi", "text": {"format": {"type": "text"}}}, None),
        ("responses: text.format unknown", "responses", {"input": "hi", "text": {"format": {"type": "yaml"}}},
         ("error", "text.format.type")),
        ("responses: text.format bad schema", "responses",
         {"input": "hi", "text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "widget"}}}},
         ("error", "text.format")),
        ("responses: tool_choice required", "responses",
         {"input": "hi", "tools": [dict(type="function", **WEATHER["function"])], "tool_choice": "required"},
         "tool calls (get_weather)"),
        ("responses: tool_choice a function", "responses",
         {"input": "hi", "tools": [dict(type="function", **WEATHER["function"]), dict(type="function", **SHELL["function"])],
          "tool_choice": {"type": "function", "name": "run_shell"}}, "tool calls (run_shell)"),
        ("responses: allowed_tools required", "responses",
         {"input": "hi", "tools": [dict(type="function", **WEATHER["function"]), dict(type="function", **SHELL["function"])],
          "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "run_shell"}]}},
         "tool calls (run_shell)"),
        ("responses: a built-in tool choice", "responses",
         {"input": "hi", "tools": [dict(type="function", **SHELL["function"])], "tool_choice": {"type": "web_search_preview"}},
         ("error", "tool_choice")),
        ("responses: a strict function", "responses",
         {"input": "hi", "tools": [dict(type="function", strict=True, **WEATHER["function"])]},
         "text or tool calls (get_weather)"),
        ("responses: a function without strict is not strict", "responses",
         {"input": "hi", "tools": [dict(type="function", **WEATHER["function"])]}, None),
        ("responses: a namespace's strict function", "responses",
         {"input": "hi", "tools": [{"type": "namespace", "name": "wx", "tools": [dict(type="function", strict=True,
                                                                                      **WEATHER["function"])]}]},
         "text or tool calls (wx__get_weather)"),
        ("responses: a custom tool named", "responses",
         {"input": "hi", "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply a patch"}],
          "tool_choice": {"type": "custom", "name": "apply_patch"}}, "tool calls (apply_patch)"),
        ("messages: tool_choice any", "messages",
         {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
          "tools": [{"name": "get_weather", "input_schema": WEATHER["function"]["parameters"]}], "tool_choice": {"type": "any"}},
         "tool calls (get_weather)"),
        ("messages: tool_choice tool, one call", "messages",
         {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
          "tools": [{"name": "get_weather", "input_schema": WEATHER["function"]["parameters"]},
                    {"name": "run_shell", "input_schema": SHELL["function"]["parameters"]}],
          "tool_choice": {"type": "tool", "name": "run_shell", "disable_parallel_tool_use": True}},
         "tool calls (run_shell; one call)"),
        ("messages: tool_choice tool without a name", "messages",
         {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
          "tools": [{"name": "get_weather", "input_schema": WEATHER["function"]["parameters"]}], "tool_choice": {"type": "tool"}},
         ("error", "")),
        ("messages: unknown tool_choice", "messages",
         {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "tool_choice": {"type": "sometimes"}}, ("error", "")),
        ("messages: output_format", "messages",
         {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
          "output_format": {"type": "json_schema", "schema": SCHEMA}}, "JSON"),
        ("messages: output_config.format", "messages",
         {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
          "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}, "JSON"),
        ("messages: output_format of another type", "messages",
         {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "output_format": {"type": "text"}}, ("error", "")),
        ("messages: a strict tool", "messages",
         {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
          "tools": [{"name": "get_weather", "strict": True, "input_schema": WEATHER["function"]["parameters"]}]},
         "text or tool calls (get_weather)"),
        ("messages: tool_choice none", "messages",
         {"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
          "tools": [{"name": "get_weather", "input_schema": WEATHER["function"]["parameters"]}], "tool_choice": {"type": "none"}},
         None),
    ]
    feed = "".join(json.dumps({"api": api, "body": dict(body, **({"messages": [{"role": "user", "content": "hi"}]}
                                                                  if api == "chat" else {}))}) + "\n"
                   for _, api, body, _ in cases)
    p = subprocess.run([os.path.join(build, "structured_check")], input=feed, capture_output=True, text=True, timeout=120)
    outs = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    check("structured_check answers every request", p.returncode == 0 and len(outs) == len(cases), p.stderr[-500:])
    for (name, _, _, want), got in zip(cases, outs):
        if isinstance(want, tuple):
            ok = got.get("status") == 400 and (not want[1] or got.get("param") == want[1])
            check(name + ": 400" + (" (" + want[1] + ")" if want[1] else ""), ok, json.dumps(got))
        elif want is None:
            check(name + ": no grammar", got.get("ok") and got.get("grammar") is None, json.dumps(got))
        else:
            check(name + ": " + want, got.get("ok") and str(got.get("grammar") or "").startswith(want), json.dumps(got))


# ---------------------------------------------------------------- against a server

def call(path, body, stream=False, timeout=900):
    """(status, JSON body) of a POST, or for a stream (status, [events' data])"""
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + API_KEY} if API_KEY else {})})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    if not stream:
        return r.status, json.loads(r.read())
    out = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[6:]))
    return r.status, out


# name, schema, prompts (each asks for what the schema holds; the system prompt says JSON)
SCHEMAS = [
    ("person", {"type": "object", "properties": {
        "name": {"type": "string"}, "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string"}, "hobbies": {"type": "array", "items": {"type": "string"}, "maxItems": 5}},
        "required": ["name", "age", "hobbies"], "additionalProperties": False},
     ["Alice Smith is 34, her email is alice@example.com, and she enjoys hiking, chess and baking.",
      "Extract the person: Bob Lee, aged seventy-one, retired sailor who loves fishing.",
      "Make up a fictional character from a fantasy novel."]),
    ("weather", {"type": "object", "properties": {
        "city": {"type": "string"}, "temperature_c": {"type": "number"},
        "conditions": {"enum": ["sunny", "cloudy", "rainy", "snowy", "windy"]},
        "forecast": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
            "type": "object", "properties": {"day": {"type": "string"}, "high_c": {"type": "integer"},
                                             "low_c": {"type": "integer"}},
            "required": ["day", "high_c", "low_c"], "additionalProperties": False}}},
        "required": ["city", "temperature_c", "conditions", "forecast"], "additionalProperties": False},
     ["Give a plausible weather report for Oslo in January.", "Weather in Lisbon this summer weekend?",
      "Report the weather for Tokyo during the rainy season."]),
    ("review", {"type": "object", "properties": {
        "sentiment": {"enum": ["positive", "negative", "neutral"]}, "stars": {"type": "integer", "minimum": 1, "maximum": 5},
        "summary": {"type": "string"}, "pros": {"type": "array", "items": {"type": "string"}},
        "cons": {"type": "array", "items": {"type": "string"}}},
        "required": ["sentiment", "stars", "summary", "pros", "cons"], "additionalProperties": False},
     ["Review: 'The battery lasts forever but the screen scratches easily. Worth it for the price.'",
      "Review: 'Arrived broken, support never answered. Avoid.'",
      "Review: 'It is a kettle. It boils water. Nothing more to say.'"]),
    ("recipe", {"type": "object", "properties": {
        "title": {"type": "string"}, "servings": {"type": "integer", "minimum": 1, "maximum": 20},
        "vegetarian": {"type": "boolean"},
        "ingredients": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
            "item": {"type": "string"}, "quantity": {"type": "number", "minimum": 0},
            "unit": {"enum": ["g", "kg", "ml", "l", "cup", "tbsp", "tsp", "piece"]}},
            "required": ["item", "quantity", "unit"], "additionalProperties": False}},
        "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
        "required": ["title", "servings", "vegetarian", "ingredients", "steps"], "additionalProperties": False},
     ["A simple pancake recipe.", "Recipe for a tomato soup for four people.", "How do I make guacamole?"]),
    ("result", {"type": "object", "properties": {"result": {"anyOf": [
        {"type": "object", "properties": {"ok": {"const": True}, "value": {"type": "integer"}},
         "required": ["ok", "value"], "additionalProperties": False},
        {"type": "object", "properties": {"ok": {"const": False}, "error": {"type": "string"}},
         "required": ["ok", "error"], "additionalProperties": False}]}},
        "required": ["result"], "additionalProperties": False},
     ["Compute 17 * 23.", "Divide 10 by zero.", "What is the square root of -4 as an integer?"]),
    ("tree", {"$defs": {"node": {"type": "object", "properties": {
        "name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#/$defs/node"}, "maxItems": 3}},
        "required": ["name", "children"], "additionalProperties": False}}, "$ref": "#/$defs/node"},
     ["The file tree of a small Python project.", "A family tree with grandparents, parents and one child.",
      "The org chart of a startup with a CEO, a CTO and two engineers."]),
]
SYSTEM = "You answer in JSON."


def server_checks():
    try:
        import jsonschema
    except ImportError:
        print("FAIL jsonschema is needed: pip install jsonschema")
        sys.exit(1)
    st, models = 0, None
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
            st, models = r.status, json.loads(r.read())
    except Exception as e:
        print("FAIL the server at " + BASE + " answers: " + str(e))
        sys.exit(1)
    model = models["data"][0]["id"]
    print("model", model)
    total = valid = 0
    t0 = time.time()
    modes = [("greedy", {"temperature": 0})] + [("T=0.7 seed %d" % s, {"temperature": 0.7, "seed": s})
                                                for s in ([7] if QUICK else [7, 11])]
    for name, schema, prompts in SCHEMAS:
        v = jsonschema.Draft202012Validator(schema)
        for k, prompt in enumerate(prompts[:2] if QUICK else prompts):
            for mode, sampling in modes:
                body = dict({"model": model, "max_tokens": 768,
                             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                             "chat_template_kwargs": {"enable_thinking": False},
                             "response_format": {"type": "json_schema",
                                                 "json_schema": {"name": name, "schema": schema, "strict": True}}},
                            **sampling)
                st, r = call("/v1/chat/completions", body)
                total += 1
                ok, why = False, ""
                if st != 200:
                    why = "status %d %s" % (st, json.dumps(r)[:300])
                else:
                    c = r["choices"][0]
                    text = c["message"].get("content") or ""
                    try:
                        doc = json.loads(text)
                        errs = sorted(v.iter_errors(doc), key=str)
                        ok = not errs and c["finish_reason"] == "stop"
                        why = "; ".join(e.message for e in errs[:3]) or c["finish_reason"]
                    except ValueError as e:
                        why = "not JSON (%s, finish %s): %r" % (e, c["finish_reason"], text[:300])
                valid += ok
                check("%s prompt %d %s: valid" % (name, k, mode), ok, why)
    check("schema-valid replies: %d of %d (%.0f s)" % (valid, total, time.time() - t0), valid == total)

    # greedy is repeatable, and a stream gives the same reply
    body = {"model": model, "max_tokens": 512, "temperature": 0,
            "messages": [{"role": "user", "content": SCHEMAS[0][2][0]}], "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMAS[0][1]}}}
    st1, r1 = call("/v1/chat/completions", body)
    st2, ev = call("/v1/chat/completions", dict(body, stream=True), stream=True)
    streamed = "".join((e.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for e in ev)
    check("greedy: the streamed reply equals the reply", st1 == 200 and st2 == 200 and
          streamed == r1["choices"][0]["message"]["content"], repr(streamed[:200]) + " vs " + repr(json.dumps(r1)[:300]))

    # any JSON object
    for k, prompt in enumerate(["List three primary colors.", "Describe a cat.", "What is 2+2?"]):
        st, r = call("/v1/chat/completions", {"model": model, "max_tokens": 512, "temperature": 0.7, "seed": k,
                                              "messages": [{"role": "user", "content": prompt + " Answer in JSON."}],
                                              "chat_template_kwargs": {"enable_thinking": False},
                                              "response_format": {"type": "json_object"}})
        try:
            ok = st == 200 and isinstance(json.loads(r["choices"][0]["message"]["content"]), dict)
        except (ValueError, KeyError):
            ok = False
        check("json_object %d: a JSON object" % k, ok, json.dumps(r)[:400])

    # reasoning on: free reasoning, then the JSON (a reasoning that runs to max_tokens never reaches the format: the
    # model's, reported, not counted)
    fact = {"type": "object", "properties": {"prime": {"type": "boolean"}, "factors": {"type": "array", "items": {"type": "integer"}}},
            "required": ["prime", "factors"], "additionalProperties": False}
    st, r = call("/v1/chat/completions", {
        "model": model, "max_tokens": 4096, "temperature": 0,
        "messages": [{"role": "user", "content": "Is 391 prime? Answer with the factorization."}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "fact", "schema": fact}}})
    try:
        c = r["choices"][0]
        m = c["message"]
        if c["finish_reason"] == "length" and not m.get("content"):
            print("SKIP reasoning on: the reasoning ran to max_tokens (%d tokens), so no format followed" %
                  r["usage"]["completion_tokens"])
            ok = True
        else:
            doc = json.loads(m["content"])
            ok = st == 200 and m.get("reasoning_content") and not list(jsonschema.Draft202012Validator(fact).iter_errors(doc))
            if ok and not (doc["prime"] is False and sorted(doc["factors"]) == [17, 23]):
                print("NOTE reasoning on: valid JSON with a wrong answer (the model's): " + m["content"][:100])
    except (ValueError, KeyError, TypeError):
        ok = False
    check("reasoning on: reasoning, then valid JSON", bool(ok), json.dumps(r)[:600])

    # Responses: text.format
    st, r = call("/v1/responses", {"model": model, "input": SCHEMAS[2][2][0], "temperature": 0,
                                   "reasoning": {"effort": "none"}, "max_output_tokens": 512,
                                   "text": {"format": {"type": "json_schema", "name": "review", "schema": SCHEMAS[2][1],
                                                       "strict": True}}})
    try:
        text = "".join(c.get("text", "") for it in r["output"] if it["type"] == "message" for c in it["content"])
        ok = st == 200 and not list(jsonschema.Draft202012Validator(SCHEMAS[2][1]).iter_errors(json.loads(text)))
    except (ValueError, KeyError):
        ok = False
    check("responses: text.format json_schema", ok, json.dumps(r)[:500])

    # tool_choice required / named; arguments valid against the tool's schema
    wv = jsonschema.Draft202012Validator(WEATHER["function"]["parameters"])
    for k, (prompt, choice) in enumerate([("Hello! How are you today?", "required"),
                                          ("What's 2+2?", {"type": "function", "function": {"name": "get_weather"}}),
                                          ("I need the forecast for Paris for the next 3 days.", "required")]):
        st, r = call("/v1/chat/completions", {"model": model, "max_tokens": 512, "temperature": 0.6, "seed": k,
                                              "messages": [{"role": "user", "content": prompt}], "tools": [WEATHER, SHELL],
                                              "tool_choice": choice, "parallel_tool_calls": False,
                                              "chat_template_kwargs": {"enable_thinking": False}})
        ok, why = False, json.dumps(r)[:500]
        try:
            c = r["choices"][0]
            calls = c["message"].get("tool_calls") or []
            ok = st == 200 and c["finish_reason"] == "tool_calls" and len(calls) == 1
            if ok and calls[0]["function"]["name"] == "get_weather":
                args = json.loads(calls[0]["function"]["arguments"])
                errs = list(wv.iter_errors(args))
                ok, why = not errs, "; ".join(e.message for e in errs)
            elif ok:
                ok = choice == "required" and calls[0]["function"]["name"] == "run_shell" and \
                    isinstance(json.loads(calls[0]["function"]["arguments"]).get("command"), str)
        except (KeyError, ValueError, IndexError):
            ok = False
        check("chat tool_choice %s (%s): one valid call" % (json.dumps(choice), prompt), ok, why)

    # reasoning on + tool_choice required: reasoning, then the call
    st, r = call("/v1/chat/completions", {"model": model, "max_tokens": 4096, "temperature": 0,
                                          "messages": [{"role": "user", "content": "Weather in Rome?"}],
                                          "tools": [WEATHER], "tool_choice": "required"})
    try:
        c = r["choices"][0]
        ok = st == 200 and c["finish_reason"] == "tool_calls" and \
            not list(wv.iter_errors(json.loads(c["message"]["tool_calls"][0]["function"]["arguments"])))
    except (KeyError, ValueError, IndexError):
        ok = False
    check("reasoning on + tool_choice required: a valid call", ok, json.dumps(r)[:500])

    # Messages: tool_choice any and tool
    for choice in [{"type": "any"}, {"type": "tool", "name": "get_weather"}]:
        st, r = call("/v1/messages", {"model": model, "max_tokens": 512, "temperature": 0,
                                      "messages": [{"role": "user", "content": "Tell me a joke."}],
                                      "tools": [{"name": "get_weather", "description": "Weather for a place",
                                                 "input_schema": WEATHER["function"]["parameters"]}],
                                      "tool_choice": choice})
        try:
            uses = [b for b in r["content"] if b["type"] == "tool_use"]
            ok = st == 200 and r["stop_reason"] == "tool_use" and len(uses) >= 1 and \
                all(u["name"] == "get_weather" and not list(wv.iter_errors(u["input"])) for u in uses)
        except (KeyError, TypeError):
            ok = False
        check("messages tool_choice %s: valid tool_use" % json.dumps(choice), ok, json.dumps(r)[:500])

    # Responses: tool_choice required
    st, r = call("/v1/responses", {"model": model, "input": "Say hi.", "temperature": 0, "reasoning": {"effort": "none"},
                                   "tools": [dict(type="function", **WEATHER["function"])], "tool_choice": "required"})
    try:
        fcs = [it for it in r["output"] if it["type"] == "function_call"]
        ok = st == 200 and fcs and not list(wv.iter_errors(json.loads(fcs[0]["arguments"])))
    except (KeyError, ValueError, TypeError):
        ok = False
    check("responses tool_choice required: a valid function_call", bool(ok), json.dumps(r)[:500])

    # a strict tool under tool_choice auto: a call it makes is valid
    strict = dict(WEATHER, function=dict(WEATHER["function"], strict=True))
    st, r = call("/v1/chat/completions", {"model": model, "max_tokens": 512, "temperature": 0,
                                          "messages": [{"role": "user", "content": "What's the weather in Berlin, in fahrenheit?"}],
                                          "tools": [strict], "chat_template_kwargs": {"enable_thinking": False}})
    try:
        c = r["choices"][0]
        calls = c["message"].get("tool_calls") or []
        ok = st == 200 and (not calls or not list(wv.iter_errors(json.loads(calls[0]["function"]["arguments"]))))
    except (KeyError, ValueError):
        ok = False
    check("strict tool, tool_choice auto: a call it makes is valid", ok, json.dumps(r)[:500])

    # refusals reach the client as 400s
    st, r = call("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "x"}],
                                          "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "widget"}}}})
    check("a schema the server cannot hold: 400 with the field", st == 400 and r.get("error", {}).get("param") == "response_format",
          json.dumps(r))

    # requests at once: two under a schema, two free (batched where the server batches: NI_BATCH=1 on the 4B)
    import threading
    out = {}
    def one(k, body):
        out[k] = call("/v1/chat/completions", body)
    bodies = []
    for k in range(4):
        b = {"model": model, "max_tokens": 400, "temperature": 0 if k % 2 == 0 else 0.7, "seed": k,
             "messages": [{"role": "user", "content": SCHEMAS[0][2][k % 3] if k < 2 else "Write a short poem about the sea."}],
             "chat_template_kwargs": {"enable_thinking": False}}
        if k < 2:
            b["response_format"] = {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMAS[0][1]}}
        bodies.append(b)
    th = [threading.Thread(target=one, args=(k, b)) for k, b in enumerate(bodies)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    pv = jsonschema.Draft202012Validator(SCHEMAS[0][1])
    for k in range(2):
        st, r = out[k]
        try:
            ok = st == 200 and not list(pv.iter_errors(json.loads(r["choices"][0]["message"]["content"])))
        except (ValueError, KeyError):
            ok = False
        check("at once with free requests: request %d under the schema is valid" % k, ok, json.dumps(r)[:400])
    check("at once: the free requests answer", all(out[k][0] == 200 and out[k][1]["choices"][0]["message"]["content"]
                                                     for k in (2, 3)), json.dumps(out.get(2, ""))[:300])


if CLI:
    cli_checks(CLI, ARGS[0] if ARGS else None)
else:
    server_checks()
print("FAILED: %d" % fails if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
