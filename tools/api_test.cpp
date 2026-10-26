// The Anthropic Messages and OpenAI Responses endpoints' translation (src/api/), CPU only, no model loaded:
//  - request mapping: multi-turn conversations with tools, thinking / reasoning round trips, system merging, images;
//  - the exact event sequences of streamed replies (text, reasoning + text, reasoning + 2 tool calls, a custom tool
//    call, a length cutoff, a stop sequence, errors) and the non-streamed responses; the Chat Completions stream's
//    bytes and its keepalive chunk;
//  - what `kapitsa claude|codex|opencode|pi` gives each client;
//  - with a model folder (its chat template and tokenizer): tool loops through the template, the generated turns parsed
//    as the engine parses them, streamed, read back as each client reads them and sent back: the next request's prompt
//    must continue the previous prompt and the generated tokens exactly (the engine then reuses its cache).
//  - with a model folder: what a draft read ahead (POST /v1/draft) reads, at every point of typing a message, is a
//    prefix of the request's prompt once the message is sent;
//   api_test [--dump DIR] [model_dir ...]   (default: the catalog's 35B and 27B folders in ~/.cache/kapitsa/models that
//   exist; --dump writes each scenario's replies for tests/sdk_check.py)
//   exit status 1 if any check fails
#include "api/anthropic.h"
#include "api/chat_completions.h"
#include "api/clients.h"
#include "api/responses.h"
#include "bench/bench.h"
#include "engine/output_parser.h"
#include "engine/disk_wait.h"
#include "engine/disk_saved.h"
#include "engine/engine.h"
#include "engine/history_gate.h"
#include "engine/reasoning_memory.h"
#include "engine/tool_args.h"
#include "hub/hub.h"
#include "model/kflash_wrap.h"
#include "tokenizer/tokenizer.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <stdlib.h>
#include <sys/stat.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace ni;
using namespace ni::api;

namespace {

int fails = 0, passes = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok || detail.empty() ? "" : ("\n     " + detail).c_str());
  ok ? ++passes : ++fails;
}

ojson J(const char* s) { return ojson::parse(s); }

struct Event {
  std::string name;
  ojson data;
};

// SSE text as events (comments skipped)
std::vector<Event> parse_sse(const std::string& s) {
  std::vector<Event> out;
  size_t p = 0;
  while (p < s.size()) {
    size_t e = s.find("\n\n", p);
    if (e == std::string::npos) e = s.size();
    const std::string frame = s.substr(p, e - p);
    p = e + 2;
    Event ev;
    bool any = false;
    size_t q = 0;
    while (q < frame.size()) {
      size_t nl = frame.find('\n', q);
      if (nl == std::string::npos) nl = frame.size();
      const std::string line = frame.substr(q, nl - q);
      q = nl + 1;
      if (line.rfind("event: ", 0) == 0) ev.name = line.substr(7), any = true;
      else if (line.rfind("data: ", 0) == 0) ev.data = ojson::parse(line.substr(6)), any = true;
    }
    if (any) out.push_back(ev);
  }
  return out;
}

std::string names(const std::vector<Event>& v) {
  std::string s;
  for (auto& e : v) s += (s.empty() ? "" : " ") + e.name;
  return s;
}

ChatResult result(const std::string& finish, int prompt = 100, int cached = 60, int completion = 20) {
  ChatResult r;
  r.finish_reason = finish;
  r.prompt_tokens = prompt;
  r.cached_tokens = cached;
  r.completion_tokens = completion;
  return r;
}

ChatDelta reasoning(const std::string& s) {
  ChatDelta d;
  d.reasoning = s;
  return d;
}
ChatDelta content(const std::string& s) {
  ChatDelta d;
  d.content = s;
  return d;
}
ChatDelta call(const std::string& id, const std::string& name, const std::string& args) {
  ChatDelta d;
  d.tool_calls.push_back({id, name, args});
  return d;
}

// Runs a writer over deltas and returns its events
std::vector<Event> run(StreamWriter& w, const std::vector<ChatDelta>& ds, const ChatResult& res) {
  std::string s = w.begin(res);
  for (auto& d : ds) s += w.delta(d);
  s += w.end(res);
  return parse_sse(s);
}

// ---------------------------------------------------------------- request mapping

void anthropic_mapping() {
  const ojson body = J(R"({
    "model": "m", "max_tokens": 1000, "stream": true, "temperature": 0.2, "top_k": 10, "stop_sequences": ["STOP"],
    "metadata": {"user_id": "u"},
    "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=12ab;"},
               {"type": "text", "text": "You are helpful."},
               {"type": "text", "text": "\nBe brief.", "cache_control": {"type": "ephemeral"}}],
    "tools": [{"name": "Read", "description": "Read a file",
               "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}},
              {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
    "thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"},
    "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                                   {"type": "text", "text": "Read /a", "cache_control": {"type": "ephemeral"}}]},
      {"role": "system", "content": [{"type": "text", "text": "SessionStart hook context"}]},
      {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "", "signature": "SIG"},
        {"type": "text", "text": "Reading."},
        {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/a"}}]},
      {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "42"}]},
        {"type": "text", "text": "<system-reminder>r</system-reminder>"}]},
      {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "Done.", "signature": "a-foreign-signature"},
        {"type": "redacted_thinking", "data": "xx"},
        {"type": "text", "text": "It is 42."}]},
      {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_2", "is_error": true, "content": "no such file"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": "What is this?"},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
        {"type": "document", "title": "spec", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBE"}},
        {"type": "document", "title": "notes", "source": {"type": "text", "media_type": "text/plain", "data": "plain notes"}}]}]})");
  ojson b = body;
  b["messages"][2]["content"][0]["signature"] = pack_reasoning("I should read it.");
  const MessagesRequest q = messages_request(b);
  const ojson want = J(R"([
    {"role": "system", "content": "You are helpful.\nBe brief."},
    {"role": "user", "content": "<system-reminder>ctx</system-reminder>\nRead /a"},
    {"role": "system", "content": "SessionStart hook context"},
    {"role": "assistant", "content": "Reading.", "reasoning_content": "I should read it.",
     "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "Read", "arguments": {"file_path": "/a"}}}]},
    {"role": "tool", "tool_call_id": "toolu_1", "content": "42"},
    {"role": "user", "content": "<system-reminder>r</system-reminder>"},
    {"role": "assistant", "content": "It is 42.", "reasoning_content": "Done."},
    {"role": "tool", "tool_call_id": "toolu_2", "content": "Error: no such file"},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
      {"type": "text", "text": "What is this?"},
      {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
      {"type": "text", "text": "[document omitted: spec: this server can't read PDF documents yet]\nnotes\n\nplain notes"}]}])");
  check(q.chat.messages == want, "anthropic: multi-turn mapping (billing header dropped, system first, mid-conversation "
        "system in place, thinking from our signature, tool results before the turn's text, images as image_url parts)",
        q.chat.messages.dump(1));
  check(q.chat.tools == J(R"([{"type": "function", "function": {"name": "Read", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}}])"),
        "anthropic: tools as functions, the server tool left out", q.chat.tools.dump());
  check(q.chat.template_kwargs == J(R"({"enable_thinking": true, "preserve_thinking": true})"),
        "anthropic: adaptive thinking on, effort high = the template's default, keep:all = preserve_thinking",
        q.chat.template_kwargs.dump());
  check(q.stream && !q.show_thinking && q.chat.max_tokens == 1000 && q.chat.stop == std::vector<std::string>{"STOP"} &&
            q.chat.temperature == 0.2f && q.chat.top_k == 10,
        "anthropic: stream, display omitted, max_tokens, stop_sequences, sampling");

  auto kw = [](const char* thinking) {
    ojson b = J(R"({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})");
    if (thinking) b["thinking"] = J(thinking);
    return messages_request(b).chat.template_kwargs;
  };
  check(kw(nullptr) == J(R"({"enable_thinking": false})"), "anthropic: no thinking parameter = thinking off", kw(nullptr).dump());
  check(kw(R"({"type": "disabled"})")["enable_thinking"] == false, "anthropic: thinking disabled");
  check(kw(R"({"type": "enabled", "budget_tokens": 2048})")["enable_thinking"] == true, "anthropic: thinking enabled");
  ojson lo = J(R"({"model": "m", "max_tokens": 10, "thinking": {"type": "enabled", "budget_tokens": 1024},
                  "output_config": {"effort": "low"}, "messages": [{"role": "user", "content": "hi"}]})");
  check(messages_request(lo).chat.template_kwargs == J(R"({"enable_thinking": true, "reasoning_effort": "low"})"),
        "anthropic: effort low passes to the template");
  ojson none = J(R"({"model": "m", "max_tokens": 10, "tool_choice": {"type": "none"},
                    "tools": [{"name": "t", "input_schema": {"type": "object"}}], "messages": [{"role": "user", "content": "hi"}]})");
  check(messages_request(none).chat.tools.empty(), "anthropic: tool_choice none = no tools");
  auto refused = [](const char* body, int status, const std::string& has) {
    try {
      messages_request(J(body));
    } catch (const ApiError& e) {
      return e.status == status && std::string(e.what()).find(has) != std::string::npos;
    }
    return false;
  };
  check(refused(R"({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "Sure,"}]})", 400, "prefill"),
        "anthropic: a final assistant message (prefill) is refused with 400");
  check(refused(R"({"model": "m", "max_tokens": 10})", 400, "messages"), "anthropic: no messages is a 400");
  check(refused(R"({"model": "m", "max_tokens": 0, "messages": [{"role": "user", "content": "hi"}]})", 400, "max_tokens"),
        "anthropic: max_tokens 0 is a 400");
  // count_tokens takes the prompt only (a final assistant message is fine there)
  const ChatRequest c = messages_prompt(J(R"({"model": "m", "system": "S", "messages": [{"role": "user", "content": "hi"},
                                              {"role": "assistant", "content": "ok"}]})"));
  check(c.messages.size() == 3 && c.messages[0]["content"] == "S", "anthropic: count_tokens prompt");
  // mid-conversation tool changes
  const ojson tc = J(R"({"model": "m", "max_tokens": 10, "tools": [{"name": "a", "input_schema": {"type": "object"}},
      {"name": "b", "input_schema": {"type": "object"}}],
    "messages": [{"role": "user", "content": "hi"},
      {"role": "system", "content": [{"type": "tool_removal", "tool": {"type": "tool_reference", "name": "a"}},
        {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": {"name": "c", "input_schema": {"type": "object"}}}},
        {"type": "text", "text": "Tools changed."}]}]})");
  const MessagesRequest tq = messages_request(tc);
  check(tq.chat.tools.size() == 2 && tq.chat.tools[0]["function"]["name"] == "b" && tq.chat.tools[1]["function"]["name"] == "c" &&
            tq.chat.messages.back() == J(R"({"role": "system", "content": "Tools changed."})"),
        "anthropic: tool_addition / tool_removal blocks change the tool list", tq.chat.tools.dump());
}

void responses_mapping() {
  ResponseStore store;
  ojson body = J(R"({
    "model": "m", "instructions": "Base instructions.", "stream": true, "store": false,
    "reasoning": {"effort": "medium", "summary": "auto"}, "include": ["reasoning.encrypted_content"],
    "prompt_cache_key": "k", "parallel_tool_calls": true, "tool_choice": "auto",
    "tools": [
      {"type": "function", "name": "exec_command", "description": "Run", "strict": false,
       "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
      {"type": "custom", "name": "apply_patch", "description": "Patch files",
       "format": {"type": "grammar", "syntax": "lark", "definition": "start: patch"}},
      {"type": "namespace", "name": "multi_agent_v1", "description": "Agents",
       "tools": [{"type": "function", "name": "spawn_agent", "parameters": {"type": "object", "properties": {}}}]},
      {"type": "web_search"}],
    "input": [
      {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Dev rules."}]},
      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<env/>"}]},
      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix it"},
                                                       {"type": "input_image", "image_url": "https://x/y.png"},
                                                       {"type": "input_image", "file_id": "file_9"}]},
      {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "plan"}],
       "encrypted_content": "ENC"},
      {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
      {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}"},
      {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
      {"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary only"}]},
      {"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch", "input": "*** Begin Patch\n*** End Patch"},
      {"type": "custom_tool_call_output", "call_id": "c2", "output": "Done"},
      {"type": "function_call", "call_id": "c3", "name": "spawn_agent", "namespace": "multi_agent_v1", "arguments": "{}"},
      {"type": "function_call_output", "call_id": "c3", "output": [{"type": "input_text", "text": "agent 7"}]},
      {"type": "message", "role": "developer", "content": "Mode changed."},
      {"role": "user", "content": "next"}]})");
  body["input"][3]["encrypted_content"] = pack_reasoning("plan, exactly");
  const ResponsesRequest q = responses_request(body, store);
  const ojson want = J(R"([
    {"role": "system", "content": "Base instructions.\n\nDev rules."},
    {"role": "user", "content": "<env/>"},
    {"role": "user", "content": [{"type": "text", "text": "Fix it"},
                                 {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                                 {"type": "image_url", "image_url": {"url": "file_id:file_9"}}]},
    {"role": "assistant", "content": "Looking.", "reasoning_content": "plan, exactly",
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "exec_command", "arguments": {"cmd": "ls"}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    {"role": "assistant", "content": "", "reasoning_content": "summary only",
     "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "apply_patch",
                     "arguments": {"input": "*** Begin Patch\n*** End Patch"}}}]},
    {"role": "tool", "tool_call_id": "c2", "content": "Done"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "multi_agent_v1__spawn_agent", "arguments": {}}}]},
    {"role": "tool", "tool_call_id": "c3", "content": "agent 7"},
    {"role": "system", "content": "Mode changed."},
    {"role": "user", "content": "next"}])");
  check(q.chat.messages == want, "responses: mapping (instructions + leading developer merged, reasoning from our "
        "encrypted_content else its summary, calls on the assistant turn, custom and namespaced calls, outputs as tool "
        "messages, a later developer message in place, images as image_url parts)", q.chat.messages.dump(1));
  std::string names_;
  for (auto& t : q.chat.tools) names_ += t["function"]["name"].get<std::string>() + " ";
  check(names_ == "exec_command apply_patch multi_agent_v1__spawn_agent ", "responses: function, custom and namespaced tools; built-in tools left out", names_);
  check(q.chat.tools[1]["function"]["parameters"]["properties"]["input"]["type"] == "string" &&
            q.chat.tools[1]["function"]["parameters"]["properties"]["input"]["description"].get<std::string>().find("start: patch") != std::string::npos,
        "responses: a custom tool is a function of one string, input, with its grammar", q.chat.tools[1].dump());
  check(q.custom.count("apply_patch") && q.namespaced.at("multi_agent_v1__spawn_agent") == std::make_pair(std::string("multi_agent_v1"), std::string("spawn_agent")),
        "responses: custom and namespaced tools remembered for the output");
  check(q.chat.template_kwargs == J(R"({"reasoning_effort": "medium"})") && q.stream && !q.store && q.chat.max_tokens == 32768,
        "responses: effort medium, stream, store false, default max_output_tokens", q.chat.template_kwargs.dump());
  auto effort = [&](const char* e) {
    ojson b = J(R"({"model": "m", "input": "hi"})");
    b["reasoning"] = {{"effort", e}};
    return responses_request(b, store).chat.template_kwargs;
  };
  check(effort("none") == J(R"({"enable_thinking": false})") && effort("minimal") == J(R"({"enable_thinking": false})") &&
            effort("high") == ojson::object() && effort("low") == J(R"({"reasoning_effort": "low"})"),
        "responses: effort none/minimal = thinking off, high = the template's default, low passes");
  ojson none = J(R"({"model": "m", "input": "hi", "tool_choice": "none", "tools": [{"type": "function", "name": "f"}]})");
  check(responses_request(none, store).chat.tools.empty(), "responses: tool_choice none = no tools");
  // previous_response_id: the stored conversation, then the new input (instructions are not inherited)
  store.put("resp_A", J(R"([{"type": "message", "role": "user", "content": "first"},
                              {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "one"}]}])"));
  const ResponsesRequest p = responses_request(J(R"({"model": "m", "previous_response_id": "resp_A", "input": "second"})"), store);
  check(p.chat.messages == J(R"([{"role": "user", "content": "first"}, {"role": "assistant", "content": "one"},
                                 {"role": "user", "content": "second"}])") && p.echo["previous_response_id"] == "resp_A",
        "responses: previous_response_id continues the stored conversation", p.chat.messages.dump());
  bool unknown = false;
  try {
    responses_request(J(R"({"model": "m", "previous_response_id": "resp_nope", "input": "x"})"), store);
  } catch (const ApiError& e) {
    unknown = e.status == 400 && e.code == "previous_response_not_found" && e.param == "previous_response_id";
  }
  check(unknown, "responses: an unknown previous_response_id is a 400");
  ResponseStore small(2);
  small.put("a", ojson::array());
  small.put("b", ojson::array());
  small.put("c", ojson::array());
  ojson tmp;
  check(!small.get("a", &tmp) && small.get("b", &tmp) && small.get("c", &tmp), "responses: the store keeps the newest");
}

// ---------------------------------------------------------------- stream events

void anthropic_events() {
  const std::string sig_text = "Think.";
  {
    MessagesStream w("msg_1", "m", true);
    auto ev = run(w, {content("Hel"), content("lo")}, result("stop"));
    check(names(ev) == "message_start ping content_block_start content_block_delta content_block_delta content_block_stop "
                       "message_delta message_stop",
          "anthropic stream: text only", names(ev));
    const ojson& st = ev[0].data["message"];
    check(st["id"] == "msg_1" && st["content"].empty() && st["stop_reason"].is_null() &&
              st["usage"] == J(R"({"input_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 60, "output_tokens": 0})"),
          "anthropic stream: message_start carries the prompt's input and cached tokens", st.dump());
    check(ev[2].data == J(R"({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})") &&
              ev[3].data["delta"] == J(R"({"type": "text_delta", "text": "Hel"})"),
          "anthropic stream: text block", ev[2].data.dump());
    check(ev[6].data["delta"] == J(R"({"stop_reason": "end_turn", "stop_sequence": null})") &&
              ev[6].data["usage"]["output_tokens"] == 20 && ev[6].data["usage"]["cache_read_input_tokens"] == 60,
          "anthropic stream: message_delta end_turn with usage", ev[6].data.dump());
  }
  {
    MessagesStream w("msg_2", "m", true);
    auto ev = run(w, {reasoning("Th"), reasoning("ink."), content("Hi")}, result("stop"));
    check(names(ev) == "message_start ping content_block_start content_block_delta content_block_delta content_block_delta "
                       "content_block_stop content_block_start content_block_delta content_block_stop message_delta message_stop",
          "anthropic stream: reasoning + text", names(ev));
    std::string text;
    check(ev[2].data["content_block"] == J(R"({"type": "thinking", "thinking": "", "signature": ""})") &&
              ev[3].data["delta"] == J(R"({"type": "thinking_delta", "thinking": "Th"})") &&
              ev[5].data["delta"]["type"] == "signature_delta" &&
              unpack_reasoning(ev[5].data["delta"]["signature"], &text) && text == sig_text && ev[7].data["index"] == 1,
          "anthropic stream: thinking block ends with a signature_delta that carries its text", ev[5].data.dump());
  }
  {
    MessagesStream w("msg_3", "m", true);
    auto ev = run(w, {reasoning("Two reads."), call("call_A", "Read", R"({"file_path":"/a"})"), call("call_B", "Read", R"({"file_path":"/b"})")},
                  result("tool_calls"));
    check(names(ev) == "message_start ping content_block_start content_block_delta content_block_delta content_block_stop "
                       "content_block_start content_block_delta content_block_stop content_block_start content_block_delta "
                       "content_block_stop message_delta message_stop",
          "anthropic stream: reasoning + 2 tool calls", names(ev));
    check(ev[6].data["content_block"] == J(R"({"type": "tool_use", "id": "toolu_A", "name": "Read", "input": {}})") &&
              ev[7].data["delta"] == J(R"({"type": "input_json_delta", "partial_json": "{\"file_path\":\"/a\"}"})") &&
              ev[9].data["index"] == 2 && ev[12].data["delta"]["stop_reason"] == "tool_use",
          "anthropic stream: each call one block with one input_json_delta; stop_reason tool_use", ev[6].data.dump());
  }
  {
    MessagesStream w("msg_4", "m", false);  // display omitted
    auto ev = run(w, {reasoning("Hidden."), content("Hi")}, result("stop"));
    std::string text;
    check(names(ev) == "message_start ping content_block_start content_block_delta content_block_stop content_block_start "
                       "content_block_delta content_block_stop message_delta message_stop" &&
              unpack_reasoning(ev[3].data["delta"]["signature"], &text) && text == "Hidden.",
          "anthropic stream: display omitted sends the thinking block with its signature only", names(ev));
  }
  {
    ChatResult r = result("length");
    MessagesStream w("m5", "m", true);
    check(run(w, {content("abc")}, r)[5].data["delta"]["stop_reason"] == "max_tokens", "anthropic stream: length = max_tokens");
    r.context_full = true;
    MessagesStream w2("m6", "m", true);
    check(run(w2, {content("abc")}, r)[5].data["delta"]["stop_reason"] == "model_context_window_exceeded",
          "anthropic stream: the context running out = model_context_window_exceeded");
    ChatResult s = result("stop");
    s.stop_sequence = "STOP";
    MessagesStream w3("m7", "m", true);
    check(run(w3, {content("abc")}, s)[5].data["delta"] == J(R"({"stop_reason": "stop_sequence", "stop_sequence": "STOP"})"),
          "anthropic stream: stop sequence");
    ChatResult e = result("error");
    e.error = "boom";
    MessagesStream w4("m8", "m", true);
    auto ev = run(w4, {}, e);
    check(names(ev) == "message_start ping error" && ev[2].data == J(R"({"type": "error", "error": {"type": "api_error", "message": "boom"}})"),
          "anthropic stream: an error after the start is an error event", names(ev));
    MessagesStream w5("m9", "m", true);
    check(parse_sse(w5.keepalive())[0].name == "ping", "anthropic stream: keepalive is a ping");
  }
  {
    ChatResult r = result("error");
    r.prompt_tokens = 140000;
    r.prompt_limit = 131070;
    auto [st, body] = messages_failure(r);
    check(st == 400 && body == J(R"({"type": "error", "error": {"type": "invalid_request_error",
                                    "message": "prompt is too long: 140000 tokens > 131070 maximum"}})"),
          "anthropic: a prompt longer than the context is 400 \"prompt is too long: N tokens > M maximum\"", body.dump());
    ChatResult t = result("error");
    t.error = "chat template: No user query found in messages.";
    check(messages_failure(t).first == 400, "anthropic: a template error is a 400");
    ChatResult o = result("error");
    o.error = "something";
    check(messages_failure(o).first == 500 && messages_failure(o).second["error"]["type"] == "api_error", "anthropic: other errors are 500 api_error");
    ChatResult im = result("error");
    im.error = "image 1: could not decode it";
    im.invalid_request = true;
    check(messages_failure(im).first == 400 && messages_failure(im).second["error"]["type"] == "invalid_request_error",
          "anthropic: an image that cannot be read is a 400 invalid_request_error");
  }
  {
    ChatResult r = result("tool_calls");
    r.reasoning = "R.";
    r.content = "Let me look.";
    r.tool_calls = {{"call_X", "Read", R"({"file_path":"/x"})"}};
    const ojson m = messages_response("msg_9", "m", r, true);
    std::string text;
    check(m["type"] == "message" && m["role"] == "assistant" && m["stop_reason"] == "tool_use" && m["stop_sequence"].is_null() &&
              m["content"].size() == 3 && m["content"][0]["type"] == "thinking" && m["content"][0]["thinking"] == "R." &&
              unpack_reasoning(m["content"][0]["signature"], &text) && text == "R." &&
              m["content"][1] == J(R"({"type": "text", "text": "Let me look."})") &&
              m["content"][2] == J(R"({"type": "tool_use", "id": "toolu_X", "name": "Read", "input": {"file_path": "/x"}})") &&
              m["usage"]["input_tokens"] == 40 && m["usage"]["cache_read_input_tokens"] == 60,
          "anthropic: non-streamed response", m.dump());
    ChatResult only = result("stop");
    only.reasoning = "only thinking";
    const ojson m2 = messages_response("msg_10", "m", only, false);
    check(m2["content"].size() == 2 && m2["content"][0]["thinking"] == "" && m2["content"][1] == J(R"({"type": "text", "text": ""})"),
          "anthropic: a reply of thinking only gets an empty text block; display omitted empties the thinking text", m2.dump());
  }
}

void responses_events() {
  ResponseStore store;
  ResponsesRequest rq = responses_request(J(R"({"model": "m", "input": "hi", "instructions": "I",
      "tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
                {"type": "custom", "name": "apply_patch"},
                {"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "f"}]}]})"), store);
  auto seq_ok = [](const std::vector<Event>& ev) {
    for (size_t i = 0; i < ev.size(); ++i)
      if (ev[i].data.value("sequence_number", -1) != int(i) || ev[i].data.value("type", "") != ev[i].name) return false;
    return true;
  };
  {
    ResponsesStream w(rq, "resp_1", "m", 123);
    auto ev = run(w, {content("Hel"), content("lo")}, result("stop"));
    check(names(ev) == "response.created response.in_progress response.output_item.added response.content_part.added "
                       "response.output_text.delta response.output_text.delta response.output_text.done "
                       "response.content_part.done response.output_item.done response.completed",
          "responses stream: text only", names(ev));
    check(seq_ok(ev), "responses stream: sequence_number counts every event");
    const ojson& c = ev.back().data["response"];
    check(c["status"] == "completed" && c["output"].size() == 1 && c["output"][0]["content"][0]["text"] == "Hello" &&
              c["usage"] == J(R"({"input_tokens": 100, "input_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 40},
                                  "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 120})") &&
              c["instructions"] == "I" && c["id"] == "resp_1" && ev[0].data["response"]["status"] == "in_progress",
          "responses stream: response.completed with the output and usage (cached tokens)", c.dump());
  }
  {
    ResponsesStream w(rq, "resp_2", "m", 123);
    auto ev = run(w, {reasoning("Pl"), reasoning("an."), content("Done")}, result("stop"));
    check(names(ev) == "response.created response.in_progress response.output_item.added response.content_part.added "
                       "response.reasoning_text.delta response.reasoning_text.delta response.reasoning_text.done "
                       "response.content_part.done response.output_item.done response.output_item.added "
                       "response.content_part.added response.output_text.delta response.output_text.done "
                       "response.content_part.done response.output_item.done response.completed",
          "responses stream: reasoning + text", names(ev));
    const ojson& item = ev[8].data["item"];
    std::string text;
    check(item["type"] == "reasoning" && item["content"] == J(R"([{"type": "reasoning_text", "text": "Plan."}])") &&
              unpack_reasoning(item["encrypted_content"], &text) && text == "Plan." && ev[9].data["output_index"] == 1,
          "responses stream: the reasoning item has its text in content and encrypted_content", item.dump());
  }
  {
    ResponsesStream w(rq, "resp_3", "m", 123);
    auto ev = run(w, {reasoning("Two."), call("call_1", "exec_command", R"({"cmd":"ls"})"), call("call_2", "ns__f", "{}")},
                  result("tool_calls"));
    check(names(ev) == "response.created response.in_progress response.output_item.added response.content_part.added "
                       "response.reasoning_text.delta response.reasoning_text.done response.content_part.done "
                       "response.output_item.done response.output_item.added response.function_call_arguments.delta "
                       "response.function_call_arguments.done response.output_item.done response.output_item.added "
                       "response.function_call_arguments.delta response.function_call_arguments.done "
                       "response.output_item.done response.completed",
          "responses stream: reasoning + 2 tool calls", names(ev));
    const ojson& fc = ev[11].data["item"];
    const ojson& ns = ev[15].data["item"];
    check(fc["type"] == "function_call" && fc["call_id"] == "call_1" && fc["name"] == "exec_command" &&
              fc["arguments"] == R"({"cmd":"ls"})" && fc["status"] == "completed" && ev[8].data["item"]["arguments"] == "" &&
              ns["name"] == "f" && ns["namespace"] == "ns",
          "responses stream: function_call items (a namespaced one with its namespace)", fc.dump() + " " + ns.dump());
  }
  {
    ResponsesStream w(rq, "resp_4", "m", 123);
    auto ev = run(w, {call("call_P", "apply_patch", R"({"input":"*** Begin Patch\n*** End Patch"})")}, result("tool_calls"));
    check(names(ev) == "response.created response.in_progress response.output_item.added "
                       "response.custom_tool_call_input.delta response.custom_tool_call_input.done "
                       "response.output_item.done response.completed",
          "responses stream: custom tool call", names(ev));
    const ojson& it = ev[5].data["item"];
    check(it["type"] == "custom_tool_call" && it["call_id"] == "call_P" && it["name"] == "apply_patch" &&
              it["input"] == "*** Begin Patch\n*** End Patch" && ev[4].data["input"] == it["input"],
          "responses stream: the custom call's input is its one string argument", it.dump());
  }
  {
    ResponsesStream w(rq, "resp_5", "m", 123);
    auto ev = run(w, {reasoning("long"), content("cut")}, result("length"));
    const ojson& r = ev.back().data["response"];
    check(ev.back().name == "response.incomplete" && r["status"] == "incomplete" &&
              r["incomplete_details"] == J(R"({"reason": "max_output_tokens"})") && r["output"][1]["status"] == "incomplete" &&
              r["output"][0]["status"] == "completed",
          "responses stream: a length cut ends with response.incomplete (max_output_tokens), the cut item incomplete", r.dump());
  }
  {
    ResponsesStream w(rq, "resp_6", "m", 123);
    ChatResult e = result("error");
    e.error = "boom";
    auto ev = run(w, {}, e);
    check(names(ev) == "response.created response.in_progress response.failed" &&
              ev[2].data["response"]["status"] == "failed" && ev[2].data["response"]["error"]["message"] == "boom",
          "responses stream: an error after the start is response.failed", names(ev));
    ResponsesStream k(rq, "resp_7", "m", 123);
    k.begin(result("stop"));
    const std::string before = k.keepalive();
    k.delta(content("x"));
    const std::string after = k.keepalive();
    const auto ka = parse_sse(before), kb = parse_sse(after);
    check(ka.size() == 1 && kb.size() == 1 && ka[0].name == "response.in_progress" && kb[0].name == "response.in_progress" &&
              kb[0].data["sequence_number"].get<int>() > ka[0].data["sequence_number"].get<int>() + 1 &&
              kb[0].data["response"]["output"].empty(),
          "responses stream: keepalive is response.in_progress, before output and after it (an event Codex's idle timer "
          "counts), numbered in sequence", after);
  }
  {
    ChatResult r = result("error");
    r.prompt_tokens = 140000;
    r.prompt_limit = 131070;
    auto [st, body] = responses_failure(r);
    check(st == 400 && body["error"]["code"] == "context_length_exceeded" && body["error"]["type"] == "invalid_request_error" &&
              body["error"]["param"] == "input",
          "responses: a prompt longer than the context is 400 context_length_exceeded", body.dump());
    ChatResult im = result("error");
    im.error = "video input is not supported";
    im.invalid_request = true;
    check(responses_failure(im).first == 400 && responses_failure(im).second["error"]["type"] == "invalid_request_error",
          "responses: a request the engine cannot take (a video) is a 400 invalid_request_error");
  }
  {
    ChatResult r = result("tool_calls");
    r.reasoning = "R";
    r.tool_calls = {{"call_9", "exec_command", R"({"cmd":"pwd"})"}};
    ojson out;
    const ojson resp = responses_response(rq, "resp_8", "m", 5, r, &out);
    check(resp["object"] == "response" && resp["status"] == "completed" && resp["output"] == out && out.size() == 2 &&
              out[0]["type"] == "reasoning" && out[1]["type"] == "function_call" && resp["created_at"] == 5,
          "responses: non-streamed response", resp.dump());
  }
}

// The Chat Completions stream: the bytes the server wrote before its chunks came from a StreamWriter, the keepalive, and
// the errors (a failure before the stream starts is chat_failure's HTTP error; after it, the same error as an event)
void chat_events() {
  auto chunk = [](const std::string& delta, const std::string& finish = "null", const std::string& usage = "") {
    return R"(data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"m","choices":[{"index":0,"delta":)" +
           delta + R"(,"finish_reason":)" + finish + "}]" + (usage.empty() ? "" : R"(,"usage":)" + usage) + "}\n\n";
  };
  const std::string role = chunk(R"({"role":"assistant","content":""})"), done = "data: [DONE]\n\n",
                    usage = R"({"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":60}})";
  auto tool = [](const char* id, const char* path, int index) {
    return std::string(R"({"id":")") + id + R"(","type":"function","function":{"name":"Read","arguments":"{\"file_path\":\")" +
           path + R"(\"}"},"index":)" + std::to_string(index) + "}";
  };
  {
    ChatCompletionsStream w("chatcmpl-1", "m", 123);
    std::string s = w.begin(ChatResult{});
    for (const ChatDelta& d : {reasoning("Two reads."), content("Hi"), call("call_A", "Read", R"({"file_path":"/a"})"),
                               call("call_B", "Read", R"({"file_path":"/b"})")})
      s += w.delta(d);
    s += w.end(result("tool_calls"));
    const std::string want = role + chunk(R"({"reasoning_content":"Two reads."})") + chunk(R"({"content":"Hi"})") +
                             chunk(R"({"tool_calls":[)" + tool("call_A", "/a", 0) + "]}") +
                             chunk(R"({"tool_calls":[)" + tool("call_B", "/b", 1) + "]}") +
                             chunk("{}", R"("tool_calls")", usage) + done;
    check(s == want, "chat stream: role chunk, reasoning, text, tool calls indexed across deltas, finish with usage, [DONE]",
          s);
  }
  {
    ChatCompletionsStream w("chatcmpl-1", "m", 123);
    ChatDelta d = reasoning("r");
    d.content = "c";
    d.tool_calls = {{"call_A", "Read", R"({"file_path":"/a"})"}, {"call_B", "Read", R"({"file_path":"/b"})"}};
    std::string s = w.begin(ChatResult{});
    s += w.delta(d);
    s += w.delta(call("call_C", "Read", R"({"file_path":"/c"})"));
    const std::string want = role + chunk(R"({"reasoning_content":"r"})") + chunk(R"({"content":"c"})") +
                             chunk(R"({"tool_calls":[)" + tool("call_A", "/a", 0) + "," + tool("call_B", "/b", 1) + "]}") +
                             chunk(R"({"tool_calls":[)" + tool("call_C", "/c", 2) + "]}");
    check(s == want, "chat stream: a delta's reasoning, text and calls are a chunk each, in that order", s);
  }
  {
    ChatResult cut = result("length");
    cut.loops = 2;
    check(ChatCompletionsStream("chatcmpl-1", "m", 123).end(cut) ==
              chunk("{}", R"("length")", usage.substr(0, usage.size() - 1) + R"(,"kapitsa":{"loops":2}})") + done,
          "chat stream: a length cut, a looped model's loop count in usage");
  }
  {
    ChatResult r = result("stop");
    r.prefill_ms = 250.04;
    ChatResult inexact = r;  // images, or a turn given to another request: the time is not the read of those tokens
    r.prefill_exact = true;
    check(ChatCompletionsStream("chatcmpl-1", "m", 123).end(r) ==
              chunk("{}", R"("stop")", usage.substr(0, usage.size() - 1) + R"(,"kapitsa":{"prefill_ms":250.0}})") + done,
          "chat stream: the engine's prefill time in usage, for a client's prefill tok/s on the uncached tokens");
    check(ChatCompletionsStream("chatcmpl-1", "m", 123).end(inexact) == chunk("{}", R"("stop")", usage) + done,
          "chat stream: no prefill time in usage when it holds image encoding or turns given to other requests");
  }
  {
    ChatResult big = result("error");
    big.prompt_tokens = 140000, big.prompt_limit = 131070;
    big.error = "prompt (140000 tokens) exceeds the context window (131072)";
    auto [st, body] = chat_failure(big);
    check(st == 400 && body == J(R"({"error": {"message": "This model's maximum context length is 131070 tokens. However, your messages resulted in 140000 tokens. Please reduce the length of the messages.",
                                                "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}})"),
          "chat: a prompt the context cannot hold is a 400 context_length_exceeded, in OpenAI's words", body.dump());
    ChatResult bad = result("error");
    bad.error = "images: the image at https://x/a.png can't be read";
    bad.invalid_request = true;
    auto [st2, body2] = chat_failure(bad);
    check(st2 == 400 && body2 == J(R"({"error": {"message": "images: the image at https://x/a.png can't be read",
                                                  "type": "invalid_request_error", "param": null, "code": null}})"),
          "chat: another failure is a 400 with the engine's message", body2.dump());
    ChatCompletionsStream w("chatcmpl-1", "m", 123);
    std::string s = w.begin(ChatResult{});
    s += w.end(big);  // the memory for the KV cache ran out after the stream started: the same error, as an event
    check(s == role + "data: " + api::dump(body) + "\n\n" + done,
          "chat stream: a failure after the start is chat_failure's error as an event, then [DONE]", s);
  }
  {
    ChatCompletionsStream w("chatcmpl-1", "m", 123);
    const std::string k = w.keepalive();
    check(k == chunk("{}"), "chat stream: the keepalive is a chunk with an empty delta, finish_reason null", k);
    std::string s = w.begin(ChatResult{});
    s += w.keepalive();
    s += w.delta(content("Hi"));
    s += w.keepalive();
    s += w.end(result("stop"));
    std::string text;
    int chunks = 0;
    for (const auto& e : parse_sse(s.substr(0, s.size() - done.size()))) {
      ++chunks;
      text += e.data["choices"][0]["delta"].value("content", "");
    }
    check(chunks == 5 && text == "Hi", "chat stream: keepalives between chunks add no text", s);
    check(w.delta(content("caf\xc3")) == chunk("{\"content\":\"caf\xef\xbf\xbd\"}"),
          "chat stream: invalid UTF-8 at the end of a piece is replaced, not thrown");
  }
}

// ---------------------------------------------------------------- the launchers

void clients() {
  AgentServer s{"http://127.0.0.1:8000", "qwen3.6-35b-a3b", 131072, ""};
  std::map<std::string, std::string> env = {{"ANTHROPIC_API_KEY", "sk-real"}};
  AgentLaunch c = agent_launch("claude", "/bin/claude", s, {"-p", "hi", "--allowedTools", "Read"}, env, -1);
  auto has = [](const AgentLaunch& l, const std::string& k, const std::string& v) {
    for (auto& [a, b] : l.set)
      if (a == k) return b == v;
    return false;
  };
  check(has(c, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8000") && has(c, "KAPITSA_API_KEY", "kapitsa") &&
            has(c, "ANTHROPIC_MODEL", "qwen3.6-35b-a3b") && has(c, "ANTHROPIC_DEFAULT_HAIKU_MODEL", "qwen3.6-35b-a3b") &&
            has(c, "ANTHROPIC_SMALL_FAST_MODEL", "qwen3.6-35b-a3b") && has(c, "CLAUDE_CODE_MAX_CONTEXT_TOKENS", "131072") &&
            has(c, "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "131072") && has(c, "CLAUDE_CODE_USE_BEDROCK", "0") &&
            has(c, "CLAUDE_CODE_USE_VERTEX", "0") && has(c, "CLAUDE_CODE_USE_FOUNDRY", "0") &&
            c.unset == std::vector<std::string>{"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"},
        "claude: environment");
  check(c.argv == std::vector<std::string>{"/bin/claude", "--settings", R"({"apiKeyHelper":"printf '%s' \"$KAPITSA_API_KEY\""})",
                                           "--model", "qwen3.6-35b-a3b", "--disallowedTools", "WebSearch", "-p", "hi",
                                           "--allowedTools", "Read"},
        "claude: arguments (the server's key through an apiKeyHelper)", "");
  check(ojson::parse(c.argv[2])["apiKeyHelper"] == "printf '%s' \"$KAPITSA_API_KEY\"", "claude: the --settings value is JSON");
  AgentServer small{"http://127.0.0.1:8000", "m", 49152, "k"};
  AgentLaunch c2 = agent_launch("claude", "/bin/claude", small, {}, {}, -1);
  check(has(c2, "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "12288") && has(c2, "KAPITSA_API_KEY", "k"),
        "claude: a smaller window caps its replies at a quarter of it; the server's key");

  AgentLaunch x = agent_launch("codex", "/bin/codex", s, {"exec", "-c", "model_verbosity=\"low\"", "--skip-git-repo-check", "fix it"}, {}, -1);
  std::string argv;
  for (auto& a : x.argv) argv += a + " | ";
  check(argv == "/bin/codex | -c | model=\"qwen3.6-35b-a3b\" | -c | model_provider=\"kapitsa\" | -c | "
                "model_providers.kapitsa={name=\"Kapitsa\",base_url=\"http://127.0.0.1:8000/v1\",env_key=\"KAPITSA_API_KEY\",wire_api=\"responses\"} | "
                "-c | model_context_window=131072 | -c | model_auto_compact_token_limit=117963 | -c | web_search=\"disabled\" | "
                "-c | show_raw_agent_reasoning=true | -c | model_verbosity=\"low\" | exec | --skip-git-repo-check | fix it | " &&
            has(x, "KAPITSA_API_KEY", "kapitsa"),
        "codex: provider settings, the user's -c moved before the subcommand", argv);

  std::map<std::string, std::string> oenv = {{"OPENCODE_CONFIG_CONTENT",
      R"({"provider": {"mine": {"npm": "x"}}, "agent": {"build": {"temperature": 0.1, "model": "other/x"}}, "theme": "dark"})"}};
  AgentLaunch o = agent_launch("opencode", "/bin/opencode", s, {"run", "hello", "--", "-x"}, oenv, 2);
  ojson cfg;
  for (auto& [k, v] : o.set)
    if (k == "OPENCODE_CONFIG_CONTENT") cfg = ojson::parse(v);
  check(cfg["model"] == "kapitsa/qwen3.6-35b-a3b" && cfg["small_model"] == "kapitsa/qwen3.6-35b-a3b" &&
            cfg["theme"] == "dark" && cfg["provider"].contains("mine") &&
            cfg["agent"]["build"] == J(R"({"temperature": 0.1, "model": "kapitsa/qwen3.6-35b-a3b"})") &&
            cfg["agent"]["title"]["model"] == "kapitsa/qwen3.6-35b-a3b" &&
            cfg["provider"]["kapitsa"]["npm"] == "@ai-sdk/openai-compatible" &&
            cfg["provider"]["kapitsa"]["options"]["baseURL"] == "http://127.0.0.1:8000/v1" &&
            cfg["provider"]["kapitsa"]["models"]["qwen3.6-35b-a3b"]["limit"] == J(R"({"context": 131072, "input": 98304, "output": 32768})") &&
            cfg["provider"]["kapitsa"]["models"]["qwen3.6-35b-a3b"]["reasoning"] == true,
        "opencode: provider, model and agents merged into the user's OPENCODE_CONFIG_CONTENT", cfg.dump());
  check(o.argv == std::vector<std::string>{"/bin/opencode", "run", "hello", "--standalone", "--", "-x"},
        "opencode 2: --standalone before --");
  check(agent_launch("opencode", "/bin/opencode", s, {"--server", "http://h"}, {}, 2).argv.size() == 3 &&
            agent_launch("opencode", "/bin/opencode", s, {}, {}, 1).argv.size() == 1,
        "opencode: no --standalone with --server, or before 2.0");
  bool refused = false;
  try {
    agent_launch("opencode", "/bin/opencode", s, {}, {{"OPENCODE_CONFIG_CONTENT", "[1]"}}, 2);
  } catch (const std::exception&) {
    refused = true;
  }
  check(refused, "opencode: an OPENCODE_CONFIG_CONTENT that is not an object is refused");

  AgentServer ks{"http://127.0.0.1:8000", "qwen3.6-35b-a3b:6bit", 131072, ""};
  AgentLaunch p = agent_launch("pi", "/bin/pi", ks, {"-p", "hi", "--thinking", "high"}, {}, -1, "/k/agents");
  check(p.argv == std::vector<std::string>{"/bin/pi", "-e", "/k/agents/pi-kapitsa.js", "--provider", "kapitsa", "--model",
                                           "qwen3.6-35b-a3b:6bit", "-p", "hi", "--thinking", "high"} &&
            has(p, "KAPITSA_PI_BASE_URL", "http://127.0.0.1:8000") && has(p, "KAPITSA_PI_MODEL", "qwen3.6-35b-a3b:6bit") &&
            has(p, "KAPITSA_PI_CONTEXT", "131072") && has(p, "KAPITSA_API_KEY", "kapitsa") && p.files.size() == 1 &&
            p.files[0].first == "/k/agents/pi-kapitsa.js" &&
            p.files[0].second.find("api: \"anthropic-messages\"") != std::string::npos &&
            p.files[0].second.find("export default function (pi)") != std::string::npos,
        "pi: its provider as an extension for the run (pi -e, nothing in ~/.pi), the server in its environment");
}

// ---------------------------------------------------------------- round trips through a chat template

struct TmplModel {
  std::string dir;
  Tokenizer tok;
  std::unique_ptr<minja::chat_template> tmpl;
};

// The deltas the engine streams for a generated `text` (what the model writes after the generation prompt)
std::vector<ChatDelta> generate(const TmplModel& m, const std::string& text, bool in_think, const ojson& tools,
                                std::vector<int>* ids) {
  OutputParser p(m.tok, m.tok.find("<think>"), m.tok.find("</think>"), m.tok.find("<tool_call>"), m.tok.find("</tool_call>"),
                 in_think, !tools.empty(), tools, true);
  std::vector<ChatDelta> out;
  // the reply continues its prompt: with a Metaspace tokenizer (Nanbeige 4.3's, prepend_scheme "first") no U+2581 goes
  // before its first word, as encode() puts before a text's; encoded after an added token, the prompt's last, and
  // without it
  const int after = m.tok.find("<|im_start|>");
  if (m.tok.strips_leading_space() && after >= 0) {
    *ids = m.tok.encode("<|im_start|>" + text);
    ids->erase(ids->begin());
  } else {
    *ids = m.tok.encode(text);
  }
  for (int id : *ids) {
    ChatDelta d = p.push(id);
    if (!d.reasoning.empty() || !d.content.empty() || !d.tool_calls.empty()) out.push_back(d);
  }
  ChatDelta t = p.finish(true);
  if (!t.reasoning.empty() || !t.content.empty() || !t.tool_calls.empty()) out.push_back(t);
  return out;
}

ChatResult finished(const std::vector<ChatDelta>& ds, int completion) {
  ChatResult r;
  for (auto& d : ds) {
    r.reasoning += d.reasoning;
    r.content += d.content;
    for (auto& c : d.tool_calls) r.tool_calls.push_back(c);
  }
  r.finish_reason = r.tool_calls.empty() ? "stop" : "tool_calls";
  r.prompt_tokens = 1000;
  r.completion_tokens = completion;
  return r;
}

// The assistant message an Anthropic client builds from the stream (as the SDK accumulates it)
ojson anthropic_reply(const std::vector<Event>& ev) {
  ojson blocks = ojson::array();
  std::map<int, std::string> json_in;
  for (auto& e : ev) {
    if (e.name == "content_block_start") blocks.push_back(e.data["content_block"]);
    else if (e.name == "content_block_delta") {
      ojson& b = blocks[e.data["index"].get<size_t>()];
      const ojson& d = e.data["delta"];
      if (d["type"] == "text_delta") b["text"] = b["text"].get<std::string>() + d["text"].get<std::string>();
      else if (d["type"] == "thinking_delta") b["thinking"] = b["thinking"].get<std::string>() + d["thinking"].get<std::string>();
      else if (d["type"] == "signature_delta") b["signature"] = d["signature"];
      else if (d["type"] == "input_json_delta") json_in[e.data["index"]] += d["partial_json"].get<std::string>();
    } else if (e.name == "content_block_stop") {
      const int i = e.data["index"];
      if (blocks[size_t(i)]["type"] == "tool_use") blocks[size_t(i)]["input"] = ojson::parse(json_in[i]);
    }
  }
  return {{"role", "assistant"}, {"content", blocks}};
}

// Whether `next`'s prompt continues `prev`'s prompt and the generated ids (+ the end of the turn) token for token
bool continues(const TmplModel& m, const std::string& prev, const std::vector<int>& gen, const std::string& next, std::string* why) {
  std::vector<int> want = m.tok.encode(prev);
  want.insert(want.end(), gen.begin(), gen.end());
  want.push_back(m.tok.find("<|im_end|>"));
  const std::vector<int> got = m.tok.encode(next);
  size_t i = 0;
  while (i < want.size() && i < got.size() && want[i] == got[i]) ++i;
  if (i == want.size()) return true;
  std::vector<int> a(want.begin() + long(i), want.begin() + long(std::min(want.size(), i + 12)));
  std::vector<int> b(got.begin() + long(i), got.begin() + long(std::min(got.size(), i + 12)));
  *why = "diverges at token " + std::to_string(i) + " of " + std::to_string(want.size()) + ": live " +
         ojson(m.tok.decode(a)).dump() + " vs next " + ojson(m.tok.decode(b)).dump();
  return false;
}

void anthropic_round_trip(const TmplModel& m, const std::string& tag) {
  // a Claude Code tool loop: the hook output as a mid-conversation system message, thinking with display omitted,
  // keep:all thinking, then the tool result with a reminder, then a final answer and a new question
  ojson body = J(R"({"model": "m", "max_tokens": 32000, "stream": true,
    "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.284.dd4; cc_entrypoint=cli;"},
               {"type": "text", "text": "You are a coding agent."}],
    "tools": [{"name": "Read", "description": "Reads a file", "input_schema": {"type": "object",
               "properties": {"file_path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["file_path"]}}],
    "thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"},
    "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    "messages": [{"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nToday is Monday.\n</system-reminder>\n"},
                                              {"type": "text", "text": "What is the value in /tmp/v.txt?"}]},
                 {"role": "system", "content": [{"type": "text", "text": "SessionStart hook additional context: none"}]}]})");
  const MessagesRequest q1 = messages_request(body);
  const std::string p1 = render_chat(*m.tmpl, q1.chat, true);
  std::vector<int> g1;
  const std::string gen1 = "The user wants the value. I'll read the file.\n</think>\n\nLet me read it.\n\n<tool_call>\n"
                           "<function=Read>\n<parameter=file_path>\n/tmp/v.txt\n</parameter>\n<parameter=limit>\n10\n"
                           "</parameter>\n</function>\n</tool_call>";
  auto d1 = generate(m, gen1, true, q1.chat.tools, &g1);
  MessagesStream w1("msg_a", "m", q1.show_thinking);
  auto ev1 = run(w1, d1, finished(d1, int(g1.size())));
  const ojson reply1 = anthropic_reply(ev1);
  check(reply1["content"].size() == 3 && reply1["content"][0]["thinking"] == "" && reply1["content"][2]["input"]["limit"] == 10,
        tag + "anthropic: the streamed reply as Claude Code assembles it (thinking omitted: signature only; integer argument)",
        reply1.dump());
  body["messages"].push_back(reply1);
  body["messages"].push_back(J(R"({"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "", "content": "1\tvalue: 42\n"},
      {"type": "text", "text": "<system-reminder>\nThe file is short.\n</system-reminder>"}]})"));
  body["messages"].back()["content"][0]["tool_use_id"] = reply1["content"][2]["id"];
  const MessagesRequest q2 = messages_request(body);
  const std::string p2 = render_chat(*m.tmpl, q2.chat, true);
  std::string why;
  check(continues(m, p1, g1, p2, &why), tag + "anthropic: the tool loop's next prompt continues the generated turn token for token", why);
  // the final answer, then a new question: with keep:all thinking (preserve_thinking) the next prompt still continues
  // the live state, reasoning included
  std::vector<int> g2;
  const std::string gen2 = "The file says 42.\n</think>\n\nThe value is 42.";
  auto d2 = generate(m, gen2, true, q2.chat.tools, &g2);
  MessagesStream w2("msg_b", "m", q2.show_thinking);
  body["messages"].push_back(anthropic_reply(run(w2, d2, finished(d2, int(g2.size())))));
  body["messages"].push_back(J(R"({"role": "user", "content": "And doubled?"})"));
  const std::string p3 = render_chat(*m.tmpl, messages_request(body).chat, true);
  check(continues(m, p2, g2, p3, &why), tag + "anthropic: after the final answer, the next question's prompt continues it too (keep:all)", why);
  check(p2.find("<|im_start|>system\nSessionStart hook additional context: none<|im_end|>\n<|im_start|>assistant") != std::string::npos &&
            p2.find("x-anthropic-billing-header") == std::string::npos,
        tag + "anthropic: the hook's system message is a system turn in place; no billing header in the prompt");
}

void responses_round_trip(const TmplModel& m, const std::string& tag, bool drop_content) {
  ResponseStore store;
  ojson input = J(R"([{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Use tools."}]},
                      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]},
                      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is in v.txt?"}]}])");
  const ojson tools = J(R"([{"type": "function", "name": "exec_command", "description": "Runs a command", "strict": false,
                             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
                            {"type": "custom", "name": "apply_patch", "description": "Edits files",
                             "format": {"type": "grammar", "syntax": "lark", "definition": "start: begin_patch"}}])");
  auto request = [&](const ojson& in) {
    ojson b = {{"model", "m"}, {"instructions", "You are Codex."}, {"stream", true}, {"store", false},
               {"reasoning", {{"summary", "auto"}}}, {"tools", tools}, {"input", in}};
    return responses_request(b, store);
  };
  const ResponsesRequest q1 = request(input);
  const std::string p1 = render_chat(*m.tmpl, q1.chat, true);
  std::vector<int> g1;
  const std::string gen1 = "I'll cat it, then patch.\n</think>\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\n"
                           "cat v.txt\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=apply_patch>\n"
                           "<parameter=input>\n*** Begin Patch\n*** Update File: v.txt\n-1\n+2\n*** End Patch\n</parameter>\n"
                           "</function>\n</tool_call>";
  auto d1 = generate(m, gen1, true, q1.chat.tools, &g1);
  ResponsesStream w1(q1, "resp_a", "m", 1);
  auto ev = run(w1, d1, finished(d1, int(g1.size())));
  // Codex keeps the finished items (output_item.done) and sends them back, then the tools' outputs; OpenAI's examples
  // send each call followed by its output
  ojson items = ojson::array(), outputs = ojson::array();
  for (auto& e : ev)
    if (e.name == "response.output_item.done") {
      ojson it = e.data["item"];
      if (drop_content && it["type"] == "reasoning") it.erase("content");  // a client that sends encrypted_content only
      items.push_back(it);
      if (it["type"] == "function_call")
        outputs.push_back({{"type", "function_call_output"}, {"call_id", it["call_id"]}, {"output", "value: 1"}});
      else if (it["type"] == "custom_tool_call")
        outputs.push_back({{"type", "custom_tool_call_output"}, {"call_id", it["call_id"]}, {"output", "Success."}});
      else
        outputs.push_back(nullptr);
    }
  check(items.size() == 3 && items[2]["type"] == "custom_tool_call" &&
            items[2]["input"] == "*** Begin Patch\n*** Update File: v.txt\n-1\n+2\n*** End Patch",
        tag + "responses: the reply's items as Codex keeps them (reasoning, function_call, custom_tool_call)", items.dump());
  ojson codex = input, interleaved = input;
  for (auto& it : items) codex.push_back(it);
  for (auto& o : outputs)
    if (!o.is_null()) codex.push_back(o);
  for (size_t i = 0; i < items.size(); ++i) {
    interleaved.push_back(items[i]);
    if (!outputs[i].is_null()) interleaved.push_back(outputs[i]);
  }
  std::string why;
  for (const auto& [in, order] : {std::make_pair(codex, std::string("calls, then outputs")),
                                  std::make_pair(interleaved, std::string("each call, then its output"))}) {
    const std::string p2 = render_chat(*m.tmpl, request(in).chat, true);
    check(continues(m, p1, g1, p2, &why),
          tag + "responses: the tool loop's next prompt continues the generated turn token for token (" + order +
              (drop_content ? "; reasoning from encrypted_content only)" : ")"), why);
  }
}

// The reasoning memory without a model (this runs in CI): which turn gets which reasoning back, and what it never touches.
void reasoning_memory_unit() {
  using Calls = std::vector<std::pair<std::string, std::string>>;
  const Calls ls = {{"bash", R"({"command":"ls","cwd":"/a"})"}};
  auto turn = [](const char* content, const Calls& calls, const char* args_text = nullptr) {
    ojson tcs = ojson::array();
    for (auto& [name, args] : calls)
      tcs.push_back({{"id", "call_x"}, {"type", "function"},
                     {"function", {{"name", name}, {"arguments", args_text ? ojson(std::string(args_text)) : ojson::parse(args)}}}});
    ojson m = {{"role", "assistant"}, {"content", content}};
    if (!calls.empty()) m["tool_calls"] = tcs;
    return m;
  };
  const ojson sys = J(R"({"role": "system", "content": "S"})");
  const ojson ua = J(R"({"role": "user", "content": "task A"})"), ub = J(R"({"role": "user", "content": "task B"})");
  const ojson tool = J(R"({"role": "tool", "tool_call_id": "call_x", "content": "ok"})");
  auto msgs = [](std::initializer_list<ojson> l) { return ojson(std::vector<ojson>(l)); };
  ReasoningMemory mem;
  // two sessions end a step with the same call and no text: each gets its own reasoning back
  mem.remember(msgs({sys, ua}), "", ls, "thinking of A");
  mem.remember(msgs({sys, ub}), "", ls, "thinking of B");
  ojson a = msgs({sys, ua, turn("", ls), tool}), b = msgs({sys, ub, turn("", ls), tool});
  check(mem.restore(a) == 1 && mem.restore(b) == 1 && a[2]["reasoning_content"] == "thinking of A" &&
            b[2]["reasoning_content"] == "thinking of B",
        "reasoning memory: two conversations with the same turn text each get their own reasoning back");
  ojson c = msgs({sys, J(R"({"role": "user", "content": "task C"})"), turn("", ls), tool});
  check(mem.restore(c) == 0 && !c[2].contains("reasoning_content"), "reasoning memory: the same turn after another history gets nothing");
  // a turn's text with the separators a naive key would use cannot collide with a call
  mem.remember(msgs({sys, ua}), "Done.", {}, "answer reasoning");
  ojson d = msgs({sys, ua, turn("Done.", {})});
  ojson e = msgs({sys, ua, turn("Done.\x01x\x02y", {})});
  check(mem.restore(d) == 1 && mem.restore(e) == 0, "reasoning memory: a final answer is found by its text; other text is not");
  // the client's own reasoning stays; client ids and key order do not matter; arguments come back as written
  ojson f = msgs({sys, ua, turn("", ls), tool});
  f[2]["reasoning_content"] = "client's";
  check(mem.restore(f) == 0 && f[2]["reasoning_content"] == "client's", "reasoning memory: a client's own reasoning is never replaced");
  ojson g = msgs({sys, ua, turn("", ls, R"({"cwd":"/a","command":"ls"})"), tool});
  g[2]["tool_calls"][0]["id"] = "other";
  check(mem.restore(g) == 1 && g[2]["tool_calls"][0]["function"]["arguments"].get<std::string>() == R"({"command":"ls","cwd":"/a"})",
        "reasoning memory: another key order and tool-call id match; the arguments are put back in the model's order");
  ojson g2 = msgs({sys, ua, turn("", ls), tool});
  g2[2]["tool_calls"][0]["function"]["arguments"] = ojson::parse(R"({"cwd":"/a","command":"ls"})");
  check(mem.restore(g2) == 1 && g2[2]["tool_calls"][0]["function"]["arguments"].dump() == R"({"command":"ls","cwd":"/a"})",
        "reasoning memory: arguments sent as an object come back in the model's key order");
  // a client that rewrote an earlier message gets nothing (the old behaviour)
  ojson h = msgs({sys, J(R"j({"role": "user", "content": "task A (edited)"})j"), turn("", ls), tool});
  check(mem.restore(h) == 0, "reasoning memory: a changed earlier message means no match");
  // arguments nested far deeper than any call: no recursion over them, no match
  std::string deep(5000, '[');
  deep += std::string(5000, ']');
  const Calls dc = {{"f", "{\"x\":" + deep + "}"}};
  mem.remember(msgs({sys, ua}), "", dc, "never kept");
  ojson dd = msgs({sys, ua, turn("", dc), tool});
  check(mem.restore(dd) == 0, "reasoning memory: arguments nested too deep are never matched (and never recursed over)");
  // a size mismatch in tool_calls leaves the arguments alone
  ojson k = msgs({sys, ua, turn("", ls), tool});
  k[2]["tool_calls"].push_back(k[2]["tool_calls"][0]);
  check(mem.restore(k) == 0, "reasoning memory: a turn with other calls than the remembered one gets nothing");
  // bounded: the oldest turns go first
  ReasoningMemory small;
  for (size_t i = 0; i < ReasoningMemory::kMaxEntries + 40; ++i)
    small.remember(msgs({sys, J(R"({"role": "user", "content": "q"})"), turn(("t" + std::to_string(i)).c_str(), {})}), "x", {}, "r");
  check(small.size() == ReasoningMemory::kMaxEntries, "reasoning memory: at most 512 turns are kept");
  ReasoningMemory big;
  for (int i = 0; i < 40; ++i) big.remember(msgs({sys, ua}), "turn " + std::to_string(i), {}, std::string(1 << 20, 'r'));
  check(big.size() <= 16 && big.size() >= 14, "reasoning memory: at most 16 MB of reasoning are kept", std::to_string(big.size()));
  // off by the environment
  setenv("NI_REASONING_MEMORY", "0", 1);
  ojson o = msgs({sys, ua, turn("", ls), tool});
  check(mem.restore(o) == 0, "reasoning memory: NI_REASONING_MEMORY=0 puts nothing back");
  ReasoningMemory off;
  off.remember(msgs({sys, ua}), "", ls, "r");
  unsetenv("NI_REASONING_MEMORY");
  check(off.size() == 0, "reasoning memory: NI_REASONING_MEMORY=0 remembers nothing");
}

// A chat-completions client that drops reasoning_content (or names it otherwise) in a tool loop: the template renders
// an empty think block where the live state holds the reasoning, and the next prompt diverges at that turn. With the
// server's reasoning memory the same request renders as the client that echoes it (the prompt continues the generated
// turn token for token); a client's own reasoning is never replaced.
void reasoning_memory(const TmplModel& m, const std::string& tag) {
  const ojson tools = J(R"([{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object",
    "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}}}])");
  ChatRequest q1;
  q1.messages = J(R"([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}])");
  q1.tools = tools;
  const std::string p1 = render_chat(*m.tmpl, q1, true);
  std::vector<int> g1;
  const std::string gen1 = "I should read the file first, then answer.\n</think>\n\nReading it.\n\n<tool_call>\n<function=f>\n"
                           "<parameter=path>\nsrc/a.py\n</parameter>\n<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>";
  auto d1 = generate(m, gen1, true, tools, &g1);
  const ChatResult res = finished(d1, int(g1.size()));
  ReasoningMemory mem;
  std::vector<std::pair<std::string, std::string>> calls;
  for (auto& tc : res.tool_calls) calls.emplace_back(tc.name, tc.arguments);
  mem.remember(q1.messages, res.content, calls, res.reasoning);
  ojson tcs = ojson::array();
  for (auto& tc : res.tool_calls)
    tcs.push_back({{"id", tc.id}, {"type", "function"}, {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
  const ojson tool = {{"role", "tool"}, {"tool_call_id", res.tool_calls.empty() ? "" : res.tool_calls[0].id}, {"content", "ok"}};
  auto next = [&](const ojson& assistant) {
    ChatRequest q2 = q1;
    q2.messages.push_back(assistant);
    q2.messages.push_back(tool);
    for (auto& mm : q2.messages)  // normalize_messages
      if (mm.contains("tool_calls"))
        for (auto& tc : mm["tool_calls"])
          if (tc["function"]["arguments"].is_string()) tc["function"]["arguments"] = ojson::parse(tc["function"]["arguments"].get<std::string>());
    return q2;
  };
  std::string why;
  ojson plain = {{"role", "assistant"}, {"content", res.content}, {"tool_calls", tcs}};  // no reasoning at all
  ChatRequest q2 = next(plain);
  check(!continues(m, p1, g1, render_chat(*m.tmpl, q2, true), &why),
        tag + "reasoning memory: a client that drops the reasoning diverges at that turn (without the memory)");
  check(res.reasoning.size() > 5 && mem.restore(q2.messages) == 1 && continues(m, p1, g1, render_chat(*m.tmpl, q2, true), &why),
        tag + "reasoning memory: the dropped reasoning is put back, the next prompt continues the generated turn token for token", why);
  // the client re-serialized the arguments (another key order, compact text): the same turn
  ojson reordered = tcs;
  reordered[0]["function"]["arguments"] = R"({"limit":10,"path":"src/a.py"})";
  q2 = next({{"role", "assistant"}, {"content", res.content}, {"tool_calls", reordered}});
  check(mem.restore(q2.messages) == 1 && continues(m, p1, g1, render_chat(*m.tmpl, q2, true), &why),
        tag + "reasoning memory: found again when the client wrote the arguments in another key order", why);
  // reasoning the client sent stays; a turn the server did not write (other content) gets nothing
  ojson own = plain;
  own["reasoning_content"] = "the client's own";
  q2 = next(own);
  check(mem.restore(q2.messages) == 0 && q2.messages[2]["reasoning_content"] == "the client's own",
        tag + "reasoning memory: a client's own reasoning is never replaced");
  ojson other = plain;
  other["content"] = "Something else.";
  q2 = next(other);
  check(mem.restore(q2.messages) == 0 && !q2.messages[2].contains("reasoning_content"),
        tag + "reasoning memory: a turn the server did not write gets no reasoning");
}

// KFlash around prompt lookup (src/model/kflash_wrap.h): the budget verifies the source's drafts until it has measured,
// then only as many as pay for their rows
void kflash_lookup_unit() {
  KfWrap w;
  w.cycle(1), w.cycle(3, 4);
  check(w.cycles == 2 && w.tokens == 4 && w.rows == 5 && w.tokens_per_cycle() == 2.0 && w.source == "none",
        "kflash wrap: counts cycles, tokens and rows");
  check(w.plan(0) == 1 && w.plan(3) == 4, "kflash wrap: with no budget a cycle verifies the source's proposals (none: 1 row, the plain step)");
  {  // through a lookup budget: unmeasured it takes the proposals, rejected ones stop being verified
    KfWrap x;
    x.lookup.emplace(4, 0.6);
    check(x.plan(0) == 1 && x.plan(3) == 4, "kflash wrap: a lookup budget unmeasured takes the proposals");
    for (int i = 0; i < 40; ++i) x.observe(4, 0, 16.0);
    int one = 0;
    for (int i = 0; i < 64; ++i) one += x.plan(4) == 1;
    check(one >= 58 && one < 64, "kflash wrap: rejected proposals are 1 row, bar the periodic probe", std::to_string(one));
  }
  {  // unmeasured: the source's own count, never more than it proposed
    KfLookup b(4, 0.6);
    check(b.pick(0) == 0 && b.pick(3) == 3 && b.pick(9) == 4, "kflash lookup: unmeasured, takes the source's count up to its maximum");
  }
  {  // drafts always accepted: most of them (an extra row at 0.6 steps still pays; the 1% margin keeps a shorter pass on a tie)
    KfLookup b(4, 0.6);
    for (int i = 0; i < 20; ++i) b.observe(4, 4, 11.0);
    int small = 0;
    for (int i = 0; i < 63; ++i) small += b.pick(4) < 3;
    check(small == 0, "kflash lookup: accepted drafts are verified");
  }
  {  // no clock in the choice: the same cycles give the same rows whatever the cycles took
    KfLookup a(4, 0.6), b(4, 0.6);
    bool same = true;
    for (int i = 0; i < 200; ++i) {
      const int da = a.pick(4), db = b.pick(4);
      same = same && da == db;
      const int acc = i % 3 == 0 ? da : 0;
      a.observe(da, acc, 10.0 + i % 7);
      b.observe(db, acc, 3.0 + 40.0 * (i % 5));
    }
    check(same, "kflash lookup: the rows of a cycle do not depend on the time the cycles took");
  }
  {  // drafts never accepted and rows costly: the pending token alone, except the periodic probe
    KfLookup b(4, 0.6);
    for (int i = 0; i < 20; ++i) b.observe(4, 0, 16.0);
    int zero = 0, full = 0;
    for (int i = 0; i < 128; ++i) (b.pick(4) == 0 ? zero : full)++;
    check(zero >= 116 && full >= 4 && full <= 12, "kflash lookup: rejected drafts are skipped, one cycle in 16 probes", std::to_string(full));
  }
  {  // two of four accepted: p(1) = p(2) = 0.97, p(3) = 0.03, so 3 rows (d = 2) win: 2.9 tokens for 2.2 steps, against 1.97 for
     // d = 1 and 3.0 for 3.4 steps at d = 4
    KfLookup b(4, 0.6);
    for (int i = 0; i < 40; ++i) b.observe(4, 2, 11.0);
    const int d = b.pick(4);
    check(d == 2, "kflash lookup: a partly accepted draft is budgeted at the count that pays", std::to_string(d));
  }
  {  // a request's budget starts fresh: the same cycles give the same rows after a request that left it skipping
    KfWrap first, second;
    first.lookup.emplace(4, 0.6);
    for (int i = 0; i < 80; ++i) first.observe(4, 0, 16.0), first.plan(4);
    second.lookup.emplace(4, 0.6);
    KfWrap third;
    third.lookup.emplace(4, 0.6);
    const bool starts_full = second.plan(4) == 5 && third.plan(4) == 5;  // (the source's own count: 4 drafts and the pending row)
    bool same = true;
    for (int i = 0; i < 100; ++i) {
      const int a = second.plan(4), c = third.plan(4);
      same = same && a == c;
      second.observe(a - 1, i % 2 ? a - 1 : 0, 11.0), third.observe(c - 1, i % 2 ? c - 1 : 0, 11.0);
    }
    check(same && starts_full, "kflash lookup: a request's budget is its own: it starts with the source's count");
  }
  {  // the regime changes (prose, then a copy): the budget skipped, then probed, accepted and re-enabled
    KfLookup b(4, 0.6);
    for (int i = 0; i < 40; ++i) b.observe(4, 0, 16.0);
    int back = -1, run = 0;
    for (int i = 0; i < 1000 && back < 0; ++i) {
      const int d = b.pick(4);
      if (d == 0) b.observe(0, 0, 10.0);
      else b.observe(d, d, 11.0);
      run = d >= 1 ? run + 1 : 0;
      if (run >= 4) back = i;
    }
    check(back >= 0 && back <= 400, "kflash lookup: accepted probes re-enable the drafts within 400 cycles", std::to_string(back));
  }
}

// Engine::prefill_history's wait for a quiet server (QuietGate): a request starting or running holds it back, and it
// opens once none started for idle_ms and none is running (two conversations on two slots: the first to finish was
// dropped by the second's next request before)
// #467: a save refused for lack of room, or one that failed while writing, used to be recorded as a save, so a slot whose
// conversation was never written looked saved and another conversation could replace it (its next request then read its
// prompt from the start). A slot counts as saved only when the write of it landed.
void disk_saved_unit() {
  check(!save_landed(SaveOutcome::kNoRoom), "save: no room at all writes nothing: the slot stays unsaved");
  check(!save_landed(SaveOutcome::kWriteFailed), "save: a piece that failed to write is not a save");
  check(!save_landed(SaveOutcome::kEntryFailed), "save: an entry that could not be written is not a save");
  check(save_landed(SaveOutcome::kLanded), "save: a write that landed is a save");
  check(save_landed(SaveOutcome::kNothingToTake), "save: a slot the SSD takes nothing from has nothing to lose");
  // what disk_save_slot and disk_leaving do with each outcome: mark the version saved, and whether to wait before the
  // next attempt (a save that could not be written is not copied again straight away)
  const SaveVerdict no_room = save_verdict(SaveOutcome::kNoRoom);
  check(!no_room.save && no_room.pace_retry, "save verdict: no room leaves the slot unsaved and waits before retrying");
  const SaveVerdict failed = save_verdict(SaveOutcome::kWriteFailed);
  check(!failed.save && failed.pace_retry, "save verdict: a failed write leaves the slot unsaved and waits");
  const SaveVerdict entry = save_verdict(SaveOutcome::kEntryFailed);
  check(!entry.save && entry.pace_retry, "save verdict: a failed entry leaves the slot unsaved and waits");
  const SaveVerdict landed = save_verdict(SaveOutcome::kLanded);
  check(landed.save && !landed.pace_retry, "save verdict: a landed write marks the slot saved");
  const SaveVerdict nothing = save_verdict(SaveOutcome::kNothingToTake);
  check(nothing.save && !nothing.pace_retry, "save verdict: nothing to take marks the slot saved and does not wait");
  check(!skip_retry(false, false, true), "retry: a slot is written when nothing is waiting");
  check(skip_retry(true, true, false), "retry: a cache with no room at all holds every slot back");
  check(skip_retry(true, false, true), "retry: a slot whose own save was refused waits");
  check(!skip_retry(true, false, false), "retry: another slot is written while that one waits");
  check(clear_retry_on_save(true), "retry: a save that landed ends a wait for room in the cache");
  check(!clear_retry_on_save(false), "retry: another slot's save does not end this slot's own wait");
  // the write-behind's pick: one slot whose conversation never fits must not block the others
  check(pick_write(std::vector<WriteCandidate>{{false, false, false, 5}, {false, false, true, 9}}) == 0,
        "write-behind: a slot waiting out a refused save is skipped, the other unsaved slot is written");
  check(pick_write(std::vector<WriteCandidate>{{false, false, false, 5}, {false, false, false, 9}}) == 1,
        "write-behind: the most recently used unsaved slot is written first");
  check(pick_write(std::vector<WriteCandidate>{{false, true, false, 9}, {true, false, false, 8}}) == -1,
        "write-behind: nothing to write when the slots are on the SSD or busy");
  check(safe_to_replace(true, 3, 3, 5000, 64), "guard: a slot already on the SSD is replaced");
  check(!safe_to_replace(true, 4, 3, 5000, 64), "guard: a slot whose save failed is not replaced");
  check(safe_to_replace(true, 4, 3, 50, 64), "guard: a slot shorter than the cache stores is replaced");
  check(safe_to_replace(false, 4, 3, 5000, 64), "guard: no SSD (or one that is off), nothing to lose");
  check(!safe_to_replace(true, 4, 3, 64, 64), "guard: a slot at the cache's minimum length is kept unless its save landed");
}

// #467: the wiring, which the helper tests above cannot see. The store's per-slot refused-version entries are sized when
// the store is built (disk_start passes the slot count); the write-behind's pass -- skip_retry, pick_write, save_verdict,
// as disk_loop applies them -- then tries a slot whose save keeps failing once per retry_ms instead of at every quiet
// spell (~10 s), and the slots behind it are written. A store whose entries were never sized (the shipped bug: no
// entries, every slot reads 0) re-picks the failing slot every spell and writes nothing else, and its retry wait is
// honoured nowhere.
void disk_retry_wiring_unit() {
  check(Engine::disk_wait_selftest(2), "retry wiring: a store built with two slots remembers a refused version per slot");
  check(Engine::disk_wait_selftest(1), "retry wiring: a store built with one slot (--slots 1) round-trips its entry");
  check(Engine::disk_wait_selftest(0), "retry wiring: a store built with no entry reads no refused version");
  // one write-behind pass every 5 s for a minute; slot 0 is the most recently used (so the pick takes it first) and its
  // conversation never fits: every attempt on it fails, the other slots' saves land
  const auto run = [](size_t entries) {
    RefusedVersions refused(entries);
    const double retry_ms = 20000;
    std::vector<uint64_t> version{1, 1, 1}, disk_version{0, 0, 0};
    std::vector<int> attempts{0, 0, 0};
    double refused_until = 0;
    bool refused_all = false;
    for (double t = 0; t <= 60000; t += 5000) {
      std::vector<WriteCandidate> cand(3);
      for (size_t i = 0; i < 3; ++i)
        cand[i] = {false, version[i] == disk_version[i],
                   skip_retry(t < refused_until, refused_all, refused.get(int(i)) == version[i]), 3 - double(i)};
      const int pick = pick_write(cand);
      if (pick < 0) continue;
      ++attempts[size_t(pick)];
      const SaveVerdict v = save_verdict(pick == 0 ? SaveOutcome::kWriteFailed : SaveOutcome::kLanded);
      if (v.save) {
        disk_version[size_t(pick)] = version[size_t(pick)];
        if (clear_retry_on_save(refused_all)) refused_until = 0, refused_all = false;
      } else {
        refused_until = t + retry_ms;
        refused_all = v.cache_full;
        if (!v.cache_full) refused.set(pick, version[size_t(pick)]);
      }
    }
    return attempts;
  };
  const std::vector<int> sized = run(3);
  check(sized[0] == 4, "retry wiring: the failing slot is tried once per retry_ms (4 times in 60 s at 20 s), not every spell");
  check(sized[1] == 1 && sized[2] == 1, "retry wiring: the slots behind a failing one are written while it waits");
  const std::vector<int> unsized = run(0);
  check(unsized[0] == 13, "retry wiring: without sized entries the failing slot is copied at every quiet spell (13 in 60 s)");
  check(unsized[1] == 0 && unsized[2] == 0, "retry wiring: without sized entries the failing slot starves the others");
}

void history_gate_unit() {
  QuietGate g(250, 10, 0);
  check(!g.ready(10, 0, 100), "history gate: 100 ms after the reply it is not yet quiet");
  check(g.ready(10, 0, 250), "history gate: no request for 250 ms and none running: ready");
  check(!g.ready(11, 0, 300), "history gate: another conversation's request started: the quiet time restarts");
  check(!g.ready(11, 0, 549), "history gate: 249 ms after that request it is still not ready");
  check(g.ready(11, 0, 550), "history gate: ready once a request-free idle_ms has passed");
  check(!g.ready(11, 1, 900), "history gate: a request running holds it back however long ago it started");
  check(g.ready(11, 0, 901), "history gate: ready again when it has ended");
  QuietGate z(0, 5, 7);
  check(z.ready(5, 0, 7), "history gate: with no idle time it opens at once on an idle server");
  // the slot a history prefill takes when its own was used meanwhile (--slots 1: the other conversation's)
  check(pick_history_slot({{true, false, false, 5}}) == 0, "history slot: with one slot it takes that one, a continued conversation's");
  check(pick_history_slot({{true, false, false, 5}, {true, false, false, 2}}) == -1,
        "history slot: with several slots all holding continued conversations it gives up");
  check(pick_history_slot({{true, false, false, 5}, {true, false, true, 9}, {true, true, false, 7}}) == 2,
        "history slot: an empty slot is taken before a one-off");
  check(pick_history_slot({{true, false, false, 5}, {true, false, true, 9}, {true, false, true, 3}}) == 2,
        "history slot: of two one-offs the least recently used");
  check(pick_history_slot({{false, true, false, 1}, {true, false, false, 2}}) == -1, "history slot: a busy slot is never taken");
  {  // which older history prefill a newer request of the conversation supersedes
    auto msgs = [](const char* sys, std::initializer_list<const char*> rest) {
      ojson m = ojson::array();
      m.push_back({{"role", "system"}, {"content", sys}});
      int i = 0;
      for (const char* c : rest) m.push_back({{"role", i++ % 2 ? "assistant" : "user"}, {"content", c}});
      return history_message_hashes(m);
    };
    const auto a = msgs("date 1", {"q1"}), a2 = msgs("date 2", {"q1", "answer", "next"});
    check(history_superseded(a, a2), "history stale: the next request of the conversation supersedes it, whatever the system prompt now says");
    check(!history_superseded(a2, a), "history stale: an older request never supersedes a newer one");
    check(!history_superseded(msgs("s", {"q1"}), msgs("s", {"other", "answer", "next"})), "history stale: another conversation does not");
    check(!history_superseded({}, a2), "history stale: no messages supersede nothing");
    // identical first messages (a replay loop): the same prefix, so the older work is obsolete, but a different later turn is not
    check(!history_superseded(msgs("s", {"q1", "x", "y"}), msgs("s", {"q1", "z", "y"})), "history stale: same first message, different turns");
  }
  check(!history_should_yield(10, 10), "history yield: no request since the work began: it goes on");
  check(history_should_yield(10, 11), "history yield: a request arrived: it stops at the chunk boundary");
}

// A float tool argument keeps its text through the Qwen3.5 / Nemotron templates' `| string` (without a model folder: a
// minimal template with their argument line, so CI runs it).
// Engine::disk_lookup's wait for a conversation's SSD copy still being written: bounded by the write and by the prefill
void disk_wait_unit() {
  using namespace disk_wait;
  // 27B/35B scale: 1.5 GB queued at 3 GB/s = 500 ms; 17k rows at 0.5 rows/ms = 34 s: wait
  Decision d = decide(1.5e9, 17000, 3e6, 0.5);
  check(d.wait, "disk wait: a copy that writes in 0.5 s saves 34 s of prefill: waits");
  check(d.limit_ms == kMaxWaitMs, "disk wait: the limit is capped at 3 s");
  // a write slower than the cap would not finish: no wait (it cost +3 s before)
  d = decide(60e9, 17000, 3e6, 0.5);
  check(!d.wait, "disk wait: a copy over 3 s to write is not waited for");
  // the copy saves only a few rows: prefilling them is quicker than the write
  d = decide(0.5e9, 200, 3e6, 10);
  check(!d.wait && d.limit_ms < 30, "disk wait: a copy saving 200 rows (20 ms of prefill) is not waited for");
  // the queue ahead counts: the same copy behind a large one
  check(decide(0.1e9, 8000, 3e6, 3).wait && !decide(20e9, 8000, 3e6, 3).wait,
        "disk wait: bytes queued ahead lengthen the wait");
  // nothing measured yet
  check(!decide(1e8, 500, 0, 0).wait, "disk wait: unmeasured, a short gain is not waited for");
  check(decide(1e8, 4000, 0, 0).wait, "disk wait: unmeasured, a long gain is waited for");
  // first wait after a start: the queued bytes are weighed at the floor write speed (400 MB/s), so 1 GiB (2.7 s) waits
  // but 2.2 GB (5.5 s, over the cap) does not
  check(decide(1.0737e9, 17000, 0, 0).wait && !decide(2.2e9, 17000, 0, 0).wait,
        "disk wait: unmeasured write speed: bounded by the queued bytes at the floor speed");
  check(!decide(1e8, 0, 3e6, 3).wait, "disk wait: nothing saved: no wait");
  // a conversation too large to copy in memory (27B, 16k tokens: 1.5 GB), written before it is replaced: the SSD takes
  // 4 GB/s (0.4 s) against 33 s of prefill: written; a drive at 0.3 GB/s (5 s) is not, nor is a short conversation
  check(decide(1.5e9, 16000, 4e6, 0.5).wait, "disk wait: 1.5 GB at 4 GB/s saves 32 s of prefill: written inline");
  check(!decide(1.5e9, 16000, 0.3e6, 0.5).wait, "disk wait: 1.5 GB at 0.3 GB/s (5 s, over the cap): not written");
  check(!decide(1.5e9, 100, 4e6, 0.5).wait, "disk wait: 1.5 GB for 100 rows (0.2 s of prefill): not written");
}

void float_arguments_text() {
  struct Form { const char* name; const char* tail; };  // the end of the real templates' argument line
  const Form forms[] = {{"plain", " else args_value | string %}"},
                        {"no spaces, -%}", " else args_value|string -%}"},
                        {"line break", "\n   else args_value | string %}"}};
  for (const Form& f : forms) {
    const std::string src =
        "{%- for m in messages %}{%- if m.tool_calls %}{%- for tool_call in m.tool_calls %}"
        "{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}"
        "{%- for args_name, args_value in tool_call.arguments|items %}"
        "{%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string)" +
        std::string(f.tail) + "{{- '<parameter=' + args_name + '>' + args_value + '</parameter>;' }}{%- endfor %}{%- endfor %}{%- endif %}{%- endfor %}";
    minja::chat_template tmpl(src, "", "");
    ChatRequest r;
    r.messages = J(R"([{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f",
      "arguments": {"a": 1.5, "b": 2.0, "c": 0.1, "d": 3, "e": "text", "f": [1.5]}}}]}])");
    const std::string out = render_chat(tmpl, r, false);
    check(out == "<parameter=a>1.5</parameter>;<parameter=b>2.0</parameter>;<parameter=c>0.1</parameter>;<parameter=d>3</parameter>;<parameter=e>text</parameter>;<parameter=f>[1.5]</parameter>;",
          std::string("float arguments: 1.5, 2.0 and 0.1 keep their text through `| string` (") + f.name + ")", out);
  }
  // a template that renders arguments some other way is left alone
  minja::chat_template other("{%- for m in messages %}{%- for t in m.tool_calls %}{{- t.function.arguments | tojson }}{%- endfor %}{%- endfor %}", "", "");
  ChatRequest r;
  r.messages = J(R"([{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": {"a": 1.5}}}]}])");
  check(render_chat(other, r, false).find("1.5") != std::string::npos && render_chat(other, r, false).find("1.500000") == std::string::npos, "float arguments: a template without `| string` renders them as before");
}

// Tool arguments keep each value's spelling (src/engine/tool_args.h) through the Qwen3.5 / Nemotron templates' `| string`
// (a minimal template with their argument line, no model folder, so CI runs it): what the model wrote (`true`, `null`,
// `2.50`, `[1,2]`, non-ASCII text) is what the next prompt holds, from an OpenAI client that sends the arguments text
// back (also re-spaced), from a Responses client, and from an Anthropic client that sends a parsed input.
void tool_arguments_spelling() {
  using namespace tool_args;
  const std::string src =
      "{%- for m in messages %}{%- if m.tool_calls %}{%- for tool_call in m.tool_calls %}"
      "{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}"
      "{%- for args_name, args_value in tool_call.arguments|items %}"
      "{%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}"
      "{{- '<parameter=' + args_name + '>' + args_value + '</parameter>;' }}{%- endfor %}{%- endfor %}{%- endif %}{%- endfor %}";
  minja::chat_template tmpl(src, "", "");
  // the argument text the output parser writes for a call whose values the model spelled like this
  const std::string written =
      "{\"s\":\"text\",\"i\":10,\"t\":true,\"f\":false,\"z\":null,\"n\":2.50,\"e\":1e3,\"a\":[1,2],\"o\":{\"k\":[true,null]},\"u\":[\"caf\xc3\xa9\",1],\"c\":\"caf\xc3\xa9\"}";
  const std::string expect =
      "<parameter=s>text</parameter>;<parameter=i>10</parameter>;<parameter=t>true</parameter>;<parameter=f>false</parameter>;"
      "<parameter=z>null</parameter>;<parameter=n>2.50</parameter>;<parameter=e>1e3</parameter>;<parameter=a>[1,2]</parameter>;"
      "<parameter=o>{\"k\":[true,null]}</parameter>;<parameter=u>[\"caf\xc3\xa9\",1]</parameter>;<parameter=c>caf\xc3\xa9</parameter>;";
  auto render = [&](const ojson& call) {
    ChatRequest r;
    r.messages = ojson::array({{{"role", "assistant"}, {"content", ""}, {"tool_calls", ojson::array({call})}}});
    return render_chat(tmpl, r, false);
  };
  auto first_diff = [](const std::string& a, const std::string& b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
    return "at " + std::to_string(i) + ": " + a.substr(i, 30) + " / " + b.substr(i, 30);
  };
  {  // the text helpers: the values as written, the object back with them
    const auto raw = raw_values(" { \"a\" : [1,2] , \"b\":true,\"c\" : \"x,}\" , \"d\":{\"k\":\"]\"} } ");
    check(raw.size() == 4 && raw[0].second == "[1,2]" && raw[1].second == "true" && raw[2].second == "\"x,}\"" && raw[3].second == "{\"k\":\"]\"}",
          "tool args: raw_values reads each top-level value's text as written (strings with delimiters, nested values)");
    check(raw_values("[1]").empty() && raw_values("{\"a\":").empty() && raw_values("not json").empty(), "tool args: raw_values of no object is nothing");
    const ojson args = ojson::parse(written);
    check(object_text(args, raw_values(written)) == written, "tool args: object_text with the raw values writes the arguments as written", first_diff(object_text(args, raw_values(written)), written));
    check(object_text(args, {{"n", "2.5000"}}).find("\"n\":2.5000,") != std::string::npos, "tool args: a raw text of the same value is written as it is");
    check(object_text(args, {{"n", "9"}}).find("\"n\":2.5") != std::string::npos, "tool args: a raw text that parses to another value is not used");
  }
  const ojson parsed_args = ojson::parse(written);
  {  // OpenAI chat: the arguments text comes back, the server parses it (normalize_messages) and keeps the spellings
    ojson fn = {{"name", "f"}, {"arguments", parsed_args}};
    note_raw(fn, written);
    const std::string out = render({{"id", "1"}, {"type", "function"}, {"function", fn}});
    check(out == expect, "tool arguments: true, false, null, 2.50, 1e3, [1,2] and non-ASCII text come back as written (OpenAI chat)", first_diff(out, expect));
    check(!fn["arguments"].is_string() && fn.contains("arguments_raw") && !fn["arguments_raw"].contains("i") && !fn["arguments_raw"].contains("s"),
          "tool arguments: strings and integers need no spelling noted");
  }
  {  // a client that re-spaces the text (a JSON pretty-printer): the spelling it sent is the spelling kept
    const std::string spaced = "{\"t\": true, \"a\": [1, 2], \"n\": 2.50, \"z\": null}";
    ojson fn = {{"name", "f"}, {"arguments", ojson::parse(spaced)}};
    note_raw(fn, spaced);
    const std::string out = render({{"id", "1"}, {"type", "function"}, {"function", fn}});
    check(out == "<parameter=t>true</parameter>;<parameter=a>[1, 2]</parameter>;<parameter=n>2.50</parameter>;<parameter=z>null</parameter>;",
          "tool arguments: a re-spaced arguments text is rendered as the client sent it", out);
  }
  {  // Responses: a function_call item's arguments text keeps its spellings the same way
    ResponseStore store;
    ojson body = {{"model", "m"}, {"input", ojson::array({{{"role", "user"}, {"content", "go"}},
                                                           {{"type", "function_call"}, {"call_id", "c1"}, {"name", "f"}, {"arguments", written}},
                                                           {{"type", "function_call_output"}, {"call_id", "c1"}, {"output", "ok"}}})}};
    const ResponsesRequest q = responses_request(body, store);
    std::string out;
    for (auto& m : q.chat.messages)
      if (m.contains("tool_calls")) {
        ChatRequest r;
        r.messages = ojson::array({m});
        out = render_chat(tmpl, r, false);
      }
    check(out == expect, "tool arguments: true, false, null, 2.50, [1,2] and non-ASCII text come back as written (Responses)", first_diff(out, expect));
  }
  {  // an arguments object without spellings (Anthropic's input): JSON's true, false, null and 2.5 for a float; lists the template's
    ojson fn = {{"name", "f"}, {"arguments", ojson::parse(R"({"t":true,"f":false,"z":null,"n":2.5,"a":[1,2],"i":3})")}};
    const std::string out = render({{"id", "1"}, {"type", "function"}, {"function", fn}});
    check(out == "<parameter=t>true</parameter>;<parameter=f>false</parameter>;<parameter=z>null</parameter>;<parameter=n>2.5</parameter>;<parameter=a>[1, 2]</parameter>;<parameter=i>3</parameter>;",
          "tool arguments: a parsed input renders true, false, null and floats as JSON writes them (Anthropic)", out);
  }
  {  // a spelling that no longer matches the value is ignored (a client that edited the arguments), and none reaches the template
    ojson fn = {{"name", "f"}, {"arguments", ojson::parse(R"({"t":false,"n":3})")}, {"arguments_raw", {{"t", "true"}, {"n", "4"}}}};
    const std::string out = render({{"id", "1"}, {"type", "function"}, {"function", fn}});
    check(out == "<parameter=t>false</parameter>;<parameter=n>3</parameter>;", "tool arguments: a spelling of another value is not used", out);
  }
  {  // a template that renders arguments some other way sees no arguments_raw
    minja::chat_template other("{%- for m in messages %}{%- for t in m.tool_calls %}{{- t.function | tojson }}{%- endfor %}{%- endfor %}", "", "");
    ChatRequest r;
    r.messages = J(R"([{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": {"a": true}, "arguments_raw": {"a": "true"}}}]}])");
    check(render_chat(other, r, false).find("arguments_raw") == std::string::npos, "tool arguments: the spelling is not part of the call the template sees");
  }
  {  // Anthropic client and the memory: the spellings of what the server generated come back with the turn's reasoning kept by the client
    ReasoningMemory mem;
    const ojson before = J(R"([{"role": "user", "content": "go"}])");
    mem.remember(before, "", {{"f", written}}, "thinking");
    ojson msgs = before;
    msgs.push_back({{"role", "assistant"}, {"content", ""}, {"reasoning_content", "thinking"},
                    {"tool_calls", ojson::array({{{"id", "1"}, {"type", "function"}, {"function", {{"name", "f"}, {"arguments", parsed_args}}}}})}});
    mem.restore(msgs);
    check(msgs[1]["tool_calls"][0]["function"].contains("arguments_raw"), "tool arguments: a turn the server wrote gets its spellings back");
    ChatRequest r;
    r.messages = ojson::array({msgs[1]});
    const std::string out = render_chat(tmpl, r, false);
    check(out == expect, "tool arguments: [1,2] and the rest come back as written for a client that sends parsed input (Anthropic, memory)", first_diff(out, expect));
    // another conversation's identical call is not matched, so no spelling is taken from it
    ojson other = J(R"([{"role": "user", "content": "elsewhere"}])");
    other.push_back({{"role", "assistant"}, {"content", ""}, {"tool_calls", ojson::array({{{"id", "1"}, {"type", "function"}, {"function", {{"name", "f"}, {"arguments", parsed_args}}}}})}});
    mem.restore(other);
    check(!other[1]["tool_calls"][0]["function"].contains("arguments_raw"), "tool arguments: a call after another conversation gets no spelling");
  }
  {  // a turn with no reasoning (thinking off): the spellings are kept anyway, and win over a client's own re-dump of the arguments
    ReasoningMemory mem;
    const ojson before = J(R"([{"role": "user", "content": "go"}])");
    mem.remember(before, "", {{"f", written}}, "");
    ojson msgs = before;
    ojson fn = {{"name", "f"}, {"arguments", parsed_args}, {"arguments_raw", {{"a", "[1, 2]"}, {"t", "True"}}}};  // the client's re-dump
    msgs.push_back({{"role", "assistant"}, {"content", ""}, {"tool_calls", ojson::array({{{"id", "1"}, {"type", "function"}, {"function", fn}}})}});
    check(mem.restore(msgs) == 1 && !msgs[1].contains("reasoning_content"),
          "tool arguments: a turn written without reasoning gets its spellings back and no reasoning");
    ChatRequest r;
    r.messages = ojson::array({msgs[1]});
    const std::string out = render_chat(tmpl, r, false);
    check(out == expect, "tool arguments: the model's spelling wins over a client's re-dump (memory, no reasoning)", first_diff(out, expect));
  }
}

// An OpenAI chat client's tool loop (what `kapitsa serve` is mostly driven by): the reply's tool call goes back as
// arguments text, the server parses it into an object (normalize_messages in src/server/main.cpp) and the template
// writes it again. The parameter the model wrote must come back as the same text, else the next prompt diverges
// inside the call and the cache restarts at the last checkpoint.
void openai_tool_arguments(const TmplModel& m, const std::string& tag) {
  const ojson tools = J(R"([{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object",
    "properties": {"s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": "number"}, "b": {"type": "boolean"},
                   "a": {"type": "array"}, "o": {"type": "object"}, "x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}}}])");
  struct Case { const char* name; const char* param; const char* value; };
  const Case cases[] = {
      {"string", "s", "src/main.py"},
      {"multi-line string with a leading space", "s", " def f():\n    return 1\n"},
      {"integer", "i", "10"},
      {"negative integer", "i", "-3"},
      {"number 1.5", "n", "1.5"},
      {"number 2.0", "n", "2.0"},
      {"number 0.1", "n", "0.1"},
      {"untyped value 1.5", "x", "1.5"},
      // a boolean as JSON writes it, and as Python does (a model that writes `True` for a boolean keeps it as the text)
      {"boolean true", "b", "true"},
      {"boolean false", "b", "false"},
      {"null", "x", "null"},
      {"number 2.50", "n", "2.50"},
      {"number 1e3", "n", "1e3"},
      {"compact array", "a", "[1,2]"},
      {"compact array of text", "a", "[\"a\",\"b\"]"},
      {"compact object", "o", "{\"k\":[true,null],\"n\":2.50}"},
      {"non-ASCII in an array", "a", "[\"caf\xc3\xa9\",1]"},
      {"boolean True", "b", "True"},
      {"boolean False", "b", "False"},
      {"array", "a", "[\"a\", \"b\"]"},
      {"array of objects", "a", "[{\"k\": \"v\", \"n\": 1}]"},
      {"object", "o", "{\"path\": \"a.py\", \"line\": 3}"},
      {"object with non-ASCII text", "o", "{\"t\": \"caf\xc3\xa9\"}"},
      {"untyped value True", "x", "True"},
      {"untyped value 42", "x", "42"},
      {"untyped text", "x", "hello"},
  };
  for (const Case& c : cases) {
    ChatRequest q1;
    q1.messages = J(R"([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}])");
    q1.tools = tools;
    const std::string p1 = render_chat(*m.tmpl, q1, true);
    std::vector<int> g1;
    const std::string gen1 = std::string("Let me call it.\n</think>\n\n<tool_call>\n<function=f>\n<parameter=") + c.param +
                             ">\n" + c.value + "\n</parameter>\n</function>\n</tool_call>";
    auto d1 = generate(m, gen1, true, tools, &g1);
    const ChatResult res = finished(d1, int(g1.size()));
    ChatRequest q2 = q1;
    ojson calls = ojson::array();
    for (auto& tc : res.tool_calls)
      calls.push_back({{"id", tc.id}, {"type", "function"}, {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
    // as the client sends it back: reasoning too (a client that keeps it), arguments as a string
    q2.messages.push_back({{"role", "assistant"}, {"content", res.content}, {"reasoning_content", res.reasoning},
                           {"tool_calls", calls}});
    q2.messages.push_back({{"role", "tool"}, {"tool_call_id", res.tool_calls.empty() ? "" : res.tool_calls[0].id}, {"content", "ok"}});
    for (auto& mm : q2.messages)  // normalize_messages
      if (mm.contains("tool_calls"))
        for (auto& tc : mm["tool_calls"]) {
          const std::string text = tc["function"]["arguments"].get<std::string>();
          tc["function"]["arguments"] = ojson::parse(text);
          tool_args::note_raw(tc["function"], text);
        }
    std::string why;
    const std::string p2 = render_chat(*m.tmpl, q2, true);
    check(res.tool_calls.size() == 1 && continues(m, p1, g1, p2, &why),
          tag + "openai: a tool call's argument as the model wrote it comes back as the same text (" + c.name + ")", why);
  }
}

// An Anthropic client sends a tool call back as a parsed input: JSON's true, false, null and 1.5 come back as written;
// a spelling JSON does not keep (`[1,2]`, `2.50`) comes back through the memory of what this server generated.
void anthropic_tool_arguments(const TmplModel& m, const std::string& tag) {
  const char* tools = R"([{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {
    "s": {"type": "string"}, "b": {"type": "boolean"}, "n": {"type": "number"}, "a": {"type": "array"}, "o": {"type": "object"}, "x": {}}}}])";
  struct Case { const char* name; const char* param; const char* value; bool memory; };
  const Case cases[] = {
      {"boolean true", "b", "true", false},
      {"boolean false", "b", "false", false},
      {"null", "x", "null", false},
      {"number 1.5", "n", "1.5", false},
      {"string", "s", "caf\xc3\xa9 text", false},
      {"array [1, 2]", "a", "[1, 2]", false},
      {"number 2.50", "n", "2.50", true},
      {"compact array", "a", "[1,2]", true},
      {"compact object with true and null", "o", "{\"k\":[true,null],\"n\":2.50}", true},
      {"non-ASCII in an array", "a", "[\"caf\xc3\xa9\",1]", true},
  };
  for (const Case& c : cases) {
    ojson body = J(R"({"model": "m", "max_tokens": 1000, "stream": true, "system": "S", "messages": [{"role": "user", "content": "Q"}],
      "thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"},
      "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}})");
    body["tools"] = J(tools);
    const MessagesRequest q1 = messages_request(body);
    const std::string p1 = render_chat(*m.tmpl, q1.chat, true);
    std::vector<int> g1;
    const std::string gen1 = std::string("Let me call it.\n</think>\n\n<tool_call>\n<function=f>\n<parameter=") + c.param + ">\n" + c.value +
                             "\n</parameter>\n</function>\n</tool_call>";
    auto d1 = generate(m, gen1, true, q1.chat.tools, &g1);
    const ChatResult res = finished(d1, int(g1.size()));
    MessagesStream w1("msg_a", "m", q1.show_thinking);
    const ojson reply1 = anthropic_reply(run(w1, d1, res));
    body["messages"].push_back(reply1);
    body["messages"].push_back(J(R"({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "", "content": "ok"}]})"));
    body["messages"].back()["content"][0]["tool_use_id"] = reply1["content"].back()["id"];
    MessagesRequest q2 = messages_request(body);
    ReasoningMemory mem;
    std::vector<std::pair<std::string, std::string>> calls;
    for (auto& tc : res.tool_calls) calls.emplace_back(tc.name, tc.arguments);
    mem.remember(q1.chat.messages, res.content, calls, res.reasoning.empty() ? "r" : res.reasoning);
    if (c.memory) mem.restore(q2.chat.messages);
    std::string why;
    const std::string p2 = render_chat(*m.tmpl, q2.chat, true);
    check(res.tool_calls.size() == 1 && continues(m, p1, g1, p2, &why),
          tag + "anthropic: a tool call's argument as the model wrote it comes back as the same text (" + c.name + (c.memory ? ", with the memory)" : ")"), why);
  }
}

void template_checks(const TmplModel& m, const std::string& tag) {
  // a later system message is a system turn in place; images become a note; the template no longer raises
  ChatRequest r;
  r.messages = J(R"([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                     {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": {}}}]},
                     {"role": "tool", "tool_call_id": "1", "content": "R"},
                     {"role": "system", "content": "Later."},
                     {"role": "user", "content": [{"type": "text", "text": "Look:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                                                  {"type": "text", "text": "what?"}]}])");
  r.tools = J(R"([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}])");
  std::string p;
  try {
    p = render_chat(*m.tmpl, r, true);
  } catch (const std::exception& e) {
    p = std::string("raised: ") + e.what();
  }
  check(p.find("</tool_response><|im_end|>\n<|im_start|>system\nLater.<|im_end|>\n<|im_start|>user\nLook:\n[image omitted: "
               "this model can't read images]\nwhat?<|im_end|>") != std::string::npos &&
            p.find("<|vision_start|>") == std::string::npos,
        tag + "template: a later system message is a system turn in place; an image is a note", p.substr(p.size() > 400 ? p.size() - 400 : 0));
  // with an image encoder: the image renders as the model's placeholder, without its data (a model whose vocabulary has
  // no image tokens, Nanbeige4.3's, has no encoder: never rendered with images)
  try {
    p = render_chat(*m.tmpl, r, true, true);
  } catch (const std::exception& e) {
    p = std::string("raised: ") + e.what();
  }
  if (m.tok.find("<|vision_start|>") >= 0)
    check(p.find("<|im_start|>user\nLook:<|vision_start|><|image_pad|><|vision_end|>what?<|im_end|>") != std::string::npos &&
              p.find("base64") == std::string::npos,
          tag + "template: with an image encoder, an image is the image placeholder (its data left out)",
          p.substr(p.size() > 400 ? p.size() - 400 : 0));
}

// Read ahead (Engine::draft, POST /v1/draft): what a draft reads is a prefix of the request's prompt once the message
// is sent, token for token, at every point of typing it (so the request continues the state the drafts left), and its
// history part is the text prefill_history reads.
void draft_checks(const TmplModel& m, const std::string& tag) {
  const std::vector<std::string> messages = {
      "Thanks! Now make it handle three numbers, and explain why, in one sentence.",
      "Here is my code:\n\n```python\nclass Stack:\n    def __init__(self):\n        self.items = []\n\n    "
      "def push(self, x):\n        self.items.append(x)\n```\n\nWhat is wrong with it?   ",
      "Numbers: 12345, 3.14159 and 2,718 -- a URL https://example.com/a?b=c&d=e.\n\n\n  Indented.\tTabbed (it's "
      "\"quoted\"), done...",
      "Unicode: café, naïve, 日本語のテキスト, emoji 🙂👍🏽 and Ελληνικά done.",
      "  leading spaces and trailing ones   ",
      "one",
  };
  const std::string answer =
      "Here it is:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nCall it as `add(2, 3)`.";
  struct Conf {
    const char* name;
    const char* kwargs;
    std::string reasoning;
  };
  const std::vector<Conf> confs = {{"thinking off", R"({"enable_thinking": false})", ""},
                                   {"thinking on", R"({"enable_thinking": true})", "The user wants an add function."},
                                   {"preserve_thinking", R"({"enable_thinking": true, "preserve_thinking": true})",
                                    "The user wants an add function."}};
  auto render = [&](const ChatRequest& q) { return render_chat(*m.tmpl, q, false); };
  for (const Conf& c : confs) {
    ChatRequest r;
    r.messages = J(R"([{"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": "Write a function that adds two numbers."}])");
    r.template_kwargs = ojson::parse(c.kwargs);
    // prefill_history's text: the conversation up to a placeholder user turn's start
    ChatRequest h = with_answer(r, answer, c.reasoning);
    h.messages.push_back({{"role", "user"}, {"content", "\x01nace-history-marker\x01"}});
    std::string hist = render_chat(*m.tmpl, h, false);
    hist.resize(hist.rfind("<|im_start|>", hist.rfind("\x01nace-history-marker\x01")));
    std::string bad, gap;
    int prefixes = 0;
    for (size_t i = 0; i < messages.size(); ++i) {
      const std::string& msg = messages[i];
      ChatRequest fin = with_answer(r, answer, c.reasoning);
      fin.messages.push_back({{"role", "user"}, {"content", msg}});
      const std::vector<int> want = m.tok.encode(render_chat(*m.tmpl, fin, true));
      for (size_t k = 0; k <= msg.size(); ++k) {
        if (k < msg.size() && (uint8_t(msg[k]) & 0xC0) == 0x80) continue;  // prefixes at character boundaries
        ++prefixes;
        const DraftPrompt dp = draft_prompt(with_answer(r, answer, c.reasoning), msg.substr(0, k), render);
        std::string why;
        int user_at = -1;
        std::vector<int> ids;
        if (dp.user_at == std::string::npos) {
          why = "no user turn";
        } else {
          ids = m.tok.encode(dp.text, dp.user_at, &user_at);
          ids.pop_back();  // the last token is left for the request
          if (dp.text.compare(0, dp.user_at, hist) != 0 || dp.user_at != hist.size()) why = "the history differs";
          else if (user_at != int(m.tok.encode(hist).size())) why = "the user turn's row differs";
          else if (ids.size() > want.size() || !std::equal(ids.begin(), ids.end(), want.begin()))
            why = "not a prefix of the request's prompt";
        }
        if (!why.empty() && bad.size() < 600)
          bad += "message " + std::to_string(i) + " at " + std::to_string(k) + " " + ojson(msg.substr(0, k)).dump() +
                 ": " + why + "; ";
        // the whole message: the request reads its last word and the end of the turn, no more
        if (k == msg.size() && why.empty()) {
          const std::vector<int> word = m.tok.encode(msg.substr(draft_readable(msg).size()));
          const size_t left = want.size() - ids.size();
          if (i == 0 && left > word.size() + 12)
            gap += "message 0 leaves " + std::to_string(left) + " tokens to the request; ";
        }
      }
    }
    check(bad.empty() && gap.empty(),
          tag + "draft (" + c.name + "): each of " + std::to_string(prefixes) +
              " drafts reads a prefix of the request's prompt, after the text prefill_history reads",
          bad + gap);
  }
}

// --dump DIR: the replies of each scenario as the server sends them, for tests/sdk_check.py (the official Python SDKs
// parse them from a fake server): <name>.sse (a stream, with keepalives), <name>.json (a response body), <name>.status
// (an error's)
void dump_cases(const std::string& dir) {
  auto write = [&](const std::string& name, const std::string& text) {
    FILE* f = fopen((dir + "/" + name).c_str(), "wb");
    if (!f) {
      check(false, "dump: can't write " + dir + "/" + name);
      return;
    }
    fwrite(text.data(), 1, text.size(), f);
    fclose(f);
  };
  struct Case {
    std::string name;
    std::vector<ChatDelta> deltas;
    ChatResult res;
  };
  ChatResult stop = result("stop"), tools = result("tool_calls"), length = result("length"), full = result("length"),
             seq = result("stop"), err = result("error");
  full.context_full = true;
  seq.stop_sequence = "STOP";
  err.error = "the engine failed";
  const std::vector<Case> cases = {
      {"text", {content("Hel"), content("lo")}, stop},
      {"reasoning-text", {reasoning("Th"), reasoning("ink."), content("Hi")}, stop},
      {"reasoning-tools", {reasoning("Two reads."), call("call_A", "Read", R"({"file_path":"/a"})"),
                           call("call_B", "Read", R"({"file_path":"/b","limit":5})")}, tools},
      {"custom-tool", {reasoning("Patch."), call("call_P", "apply_patch", R"({"input":"*** Begin Patch\n*** End Patch"})")}, tools},
      {"length", {reasoning("long"), content("cut")}, length},
      {"context-full", {content("cut")}, full},
      {"stop-sequence", {content("abc")}, seq},
      {"error", {}, err},
  };
  ResponseStore store;
  const ResponsesRequest rq = responses_request(J(R"({"model": "m", "input": "hi", "instructions": "I",
      "tools": [{"type": "function", "name": "Read", "parameters": {"type": "object"}},
                {"type": "custom", "name": "apply_patch"}]})"), store);
  for (const auto& c : cases) {
    ChatResult r = c.res;
    for (auto& d : c.deltas) {
      r.reasoning += d.reasoning;
      r.content += d.content;
      for (auto& t : d.tool_calls) r.tool_calls.push_back(t);
    }
    for (bool show : {true, false}) {
      MessagesStream w("msg_" + c.name, "m", show);
      std::string sse = w.begin(r) + w.keepalive();
      for (auto& d : c.deltas) sse += w.delta(d) + (d.content.empty() ? "" : w.keepalive());
      sse += w.end(r);
      write("anthropic-" + c.name + (show ? "" : "-omitted") + ".sse", sse);
      if (c.name != "error") write("anthropic-" + c.name + (show ? "" : "-omitted") + ".json", api::dump(messages_response("msg_" + c.name, "m", r, show)));
    }
    ResponsesStream w(rq, "resp_" + c.name, "m", 1760000000);
    std::string sse = w.begin(r);
    sse += w.keepalive();  // a statement each: keepalive() numbers its event after the ones before it
    for (auto& d : c.deltas) {
      sse += w.delta(d);
      sse += w.keepalive();
    }
    sse += w.end(r);
    write("responses-" + c.name + ".sse", sse);
    if (c.name != "error") write("responses-" + c.name + ".json", api::dump(responses_response(rq, "resp_" + c.name, "m", 1760000000, r)));
    ChatCompletionsStream cw("chatcmpl-" + c.name, "m", 1760000000);
    std::string chat = cw.begin(r);
    chat += cw.keepalive();
    for (auto& d : c.deltas) {
      chat += cw.delta(d);
      chat += cw.keepalive();
    }
    chat += cw.end(r);
    write("chat-" + c.name + ".sse", chat);
  }
  ChatResult big = result("error");
  big.prompt_tokens = 140000;
  big.prompt_limit = 131070;
  auto [as, ab] = messages_failure(big);
  write("anthropic-too-long.json", api::dump(ab));
  write("anthropic-too-long.status", std::to_string(as));
  auto [rs, rb] = responses_failure(big);
  write("responses-too-long.json", api::dump(rb));
  write("responses-too-long.status", std::to_string(rs));
  auto [cs, cb] = chat_failure(big);
  write("chat-too-long.json", api::dump(cb));
  write("chat-too-long.status", std::to_string(cs));
  write("anthropic-count.json", api::dump(ojson{{"input_tokens", 1234}}));
}

// paren_call_postfix: minja reads no `.`/`[` after a call (Nanbeige 4.3's template, `visible_text(m).startswith(...)`,
// aborted kapitsa serve at load), so chat_template_source puts such calls in parentheses. CPU only, no model.
void paren_call_postfix_unit() {
  const std::string macro = "{%- macro f(x) %}{{ x }}{% endmacro %}";
  auto render = [&](const std::string& src) {
    try {
      auto t = minja::Parser::parse(paren_call_postfix(src), {});
      auto ctx = minja::Context::make(minja::Value(nlohmann::ordered_json{{"s", "<a>x</a>"}}));
      return t->render(ctx);
    } catch (const std::exception& e) {
      return std::string("error: ") + e.what();
    }
  };
  // the Nanbeige 4.3 template's line, cut down
  check(render(macro + "{{ s is string and not(f(s).startswith('<a>') and f(s).endswith('</a>')) }}") == "False",
        "paren_call_postfix: not(f(s).startswith(..) and f(s).endswith(..)) renders as Jinja does");
  check(render(macro + "{{ f(f(s)[1:3]).upper() }}") == "A>", "paren_call_postfix: nested calls with a subscript");
  // left as written: keywords, method calls, filters, strings, comments and text outside the tags
  for (const std::string& same :
       {std::string("{{ not(s).startswith('<') }}"), std::string("{{ s.upper().lower()[0] }}"),
        std::string("{{ s|replace('a', 'b')|length }}"), std::string("{{ 'f(x).y' }}{# f(x).y #} f(x).y"),
        std::string("{% if (s) %}{% for c in range(2) %}{{ c }}{% endfor %}{% endif %}")})
    check(paren_call_postfix(same) == same, "paren_call_postfix: unchanged: " + same);
  check(paren_call_postfix(macro + "{{ f(s).y }}") == macro + "{{ (f(s)).y }}", "paren_call_postfix: the rewrite");
}

// A folder's template as written: what chat_template_source reads (chat_template.jinja, else tokenizer_config.json's
// chat_template string), before its rewrite
std::string template_as_written(const std::string& dir) {
  auto slurp = [](const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), {});
  };
  std::string src = slurp(dir + "/chat_template.jinja");
  if (src.empty()) {
    auto cfg = nlohmann::json::parse(slurp(dir + "/tokenizer_config.json"), nullptr, false);
    if (cfg.is_object() && cfg.contains("chat_template") && cfg["chat_template"].is_string())
      src = cfg["chat_template"].get<std::string>();
  }
  return src;
}

// Every chat template kapitsa loads, through chat_template_source's rewrite: each catalog variant's template at its
// pinned revision, the model folders the fixture was made from (tests/data/chat_templates.json, regenerated by
// tests/chat_templates_fixture.py), and the model folders on this Mac. The rewrite hands minja each of them exactly as
// written, except the template it exists for, Nanbeige 4.3's, whose two rewritten calls are spelled out here. A catalog
// variant without a fixture entry fails, so a model added to the catalog cannot skip the sweep.
void chat_template_sweep(std::vector<std::string> dirs) {
  const std::string here = __FILE__;
  const std::string path = here.substr(0, here.rfind('/')) + "/../tests/data/chat_templates.json";
  std::ifstream f(path, std::ios::binary);
  const auto fx = nlohmann::json::parse(std::string(std::istreambuf_iterator<char>(f), {}), nullptr, false);
  const bool read = fx.is_object() && fx.contains("sources") && fx.contains("templates");
  check(read, "chat templates: " + path + " read");
  if (!read) return;

  // Nanbeige 4.3's template (fcmeyer/Nanbeige4.3-33B-MLX-4bit's chat_template.jinja), by sha256, and the exact rewrite
  // of its two calls
  const std::string nanbeige43 = "f2b1e2a1be25d277e8c903a2aa4d69ca5fd8a0fcb37d4792088e72426df5b91c";
  const std::vector<std::pair<std::string, std::string>> nanbeige43_calls = {
      {"not(visible_text(message.content).startswith(", "not((visible_text(message.content)).startswith("},
      {" and visible_text(message.content).endswith(", " and (visible_text(message.content)).endswith("}};
  auto expected = [&](const std::string& t) {
    if (bench::sha256_hex(t) != nanbeige43) return t;
    std::string want = t;
    for (const auto& [from, to] : nanbeige43_calls) {
      const size_t at = want.find(from);
      check(at != std::string::npos && want.find(from, at + 1) == std::string::npos,
            "chat templates: Nanbeige 4.3's template has `" + from + "` once");
      if (at != std::string::npos) want.replace(at, from.size(), to);
    }
    return want;
  };

  int nanbeige_sources = 0;
  for (const auto& [key, sha] : fx["sources"].items()) {
    const std::string h = sha.get<std::string>();
    const std::string t = fx["templates"].contains(h) ? fx["templates"][h].get<std::string>() : std::string();
    check(bench::sha256_hex(t) == h, "chat templates: " + key + "'s text has its sha256");
    nanbeige_sources += h == nanbeige43;
    check(paren_call_postfix(t) == expected(t),
          "chat template " + key +
              (h == nanbeige43 ? ": the rewrite parenthesizes Nanbeige 4.3's two calls, nothing else"
                               : ": the rewrite leaves it byte-for-byte as written"));
  }
  check(nanbeige_sources > 0, "chat templates: the fixture has Nanbeige 4.3's template");

  int variants = 0;
  for (const auto& m : hub::catalog())
    for (const auto& v : m.variants) {
      ++variants;
      const std::string key = "hf:" + v.repo + "@" + v.revision + (v.gguf.path.empty() ? "" : "/" + v.gguf.path);
      check(fx["sources"].contains(key), "chat templates: the fixture has " + m.name + ":" + v.name + "'s (" + key + ")",
            "rerun python3 tests/chat_templates_fixture.py");
    }

  // this Mac's model folders: the ones given, and every folder in kapitsa's and $NACE_MODELS' model directories
  const char* home = getenv("HOME");
  const char* nace = getenv("NACE_MODELS");
  for (const std::string root : {std::string(home ? home : "") + "/.cache/kapitsa/models", std::string(nace ? nace : "")}) {
    std::error_code ec;
    if (!root.empty())
      for (const auto& e : std::filesystem::directory_iterator(root, ec))
        if (e.is_directory(ec)) dirs.push_back(e.path().string());
  }
  int local = 0;
  for (const auto& dir : dirs) {
    const std::string t = template_as_written(dir);
    if (t.empty()) continue;
    ++local;
    check(chat_template_source(dir) == expected(t),
          "chat template " + dir + ": chat_template_source returns it as written" +
              (bench::sha256_hex(t) == nanbeige43 ? ", Nanbeige 4.3's two calls parenthesized" : ""));
  }
  printf("chat templates: %zu fixture sources (%zu distinct texts), %d catalog variants, %d in this Mac's folders\n",
         fx["sources"].size(), fx["templates"].size(), variants, local);
}

// A folder whose config.json is an architecture Kapitsa does not run is refused by unsupported_model(), the function
// the engine's constructor, model_plan and `kapitsa serve|bench|doctor` all refuse with (issue 447: Gemma 4 fell
// through to the Qwen3.5 loader and aborted in nlohmann instead of a message). CPU only: config.json files, no model.
void unsupported_model_unit() {
  char tmpl[] = "/tmp/kap-api-unsupported-XXXXXX";
  const char* root = mkdtemp(tmpl);
  check(root != nullptr, "unsupported_model: a temporary folder for the config.json files");
  if (!root) return;
  auto folder = [&](const std::string& name, const std::string& config) {
    const std::string dir = std::string(root) + "/" + name;
    mkdir(dir.c_str(), 0700);
    if (FILE* f = fopen((dir + "/config.json").c_str(), "w")) {
      fwrite(config.data(), 1, config.size(), f);
      fclose(f);
    }
    return dir;
  };
  auto refused = [&](const std::string& dir) { return unsupported_model(dir); };
  // the reporter's config.json (mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned), cut to what is read
  const std::string gemma = refused(folder("gemma4", R"({"architectures": ["Gemma4ForConditionalGeneration"],
      "model_type": "gemma4", "text_config": {"model_type": "gemma4_text", "hidden_size": 2816,
      "num_hidden_layers": 30, "num_attention_heads": 16, "vocab_size": 262144},
      "quantization": {"group_size": 32, "bits": 4, "mode": "affine"}})"));
  check(gemma.find("is a gemma4 model") != std::string::npos && gemma.find("qwen3_5") != std::string::npos &&
            gemma.find("nemotron_h") != std::string::npos && gemma.find("flint") != std::string::npos,
        "unsupported_model: another architecture is named by its model_type, with the ones Kapitsa runs");
  // the Qwen3.5 family is accepted by its model_type, and by the hybrid's shape when the config names none
  check(refused(folder("qwen35", R"({"model_type": "qwen3_5"})")).empty(),
        "unsupported_model: a qwen3_5 config is not refused");
  check(refused(folder("qwen35moe", R"({"model_type": "qwen3_5_moe"})")).empty(),
        "unsupported_model: a qwen3_5_moe config is not refused");
  check(refused(folder("qwen35text", R"({"text_config": {"model_type": "qwen3_5_moe_text"}})")).empty(),
        "unsupported_model: a qwen3_5_moe_text text_config is not refused");
  check(refused(folder("hybrid", R"({"linear_num_value_heads": 8, "num_hidden_layers": 4})")).empty(),
        "unsupported_model: a config with no model_type and the hybrid's shape is not refused");
  // a drafter's config is another architecture here: it is refused, with its type named
  check(refused(folder("drafter", R"({"model_type": "qwen3"})")).find("is a qwen3 model") != std::string::npos,
        "unsupported_model: a qwen3 drafter config is refused by its model_type");
  // no model_type at all, and a config.json that is not JSON: each says so instead of blaming an architecture
  check(refused(folder("notype", R"({"hidden_size": 2816})")).find("names no model_type") != std::string::npos,
        "unsupported_model: a config that names no model_type says that");
  check(refused(folder("badjson", "{ not json")).find("not valid JSON") != std::string::npos,
        "unsupported_model: a config.json that is not JSON says that");
  // a folder with no config.json is not refused here: it is refused where the folder is resolved
  const std::string noconfig = std::string(root) + "/noconfig";
  mkdir(noconfig.c_str(), 0700);
  check(refused(noconfig).empty(), "unsupported_model: a folder with no config.json returns nothing");
}

bool exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

}  // namespace

// a dense Llama's config.json (Nanbeige4.3's): what Kapitsa runs, and the loader's refusals that `kapitsa pull` gives too
void llama_config_unit() {
  const std::string ok = R"({"model_type": "llama", "hidden_size": 7168, "intermediate_size": 22016,
    "num_hidden_layers": 52, "num_attention_heads": 56, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 166912,
    "rope_theta": 120000000, "rope_scaling": null, "hidden_act": "silu", "tie_word_embeddings": false})";
  check(Flint::llama_unsupported(ok).empty(), "llama config: Nanbeige4.3's runs", Flint::llama_unsupported(ok));
  auto refused = [&](const std::string& from, const std::string& to, const std::string& want) {
    std::string j = ok;
    j.replace(j.find(from), from.size(), to);
    const std::string why = Flint::llama_unsupported(j);
    check(why.find(want) != std::string::npos, "llama config: refused, " + want, why);
  };
  refused(R"("rope_scaling": null)", R"("rope_scaling": {"rope_type": "llama3", "factor": 8.0})", "rope_scaling");
  refused(R"("tie_word_embeddings": false)", R"("tie_word_embeddings": true)", "tied embeddings");
  refused(R"("hidden_act": "silu")", R"("hidden_act": "gelu")", "activation gelu");
  refused(R"("head_dim": 128)", R"("head_dim": 64)", "shape");
  refused(R"("num_key_value_heads": 8)", R"("num_key_value_heads": 4)", "shape");  // 14 query heads per KV head
  refused(R"("tie_word_embeddings": false)", R"("tie_word_embeddings": false, "quantization": {"bits": 3, "group_size": 64})",
          "3-bit group 64");
  check(!Flint::llama_unsupported("{not json").empty(), "llama config: a broken config.json is refused");
  // the chat template: ChatML runs (Nanbeige4.3's), Llama 3's header markup is refused, and so is none
  check(Flint::llama_template_unsupported("{{ '<|im_start|>' + message['role'] + '\\n' }}").empty(),
        "llama template: ChatML runs");
  check(Flint::llama_template_unsupported("{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}")
                .find("ChatML") != std::string::npos,
        "llama template: Llama 3's is refused");
  check(!Flint::llama_template_unsupported("").empty(), "llama template: none is refused");
}

int main(int argc, char** argv) {
  anthropic_mapping();
  responses_mapping();
  anthropic_events();
  responses_events();
  chat_events();
  clients();
  reasoning_memory_unit();
  float_arguments_text();
  tool_arguments_spelling();
  disk_wait_unit();
  disk_saved_unit();
  disk_retry_wiring_unit();
  history_gate_unit();
  kflash_lookup_unit();
  unsupported_model_unit();
  llama_config_unit();
  paren_call_postfix_unit();
  std::vector<std::string> dirs;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--dump" && i + 1 < argc) dump_cases(argv[++i]);
    else dirs.push_back(argv[i]);
  }
  chat_template_sweep(dirs);
  if (dirs.empty()) {
    const char* home = getenv("HOME");
    const std::string base = std::string(home ? home : "") + "/.cache/kapitsa/models/";
    for (const char* d : {"qwen3.6-35b-a3b-4bit", "qwen3.8-27b-4bit"})
      if (exists(base + d + "/tokenizer.json")) dirs.push_back(base + d);
  }
  if (dirs.empty()) printf("SKIP round trips through a chat template: no model folder (api_test <model_dir>)\n");
  for (const auto& dir : dirs) {
    TmplModel m;
    m.dir = dir;
    m.tok.load(dir + "/tokenizer.json");
    // (a dense Llama on Flint's backend, Nanbeige4.3: its template rewritten for minja, as the engine renders it)
    std::string src = chat_template_source(dir);
    if (Flint::is_model(dir)) src = flint_template(src);
    m.tmpl = std::make_unique<minja::chat_template>(src, "", "<|im_end|>");
    const std::string tag = "[" + dir.substr(dir.find_last_of('/') + 1) + "] ";
    anthropic_round_trip(m, tag);
    reasoning_memory(m, tag);
    responses_round_trip(m, tag, false);
    responses_round_trip(m, tag, true);
    openai_tool_arguments(m, tag);
    anthropic_tool_arguments(m, tag);
    template_checks(m, tag);
    draft_checks(m, tag);
  }
  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
