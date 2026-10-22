// Streaming parser for Qwen3.5+ output: <think>...</think> reasoning, content, and tool calls in the
// template's <tool_call><function=...><parameter=...> XML, turned into OpenAI-style deltas; with CallFormat::Glm, calls
// in GLM-4.5's <tool_call>NAME<arg_key>...</arg_key><arg_value>...</arg_value></tool_call> (Flint's) instead. With
// TextTags, tags that are no tokens of the vocabulary (Nemotron Nano 9B v2's <think> and <TOOLCALL>[...] JSON calls)
// are matched in the decoded text.
#pragma once
#include "engine/engine.h"
#include "tokenizer/tokenizer.h"

#include <random>
#include <string>
#include <vector>

namespace ni {

inline std::string random_id(const char* prefix) {
  static std::mt19937_64 rng(std::random_device{}());
  static const char* al = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::string s = prefix;
  for (int i = 0; i < 24; ++i) s += al[rng() % 62];
  return s;
}

// The markup of a call between its <tool_call> and </tool_call>
enum class CallFormat {
  QwenXml,  // <function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function> (Qwen3.5+'s templates)
  Glm,      // NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>... (GLM-4.5's and 4.7's, Flint's)
};

class OutputParser {
 public:
  // think_tools (NI_THINK_TOOLCALLS): a <tool_call> the model writes inside its reasoning is held back
  // instead of streamed as reasoning. If the turn ends with it (nothing but whitespace, or </think> and
  // whitespace, follows) and it names one of the request's tools, finish() returns it as a tool call:
  // Qwen3.6 sometimes writes its call and stops without ever closing the think block. If the reasoning
  // goes on instead (a call drafted while thinking), the held tokens are replayed as reasoning, exactly
  // as without think_tools.
  // think_ws: the whitespace the model writes before </think> stays in the reasoning (Nemotron 3.5's template
  // renders a returned turn as "<think>\n" + reasoning + "</think>", so a reasoning that ended with the model's own
  // "\n" renders as generated and the next request reuses the cache through the reply; Qwen's templates add theirs)
  // text_tags: the tags matched in the text (think_tools is then off: it holds tokens)
  // call_format: the markup between a call's tags; a call not written in it stays the text the model wrote
  OutputParser(const Tokenizer& t, int think_open, int think_close, int tc_open, int tc_close, bool start_in_think,
               bool parse_tools, const ojson& tools, bool think_tools = false, bool think_ws = false,
               const TextTags& text_tags = {}, CallFormat call_format = CallFormat::QwenXml)
      : dec_(t), hdec_(t), tt_(text_tags), think_open_(think_open), think_close_(think_close), tc_open_(tc_open),
        tc_close_(tc_close), in_think_(start_in_think), parse_tools_(parse_tools),
        think_tools_(think_tools && parse_tools && !text_tags.any()), think_ws_(think_ws), fmt_(call_format),
        tools_(tools) {}

  // Returns the delta produced by this token.
  ChatDelta push(int id) {
    ChatDelta d;
    if (think_tools_ && (holding_ || (in_think_ && id == tc_open_))) hold(d, id);
    else push_plain(d, id);
    return d;
  }

  // End of the turn. `ended`: the model stopped by itself (not cut by max_tokens or the client), which
  // a held call needs to become the turn's tool call.
  ChatDelta finish(bool ended = true) {
    ChatDelta d;
    if (holding_) {
      std::vector<ToolCall> calls;
      if (ended && held_calls(calls)) {
        held_.clear();  // whitespace before the call
        d.tool_calls = std::move(calls);
        promoted_ += int(d.tool_calls.size());
        std::vector<int>().swap(hold_ids_);
        end_hold();
      } else {
        replay(d);
      }
    }
    std::string s = dec_.flush();
    if (tt_.any()) {  // the text held as a possible tag's start is text after all
      text_push(d, s);
      s = pend_;
      pend_.clear();
    }
    if (in_tool_) {
      // unterminated tool call: surface as text
      emit(d, (tt_.tc_open.empty() ? std::string("<tool_call>") : tt_.tc_open) + tool_buf_ + s);
      in_tool_ = false;
    } else {
      emit(d, s);
    }
    held_.clear();  // trailing whitespace
    return d;
  }

  bool in_think() const { return in_think_; }
  int promoted() const { return promoted_; }  // tool calls returned from the reasoning (think_tools)

 private:
  void push_plain(ChatDelta& d, int id) {
    if (id == think_open_) {
      flush_text(d);
      in_think_ = true;
      return;
    }
    if (id == think_close_) {
      flush_text(d);
      if (think_ws_ && in_think_) d.reasoning += held_;
      held_.clear();  // drop whitespace before </think> (think_ws: kept above)
      in_think_ = false;
      skip_leading_ws_ = true;
      return;
    }
    if (parse_tools_ && id == tc_open_ && !in_think_) {
      flush_text(d);
      held_.clear();
      in_tool_ = true;
      tool_buf_.clear();
      return;
    }
    if (parse_tools_ && id == tc_close_ && in_tool_) {
      tool_buf_ += dec_.flush();
      in_tool_ = false;
      ToolCall tc;
      if (parse_tool(tool_buf_, tc)) d.tool_calls.push_back(tc);
      else emit(d, "<tool_call>" + tool_buf_ + "</tool_call>");
      skip_leading_ws_ = true;
      return;
    }
    std::string s = dec_.push(id);
    if (tt_.any()) {
      text_push(d, s);
      return;
    }
    if (in_tool_) {
      tool_buf_ += s;
      return;
    }
    emit(d, s);
  }

  // TextTags: the decoded text s after the text held so far, split at the tags the parser's state looks for (the think
  // tags anywhere outside a call, a call's open tag outside the reasoning, its close tag inside a call), with the same
  // transitions as the tag tokens; a tail that may be a tag's start is held until the next text decides it.
  void text_push(ChatDelta& d, const std::string& s) {
    static const std::string topen = "<think>", tclose = "</think>";
    pend_ += s;
    while (!pend_.empty()) {
      const std::string* tags[3];
      int n = 0;
      if (in_tool_) {
        tags[n++] = &tt_.tc_close;
      } else {
        if (tt_.think) tags[n++] = &topen, tags[n++] = &tclose;
        if (parse_tools_ && !in_think_ && !tt_.tc_open.empty()) tags[n++] = &tt_.tc_open;
      }
      size_t at = std::string::npos;
      const std::string* tag = nullptr;
      for (int i = 0; i < n; ++i)
        if (size_t k = pend_.find(*tags[i]); k < at) at = k, tag = tags[i];
      if (!tag) {
        size_t keep = 0;  // the longest end of the text that begins a tag
        for (int i = 0; i < n; ++i)
          for (size_t m = std::min(pend_.size(), tags[i]->size() - 1); m > keep; --m)
            if (pend_.compare(pend_.size() - m, m, *tags[i], 0, m) == 0) {
              keep = m;
              break;
            }
        text_out(d, pend_.substr(0, pend_.size() - keep));
        pend_.erase(0, pend_.size() - keep);
        return;
      }
      text_out(d, pend_.substr(0, at));
      pend_.erase(0, at + tag->size());
      if (tag == &topen) {
        in_think_ = true;
      } else if (tag == &tclose) {
        if (think_ws_ && in_think_) d.reasoning += held_;
        held_.clear();  // drop whitespace before </think> (think_ws: kept above)
        in_think_ = false;
        skip_leading_ws_ = true;
      } else if (tag == &tt_.tc_open) {
        held_.clear();
        in_tool_ = true;
        tool_buf_.clear();
      } else {  // the call's close tag
        in_tool_ = false;
        std::vector<ToolCall> calls;
        if (tt_.json_calls ? parse_json_calls(tool_buf_, calls) : parse_tool(tool_buf_, calls.emplace_back()))
          for (auto& c : calls) d.tool_calls.push_back(std::move(c));
        else
          emit(d, tt_.tc_open + tool_buf_ + tt_.tc_close);
        skip_leading_ws_ = true;
      }
    }
  }
  void text_out(ChatDelta& d, const std::string& t) {
    if (in_tool_) tool_buf_ += t;
    else emit(d, t);
  }

  // A JSON list of calls (or one call): [{"name": "f", "arguments": {...}}, ...]; arguments may be an object or the
  // JSON text of one
  bool parse_json_calls(const std::string& body, std::vector<ToolCall>& calls) const {
    ojson j = ojson::parse(body, nullptr, false);
    if (j.is_discarded()) return false;
    if (j.is_object()) j = ojson::array({j});
    if (!j.is_array() || j.empty()) return false;
    for (const auto& c : j) {
      if (!c.is_object() || !c.contains("name") || !c["name"].is_string()) return false;
      ToolCall tc;
      tc.name = c["name"].get<std::string>();
      const ojson a = c.contains("arguments") ? c["arguments"] : ojson::object();
      if (a.is_string()) {
        const ojson pa = ojson::parse(a.get<std::string>(), nullptr, false);
        tc.arguments = !pa.is_discarded() && pa.is_object() ? pa.dump() : a.get<std::string>();
      } else {
        tc.arguments = a.dump();
      }
      tc.id = random_id("call_");
      calls.push_back(std::move(tc));
    }
    return true;
  }

  // think_tools: holds the tokens of a <tool_call> written inside the reasoning, and what follows it,
  // while the turn may still end with it. Anything else (more reasoning, a call after </think>, content,
  // malformed tags) replays the held tokens through push_plain, i.e. as reasoning.
  void hold(ChatDelta& d, int id) {
    if (!holding_) {
      flush_text(d);  // the reasoning before the call (its trailing whitespace stays held)
      holding_ = true;
      in_body_ = false;
      closed_ = false;
      bodies_.clear();
      body_.clear();
    }
    hold_ids_.push_back(id);
    bool goes_on = false;
    if (in_body_) {
      if (id == tc_close_) {
        body_ += hdec_.flush();
        bodies_.push_back(body_);
        body_.clear();
        in_body_ = false;
      } else if (id == tc_open_ || id == think_open_ || id == think_close_) {
        goes_on = true;
      } else {
        body_ += hdec_.push(id);
      }
    } else if (id == tc_open_) {
      if (closed_) goes_on = true;  // a call after </think>: the held ones were drafts
      else in_body_ = true;
    } else if (id == think_close_ && !closed_) {
      closed_ = true;
    } else if (id == think_open_ || id == think_close_ || id == tc_close_) {
      goes_on = true;
    } else if (hdec_.push(id).find_first_not_of(" \n\r\t") != std::string::npos) {
      goes_on = true;
    }
    if (goes_on) replay(d);
  }

  void replay(ChatDelta& d) {
    std::vector<int> ids;
    ids.swap(hold_ids_);
    end_hold();
    for (int i : ids) push_plain(d, i);
  }

  void end_hold() {
    holding_ = false;
    in_body_ = false;
    closed_ = false;
    bodies_.clear();
    body_.clear();
    hdec_.flush();
  }

  // The held calls, if all of them are complete, parse and name a tool of the request.
  bool held_calls(std::vector<ToolCall>& calls) const {
    if (in_body_ || bodies_.empty()) return false;
    for (auto& b : bodies_) {
      ToolCall tc;
      if (!parse_tool(b, tc) || !has_tool(tc.name)) return false;
      calls.push_back(tc);
    }
    return true;
  }

  bool has_tool(const std::string& name) const {
    for (auto& t : tools_) {
      const ojson& f = t.contains("function") ? t["function"] : t;
      if (f.value("name", "") == name) return true;
    }
    return false;
  }

  // Holds back trailing whitespace so it can be dropped before </think>, <tool_call> or the end.
  void emit(ChatDelta& d, const std::string& s) {
    if (s.empty()) return;
    std::string t = held_ + s;
    held_.clear();
    if (skip_leading_ws_) {
      size_t k = t.find_first_not_of(" \n\r\t");
      if (k == std::string::npos) return;
      t = t.substr(k);
      skip_leading_ws_ = false;
    }
    size_t e = t.find_last_not_of(" \n\r\t");
    if (e == std::string::npos) {
      held_ = t;
      return;
    }
    held_ = t.substr(e + 1);
    t.resize(e + 1);
    (in_think_ ? d.reasoning : d.content) += t;
  }
  void flush_text(ChatDelta& d) { emit(d, dec_.flush()); }

  static std::string trim_nl(const std::string& v) {
    size_t a = 0, b = v.size();
    if (a < b && v[a] == '\n') ++a;
    if (b > a && v[b - 1] == '\n') --b;
    return v.substr(a, b - a);
  }
  static std::string trim_ws(const std::string& v) {
    const size_t a = v.find_first_not_of(" \n\r\t");
    return a == std::string::npos ? std::string() : v.substr(a, v.find_last_not_of(" \n\r\t") - a + 1);
  }

  std::string param_type(const std::string& fn, const std::string& param) const {
    for (auto& t : tools_) {
      const ojson& f = t.contains("function") ? t["function"] : t;
      if (f.value("name", "") != fn) continue;
      if (f.contains("parameters") && f["parameters"].contains("properties") &&
          f["parameters"]["properties"].contains(param)) {
        const ojson& p = f["parameters"]["properties"][param];
        if (p.contains("type") && p["type"].is_string()) return p["type"];
        return "any";
      }
    }
    return "string";
  }

  // A value as the tool's schema types its parameter: a string's text as written, anything else as JSON (the text
  // itself when it is no JSON)
  ojson typed(const std::string& fn, const std::string& param, const std::string& val) const {
    if (param_type(fn, param) == "string") return val;
    ojson v = ojson::parse(val, nullptr, false);
    return v.is_discarded() ? ojson(val) : v;
  }

  // The body of a call (between its tags), in the parser's call format
  bool parse_tool(const std::string& body, ToolCall& tc) const {
    return fmt_ == CallFormat::Glm ? parse_glm(body, tc) : parse_qwen(body, tc);
  }

  bool parse_qwen(const std::string& body, ToolCall& tc) const {
    size_t f = body.find("<function=");
    if (f == std::string::npos) return false;
    size_t fe = body.find('>', f);
    if (fe == std::string::npos) return false;
    tc.name = body.substr(f + 10, fe - f - 10);
    ojson args = ojson::object();
    size_t p = fe + 1;
    while (true) {
      size_t ps = body.find("<parameter=", p);
      if (ps == std::string::npos) break;
      size_t pe = body.find('>', ps);
      if (pe == std::string::npos) break;
      std::string pname = body.substr(ps + 11, pe - ps - 11);
      size_t ve = body.find("</parameter>", pe);
      if (ve == std::string::npos) ve = body.find("</function>", pe);
      if (ve == std::string::npos) ve = body.size();
      args[pname] = typed(tc.name, pname, trim_nl(body.substr(pe + 1, ve - pe - 1)));
      p = ve + 1;
    }
    tc.arguments = args.dump();
    tc.id = random_id("call_");
    return true;
  }

  // GLM-4.5's call (Flint's template renders it): NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>..., a value
  // the text of a string or the JSON of anything else; no arguments, NAME alone. Whitespace may surround the name and a
  // key and separate the elements; a value is everything up to the first </arg_value>. Anything else (no name, a name
  // with whitespace or a tag in it, a tag in a key, text between the elements, a tag left open) is no call.
  bool parse_glm(const std::string& body, ToolCall& tc) const {
    static const std::string ko = "<arg_key>", kc = "</arg_key>", vo = "<arg_value>", vc = "</arg_value>";
    static const char* ws = " \n\r\t";
    constexpr size_t npos = std::string::npos;
    size_t p = body.find(ko);
    const std::string name = trim_ws(body.substr(0, p));
    if (name.empty() || name.find_first_of(" \n\r\t<") != npos) return false;
    ojson args = ojson::object();
    while (p != npos) {  // at an <arg_key>
      const size_t ke = body.find(kc, p + ko.size());
      if (ke == npos) return false;
      const std::string key = trim_ws(body.substr(p + ko.size(), ke - p - ko.size()));
      if (key.empty() || key.find(ko) != npos || key.find(vo) != npos || key.find(vc) != npos) return false;
      size_t vs = body.find_first_not_of(ws, ke + kc.size());
      if (vs == npos || body.compare(vs, vo.size(), vo) != 0) return false;
      vs += vo.size();
      const size_t ve = body.find(vc, vs);
      if (ve == npos) return false;
      args[key] = typed(name, key, body.substr(vs, ve - vs));
      p = body.find_first_not_of(ws, ve + vc.size());
      if (p != npos && body.compare(p, ko.size(), ko) != 0) return false;
    }
    tc.name = name;
    tc.arguments = args.dump();
    tc.id = random_id("call_");
    return true;
  }

  StreamDecoder dec_;
  StreamDecoder hdec_;  // text of held tokens (think_tools)
  TextTags tt_;
  std::string pend_;    // TextTags: decoded text that may begin a tag
  int think_open_, think_close_, tc_open_, tc_close_;
  bool in_think_;
  bool parse_tools_, think_tools_, think_ws_;
  CallFormat fmt_;
  const ojson& tools_;
  bool in_tool_ = false, skip_leading_ws_ = false;
  std::string tool_buf_, held_;
  // think_tools hold: token ids, complete call bodies, the body being read, </think> seen after them
  bool holding_ = false, in_body_ = false, closed_ = false;
  std::vector<int> hold_ids_;
  std::vector<std::string> bodies_;
  std::string body_;
  int promoted_ = 0;
};

} // namespace ni
