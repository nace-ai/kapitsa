// Drex systemone on Kapitsa: the request encoder, the answer formatter and the pointer head. See systemone.h.
//
// The encoder and the formatting follow the reference implementation exactly (kev/api.py and kev/model.py, Apache-2.0,
// and the llama.cpp fork's tools/server/systemone.cpp, Apache-2.0): the same text rendering, the same delimiters, the
// same truncation and the same answer shapes, so that the same weights give the same probabilities as the reference
// server (up to floating point noise and the quantization).
#include "systemone/systemone.h"

#include "util/bf16.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace ni::systemone {
namespace {

const char* kDelims[] = {
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
};

std::string render_value(const ojson& value, int depth);

// A request is caller JSON and the renderer recurses once per array or object nesting level: a body of 64 MiB (the
// server's cap) can nest hundreds of thousands deep, which overflows the render thread's stack (a secondary thread on
// macOS gets 512 KB). The reference renders a handful of levels, so a depth bound costs nothing and rejects it as a 400.
constexpr int kMaxRenderDepth = 64;

// Caller text can never produce a delimiter: `<|name|>` becomes `<¦name¦>` before tokenizing (the tokenizer matches
// added tokens wherever they appear, so the shield is what makes option boundaries unforgeable).
std::string shield_specials(const std::string& text) {
  static const std::regex re("<\\|([A-Za-z0-9_]+)\\|>");
  return std::regex_replace(text, re, "<\xC2\xA6$1\xC2\xA6>");
}

// Python's str(float): shortest round-trip digits, fixed notation for exponents -4 .. 15, else d[.ddd]e±nn.
std::string render_float(double value) {
  if (!std::isfinite(value)) throw BadRequest("numbers must be finite");
  std::array<char, 64> buffer{};
  const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), std::fabs(value), std::chars_format::scientific);
  if (converted.ec != std::errc()) throw BadRequest("could not render number");
  const std::string scientific(buffer.data(), converted.ptr);
  const size_t e = scientific.find('e');
  const int exponent = std::stoi(scientific.substr(e + 1));
  std::string digits;
  for (size_t i = 0; i < e; ++i)
    if (scientific[i] != '.') digits += scientific[i];
  std::string out = std::signbit(value) ? "-" : "";
  if (exponent >= -4 && exponent < 16) {
    const int point = exponent + 1;
    if (point <= 0) {
      out += "0." + std::string(-point, '0') + digits;
    } else if (point >= (int) digits.size()) {
      out += digits + std::string(point - digits.size(), '0') + ".0";
    } else {
      out += digits.substr(0, point) + "." + digits.substr(point);
    }
  } else {
    out += digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    std::string power = std::to_string(std::abs(exponent));
    if (power.size() < 2) power.insert(0, 1, '0');
    out += exponent < 0 ? "e-" : "e+";
    out += power;
  }
  return out;
}

std::string render_scalar(const ojson& value) {
  if (value.is_null()) return "";
  if (value.is_string()) return value.get<std::string>();
  if (value.is_boolean()) return value.get<bool>() ? "True" : "False";
  if (value.is_number_integer()) return value.dump();
  if (value.is_number()) return render_float(value.get<double>());
  return render_value(value, 0);
}

// str | object | array flattened into the text the model sees, field names kept as labels (kev/api.py's render).
std::string render_value(const ojson& value, int depth) {
  if (depth >= kMaxRenderDepth) {
    throw BadRequest("the request nests values more than " + std::to_string(kMaxRenderDepth) + " levels deep");
  }
  const std::string pad(depth * 2, ' ');
  if (value.is_array()) {
    std::string out;
    for (size_t i = 0; i < value.size(); ++i) {
      if (i) out += '\n';
      std::string child = render_value(value.at(i), depth + 1);
      size_t cut = 0;
      while (cut < child.size() && (child[cut] == ' ' || child[cut] == '\t')) ++cut;
      out += pad + "- " + child.substr(cut);
    }
    return out;
  }
  if (value.is_object()) {
    std::string out;
    bool first = true;
    for (const auto& [key, child] : value.items()) {
      if (!first) out += '\n';
      first = false;
      if (child.is_object() || child.is_array()) {
        out += pad + key + ":\n" + render_value(child, depth + 1);
      } else {
        out += pad + key + ": " + render_scalar(child);
      }
    }
    return out;
  }
  return render_scalar(value);
}

int one_special(const Tokenizer& tok, const char* text) {
  const std::vector<int> ids = tok.encode(text);
  if (ids.size() != 1 || tok.find(text) != ids[0]) {
    throw BadRequest(std::string("tokenizer has no single token for the delimiter ") + text);
  }
  return ids[0];
}

std::vector<int32_t> user_tokens(const Tokenizer& tok, const std::string& text) {
  if (text.empty()) return {};
  std::vector<int32_t> out;
  for (int id : tok.encode(shield_specials(text))) out.push_back(id);
  return out;
}

std::string option_text(const std::string& name, const ojson* desc) {
  if (desc == nullptr || desc->is_null() || (desc->is_string() && desc->get<std::string>().empty())) return name;
  return name + ": " + render_value(*desc, 0);
}

void validate_request(const ojson& body) {
  if (body.is_object() && body.contains("requests")) {
    throw BadRequest("requests batch wrapper is not supported here; send one state and questions object");
  }
  if (!body.is_object() || !body.contains("state") || !body.contains("questions") || !body.at("questions").is_object()) {
    throw BadRequest("request needs state and questions");
  }
  if (body.at("questions").size() == 0) throw BadRequest("request needs at least one question");
  for (const auto& [qid, question] : body.at("questions").items()) {
    (void) qid;
    if (!question.is_object() || !question.contains("type") || !question.at("type").is_string()) {
      throw BadRequest("each question needs a type");
    }
    if (question.at("type").get<std::string>() == "noul" && question.contains("criteria")) {
      const auto& criteria = question.at("criteria");
      if (!criteria.is_null() && !criteria.is_object()) throw BadRequest("noul criteria must be an object or null");
    }
  }
}

void append_py_string(std::string& out, const std::string& text);
void append_py_json(std::string& out, const ojson& value);

// Python json.dumps(value, ensure_ascii=False) with the default separators, for prompt text.
void dumps_string(std::string& out, const std::string& text) {
  append_py_string(out, text);
}

void dumps(std::string& out, const ojson& value) {
  append_py_json(out, value);
}

}  // namespace

Limits limits_from_env() {
  auto read_max = [](const char* name, int fallback) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return fallback;
    char* end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 1 || value > kPositionLimit) {
      std::fprintf(stderr, "%s=%s is invalid or above %d, using %d\n", name, env, kPositionLimit, fallback);
      return fallback;
    }
    return (int) value;
  };
  auto limit = [&](const char* specific) {
    const char* env = std::getenv(specific);
    if (env != nullptr && env[0] != '\0') return read_max(specific, kRecommendedContext);
    return read_max("SYSTEMONE_CONTEXT", kRecommendedContext);
  };
  Limits out;
  out.max_state = limit("SYSTEMONE_MAX_STATE");
  out.max_branch = limit("SYSTEMONE_MAX_BRANCH");
  out.max_packed = limit("SYSTEMONE_MAX_PACKED");
  return out;
}

size_t Encoded::max_branch() const {
  size_t m = 0;
  for (const Branch& b : branches) m = std::max(m, b.tokens.size());
  return m;
}

size_t Encoded::total_tokens() const {
  size_t n = state.size();
  for (const Branch& b : branches) n += b.tokens.size();
  return n;
}

void check_fits(const Encoded& enc, int context) {
  const int n_state = enc.n_state();
  const int row = n_state + (int) enc.max_branch();
  if (n_state >= context) {
    throw BadRequest("state is " + std::to_string(n_state) + " tokens and the context's last usable position is " +
                     std::to_string(context - 1) + " (raise --context)");
  }
  if (row >= context) {
    throw BadRequest("a question row ends at position " + std::to_string(row) + " with the state and the context's " +
                     "last usable position is " + std::to_string(context - 1) + " (raise --context)");
  }
}

Encoded encode(const Tokenizer& tok, const ojson& body, const Limits& limits) {
  validate_request(body);
  int delim[5];
  for (int i = 0; i < 5; ++i) delim[i] = one_special(tok, kDelims[i]);

  const std::vector<int32_t> state_tokens = user_tokens(tok, render_value(body.at("state"), 0));
  if ((int) state_tokens.size() + 1 > limits.max_state) {
    throw BadRequest("state exceeds " + std::to_string(limits.max_state) + " tokens");
  }

  Encoded out;
  out.state.push_back((int32_t) delim[0]);
  out.state.insert(out.state.end(), state_tokens.begin(),
                   state_tokens.begin() + std::min<size_t>(state_tokens.size(), size_t(limits.max_state - 1)));
  const int32_t n_state = (int32_t) out.state.size();

  size_t packed = out.state.size();
  for (const auto& [qid, question] : body.at("questions").items()) {
    const std::string type = question.at("type").get<std::string>();
    QuestionMeta meta;
    meta.id = qid;
    meta.type = type;

    std::vector<std::string> options;
    if (type == "noul") {
      const ojson* criteria = question.contains("criteria") ? &question.at("criteria") : nullptr;
      const ojson* no = nullptr;
      const ojson* yes = nullptr;
      if (criteria && criteria->is_object()) {
        if (criteria->contains("false")) no = &criteria->at("false");
        if (criteria->contains("true")) yes = &criteria->at("true");
      }
      options.push_back(option_text("no", no));
      options.push_back(option_text("yes", yes));
      meta.keys = {"false", "true"};
    } else if (type == "choice") {
      if (!question.contains("criteria") || !question.at("criteria").is_object() || question.at("criteria").size() == 0) {
        throw BadRequest("choice needs criteria");
      }
      for (const auto& [name, desc] : question.at("criteria").items()) {
        options.push_back(option_text(name, &desc));
        meta.keys.push_back(name);
      }
    } else if (type == "score") {
      if (!question.contains("criteria") || !question.at("criteria").is_array() || question.at("criteria").size() == 0) {
        throw BadRequest("score needs criteria");
      }
      for (size_t i = 0; i < question.at("criteria").size(); ++i) {
        const std::string text = render_value(question.at("criteria").at(i), 0);
        options.push_back(text);
        meta.keys.push_back(std::to_string(i));
        meta.legend.push_back(text);
      }
    } else {
      throw BadRequest("question type must be choice, noul, or score");
    }
    if (options.empty() || (int) options.size() > kMaxOptions) throw BadRequest("a question needs 1 to 255 options");

    std::vector<int32_t> branch;
    branch.push_back((int32_t) delim[1]);
    const std::vector<int32_t> instr =
        user_tokens(tok, render_value(question.contains("instructions") ? question.at("instructions") : ojson(nullptr), 0));
    branch.insert(branch.end(), instr.begin(), instr.end());
    std::vector<int> ends;
    for (const std::string& option : options) {
      branch.push_back((int32_t) delim[2]);
      const std::vector<int32_t> words = user_tokens(tok, option);
      branch.insert(branch.end(), words.begin(), words.end());
      branch.push_back((int32_t) delim[3]);
      ends.push_back((int) branch.size() - 1);
    }
    branch.push_back((int32_t) delim[4]);
    if ((int) branch.size() > limits.max_branch - (int) n_state) {
      throw BadRequest("question branch exceeds the " + std::to_string(limits.max_branch) + "-token row limit");
    }
    packed += branch.size();
    if ((int) packed > limits.max_packed) {
      throw BadRequest("request exceeds the " + std::to_string(limits.max_packed) + "-token packed limit");
    }

    Branch b;
    b.group.decide = (int) branch.size() - 1;
    b.group.options = ends;
    b.tokens = std::move(branch);
    out.branches.push_back(std::move(b));
    out.meta.push_back(std::move(meta));
  }
  return out;
}

ojson format_answers(const std::vector<std::vector<float>>& logits, const std::vector<QuestionMeta>& meta) {
  if (logits.size() != meta.size()) throw std::runtime_error("pointer readout did not return one group per question");
  auto round4 = [](double value) { return std::round(value * 10000.0) / 10000.0; };
  ojson answers = ojson::object();
  for (size_t q = 0; q < meta.size(); ++q) {
    const std::vector<float>& row = logits[q];
    if (row.empty()) throw std::runtime_error("question has no options");
    double peak = row[0];
    for (float v : row) peak = std::max(peak, (double) v);
    std::vector<double> probs(row.size());
    double sum = 0.0;
    for (size_t i = 0; i < row.size(); ++i) {
      probs[i] = std::exp((double) row[i] - peak);
      sum += probs[i];
    }
    if (sum == 0.0) throw std::runtime_error("pointer logits could not be normalized");
    for (double& p : probs) p /= sum;
    if (probs.size() != meta[q].keys.size()) throw std::runtime_error("pointer readout width does not match the question");
    size_t best = 0;
    for (size_t i = 1; i < probs.size(); ++i)
      if (probs[i] > probs[best]) best = i;

    ojson item = ojson::object();
    item["type"] = meta[q].type;
    if (meta[q].type == "noul") {
      item["noul"] = round4(probs[1]);
    } else if (meta[q].type == "choice") {
      ojson dist = ojson::object();
      for (size_t i = 0; i < probs.size(); ++i) dist[meta[q].keys[i]] = round4(probs[i]);
      const double uniform = 1.0 / (double) probs.size();
      const double confidence = probs.size() == 1 ? 1.0 : (probs[best] - uniform) / (1.0 - uniform);
      item["choice"] = meta[q].keys[best];
      item["confidence"] = round4(confidence);
      item["probabilities"] = std::move(dist);
    } else {
      double score = 0.0, spread = 0.0;
      ojson dist = ojson::object();
      ojson legend = ojson::array();
      for (size_t i = 0; i < probs.size(); ++i) {
        score += (double) i * probs[i];
        spread += probs[i] * std::fabs((double) i - (double) best);
        dist[meta[q].keys[i]] = round4(probs[i]);
        if (i < meta[q].legend.size()) legend.push_back(meta[q].legend[i]);
      }
      const double confidence = probs.size() == 1 ? 1.0 : 1.0 - spread / (double) (probs.size() - 1);
      item["score"] = round4(score);
      item["legend"] = std::move(legend);
      item["probabilities"] = std::move(dist);
      item["confidence"] = round4(confidence);
    }
    answers[meta[q].id] = std::move(item);
  }
  return answers;
}

namespace {

std::string py_float(double value) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4f", value);
  std::string text(buf);
  const auto dot = text.find('.');
  if (dot == std::string::npos) return text + ".0";
  while (text.size() > dot + 2 && text.back() == '0') text.pop_back();
  return text;
}

void append_py_string(std::string& out, const std::string& text) {
  out.push_back('"');
  for (size_t i = 0; i < text.size();) {
    const unsigned char c = (unsigned char) text[i];
    if (c == '"') { out += "\\\""; ++i; continue; }
    if (c == '\\') { out += "\\\\"; ++i; continue; }
    if (c == '\b') { out += "\\b"; ++i; continue; }
    if (c == '\f') { out += "\\f"; ++i; continue; }
    if (c == '\n') { out += "\\n"; ++i; continue; }
    if (c == '\r') { out += "\\r"; ++i; continue; }
    if (c == '\t') { out += "\\t"; ++i; continue; }
    uint32_t cp = c;
    size_t n = 1;
    bool ok = true;
    if (c >= 0x80) {
      if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
        cp = ((uint32_t) (c & 0x1F) << 6) | ((unsigned char) text[i + 1] & 0x3F);
        n = 2;
        ok = (c & 0xFE) != 0xC0;
      } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
        cp = ((uint32_t) (c & 0x0F) << 12) | (((unsigned char) text[i + 1] & 0x3F) << 6) | ((unsigned char) text[i + 2] & 0x3F);
        n = 3;
        ok = cp >= 0x800;
      } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
        cp = ((uint32_t) (c & 0x07) << 18) | (((unsigned char) text[i + 1] & 0x3F) << 12) |
             (((unsigned char) text[i + 2] & 0x3F) << 6) | ((unsigned char) text[i + 3] & 0x3F);
        n = 4;
        ok = cp >= 0x10000 && cp <= 0x10FFFF;
      } else {
        ok = false;
      }
    }
    if (!ok) {
      out += "\\ufffd";
      ++i;
      continue;
    }
    if (cp < 0x20 || cp >= 0x80) {
      char esc[16];
      if (cp >= 0x10000) {
        const uint32_t u = cp - 0x10000;
        std::snprintf(esc, sizeof(esc), "\\u%04x\\u%04x", 0xD800 + (u >> 10), 0xDC00 + (u & 0x3FF));
      } else {
        std::snprintf(esc, sizeof(esc), "\\u%04x", cp);
      }
      out += esc;
    } else {
      out.push_back((char) cp);
    }
    i += n;
  }
  out.push_back('"');
}

void append_py_json(std::string& out, const ojson& value) {
  if (value.is_null()) {
    out += "null";
  } else if (value.is_boolean()) {
    out += value.get<bool>() ? "true" : "false";
  } else if (value.is_string()) {
    append_py_string(out, value.get<std::string>());
  } else if (value.is_number()) {
    out += py_float(value.get<double>());
  } else if (value.is_array()) {
    out.push_back('[');
    for (size_t i = 0; i < value.size(); ++i) {
      if (i) out += ", ";
      append_py_json(out, value.at(i));
    }
    out.push_back(']');
  } else if (value.is_object()) {
    out.push_back('{');
    bool first = true;
    for (const auto& [key, child] : value.items()) {
      if (!first) out += ", ";
      first = false;
      append_py_string(out, key);
      out += ": ";
      append_py_json(out, child);
    }
    out.push_back('}');
  } else {
    throw std::runtime_error("answer could not be serialized");
  }
}

}  // namespace

int output_tokens(const Tokenizer& tok, const ojson& answers) {
  std::string text;
  append_py_json(text, answers);
  return (int) tok.encode(text).size();
}

// ---- the pointer head ----

PointerHead PointerHead::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open the pointer head " + path);
  auto read = [&](void* dst, size_t bytes, const char* what) {
    f.read(static_cast<char*>(dst), std::streamsize(bytes));
    if ((size_t) f.gcount() != bytes) throw std::runtime_error(std::string("pointer head is truncated: ") + what);
  };
  char magic[4];
  read(magic, 4, "magic");
  if (std::memcmp(magic, "KPS1", 4) != 0) throw std::runtime_error("pointer head has no KPS1 magic: " + path);
  uint32_t version = 0, hidden = 0, dim = 0;
  read(&version, 4, "version");
  read(&hidden, 4, "hidden");
  read(&dim, 4, "dim");
  if (version != 1) throw std::runtime_error("pointer head version " + std::to_string(version) + " is not supported");
  if (hidden == 0 || dim == 0 || dim > 4096) throw std::runtime_error("pointer head has a bad shape");
  PointerHead head;
  head.hidden_ = (int) hidden;
  head.dim_ = (int) dim;
  head.q_w_.resize(size_t(hidden) * dim);
  head.q_b_.resize(dim);
  head.k_w_.resize(size_t(hidden) * dim);
  head.k_b_.resize(dim);
  read(head.q_w_.data(), head.q_w_.size() * 4, "q weights");
  read(head.q_b_.data(), head.q_b_.size() * 4, "q bias");
  read(head.k_w_.data(), head.k_w_.size() * 4, "k weights");
  read(head.k_b_.data(), head.k_b_.size() * 4, "k bias");
  read(&head.temperature_, 4, "temperature");
  if (!(head.temperature_ > 0.0f) || !std::isfinite(head.temperature_)) {
    throw std::runtime_error("pointer head temperature must be positive and finite");
  }
  size_t bad = 0;
  for (const std::vector<float>* v : {&head.q_w_, &head.q_b_, &head.k_w_, &head.k_b_})
    for (float x : *v)
      if (!std::isfinite(x)) ++bad;
  if (bad) throw std::runtime_error("pointer head has " + std::to_string(bad) + " non-finite weights");
  return head;
}

std::string PointerHead::find(const std::string& model_dir) {
  for (const std::string& candidate : {model_dir + "/pointer.bin", model_dir + "/.kapitsa/pointer.bin"}) {
    std::ifstream f(candidate, std::ios::binary);
    if (f.good()) return candidate;
  }
  return "";
}

void PointerHead::logits(const float* h_decide, const std::vector<const float*>& option_rows, std::vector<float>* out) const {
  const int H = hidden_, D = dim_;
  // q(h_decide) once, then one dot per option.
  std::vector<float> qv(size_t(D), 0.0f), kv(size_t(D), 0.0f);
  for (int d = 0; d < D; ++d) {
    const float* row = q_w_.data() + size_t(d) * H;
    float sum = q_b_[d];
    for (int i = 0; i < H; ++i) sum += row[i] * h_decide[i];
    qv[d] = sum;
  }
  const float scale = 1.0f / std::sqrt((float) D) / temperature_;
  out->assign(option_rows.size(), 0.0f);
  for (size_t o = 0; o < option_rows.size(); ++o) {
    const float* h = option_rows[o];
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
      const float* row = k_w_.data() + size_t(d) * H;
      float sum = k_b_[d];
      for (int i = 0; i < H; ++i) sum += row[i] * h[i];
      dot += sum * qv[d];
    }
    (*out)[o] = dot * scale;
  }
}

void final_norm_row(const float* x, const float* w, int hidden, float eps, std::vector<float>* out) {
  double ss = 0.0;
  for (int i = 0; i < hidden; ++i) ss += (double) x[i] * x[i];
  const float inv = 1.0f / std::sqrt((float) (ss / (double) hidden) + eps);
  out->resize(size_t(hidden));
  for (int i = 0; i < hidden; ++i) (*out)[i] = x[i] * inv * w[i];
}

}  // namespace ni::systemone
