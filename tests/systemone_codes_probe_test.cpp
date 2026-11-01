// Test-only Tokenizer definitions measure calls from the real encode_max.
// Real BPE equivalence is checked separately by test_systemone_codes.py.
#include "systemone/systemone.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
std::vector<std::string> probes;
std::set<std::string> accepted;
bool accept_all = true;
}

namespace ni {
std::vector<int> Tokenizer::encode(const std::string& text, size_t, int*) const {
  if (!text.empty() && text.size() <= 2 &&
      std::all_of(text.begin(), text.end(), [](char c) { return c >= 'A' && c <= 'Z'; })) {
    probes.push_back(text);
    return accept_all || accepted.count(text) ? std::vector<int>{1} : std::vector<int>{1, 2};
  }
  return {1, 2, 3};
}
int Tokenizer::find(const std::string&) const { return -1; }
}

using namespace ni::systemone;

void check(const ni::Tokenizer& tok, const std::vector<int>& widths,
           size_t expected_probes, bool should_reject = false) {
  ojson questions = ojson::object();
  for (size_t q = 0; q < widths.size(); ++q) {
    ojson criteria = ojson::object();
    for (int i = 0; i < widths[q]; ++i) criteria["key" + std::to_string(i)] = nullptr;
    questions[std::to_string(q)] = {{"type", "choice"}, {"criteria", criteria}};
  }
  probes.clear();
  bool rejected = false;
  try { (void)encode_max(tok, {{"state", "s"}, {"questions", questions}}, Limits{}); }
  catch (const BadRequest& e) {
    if (std::string(e.what()) != "a question needs 1 to 255 options") throw;
    rejected = true;
  }
  if (rejected != should_reject) throw std::runtime_error("wrong rejection status");
  if (probes.size() != expected_probes)
    throw std::runtime_error("probe count: expected " + std::to_string(expected_probes) +
                             ", got " + std::to_string(probes.size()));
  std::vector<std::string> order;
  for (char a = 'A'; a <= 'Z'; ++a) order.emplace_back(1, a);
  for (char a = 'A'; a <= 'Z'; ++a)
    for (char b = 'A'; b <= 'Z'; ++b) order.push_back(std::string{a, b});
  if (!std::equal(probes.begin(), probes.end(), order.begin()))
    throw std::runtime_error("candidate order or no-rescan invariant failed");
}

int main() {
  try {
    ni::Tokenizer first, second;
    check(first, {2}, 2);
    check(first, {2, 40, 3}, 40);
    check(first, {26, 27, 2}, 27);
    check(first, {255}, 255);
    check(first, {256}, 255, true);
    check(first, {2}, 2);
    check(second, {2}, 2);
    accept_all = false;
    accepted = {"B", "Z", "AB", "ZZ"};
    check(first, {2, 4, 3}, 702);
    check(first, {5}, 702, true);
    accepted = {"ZZ"};
    check(second, {1}, 702);
    accepted.clear();
    check(first, {1}, 702, true);
    std::cout << "11 production-encoder probe-count/order checks passed with test-only Tokenizer\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
