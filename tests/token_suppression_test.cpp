#include "model/token_suppression.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  using nlohmann::json;
  int fails = 0;
  auto check = [&](bool ok, const char* name) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++fails;
  };

  const auto from_obj =
      ni::token_suppression_mask(json{{"target_token_ids", {0, 31, 32, 64}}, {"n_target", 4}}, 65, {63});
  check(from_obj.count == 4 && from_obj.words == std::vector<uint32_t>{0x7ffffffeu, 0xfffffffeu, 0xfffffffeu},
        "target_token_ids JSON object clears exact bits across word boundaries");

  const auto from_arr = ni::token_suppression_mask(std::string(" [0, 31, 32, 64] "), 65, {63});
  check(from_arr.count == 4 && from_arr.words == from_obj.words,
        "plain JSON integer array matches target_token_ids mask");

  const auto from_txt = ni::token_suppression_mask(
      std::string("# comment\n\n  0\n31 # inline comment\n32\n64\n"), 65, {63});
  check(from_txt.count == 4 && from_txt.words == from_obj.words,
        "newline-separated integer text matches JSON mask");

  for (const auto& j : std::vector<json>{
           json::object(),
           json::array(),
           json{{"suppress_tokens", {2}}},
           json{{"target_token_ids", json::array()}},
           json{{"target_token_ids", {-1}}},
           json{{"target_token_ids", {65}}},
           json{{"target_token_ids", {1.5}}},
           json{{"target_token_ids", {"1"}}},
           json{{"target_token_ids", {true}}},
           json{{"target_token_ids", {1, 1}}},
           json{{"target_token_ids", {63}}},
           json{{"target_token_ids", {1}}, {"n_target", 2}},
           json{{"target_token_ids", {1}}, {"n_target", "1"}},
           json{-1},
           json{65},
           json{1, 1},
           json{63}}) {
    bool rejected = false;
    try {
      ni::token_suppression_mask(j, 65, {63});
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    check(rejected, ("reject invalid JSON mask " + j.dump()).c_str());
  }

  for (const std::string& bad : std::vector<std::string>{
           "", "   \n# only comments\n", "12abc\n", "1.5\n", "1 2\n", "-1\n", "65\n", "1\n1\n", "63\n"}) {
    bool rejected = false;
    try {
      ni::token_suppression_mask(bad, 65, {63});
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    check(rejected, ("reject invalid text mask " + bad).c_str());
  }

  bool rejected_all = false;
  try {
    ni::token_suppression_mask(json{0, 1}, 2, {});
  } catch (const std::runtime_error&) {
    rejected_all = true;
  }
  check(rejected_all, "reject suppressing the entire vocabulary");

  if (argc == 2) {
    std::ifstream f(argv[1]);
    std::stringstream ss;
    ss << f.rdbuf();
    const auto t1 = ni::token_suppression_mask(ss.str(), 248320, {});
    size_t cleared = 0;
    for (uint32_t w : t1.words) cleared += size_t(__builtin_popcount(~w));
    check(t1.count == 32211 && cleared == 32211, "external mask file count matches cleared bits");
  }

  printf("token suppression: %d failures\n", fails);
  return fails ? 1 : 0;
}
