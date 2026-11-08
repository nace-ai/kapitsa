// Portable adaptation of the measured fixed-seed client; worker and machine probes omitted.
#include "bench/bench.h"

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  if (!ni::bench::GpuLock::held_by_caller()) {
    std::cerr << "Hold the exclusive GPU lock across server and client, then set a caller-held marker.\n";
    return 2;
  }
  ni::bench::Settings settings;
  settings.host = argv[1];
  settings.port = std::stoi(argv[2]);
  settings.model = argv[3];
  settings.decode_runs = settings.greedy_runs = 3;
  settings.sizes = {4000};
  settings.ref_prompt_tokens = ni::bench::reference_prompt_tokens("nemotron-h");
  ni::bench::Nonces nonces;
  nonces.fixed = true;
  ni::bench::ojson output = {{"format", "fixed-request fixed-seed measurement"},
                            {"cohort_seeds", {1, 2, 3}},
                            {"rows", ni::bench::ojson::array()}};
  try {
    output["warmup"] = ni::bench::warmup(settings, nonces);
    if (!output["warmup"]["ok"].get<bool>()) throw std::runtime_error("warmup failed");
    for (bool greedy : {false, true}) {
      for (int seed : {1, 2, 3}) {
        std::string nonce = greedy ? "0.20000" : "0.10000";
        nonce += std::to_string(seed);
        auto body = ni::bench::body_of(settings, ni::bench::decode_messages(nonce), 512,
                                      greedy, false, {{"seed", seed}});
        body["stream"] = true;
        body["stream_options"] = {{"include_usage", true}};
        const auto response = ni::bench::stream_chat(settings, body, "", 1800, true);
        const int tokens = response.usage.value("completion_tokens", 0);
        const auto rate = ni::bench::decode_rates(response.ev_t, response.ev_c, tokens,
                                                 settings.window_head, settings.window_tail);
        const auto d4 = ni::bench::distinct4(response.content);
        auto invalid = ni::bench::ojson::array();
        auto flags = ni::bench::ojson::array();
        if (response.content.empty()) invalid.push_back("empty generated content");
        if (!rate.mid) invalid.push_back("too few streamed events for middle-window rate");
        if (tokens != 512) flags.push_back("completion token count differs from requested 512");
        if (response.finish != "length") flags.push_back("finish reason differs from length");
        if (d4 && *d4 < settings.rep_threshold) flags.push_back("repetition distinct-4 below 0.5");
        for (const auto& why : ni::bench::thinking_leak(response)) invalid.push_back(why);
        ni::bench::ojson row = {{"greedy", greedy}, {"seed", seed}, {"nonce", nonce},
                               {"request_sha256", ni::bench::sha256_hex(body.dump())},
                               {"content_sha256", ni::bench::sha256_hex(response.content)},
                               {"content_nonempty", !response.content.empty()},
                               {"completion_tokens", tokens}, {"finish_reason", response.finish},
                               {"chunks", response.n_chunks}, {"ttft_s", response.ttft},
                               {"total_s", response.total}, {"flags", flags}, {"invalid", invalid}};
        row["middle_window_tps"] = rate.mid ? ni::bench::ojson(*rate.mid) : ni::bench::ojson(nullptr);
        row["rate_method"] = rate.info;
        row["e2e_tps"] = rate.e2e ? ni::bench::ojson(*rate.e2e) : ni::bench::ojson(nullptr);
        row["distinct4"] = d4 ? ni::bench::ojson(*d4) : ni::bench::ojson(nullptr);
        output["rows"].push_back(row);
        std::cerr << "seed " << seed << " greedy=" << greedy << " tokens=" << tokens
                  << " finish=" << response.finish << " flags=" << flags.dump()
                  << " invalid=" << invalid.dump() << '\n';
      }
    }
    settings.temperature = 0;
    settings.top_p = 1;
    settings.top_k = 1;
    output["prefill_greedy"] = ni::bench::t_prefill(settings, 4000, nonces);
    std::cout << output.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    output["error"] = error.what();
    std::cout << output.dump(2) << '\n';
    return 1;
  }
}
