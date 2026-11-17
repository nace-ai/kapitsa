#include "bench/bench.h"

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  ni::bench::Settings settings;
  settings.port = std::stoi(argv[1]);
  settings.model = argv[2];
  const int server_pid = std::stoi(argv[3]);
  settings.decode_runs = settings.greedy_runs = 3;
  settings.sizes = {4000};
  settings.ref_prompt_tokens = ni::bench::reference_prompt_tokens("qwen3.5");
  ni::bench::Nonces nonces;
  nonces.fixed = true;
  const auto own = [&] { return ni::bench::process_tree({server_pid}); };
  const auto ambient = ni::bench::ancestors();
  ni::bench::Monitor monitor(own, ambient);
  const double monitoring_started = ni::bench::Monitor::now();
  monitor.start();
  ni::bench::ojson output = {{"format", "fixed-request fixed-seed measurement"},
                            {"cohort_seeds", {1, 2, 3}},
                            {"rows", ni::bench::ojson::array()}};
  try {
    output["warmup"] = ni::bench::warmup(settings, nonces);
    if (!output["warmup"]["ok"].get<bool>()) throw std::runtime_error("warmup failed");
    double waited = 0;
    std::vector<ni::bench::Busy> busy;
    output["quiet_gate"] = ni::bench::wait_quiet(5, 18, 300, own, ambient, &waited, &busy);
    if (!output["quiet_gate"].get<bool>()) throw std::runtime_error("GPU quiet gate failed");
    for (bool greedy : {false, true}) {
      for (int seed : {1, 2, 3}) {
        std::string nonce = greedy ? "0.20000" : "0.10000";
        nonce += std::to_string(seed);
        auto body = ni::bench::body_of(settings, ni::bench::decode_messages(nonce), 512,
                                      greedy, false, {{"seed", seed}});
        body["stream"] = true;
        body["stream_options"] = {{"include_usage", true}};
        const double phase_started = ni::bench::Monitor::now();
        const auto vm_before = ni::bench::vm();
        const auto response = ni::bench::stream_chat(settings, body, "", 1800, true);
        const auto vm_after = ni::bench::vm();
        const double phase_finished = ni::bench::Monitor::now();
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
        row["phase_started_private"] = phase_started;
        row["phase_finished_private"] = phase_finished;
        row["swapouts_delta_private"] = vm_before.swapouts >= 0 && vm_after.swapouts >= 0
                                            ? ni::bench::ojson(vm_after.swapouts - vm_before.swapouts)
                                            : ni::bench::ojson(nullptr);
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
    const double prefill_started = ni::bench::Monitor::now();
    const auto prefill_vm_before = ni::bench::vm();
    output["prefill_greedy"] = ni::bench::t_prefill(settings, 4000, nonces);
    const auto prefill_vm_after = ni::bench::vm();
    const double prefill_finished = ni::bench::Monitor::now();
    monitor.stop();
    const auto samples = monitor.samples();
    const auto apply_conditions = [&](ni::bench::ojson& row, double start, double end) {
      auto conditions = ni::bench::summarize_samples(samples, start, end);
      row["monitor_private"] = conditions;
      if (conditions.value("samples", 0) == 0) row["invalid"].push_back("no monitor samples for phase");
      if (conditions.value("flagged", false)) row["flags"].push_back("contention gate failed");
      if (conditions.contains("thermal") && ni::bench::thermal_rank(conditions["thermal"].get<std::string>()) >= 2)
        row["flags"].push_back("thermal gate failed");
      if (conditions.value("on_battery", false)) row["flags"].push_back("power source gate failed");
      if (conditions.value("low_power_mode", false)) row["flags"].push_back("low power mode gate failed");
      if (row["swapouts_delta_private"].is_number() && row["swapouts_delta_private"].get<int64_t>() > 1000)
        row["flags"].push_back("swap activity gate failed");
    };
    for (auto& row : output["rows"])
      apply_conditions(row, row["phase_started_private"].get<double>(), row["phase_finished_private"].get<double>());
    auto& prefill = output["prefill_greedy"];
    prefill["swapouts_delta_private"] = prefill_vm_before.swapouts >= 0 && prefill_vm_after.swapouts >= 0
                                           ? ni::bench::ojson(prefill_vm_after.swapouts - prefill_vm_before.swapouts)
                                           : ni::bench::ojson(nullptr);
    apply_conditions(prefill, prefill_started, prefill_finished);
    output["gpu_monitor_private"] = ni::bench::summarize_samples(monitor.samples(), monitoring_started,
                                                               ni::bench::Monitor::now());
    std::cout << output.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    monitor.stop();
    output["error"] = error.what();
    std::cout << output.dump(2) << '\n';
    return 1;
  }
}
