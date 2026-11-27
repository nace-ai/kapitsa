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
  ni::bench::ojson output = {{"format", "cold-prefill-only recovery measurement"},
                            {"cohort_seeds", {1, 2, 3}},
                            {"rows", ni::bench::ojson::array()}};
  try {
    output["warmup"] = ni::bench::warmup(settings, nonces);
    if (!output["warmup"]["ok"].get<bool>()) throw std::runtime_error("warmup failed");
    double waited = 0;
    std::vector<ni::bench::Busy> busy;
    output["quiet_gate"] = ni::bench::wait_quiet(5, 18, 300, own, ambient, &waited, &busy);
    if (!output["quiet_gate"].get<bool>()) throw std::runtime_error("GPU quiet gate failed");
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
