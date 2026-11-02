// Native helper for the macOS app and managed instances. Configuration enters
// as JSON; stdout/stderr are diagnostics, never the observation protocol.
#include "server/service.h"
#include "server/runtime.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

int main(int argc, char** argv) {
  try {
    bool check = false;
    std::ifstream file;
    std::istream* input = &std::cin;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--check") check = true;
      else if (a == "--config" && i + 1 < argc) {
        file.open(argv[++i]);
        if (!file) throw std::invalid_argument("cannot open configuration");
        input = &file;
      } else if (a == "--help") {
        std::cout << "kapitsa-runtime [--config FILE] [--check]\n"
                     "Reads one protocol-v1 JSON configuration from FILE or stdin.\n"
                     "--check validates configuration without loading a model or using Metal.\n";
        return 0;
      } else throw std::invalid_argument("unknown runtime option");
    }
    std::string bytes;
    char ch;
    while (input->get(ch)) {
      if (bytes.size() >= 16384) throw std::invalid_argument("configuration exceeds 16 KiB");
      bytes += ch;
    }
    auto options = ni::server::native_options(ni::ojson::parse(bytes));
    if (check) {
      std::cout << "{\"protocol\":1,\"status\":\"valid\"}\n";
      return 0;
    }
    // A managed runtime outlives the app/CLI which launched it. It must not
    // inherit a coding-agent launcher's parent-death policy or its terminal.
    unsetenv("KAPITSA_PARENT_PID");
    setsid();
    return ni::server::run_service(std::move(options));
  } catch (const std::exception& e) {
    std::cerr << "kapitsa-runtime: " << e.what() << "\n";
    return 2;
  }
}
