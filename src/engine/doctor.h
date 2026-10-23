// `kapitsa doctor`: what this Mac is and which models fit it (see doctor.cpp).
#pragma once
#include <string>

namespace ni {
// argv: the arguments after "doctor". Prints the report (or JSON with --json); returns the exit code.
int run_doctor(int argc, char** argv);
// Memory bandwidth (GB/s) of a chip (Gpu::name(), "Apple M4 Pro") with `cores` GPU cores, from Apple's
// specifications (Max chips by their cores); 0 when unknown
double bandwidth_gbs(const std::string& chip, int cores);
}  // namespace ni
