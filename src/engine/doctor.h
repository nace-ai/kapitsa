// `kapitsa doctor`: what this Mac is and which models fit it (see doctor.cpp).
#pragma once

namespace ni {
// argv: the arguments after "doctor". Prints the report (or JSON with --json); returns the exit code.
int run_doctor(int argc, char** argv);
}  // namespace ni
