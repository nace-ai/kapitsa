#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-}"
if [[ -z "$binary" ]]; then binary="$(mktemp "${TMPDIR:-/tmp}/kapitsa-execution-session.XXXXXX")"; fi
printf 'CPU test binary: %s\n' "$binary"
"${CXX:-clang++}" -std=c++20 -pthread -Wall -Wextra -Werror \
  -I"$root/src" -I"$root/third_party" \
  "$root/src/engine/execution_session.cpp" "$root/tools/execution_session_test.cpp" -o "$binary"
"$binary"
