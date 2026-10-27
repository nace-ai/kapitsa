#!/usr/bin/env bash
# CPU only, no native/UI linkage or sockets. Run after host/queue preflight.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-}"
if [[ -z "$binary" ]]; then binary="$(mktemp "${TMPDIR:-/tmp}/kapitsa-http-stream.XXXXXX")"; fi
printf 'CPU test binary: %s\n' "$binary"
"${CXX:-clang++}" -std=c++20 -pthread -Wall -Wextra -Werror \
  -I"$root/src" -I"$root/third_party" \
  "$root/src/engine/execution_session.cpp" "$root/src/server/http_stream.cpp" \
  "$root/tools/http_stream_test.cpp" -o "$binary"
"$binary"
