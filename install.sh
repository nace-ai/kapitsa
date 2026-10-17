#!/bin/bash
# Kapitsa installer: builds this checkout and puts `kapitsa` on your PATH. Safe to run again (updates in place).
#   ./install.sh                 build and install (asks before installing CMake with Homebrew)
#   ./install.sh --yes           don't ask: install CMake with Homebrew if it is missing
#   ./install.sh --prefix DIR    install into DIR (default: /opt/homebrew/bin if writable, else ~/.local/bin)
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '%s\n' "$*"; }
die() { printf '\ninstall.sh: %s\n' "$*" >&2; exit 1; }

PREFIX="" YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="${2:?--prefix needs a folder}"; shift 2 ;;
    --prefix=*) PREFIX="${1#*=}"; shift ;;
    -y | --yes) YES=1; shift ;;
    -h | --help) sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option $1 (./install.sh --help)" ;;
  esac
done

# 1. An Apple silicon Mac with macOS 14 or later
[ "$(uname -s)" = Darwin ] || die "Kapitsa runs on Macs with Apple silicon only."
if [ "$(uname -m)" != arm64 ]; then
  if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
    die "this terminal runs under Rosetta (Intel mode). Open a native terminal and run ./install.sh again."
  fi
  die "Kapitsa needs a Mac with Apple silicon (M1 or newer); this one is $(uname -m)."
fi
macos=$(sw_vers -productVersion)
[ "${macos%%.*}" -ge 14 ] || die "Kapitsa needs macOS 14 or later; this Mac runs macOS $macos."

# 2. Apple's Command Line Tools (compilers; Xcode itself is not needed)
if ! xcode-select -p > /dev/null 2>&1 || ! xcrun --find clang++ > /dev/null 2>&1; then
  say "Kapitsa is built with Apple's Command Line Tools. Opening their installer..."
  xcode-select --install > /dev/null 2>&1 || true
  die "finish installing the Command Line Tools in the window that opened, then run ./install.sh again."
fi

# 3. CMake 3.24 or later
need_cmake() {
  command -v cmake > /dev/null 2>&1 || return 0
  local v
  v=$(cmake --version | head -1 | awk '{print $3}')
  [ "$(printf '%s\n3.24\n' "$v" | sort -V | head -1)" != 3.24 ]
}
if need_cmake; then
  if command -v brew > /dev/null 2>&1; then
    ans=y
    if [ "$YES" != 1 ]; then
      if [ -t 0 ]; then read -r -p "Kapitsa needs CMake 3.24+ to build. Install it with Homebrew (brew install cmake)? [Y/n] " ans || ans=n
      else ans=n; fi
    fi
    case "${ans:-y}" in
      [Yy]*) brew install cmake || brew upgrade cmake ;;
      *) die "install CMake 3.24 or later (brew install cmake, or https://cmake.org/download), then run ./install.sh again (--yes installs it for you)." ;;
    esac
  else
    die "Kapitsa needs CMake 3.24 or later to build. Install it (https://cmake.org/download, or Homebrew: brew install cmake), then run ./install.sh again."
  fi
fi

# 4. Build (Release, the server only; the Metal kernels compile when a model loads, so no Xcode)
log=$(mktemp -t kapitsa-install)
jobs=$(sysctl -n hw.perflevel0.logicalcpu 2> /dev/null || sysctl -n hw.ncpu)
say "Building Kapitsa (a minute or two the first time)..."
if ! { cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target kapitsa -j "$jobs"; } > "$log" 2>&1; then
  tail -25 "$log" >&2
  die "the build failed (full log: $log)."
fi

# 5. Install
if [ -z "$PREFIX" ]; then
  if [ -d /opt/homebrew/bin ] && [ -w /opt/homebrew/bin ]; then PREFIX=/opt/homebrew/bin; else PREFIX="$HOME/.local/bin"; fi
fi
mkdir -p "$PREFIX" || die "can't create $PREFIX (--prefix picks another folder)."
install -m 0755 build/kapitsa "$PREFIX/kapitsa" || die "can't write $PREFIX/kapitsa (--prefix picks another folder)."
version=$("$PREFIX/kapitsa" --version) || die "the installed $PREFIX/kapitsa does not run."
rm -f "$log"

say "Installed $version: $PREFIX/kapitsa"
run=kapitsa
case ":$PATH:" in
  *":$PREFIX:"*) ;;
  *)
    run="$PREFIX/kapitsa"
    say ""
    say "$PREFIX is not on your PATH yet. To add it (zsh):"
    say "  echo 'export PATH=\"$PREFIX:\$PATH\"' >> ~/.zprofile && export PATH=\"$PREFIX:\$PATH\""
    ;;
esac
say ""
say "Next:"
say "  $run serve        # starts the best model for this Mac; the first run downloads it"
say "  $run --help       # models and commands"
