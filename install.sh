#!/bin/bash
# Kapitsa installer: puts `kapitsa` on your PATH. Safe to run again (updates in place).
#   curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash
#                  installs the latest release, a prebuilt binary (no compiler, Homebrew or sudo needed)
#   ./install.sh   in a clone: builds that checkout instead (Command Line Tools and CMake)
# Options: ./install.sh --help, or curl ... | bash -s -- --help
set -euo pipefail

KAPITSA_REPO=nace-ai/kapitsa
KAPITSA_SELF="${BASH_SOURCE[0]:-}" # empty when piped into bash (read here: inside a function it says "bash")

say() { printf '%s\n' "$*"; }
die() {
  printf '\ninstall.sh: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat << EOF
Installs Kapitsa (https://github.com/$KAPITSA_REPO) and puts \`kapitsa\` on your PATH. Run it again to update.

  curl -fsSL https://raw.githubusercontent.com/$KAPITSA_REPO/main/install.sh | bash
      the latest release: a prebuilt binary, checked against its published sha256
  ./install.sh
      in a clone: builds that checkout (needs the Command Line Tools and CMake 3.24+)

Options (when piped: curl ... | bash -s -- OPTIONS):
  --version X.Y.Z    install that release instead of the latest (or KAPITSA_VERSION=X.Y.Z)
  --from-source      build from source (in a clone)
  --from-file FILE   install a release tarball that is on this Mac (checked against FILE.sha256 when present)
  --prefix DIR       install into DIR (default: /opt/homebrew/bin if writable, else ~/.local/bin)
  -y, --yes          let a source build install CMake with Homebrew without asking
KAPITSA_RELEASE_URL=URL downloads the tarball from URL, and its checksum from URL.sha256.
EOF
}

# An Apple silicon Mac with macOS 14 or later. $1: the install mode.
check_mac() {
  [ "$(uname -s)" = Darwin ] || die "Kapitsa runs on Macs with Apple silicon only."
  if [ "$(uname -m)" != arm64 ]; then
    [ "$(sysctl -in sysctl.proc_translated 2> /dev/null)" = 1 ] ||
      die "Kapitsa needs a Mac with Apple silicon (M1 or newer); this one is $(uname -m)."
    # a terminal under Rosetta: the arm64 release binary still runs natively, but a build there would not
    [ "$1" != source ] || die "this terminal runs under Rosetta (Intel mode). Open a native terminal and run ./install.sh again."
  fi
  local macos
  macos=$(sw_vers -productVersion)
  [ "${macos%%.*}" -ge 14 ] || die "Kapitsa needs macOS 14 or later; this Mac runs macOS $macos."
  [ "${macos%%.*}" -ge 15 ] || say "Note: Kapitsa is tested on macOS 15 and later; this Mac runs $macos. Continuing."
}

# The latest release's version, from where GitHub's /releases/latest redirects (no API rate limit)
latest_version() {
  local got
  got=$(curl -sSLI -o /dev/null -w '%{http_code} %{url_effective}' "https://github.com/$KAPITSA_REPO/releases/latest") ||
    die "can't reach github.com to find the latest release."
  case "$got" in
    "200 "*/releases/tag/v*) printf '%s\n' "${got##*/tag/v}" ;;
    *) die "found no release at https://github.com/$KAPITSA_REPO/releases." ;;
  esac
}

# A release tarball: downloaded (or --from-file), checked against its sha256, unpacked into $KAPITSA_TMP/x
fetch_release() {
  local tarball sums url want got
  if [ "$mode" = file ]; then
    [ -f "$from_file" ] || die "there is no file $from_file."
    tarball=$from_file
    sums="$from_file.sha256"
    if [ ! -f "$sums" ]; then
      say "Note: there is no $sums, so the tarball's checksum is not checked."
      sums=""
    fi
  else
    if [ -n "${KAPITSA_RELEASE_URL:-}" ]; then
      url=$KAPITSA_RELEASE_URL
    else
      if [ -z "$version" ]; then version=$(latest_version) || exit 1; fi
      version=${version#v}
      url="https://github.com/$KAPITSA_REPO/releases/download/v$version/kapitsa-$version-darwin-arm64.tar.gz"
    fi
    say "Downloading ${url##*/}..."
    tarball="$KAPITSA_TMP/kapitsa.tar.gz"
    sums="$tarball.sha256"
    curl -fsSL --retry 3 -o "$tarball" "$url" ||
      die "can't download $url${version:+ (is $version a release? https://github.com/$KAPITSA_REPO/releases)}."
    curl -fsSL --retry 3 -o "$sums" "$url.sha256" || die "can't download its checksum, $url.sha256."
  fi
  if [ -n "$sums" ]; then
    want=$(awk 'NR == 1 {print $1}' "$sums")
    got=$(shasum -a 256 "$tarball" | awk '{print $1}')
    [ "${#want}" = 64 ] || die "$sums holds no sha256."
    [ "$want" = "$got" ] || die "the tarball's sha256 is $got, not the published $want. Nothing was installed."
    say "Checksum OK."
  fi
  mkdir "$KAPITSA_TMP/x"
  tar -xzf "$tarball" -C "$KAPITSA_TMP/x" --strip-components 1 || die "can't unpack ${tarball##*/}."
  [ -f "$KAPITSA_TMP/x/kapitsa" ] && [ -f "$KAPITSA_TMP/x/LICENSE" ] || die "${tarball##*/} is not a Kapitsa release."
}

# Builds the clone at $clone into $clone/build/kapitsa
build_from_source() {
  cd "$clone"
  # Apple's Command Line Tools (compilers; Xcode itself is not needed)
  if ! xcode-select -p > /dev/null 2>&1 || ! xcrun --find clang++ > /dev/null 2>&1; then
    say "Kapitsa is built with Apple's Command Line Tools. Opening their installer..."
    xcode-select --install > /dev/null 2>&1 || true
    die "finish installing the Command Line Tools in the window that opened, then run ./install.sh again."
  fi

  # CMake 3.24 or later
  need_cmake() {
    command -v cmake > /dev/null 2>&1 || return 0
    local v
    v=$(cmake --version | head -1 | awk '{print $3}')
    [ "$(printf '%s\n3.24\n' "$v" | sort -V | head -1)" != 3.24 ]
  }
  if need_cmake; then
    if command -v brew > /dev/null 2>&1; then
      local ans=y
      if [ "$yes" != 1 ]; then
        if [ -t 0 ]; then
          read -r -p "Kapitsa needs CMake 3.24+ to build. Install it with Homebrew (brew install cmake)? [Y/n] " ans || ans=n
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

  # Build (Release, the server only; the Metal kernels compile when a model loads, so no Xcode)
  local log jobs
  log=$(mktemp -t kapitsa-install)
  jobs=$(sysctl -n hw.perflevel0.logicalcpu 2> /dev/null || sysctl -n hw.ncpu)
  say "Building Kapitsa (a minute or two the first time)..."
  if ! { cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target kapitsa -j "$jobs"; } > "$log" 2>&1; then
    tail -25 "$log" >&2
    die "the build failed (full log: $log)."
  fi
  rm -f "$log"
}

# The folder to install into: --prefix, else /opt/homebrew/bin if writable, else ~/.local/bin
pick_prefix() {
  if [ -z "$prefix" ]; then
    if [ -d /opt/homebrew/bin ] && [ -w /opt/homebrew/bin ]; then prefix=/opt/homebrew/bin; else prefix="$HOME/.local/bin"; fi
  fi
  mkdir -p "$prefix" || die "can't create $prefix (--prefix picks another folder)."
  prefix="$(cd "$prefix" && pwd)" || die "can't resolve the install folder $prefix."
  case "$(readlink "$prefix/kapitsa" 2> /dev/null || true)" in
    *Cellar/kapitsa/*) die "$prefix/kapitsa is Homebrew's: brew upgrade kapitsa updates it (--prefix installs elsewhere)." ;;
  esac
}

# Installs $1 (the binary) into $prefix, and LICENSE and THIRD_PARTY_NOTICES from folder $2 beside it
install_kapitsa() {
  local notices notice installed first run=kapitsa
  notices="$(dirname "$prefix")/share/kapitsa"
  mkdir -p "$notices" || die "can't create the license notices folder $notices."
  for notice in LICENSE THIRD_PARTY_NOTICES; do
    install -m 0644 "$2/$notice" "$notices/$notice" || die "can't install $notice in $notices."
  done
  install -m 0755 "$1" "$prefix/kapitsa" || die "can't write $prefix/kapitsa (--prefix picks another folder)."
  installed=$("$prefix/kapitsa" --version) || die "the installed $prefix/kapitsa does not run."

  say "Installed $installed: $prefix/kapitsa"
  say "License notices: $notices"
  case ":$PATH:" in
    *":$prefix:"*)
      first=$(command -v kapitsa || true)
      if [ -n "$first" ] && [ "$first" != "$prefix/kapitsa" ]; then
        run="$prefix/kapitsa"
        say ""
        say "Note: $first comes first on your PATH, so \`kapitsa\` runs that one."
      fi
      ;;
    *)
      run="$prefix/kapitsa"
      say ""
      say "$prefix is not on your PATH yet. To add it (zsh):"
      say "  echo 'export PATH=\"$prefix:\$PATH\"' >> ~/.zprofile && export PATH=\"$prefix:\$PATH\""
      ;;
  esac
  say ""
  say "Next:"
  say "  $run serve        # starts the best model for this Mac; the first run downloads it"
  say "  $run --help       # models and commands"
}

main() {
  prefix="" yes=0 mode="" from_file="" version="${KAPITSA_VERSION:-}" clone=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --prefix) [ $# -ge 2 ] || die "--prefix needs a folder."; prefix=$2; shift 2 ;;
      --prefix=*) prefix=${1#*=}; shift ;;
      --version) [ $# -ge 2 ] || die "--version needs a release, e.g. 0.2.5."; version=$2; shift 2 ;;
      --version=*) version=${1#*=}; shift ;;
      --from-source) mode=source; shift ;;
      --from-file) [ $# -ge 2 ] || die "--from-file needs a tarball."; mode="file"; from_file=$2; shift 2 ;;
      --from-file=*) mode="file"; from_file=${1#*=}; shift ;;
      -y | --yes) yes=1; shift ;;
      -h | --help) usage; exit 0 ;;
      *) die "unknown option $1 (--help lists them)." ;;
    esac
  done
  case "$prefix" in /* | "") ;; *) prefix="$PWD/$prefix" ;; esac

  # Run from a clone (./install.sh), it builds that checkout unless a release is asked for
  if [ -n "$KAPITSA_SELF" ] && [ -f "$KAPITSA_SELF" ]; then
    local dir
    dir="$(cd "$(dirname "$KAPITSA_SELF")" && pwd)"
    if [ -f "$dir/CMakeLists.txt" ] && [ -f "$dir/src/server/main.cpp" ]; then clone=$dir; fi
  fi
  if [ -z "$mode" ]; then
    if [ -n "$clone" ] && [ -z "$version" ] && [ -z "${KAPITSA_RELEASE_URL:-}" ]; then mode=source; else mode=release; fi
  fi
  [ "$mode" != source ] || [ -n "$clone" ] ||
    die "--from-source builds a clone: git clone https://github.com/$KAPITSA_REPO && cd kapitsa && ./install.sh"

  check_mac "$mode"
  pick_prefix
  if [ "$mode" = source ]; then
    build_from_source
    install_kapitsa "$clone/build/kapitsa" "$clone"
  else
    KAPITSA_TMP="$(mktemp -d -t kapitsa-install)"
    trap 'rm -rf "$KAPITSA_TMP"' EXIT
    fetch_release
    install_kapitsa "$KAPITSA_TMP/x/kapitsa" "$KAPITSA_TMP/x"
  fi
}

# Last, so that a download cut short runs nothing
main "$@"
