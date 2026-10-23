#!/bin/bash
# Builds a Kapitsa release: build/release/kapitsa-VERSION-darwin-arm64.tar.gz (kapitsa, kapitsa-systemone and
# kapitsa-drex-pointer for Drex decision models (docs/systemone.md), LICENSE, THIRD_PARTY_NOTICES and bench/harness,
# which `kapitsa bench compare` runs) and its .sha256. The Homebrew formula
# and install.sh install this tarball; the harness carries VERSION (this release) since a copy has no checkout.
#   packaging/release/build.sh [VERSION]    VERSION must be CMakeLists.txt's KAPITSA_VERSION (the default)
# It builds the committed tree (HEAD, exported with git archive), never the working tree: Release (-O3), arm64,
# CMakeLists.txt's deployment target. It strips and signs the binaries, then checks their architecture, minimum macOS,
# libraries (system ones only), kapitsa's --version and kapitsa-systemone's --help.
# Signing is ad hoc, unless KAPITSA_SIGN_IDENTITY names a "Developer ID Application" certificate in the keychain
# (KAPITSA_KEYCHAIN: in that keychain file). KAPITSA_NOTARY_PROFILE, a profile saved with
# `xcrun notarytool store-credentials`, then also has Apple notarize it. .github/workflows/release.yml runs it for each release tag.
set -euo pipefail

die() { printf 'build.sh: %s\n' "$*" >&2; exit 1; }

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
[ "$(uname -s)-$(uname -m)" = Darwin-arm64 ] || die "build on an Apple silicon Mac, in a native (not Rosetta) terminal."
for tool in git cmake strip codesign lipo vtool otool shasum tar gzip; do
  command -v "$tool" > /dev/null || die "needs $tool (the Command Line Tools and CMake 3.24+)."
done

commit=$(git rev-parse HEAD)
short=$(git rev-parse --short HEAD)
lists=$(git show "$commit:CMakeLists.txt")
cmake_version=$(printf '%s\n' "$lists" | sed -n 's/^set(KAPITSA_VERSION "\([^"]*\)").*/\1/p')
minos=$(printf '%s\n' "$lists" | sed -n 's/^set(CMAKE_OSX_DEPLOYMENT_TARGET "\([^"]*\)".*/\1/p')
[ -n "$cmake_version" ] && [ -n "$minos" ] || die "can't read KAPITSA_VERSION and CMAKE_OSX_DEPLOYMENT_TARGET from CMakeLists.txt."
version=${1:-$cmake_version}
version=${version#v}
[ "$version" = "$cmake_version" ] ||
  die "CMakeLists.txt at $short says $cmake_version, not $version: set KAPITSA_VERSION to $version in a commit first."
[ -z "$(git status --porcelain --untracked-files=no)" ] || echo "Note: building HEAD ($short); uncommitted changes are left out."

name="kapitsa-$version-darwin-arm64"
out="$root/build/release"
work="$out/work"
rm -rf "$work" "$out/$name.tar.gz" "$out/$name.tar.gz.sha256"
mkdir -p "$work/src" "$work/stage/$name"
git archive --format=tar "$commit" | tar -x -C "$work/src"

# 1. Build, with no compiler flags or deployment target from the environment
clean_env=(env -u CFLAGS -u CXXFLAGS -u OBJCFLAGS -u OBJCXXFLAGS -u CPPFLAGS -u LDFLAGS -u MACOSX_DEPLOYMENT_TARGET)
"${clean_env[@]}" cmake -S "$work/src" -B "$work/obj" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET="$minos" -DKAPITSA_GIT="$short"
"${clean_env[@]}" cmake --build "$work/obj" --target kapitsa systemone -j "$(sysctl -n hw.ncpu)"
for target in naceinfer kapitsa systemone; do
  flags=$(grep -E '^(C|CXX|OBJCXX)_FLAGS = ' "$work/obj/CMakeFiles/$target.dir/flags.make")
  if [ -z "$flags" ] || printf '%s\n' "$flags" | grep -v -q -- ' -O3'; then
    die "the $target target did not compile with -O3: $flags"
  fi
done

# 2. Strip and sign (hardened runtime either way, so the ad hoc build runs as the notarized one would)
bin="$work/stage/$name/kapitsa"
sysbin="$work/stage/$name/kapitsa-systemone"
install -m 0755 "$work/obj/kapitsa" "$bin"
install -m 0755 "$work/obj/systemone" "$sysbin"
if [ -z "${KAPITSA_SIGN_IDENTITY:-}" ] && [ -n "${KAPITSA_NOTARY_PROFILE:-}" ]; then
  die "notarizing needs a Developer ID signature: set KAPITSA_SIGN_IDENTITY too."
fi
for b in "$bin" "$sysbin"; do
  strip "$b"
  sign=(codesign --force --options runtime --identifier "ai.nace.$(basename "$b")")
  if [ -n "${KAPITSA_SIGN_IDENTITY:-}" ]; then
    "${sign[@]}" --timestamp --sign "$KAPITSA_SIGN_IDENTITY" ${KAPITSA_KEYCHAIN:+--keychain "$KAPITSA_KEYCHAIN"} "$b"
    signed="Developer ID ($KAPITSA_SIGN_IDENTITY)"
  else
    "${sign[@]}" --sign - "$b"
    signed="ad hoc"
  fi
  codesign --verify --strict "$b" || die "the signature of $(basename "$b") does not verify."
done
if [ -n "${KAPITSA_NOTARY_PROFILE:-}" ]; then
  mkdir "$work/notarize"
  cp "$bin" "$sysbin" "$work/notarize/"
  ditto -c -k --keepParent "$work/notarize" "$work/notarize.zip"
  xcrun notarytool submit "$work/notarize.zip" --keychain-profile "$KAPITSA_NOTARY_PROFILE" \
    ${KAPITSA_KEYCHAIN:+--keychain "$KAPITSA_KEYCHAIN"} --wait --timeout 1h --output-format json > "$work/notary.json" || true
  status=$(plutil -extract status raw -o - "$work/notary.json" 2> /dev/null || echo "no answer")
  if [ "$status" != Accepted ]; then
    id=$(plutil -extract id raw -o - "$work/notary.json" 2> /dev/null || true)
    [ -z "$id" ] || xcrun notarytool log "$id" --keychain-profile "$KAPITSA_NOTARY_PROFILE" \
      ${KAPITSA_KEYCHAIN:+--keychain "$KAPITSA_KEYCHAIN"} >&2 || true
    die "notarization: $status."
  fi
  signed="$signed, notarized"
fi

# 3. Check what ships: arm64 only, the deployment target, system libraries only, the version
for b in "$bin" "$sysbin"; do
  what=$(basename "$b")
  [ "$(lipo -archs "$b")" = arm64 ] || die "$what is $(lipo -archs "$b"), not arm64 only."
  build_info=$(vtool -show-build "$b")
  platform=$(printf '%s\n' "$build_info" | awk '$1 == "platform" {print $2; exit}')
  got_minos=$(printf '%s\n' "$build_info" | awk '$1 == "minos" {print $2; exit}')
  [ "$platform" = MACOS ] && [ "$got_minos" = "$minos" ] ||
    die "$what targets $platform $got_minos, not MACOS $minos (vtool -show-build)."
  libs=$(otool -L "$b" | tail -n +2 | awk '{print $1}')
  outside=$(printf '%s\n' "$libs" | grep -v -E '^(/usr/lib/|/System/Library/)' || true)
  [ -z "$outside" ] || die "$what links libraries that are not part of macOS: $outside"
done
got_version=$("$bin" --version)
[ "$got_version" = "kapitsa $version ($short)" ] || die "--version prints \"$got_version\", not \"kapitsa $version ($short)\"."
# --help exits before the GPU lock and the model load
sys_help=$("$sysbin" --help 2>&1) || die "kapitsa-systemone --help fails."
[[ "$sys_help" == *"usage: systemone"* ]] || die "kapitsa-systemone --help does not print its usage."

# 4. The tarball: fixed owner, times (the commit's) and order, no extended attributes, so a rebuild matches
install -m 0644 "$work/src/LICENSE" "$work/src/THIRD_PARTY_NOTICES" "$work/stage/$name/"
install -m 0755 "$work/src/tools/drex_pointer.py" "$work/stage/$name/kapitsa-drex-pointer"
# `kapitsa bench compare` runs this harness; install.sh and the formula put it beside the notices, where find_harness()
# looks: <the binary's folder>/../share/kapitsa/bench/harness. The copy has no checkout, so its VERSION file says which
# release it came with, and a report made from an install names its harness instead of "unknown".
mkdir -p "$work/stage/$name/bench"
cp -R "$work/src/bench/harness" "$work/stage/$name/bench/harness"
printf '%s\n' "$version" > "$work/stage/$name/bench/harness/VERSION"
stamp=$(TZ=UTC git show -s --format=%cd --date=format-local:%Y%m%d%H%M.%S "$commit")
find "$work/stage/$name" -exec env TZ=UTC touch -t "$stamp" {} +
(cd "$work/stage" && find "$name" | LC_ALL=C sort |
  COPYFILE_DISABLE=1 tar -c -n --format ustar --uid 0 --gid 0 --uname root --gname wheel \
    --no-xattrs --no-acls --no-fflags -f - -T -) | gzip -9n > "$out/$name.tar.gz"
(cd "$out" && shasum -a 256 "$name.tar.gz" > "$name.tar.gz.sha256")
tar -tzf "$out/$name.tar.gz" | grep -qx "$name/bench/harness/kbench.py" ||
  die "the tarball has no bench/harness: kapitsa bench compare would not find it."
for f in kapitsa-systemone kapitsa-drex-pointer; do
  tar -tzf "$out/$name.tar.gz" | grep -qx "$name/$f" || die "the tarball has no $f."
done
rm -rf "$work"

printf '\n%s\n' "$out/$name.tar.gz"
printf '  sha256    %s\n' "$(cut -d ' ' -f 1 "$out/$name.tar.gz.sha256")"
printf '  size      %s bytes (binary %s bytes)\n' "$(stat -f %z "$out/$name.tar.gz")" "$(tar -xOzf "$out/$name.tar.gz" "$name/kapitsa" | wc -c | tr -d ' ')"
printf '  harness   %s files (bench/harness, run by `kapitsa bench compare`)\n' \
  "$(tar -tzf "$out/$name.tar.gz" | grep "/bench/harness/" | grep -vc "/$")"
printf '  binaries  %s and kapitsa-systemone, arm64, macOS %s or later, system libraries only, signed %s\n' \
  "$got_version" "$minos" "$signed"
