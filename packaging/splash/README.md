# KFlash Splash engine: patches to Splash

`kapitsa serve --engine splash` runs a Splash engine child (`splash serve-native`, see
[docs/splash-backend.md](../../docs/splash-backend.md)). This directory holds Kapitsa's changes to that engine as
patches against one pinned Splash release, and the script that builds it. The patched engine is KFlash's Splash
engine; benchmark rows measured with it say KFlash, not Splash.

## Why patches, not a fork or third_party/

- Splash is Inco's, Apache License 2.0. The changes stay a readable diff against a named upstream commit, so a reviewer
  sees exactly what Kapitsa changed, and each one can go upstream as it is.
- `third_party/` is vendored code Kapitsa does not edit. Splash is not compiled into Kapitsa: it is a separate binary
  that Kapitsa starts, so it is cloned and built here, not vendored.
- A patch that stops applying fails the build, by name. Kapitsa never builds a half-patched Splash.

## Contents

| File | What |
|---|---|
| `pin.json` | The upstream repo, tag and commit, and the sha256 of that release's engine binary and both metallibs |
| `patches/NNNN-*.patch` | Applied in order with `git apply`. Each starts with its license header; each file it changes gets a `Modified by Nace AI` line, and each change is marked `KSX` |
| `kernels/ksx_h1_kernels.metal` | The Metal kernels patch 0002 dispatches, written by hand |
| `gen_kernels.py` | Writes `ksx_h1.metal` at build time: Splash's repack helpers from the pinned source, with staging read through system-coherent loads, followed by the kernels above |
| `build.py` | Clones, checks the commit, applies the patches, builds, writes the kernels and `kflash.json` |

## The patches

| Patch | Change | Off when |
|---|---|---|
| `0001-runtime-kernels` | Splash compiles the `.metal` files in `$SPLASH_KSX_KERNELS` at startup; their functions replace the metallib's of the same name | `SPLASH_KSX_KERNELS` unset |
| `0002-h1-gate-replay` | H1=3: a decode step's expert-load host step runs inside the command buffer, with a GPU gate that replays the rest of the command when the CPU did not release the step in time | `SPLASH_KSX_H1` is not `3` |

With neither variable set, the patched binary runs Splash's own code paths. `SPLASH_KSX_H1=3` needs `SPLASH_KSX_KERNELS`
too: its kernels are not in Splash's metallib, so with H1=3 alone the first decode fails on a missing function rather
than falling back to stock. Kapitsa always sets both (`kflash.json`'s `env`); patch 0002's header line "Off unless
SPLASH_KSX_H1=3 and SPLASH_KSX_KERNELS are set" means that pair.

## Build

```bash
python3 packaging/splash/build.py --release ~/path/to/splash-1.4.0     # Splash's unpacked 1.4.0 release archive
python3 packaging/splash/build.py --check                              # apply only, no build (any OS)
python3 tests/test_splash_patches.py --apply                           # the CPU test; --apply clones Splash
```

The build takes the release's metallibs (checked against `pin.json`) and compiles only the host code. Stock Splash
1.4.0 built this way reproduces the release binary byte for byte (sha256 `20b36bcc…`) with Xcode 26.5's SDK. So the
KFlash binary differs from Splash's release by the patches alone. A different SDK builds a different binary.
`pin.json`'s `kflash_binary_sha256` is the binary the output check and the A/B measured (`92a5dfed…`, Xcode 26.5's
SDK); a build that does not match it gets a warning from `build.py`, and is a binary nobody measured.

## Moving to a new Splash release

1. Change `pin.json` and run `build.py --check`. A patch that fails is rebased by hand onto the new tag.
2. `gen_kernels.py` refuses a release whose repack helpers changed shape. Review the transform before changing it.
3. The new binary is a new engine: rerun the output check and the interleaved A/B against that release's own binary
   before a default or a benchmark row moves to it.
