#!/usr/bin/env python3
"""The KFlash Splash build's inputs (packaging/splash/): pin.json's shape, each patch's license header and its
"Modified by Nace AI" line in every file it changes, gen_kernels.py on a throwaway Splash tree (and its refusal when
Splash's repack helpers change), and, with --apply, every patch applied in order to the pinned Splash commit (clones
github.com/incoai/splash, or uses SPLASH_CACHE, an existing clone). CPU only, stdlib only; no build, no GPU.

usage: test_splash_patches.py [--apply]
"""
import json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "packaging", "splash")
sys.path.insert(0, PKG)
import gen_kernels  # noqa: E402

FAILED = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


with open(os.path.join(PKG, "pin.json")) as f:
    pin = json.load(f)
check(pin["repo"] == "https://github.com/incoai/splash.git", "pin: upstream Splash", pin["repo"])
check(re.fullmatch(r"[0-9a-f]{40}", pin["commit"]) is not None, "pin: full commit hash", pin["commit"])
for k in ("release_binary_sha256", "release_metallib_sha256", "release_host_step_metallib_sha256", "kflash_binary_sha256"):
    check(re.fullmatch(r"[0-9a-f]{64}", pin[k]) is not None, f"pin: {k}", pin[k])

PATCHES = sorted(n for n in os.listdir(os.path.join(PKG, "patches")) if n.endswith(".patch"))
check(len(PATCHES) > 0, "patches: present")
check([n[:5] for n in PATCHES] == [f"{i:04d}-" for i in range(1, len(PATCHES) + 1)], "patches: numbered 0001.. in order",
      PATCHES)
marked = set()  # a file the series already marked needs no second line
for name in PATCHES:
    with open(os.path.join(PKG, "patches", name)) as f:
        text = f.read()
    check(text.startswith("This file changes Splash by Inco (https://github.com/incoai/splash), licensed under the "
                          "Apache License 2.0"), f"{name}: license header")
    # Apache 2.0 section 4(b): every modified file carries a notice that it was changed.
    files = re.findall(r"^\+\+\+ b/(\S+)$", text, re.M)
    check(len(files) > 0, f"{name}: changes files")
    for path in files:
        part = text.split(f"+++ b/{path}\n", 1)[1].split("\ndiff --git ", 1)[0]
        if "\n+// Modified by Nace AI" in "\n" + part:
            marked.add(path)
        check(path in marked, f"{name}: {path} marked modified")

REPACK = """#pragma once
#include <metal_stdlib>
using namespace metal;
inline void a(device const uchar *p0, device const uchar *p1, device const ushort *p2, device const ushort *p3) {}
inline void b(device const uchar *p4, device const uchar *p5, device const ushort *p6) {
  auto q = (device const ushort *)p4;
}
kernel void gguf_repack(device const uchar *in [[buffer(0)]]) {}
"""


def tree(repack):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "runtime", gen_kernels.REPACK)
    os.makedirs(os.path.dirname(p))
    with open(p, "w") as f:
        f.write(repack)
    return d


out = gen_kernels.generate(tree(REPACK))
check(out.startswith("#pragma METAL internals : enable\n"), "gen: internals pragma first")
helpers = out.split("device const\n", 1)[1].split("// ---- KSX H1", 1)[0]
check(helpers.count("KSXC u") == 8 and "(KSXC ushort *)" in helpers and "device const" not in helpers,
      "gen: 8 staging pointers coherent", helpers)
check("device const uchar *in" not in out and "kernel void gguf_repack" not in out, "gen: stops before gguf_repack")
check("kernel void ksx_h1_gate_step" in out and "kernel void ksx_moe_expert_load_c" in out, "gen: H1 kernels appended")
for broken, what in ((REPACK.replace("device const ushort *p6", "ushort *p6"), "7 pointers"),
                     (REPACK.replace("kernel void gguf_repack", "kernel void repack2"), "no gguf_repack")):
    try:
        gen_kernels.generate(tree(broken))
        check(False, f"gen: refuses {what}", "generated")
    except ValueError:
        check(True, f"gen: refuses {what}")


# build.py --out: an earlier build (marker) or an empty dir is replaced; anything else is refused and left as it was.
import build  # noqa: E402

t = tempfile.mkdtemp()
mine, empty, theirs, afile = (os.path.join(t, n) for n in ("mine", "empty", "theirs", "afile"))
for d in (mine, empty, theirs):
    os.makedirs(d)
open(os.path.join(mine, build.MARKER), "w").close()
open(os.path.join(mine, "stale"), "w").close()
open(os.path.join(theirs, "keep.txt"), "w").write("user data")
open(afile, "w").close()
for d in (mine, empty, os.path.join(t, "new")):
    build.clear_out(d)
    check(os.listdir(d) == [build.MARKER], f"out: {os.path.basename(d)} becomes an empty build dir", os.listdir(d))
for d, what in ((theirs, "a directory that is not a build"), (afile, "a file")):
    try:
        build.clear_out(d)
        check(False, f"out: refuses {what}", "cleared")
    except build.BuildError:
        check(True, f"out: refuses {what}")
check(open(os.path.join(theirs, "keep.txt")).read() == "user data", "out: the refused directory's files survive")

def install_kflash(version):
    """install.sh's build_kflash against a fake Splash `version` release; build.py replaced by a marker writer."""
    src = open(os.path.join(HERE, "..", "install.sh")).read()
    fn = re.search(r"^build_kflash\(\) \{\n.*?^\}\n", src, re.S | re.M).group(0)
    t = tempfile.mkdtemp()
    clone, root = os.path.join(t, "clone"), os.path.join(t, "Splash")
    os.makedirs(os.path.join(clone, "packaging", "splash")); os.makedirs(os.path.join(root, "engine"))
    with open(os.path.join(clone, "packaging", "splash", "pin.json"), "w") as f:
        json.dump(pin, f)
    with open(os.path.join(clone, "packaging", "splash", "build.py"), "w") as f:
        f.write("import sys\nopen(%r, 'w').write(' '.join(sys.argv[1:]))\n" % os.path.join(t, "built"))
    engine = os.path.join(root, "engine", "splash")
    with open(engine, "w") as f:
        f.write("#!/bin/sh\n")
    os.chmod(engine, 0o755)
    with open(os.path.join(root, "release.json"), "w") as f:
        json.dump({"version": version}, f)
    before = {n: open(os.path.join(d, n), "rb").read() for d, _, ns in os.walk(root) for n in ns}
    p = subprocess.run(["bash", "-c", 'say() { printf "%s\\n" "$*"; }\n' + fn + 'build_kflash "$1"', "x", clone],
                       capture_output=True, text=True, env=dict(os.environ, HOME=t, SPLASH_HOME=root))
    after = {n: open(os.path.join(d, n), "rb").read() for d, _, ns in os.walk(root) for n in ns}
    return p, os.path.exists(os.path.join(t, "built")), before == after


p, built, same = install_kflash("1.5.0")
check(p.returncode == 0 and not built and same, "install: a newer Splash release is left alone, not built", p.stdout + p.stderr)
check("Splash 1.5.0 is installed" in p.stdout and "patches are for Splash " + pin["tag"] in p.stdout,
      "install: says why KFlash was not built", p.stdout)
p, built, same = install_kflash(pin["tag"])
check(p.returncode == 0 and built and same, "install: the pinned release builds KFlash, its files untouched", p.stdout + p.stderr)

if "--apply" in sys.argv[1:]:
    cache = os.environ.get("SPLASH_CACHE") or os.path.join(tempfile.mkdtemp(), "splash")
    p = subprocess.run([sys.executable, os.path.join(PKG, "build.py"), "--check", "--cache", cache],
                       capture_output=True, text=True)
    check(p.returncode == 0, "apply: every patch applies to the pinned Splash", p.stdout + p.stderr)

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {', '.join(FAILED)}")
print("all passed")
