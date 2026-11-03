#!/usr/bin/env python3
# the tensor library's source as cmake/embed_kernels_tensor.cmake concatenates it. usage: mksrc.py REPO OUT
import glob, os, re, sys
repo, out = sys.argv[1:3]
k = os.path.join(repo, "src/kernels")
parts = [open(os.path.join(k, f)).read() for f in ("common.h", "iq.h", "nemotron/nh_route.h", "kv_int8.h")]
for m in sorted(glob.glob(os.path.join(k, "tensor/*.metal"))):
    parts.append(re.sub(r'#include "[a-z_0-9]+\.h"', "", open(m).read()))
open(out, "w").write("\n".join(parts))
