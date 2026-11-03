#!/usr/bin/env python3
# Flint's library source as cmake/embed_kernels_flint.cmake concatenates it (common.h, quant.h, kflash.h, flint/*.metal).
# usage: mkflint.py REPO OUT
import glob, os, re, sys
repo, out = sys.argv[1:3]
k = os.path.join(repo, "src/kernels")
parts = [open(os.path.join(k, f)).read() for f in ("common.h", "quant.h", "kflash.h")]
for m in sorted(glob.glob(os.path.join(k, "flint/*.metal"))):
    parts.append(re.sub(r'#include "[a-z_0-9]+\.h"', "", open(m).read()))
open(out, "w").write("\n".join(parts))
