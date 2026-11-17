#!/usr/bin/env python3
"""vmlog.py FILE: every 0.5 s, the time and vm_stat's free pages, pageins, compressions, decompressions, swapins and
the compressor's pages, until killed."""
import re, subprocess, sys, time
keys = ("Pages free", "Pageins", "Compressions", "Decompressions", "Swapins", "Pages occupied by compressor")
with open(sys.argv[1], "a") as f:
    while True:
        o = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        v = {k: m.group(1) for k in keys for m in [re.search(re.escape(k) + r":\s+(\d+)", o)] if m}
        f.write(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 10) % 10} " + " ".join(f"{k.split()[-1]}={v.get(k)}" for k in keys) + "\n")
        f.flush()
        time.sleep(0.5)
