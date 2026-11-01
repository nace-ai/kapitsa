#!/usr/bin/env python3
import json
from pathlib import Path
import re
import sys

FIELDS = ('free_percent', 'swap_used_MiB', 'outside_cpu_percent')


def summarize(path):
    rows = [dict(re.findall(r'(\w+)=([0-9.]+)', line))
            for line in path.read_text().splitlines() if ' leg=' in line]
    assert len(rows) == 8, f'{path}: expected eight pre-leg samples'
    return {key: {'min': min(float(row[key]) for row in rows),
                  'max': max(float(row[key]) for row in rows)} for key in FIELDS}


if __name__ == '__main__':
    print(json.dumps({str(path): summarize(path) for path in map(Path, sys.argv[1:])}, indent=2))
