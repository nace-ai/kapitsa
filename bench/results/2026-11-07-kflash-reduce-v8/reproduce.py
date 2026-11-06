#!/usr/bin/env python3
"""Build and run the isolated reducer comparison under the Kapitsa GPU lock."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkout', type=Path)
parser.add_argument('output', type=Path, help='new output directory; existing evidence is never replaced')
parser.add_argument('--correctness-only', action='store_true')
args = parser.parse_args()
root, out = args.checkout.resolve(), args.output.resolve()
here = Path(__file__).resolve().parent
out.mkdir(parents=True, exist_ok=False)
cmd = ['c++', '-std=c++20', '-O3', '-arch', 'arm64', '-mmacosx-version-min=14.0']
for include in ['src', 'include', 'third_party']:
    cmd += ['-I', str(root / include)]
cmd += [str(here / 'check.cpp'), str(root / 'build/libnaceinfer.a')]
for framework in ['Metal', 'Foundation', 'CoreFoundation', 'IOSurface', 'IOKit', 'ImageIO', 'CoreGraphics',
                  'Security', 'SystemConfiguration', 'Accelerate']:
    cmd += ['-framework', framework]
cmd += ['-o', str(out / 'check')]
with (out / 'build.log').open('w') as log:
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
env = {k: v for k, v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'MTL_'))}
env['NI_LOCK_HELD'] = '1'
records = [{'build_command': cmd}]
with open(os.environ.get('BENCH_GPU_LOCK', '/tmp/kapitsa-gpu.lock'), 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for label, reps in [('validation', 0)] + ([] if args.correctness_only else [('timing', 96)]):
        run_env = dict(env)
        if label == 'validation':
            run_env.update(MTL_SHADER_VALIDATION='1', MTL_DEBUG_LAYER='1')
        call = [str(out / 'check'), str(here / 'original-reduce.metal'), str(reps)]
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(call, env=run_env, stdout=log, stderr=subprocess.STDOUT)
        records.append({'label': label, 'command': call, 'exit': result.returncode})
        (out / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        result.check_returncode()
print('PASS: see', out)
