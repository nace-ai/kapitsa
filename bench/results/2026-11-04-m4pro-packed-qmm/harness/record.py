#!/usr/bin/env python3
"""Run a validation command with an exclusive GPU lock and an immutable transcript."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'build-m4pro-validation-20261104'
EVID = WORK / 'evidence'


def run(label, command, extra=None, gpu=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith('NI_')}
    env.pop('NACE_GPU_LOCK_HELD', None)
    env.pop('KAPITSA_BENCH_SERVER', None)
    env.update(KAPITSA_NO_AUTO_UPDATE='1', KAPITSA_NO_PROMPT='1',
               KAPITSA_HOME=str(WORK / 'home'))
    env.update(extra or {})
    meta = EVID / (label + '.command.json')
    log = EVID / (label + '.log')
    if meta.exists() or log.exists():
        raise RuntimeError(f'Label already exists: {label}')
    info = {'command': [str(x) for x in command], 'cwd': str(ROOT),
            'environment_overrides': extra or {}, 'gpu_lock': gpu,
            'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    executable = Path(command[0])
    if executable.is_file():
        info['executable_sha256'] = hashlib.sha256(executable.read_bytes()).hexdigest()
    meta.write_text(json.dumps(info, indent=2)+'\n')
    lock = None
    start = time.monotonic()
    try:
        if gpu:
            lock = open(env.get('BENCH_GPU_LOCK', '/tmp/kapitsa-gpu.lock'), 'a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env['NI_LOCK_HELD'] = '1'
            env['NACE_GPU_LOCK_HELD'] = '1'
        print('Running', label, flush=True)
        with log.open('x') as out:
            p = subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                               stdout=out, stderr=subprocess.STDOUT)
        info['exit_code'] = p.returncode
        print(f'{label}: exit {p.returncode}, {time.monotonic()-start:.1f}s', flush=True)
        print(log.read_text(errors='replace')[-2000:], flush=True)
        return p.returncode
    finally:
        info['elapsed_s'] = time.monotonic()-start
        info['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta.write_text(json.dumps(info, indent=2)+'\n')
        if lock:
            lock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        ap.error('missing command')
    raise SystemExit(run(args.label, command, dict(x.split('=', 1) for x in args.env), not args.cpu))


if __name__ == '__main__':
    main()
