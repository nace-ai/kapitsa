#!/usr/bin/env python3
"""Own one private server under gpu_leg.py's lock and stop exactly that server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = Path('$REPO')


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pair', type=int, default=0)
    parser.add_argument('--workloads', default='0,4000,16000')
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--messages', type=Path, help='fixed messages JSON for an additional real API request')
    parser.add_argument('--suite', type=Path, help='JSON list of names and fixed messages file paths')
    parser.add_argument('--check-server', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    if os.environ.get('NI_LOCK_HELD') != '1' or os.environ.get('NACE_GPU_LOCK_HELD') != '1':
        parser.error('launch only through gpu_leg.py, which owns the GPU lock')
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env['KAPITSA_PARENT_PID'] = str(os.getpid())
    command = [str(args.server), 'serve', '--model', str(args.model), '--model-id', 'bench-model',
               '--host', '127.0.0.1', '--private', '--port', str(port), '--ctx', '32768',
               '--slots', '1', '--disk-cache', '0']
    with args.server.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    save(out / 'server-command.json', {'argv': command, 'binary_sha256': sha,
         'client_sha256': hashlib.sha256((ROOT / 'api_probe').read_bytes()).hexdigest(),
         'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'env': {k: v for k, v in env.items() if k.startswith(('NI_', 'NACE_'))},
         'KAPITSA_HOME': env['KAPITSA_HOME'], 'parent_pid': os.getpid()})
    server = None
    launched = time.time()
    try:
        with (out / 'server-stdout.txt').open('x') as stdout, (out / 'server-stderr.txt').open('x') as stderr:
            server = subprocess.Popen(command, env=env, cwd=REPO, stdout=stdout, stderr=stderr)
            save(out / 'server-pid.json', {'pid': server.pid, 'launched_epoch': launched})
            deadline = time.monotonic() + 240
            while True:
                if server.poll() is not None:
                    raise RuntimeError('our server exited before readiness')
                try:
                    replies = {}
                    for endpoint in ('health', 'v1/models'):
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/{endpoint}', timeout=2) as response:
                            replies[endpoint] = json.load(response)
                    break
                except (OSError, urllib.error.URLError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('our server did not become ready')
                    time.sleep(0.25)
            save(out / 'ready.json', {'epoch': time.time(), 'load_wall_s': time.time() - launched,
                                     'replies': replies})
            client = str(ROOT / 'api_probe')
            if args.check_server:
                with (out / 'server-tests.txt').open('x') as log:
                    subprocess.run([sys.executable, str(REPO / 'tests/test_server.py'),
                                    f'http://127.0.0.1:{port}', '--model-id', 'bench-model'],
                                   stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=True)
                return 0
            # Loaded-model timing on 4B must not mix prefill before and after its asynchronous ANE program is ready.
            startup = (out / 'server-stderr.txt').read_text()
            ane_wait_start = time.monotonic()
            if 'compiling the Neural Engine program' in startup:
                while not re.search(r'NI_ANE: (Neural Engine program ready|Neural Engine split unavailable)', startup):
                    if server.poll() is not None or time.monotonic() - ane_wait_start > 240:
                        raise RuntimeError('ANE preparation did not settle before warmup')
                    time.sleep(0.25)
                    startup = (out / 'server-stderr.txt').read_text()
            save(out / 'ane-preparation.json', {'wait_s': time.monotonic() - ane_wait_start,
                                               'ready': 'Neural Engine program ready' in startup})
            subprocess.run([client, 'warmup', str(port), str(out / 'warmup'), str(args.pair)],
                           cwd=REPO, check=True)
            subprocess.run([client, 'memory', str(port), str(out / 'memory-warm'), str(server.pid)],
                           cwd=REPO, check=True)
            contexts = [int(c) for c in args.workloads.split(',') if c]
            for context in contexts:
                nonce = f'overnight-p{args.pair:02d}-c{context}'
                subprocess.run([client, 'request', str(port), str(out / f'c{context}'), nonce,
                                str(context), str(args.tokens)], cwd=REPO, check=True)
            if args.messages:
                subprocess.run([client, 'request-file', str(port), str(out / 'custom'),
                                f'overnight-p{args.pair:02d}-custom', str(args.messages), str(args.tokens)],
                               cwd=REPO, check=True)
            if args.suite:
                for case in json.loads(args.suite.read_text()):
                    mark = len((out / 'server-stderr.txt').read_text())
                    subprocess.run([client, 'request-file', str(port), str(out / case['name']),
                                    f"overnight-p{args.pair:02d}-{case['name']}", case['messages'], str(args.tokens)],
                                   cwd=REPO, check=True)
                    # After the timed HTTP request, retain evidence of the path this request exercised.
                    time.sleep(0.05)
                    with (out / case['name'] / 'server-during.txt').open('x') as request_log:
                        request_log.write((out / 'server-stderr.txt').read_text()[mark:])
            subprocess.run([client, 'memory', str(port), str(out / 'memory-end'), str(server.pid)],
                           cwd=REPO, check=True)
            text = (out / 'server-stderr.txt').read_text()
            expected_window = os.environ.get('NI_DFLASH_WINDOW', '2048')
            expected = expected_window not in ('', 'auto', '2048')
            active = set(re.findall(r'NI_DFLASH_WINDOW=(\d+): v1 drafter', text))
            if active != ({expected_window} if expected else set()):
                raise RuntimeError('drafter-window server load line does not match the selected arm')
            return 0
    finally:
        if server is not None:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
            save(out / 'server-stopped.json', {'pid': server.pid, 'returncode': server.returncode,
                                              'epoch': time.time()})


if __name__ == '__main__':
    raise SystemExit(main())
