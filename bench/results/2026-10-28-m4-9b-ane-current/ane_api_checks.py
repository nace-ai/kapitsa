"""Local functional/quality checks for the optional ANE path; never speed evidence."""
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


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ane', action='store_true')
    p.add_argument('--native', action='store_true')
    p.add_argument('--slots', type=int, default=1)
    p.add_argument('--server-test', type=Path, default=REPO / 'tests/test_server.py')
    args = p.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    assert os.environ.get('NI_LOCK_HELD') == '1' and os.environ.get('NACE_GPU_LOCK_HELD') == '1'
    args.out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    command = [str(args.server), 'serve', '--model', str(args.model), '--model-id', 'ane-quality',
               '--host', '127.0.0.1', '--private', '--port', str(port), '--ctx', '32768',
               '--slots', str(args.slots), '--disk-cache', '0']
    save(args.out / 'command.json', {'argv': command, 'binary_sha256': hashlib.sha256(args.server.read_bytes()).hexdigest(),
         'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
         'env': {k: v for k, v in os.environ.items() if k.startswith(('NI_', 'NACE_', 'KAPITSA_'))},
         'purpose': 'Functional only. No times from this leg are speed evidence.'})
    server = None
    results = []
    native_results = []

    def log():
        return (args.out / 'server.log').read_text(errors='replace')

    def post(body):
        request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=240) as response:
            assert response.status == 200
            return json.load(response)

    try:
        with (args.out / 'server.log').open('x') as out:
            server = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT)
            save(args.out / 'server-pid.json', {'pid': server.pid})
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError('server exited before readiness')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('server health timeout')
                time.sleep(0.25)
            if 'compiling the Neural Engine program' in log():
                while not re.search(r'NI_ANE: (Neural Engine program ready|Neural Engine split unavailable)', log()):
                    if time.monotonic() > deadline or server.poll() is not None:
                        raise TimeoutError('ANE preparation timeout')
                    time.sleep(0.25)
            post({'model': 'ane-quality', 'messages': [{'role': 'user', 'content': 'hi'}],
                  'max_tokens': 1, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
            active = 'NI_ANE_H4096=1: experimental' in log()
            assert active == args.ane, 'wrong prefill path'
            for case in json.loads(args.cases.read_text()):
                mark = len(log())
                response = post(case['body'])
                choice = response['choices'][0]
                message = choice['message']
                usage = response['usage']
                error = None
                actual = None
                try:
                    if case['mode'] == 'tool':
                        calls = message.get('tool_calls', [])
                        assert len(calls) == 1 and calls[0]['function']['name'] == 'report_record'
                        actual = json.loads(calls[0]['function']['arguments'])
                        assert choice['finish_reason'] == 'tool_calls'
                    elif case['mode'] == 'json':
                        actual = json.loads(message['content'])
                        assert choice['finish_reason'] == 'stop'
                    else:
                        actual = message.get('content', '').strip()
                        assert choice['finish_reason'] == 'stop'
                    assert actual == case['expected'], f'expected {case["expected"]!r}, got {actual!r}'
                    assert usage['prompt_tokens'] >= 512 and usage['completion_tokens'] > 0
                except (AssertionError, KeyError, ValueError, TypeError) as exc:
                    error = str(exc) or type(exc).__name__
                # The log is emitted at completion; allow the writer a short scheduling turn.
                time.sleep(0.05)
                tail = log()[mark:]
                split = re.search(r'prefill: Neural Engine busy .*\((\d+) evaluations\)', tail)
                path_ok = bool(split and int(split.group(1)) > 0) if args.ane else split is None
                if not path_ok:
                    error = (error or '') + ' prefill path was not exercised as intended'
                if 'reruns on the GPU' in tail or 'has not finished' in tail:
                    error = (error or '') + ' unexpected ANE fallback'
                result = {'name': case['name'], 'mode': case['mode'], 'ok': error is None,
                          'expected': case['expected'], 'actual': actual, 'error': error,
                          'usage': usage, 'finish_reason': choice['finish_reason'],
                          'split_evals': int(split.group(1)) if split else 0, 'response': response}
                results.append(result)
                save(args.out / (case['name'] + '.json'), result)
                print(('PASS ' if result['ok'] else 'FAIL ') + case['name'] + ' ' + str(error or usage), flush=True)
            if args.native:
                for name, extra in [('test_server', ['--model-id', 'ane-quality']), ('test_structured', ['--quick'])]:
                    with (args.out / (name + '.log')).open('x') as native:
                        test_path = args.server_test if name == 'test_server' else REPO / 'tests/test_structured.py'
                        proc = subprocess.run([sys.executable, str(test_path), base, *extra],
                                              cwd=REPO, stdout=native, stderr=subprocess.STDOUT, timeout=1200)
                    native_results.append({'name': name, 'returncode': proc.returncode})
                    print(('PASS ' if proc.returncode == 0 else 'FAIL ') + name, flush=True)
            save(args.out / 'summary.json', {'cases': len(results), 'correct': sum(r['ok'] for r in results),
                                            'results': [{k: v for k, v in r.items() if k != 'response'} for r in results],
                                            'native': native_results})
    finally:
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        if server is not None:
            save(args.out / 'stopped.json', {'pid': server.pid, 'returncode': server.returncode})
    return int(any(not r['ok'] for r in results) or any(r['returncode'] for r in native_results))


if __name__ == '__main__':
    raise SystemExit(main())
