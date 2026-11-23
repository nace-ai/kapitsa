#!/usr/bin/env python3
"""Private PR1396 A/B evidence runner. --manifest-only never starts a server."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request

ROOT = Path('/Users/alisheramantay/Documents/nace/kapitsa')
BASE = Path(__file__).resolve().parent
MODEL = '/Volumes/Extreme Pro/downloads/rl1368-inkling2-mini'
COMMITS = {'A': '37f6f2fb1cd89541cc04019e63b5790332b87037',
           'B': '8eb6db9b44940f26d693d913c92f09e89d340229'}
BINARIES = {'A': ROOT / '.worktrees/k758-main-37f6f2fb/build/kapitsa',
            'B': ROOT / '.worktrees/review-1396-20261123-head/build/kapitsa'}
OPTIONS = {'temperature': 0, 'top_k': 1, 'top_p': 1, 'seed': 0, 'num_predict': 256}
SCHEDULE = [(n, arm, rep) for n in (4096, 32768) for rep in (1, 2) for arm in ('A', 'B')]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with Path(path).open('xb') as f:
        f.write(encoded(value) + b'\n')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command(binary, model=MODEL):
    return [str(binary), 'serve', '--model', model, '--profile', 'fast', '--context', '65536',
            '--slots', '1', '--disk-cache', '0', '--dflash', 'none', '--spec', '0',
            '--prefill-chunk', '512', '--idle-unload', '0', '--local', '--private', '--port', '18758']


def request_body(text):
    return {'model': '', 'messages': [{'role': 'user', 'content': text}],
            'stream': False, 'think': True, 'options': dict(OPTIONS)}


def count_body(body):
    if any(m.get('role') not in ('user', 'assistant') or not isinstance(m.get('content'), str)
           for m in body['messages']):
        raise ValueError('count equivalence supports plain user/assistant messages only')
    if body.get('tools') or not isinstance(body.get('think'), bool):
        raise ValueError('count equivalence needs no tools and an explicit boolean think')
    return {'model': body.get('model', ''), 'messages': body['messages'],
            'thinking': {'type': 'enabled' if body['think'] else 'disabled'}}


def http(path, body=None, timeout=60, raw_path=None):
    data = None if body is None else body if isinstance(body, bytes) else encoded(body)
    req = urllib.request.Request('http://127.0.0.1:18758' + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        if raw_path:
            Path(raw_path).write_bytes(raw)
        raise RuntimeError(f'HTTP {exc.code}: {raw[:2000]!r}') from exc
    if raw_path:
        with Path(raw_path).open('xb') as f:
            f.write(raw)
    return json.loads(raw)


def capture(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {'argv': argv, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except Exception as exc:
        return {'argv': argv, 'error': str(exc)}


def resource_snapshot():
    commands = {'power': ['pmset', '-g', 'batt'], 'powermode': ['pmset', '-g', 'custom'],
                'free_percent': ['memory_pressure', '-Q'], 'thermal': ['pmset', '-g', 'therm'],
                'pressure': ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                'free_pages': ['sysctl', '-n', 'vm.page_free_count'],
                'page_size': ['sysctl', '-n', 'hw.pagesize'],
                'swap': ['sysctl', 'vm.swapusage'], 'vm_stat': ['vm_stat'],
                'storage': ['df', '-k', str(BASE)], 'host': ['sw_vers'],
                'chip': ['sysctl', '-n', 'machdep.cpu.brand_string']}
    result = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'commands': {k: capture(v) for k, v in commands.items()}}
    c = result['commands']
    failures = []
    if "'AC Power'" not in c['power'].get('stdout', ''):
        failures.append('AC power absent or unverified')
    try:
        free = int(c['free_pages']['stdout']) * int(c['page_size']['stdout']) / 2**30
        result['free_gib'] = free
    except (KeyError, ValueError):
        result['free_gib'] = None
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', c['free_percent'].get('stdout', ''))
    if not match or int(match[1]) < 3:
        failures.append('memory free percentage below 3 or unverified')
    ac_settings = c['powermode'].get('stdout', '').split('AC Power:', 1)
    mode = re.search(r'\bpowermode\s+(\d+)', ac_settings[1] if len(ac_settings) == 2 else '')
    if not mode or int(mode[1]) != 2:
        failures.append('AC powermode is not 2 or unverified')
    thermal = c['thermal']
    therm_text = thermal.get('stdout', '')
    if thermal.get('returncode') != 0 or not therm_text.strip():
        failures.append('thermal state unverified')
    for field in ('CPU_Speed_Limit', 'CPU_Scheduler_Limit', 'GPU_Speed_Limit'):
        for value in re.findall(field + r'\s*=\s*(\d+)', therm_text):
            if int(value) < 100:
                failures.append('thermal throttling ' + field)
    for field in ('Thermal_Level', 'CPU_Available_CPUs'):
        if field == 'Thermal_Level' and re.search(field + r'\s*=\s*[1-9]', therm_text):
            failures.append('thermal warning')
    if re.search(r'(?:thermal|performance) warning level\s*[:=]\s*[1-9]', therm_text, re.I):
        failures.append('thermal warning')
    try:
        pressure = int(c['pressure']['stdout'])
        result['pressure_level'] = pressure
        if pressure >= 4:
            failures.append('memory pressure level at least 4')
    except (KeyError, ValueError):
        failures.append('memory pressure unverified')
    match = re.search(r'used\s*=\s*([\d.]+)([KMGT])', c['swap'].get('stdout', ''))
    if match:
        used = float(match[1]) * {'K': 2**10, 'M': 2**20, 'G': 2**30, 'T': 2**40}[match[2]]
        result['swap_used_gib'] = used / 2**30
        if used > 16 * 2**30:
            failures.append('swap exceeds 16 GiB')
    else:
        failures.append('swap unverified')
    result['timing_invalid_reasons'] = failures
    return result


def parse_result(response, target):
    if response.get('error') or response.get('done') is not True:
        raise ValueError('response error or incomplete response')
    if response.get('prompt_eval_count') != target:
        raise ValueError(f'actual prompt count {response.get("prompt_eval_count")} != {target}')
    count, duration = response.get('eval_count'), response.get('eval_duration')
    if type(count) is not int or count != 256:
        raise ValueError('output token count is not exactly 256')
    if response.get('done_reason') != 'length':
        raise ValueError('completion did not terminate at the requested token limit')
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid decode duration')
    cached = response.get('prompt_eval_cached_count', 0)
    if cached != 0:
        raise ValueError('measured request reused cached tokens')
    message = response.get('message', {})
    generated = {'thinking': message.get('thinking', ''), 'content': message.get('content', '')}
    if any(not isinstance(v, str) for v in generated.values()) or not any(generated.values()):
        raise ValueError('empty or invalid generated text')
    return {'actual_prompt_tokens': target, 'eval_count': count, 'eval_duration_ns': duration,
            'decode_tps': (count - 1) * 1e9 / duration, 'cached_tokens': cached,
            'cache_counter_present': 'prompt_eval_cached_count' in response,
            'generated_text_sha256': sha(encoded(generated)),
            'canonical_text_format': 'UTF-8 JSON, sorted keys content/thinking, compact separators, ensure_ascii=False',
            'generated_utf8_bytes': {k: len(v.encode('utf-8')) for k, v in generated.items()}}


class ResourceAbort(RuntimeError):
    pass


def check_port_available(port=18758):
    with socket.socket() as sock:
        # Match the server's reusable binding; TIME_WAIT is not an active listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))


class Server:
    def __init__(self, binary, directory, startup_timeout, request_timeout, model=MODEL):
        self.binary, self.directory, self.model = binary, directory, model
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self.process = None
        self.log = None
        self.stop_guard = threading.Event()
        self.guard_worker = None
        self.guard_error = None
        self.initial_swap = 0
        self.guard_count = 0

    def check_snapshot(self, snapshot):
        reasons = list(snapshot['timing_invalid_reasons'])
        if snapshot.get('swap_used_gib', 0) - self.initial_swap > 1:
            reasons.append('swap grew more than 1 GiB during server lifecycle')
        if reasons:
            self.guard_error = '; '.join(reasons)
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            raise ResourceAbort(self.guard_error)

    def monitor(self):
        while not self.stop_guard.wait(5):
            try:
                snap = resource_snapshot()
                self.guard_count += 1
                save(self.directory / f'guard-{self.guard_count:04d}.json', snap)
                self.check_snapshot(snap)
            except Exception as exc:
                self.guard_error = self.guard_error or 'resource guard failed: ' + str(exc)
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                return

    def __enter__(self):
        self.directory.mkdir()
        env = dict(os.environ)
        env['KAPITSA_HOME'] = str(self.directory / 'home')
        env['KAPITSA_PROFILE'] = 'fast'
        save(self.directory / 'launch.json', {'argv': command(self.binary, self.model), 'env': {
            k: v for k, v in env.items() if k.startswith(('NI_', 'NACE_GPU_', 'KAPITSA_', 'BENCH_GPU_'))}})
        check_port_available()
        initial = resource_snapshot()
        save(self.directory / 'prelaunch.json', initial)
        self.initial_swap = initial.get('swap_used_gib', 0)
        self.check_snapshot(initial)
        self.log = (self.directory / 'server.log').open('xb')
        self.process = subprocess.Popen(command(self.binary, self.model), stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self.guard_worker = threading.Thread(target=self.monitor, name='resource-guard', daemon=True)
        self.guard_worker.start()
        try:
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'server exited {self.process.returncode}')
                try:
                    http('/health', timeout=2)
                    return self
                except (OSError, ValueError, RuntimeError):
                    time.sleep(1)
            raise TimeoutError('server startup timeout')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        self.stop_guard.set()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=30)
            save(self.directory / 'server-exit.json', {'returncode': self.process.returncode, 'pid': self.process.pid})
        if self.log:
            self.log.close()
        if self.guard_worker:
            self.guard_worker.join(timeout=180)
            if self.guard_worker.is_alive():
                self.guard_error = self.guard_error or 'resource guard failed to join'
        if self.guard_error:
            raise ResourceAbort(self.guard_error)

    def count(self, body):
        value = http('/v1/messages/count_tokens', count_body(body), timeout=60)
        n = value.get('input_tokens')
        if type(n) is not int:
            raise ValueError('invalid token count response')
        return n


def fit_prompt(server, target):
    prefix = 'Continue this document with a detailed numbered technical analysis. Do not conclude early.\n'
    suffix = '\nContinue for at least 1000 words without a conclusion.'
    def body(n):
        return request_body(prefix + ' x' * n + suffix)
    lo, hi = 0, target * 2
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = body(mid)
        n = server.count(candidate)
        if n == target:
            return encoded(candidate)
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    raise ValueError(f'cannot fit an exact {target}-token private prompt; no relabeling permitted')


def check_body(body):
    if body.get('stream') is not False or body.get('options') != OPTIONS:
        raise ValueError('fixture must already have exact greedy measurement options and stream:false')
    count_body(body)


def summarize(records):
    result = {'pass': False, 'slowdown_gate': 0.02, 'groups': {}}
    for n in (4096, 32768):
        rows = [r for r in records if r['prompt_tokens'] == n]
        valid = len(rows) == 4 and all(r.get('status') == 'ok' for r in rows)
        group = {'valid': valid}
        if valid:
            hashes = {r['measurement']['generated_text_sha256'] for r in rows}
            rates = {arm: [r['measurement']['decode_tps'] for r in rows if r['arm'] == arm]
                     for arm in ('A', 'B')}
            ratio = statistics.mean(rates['B']) / statistics.mean(rates['A'])
            group.update({'rates': rates, 'head_over_main': ratio, 'same_generated_text': len(hashes) == 1,
                          'paired_head_over_main': [rates['B'][i] / rates['A'][i] for i in range(2)],
                          'pass': len(hashes) == 1 and ratio >= 0.98})
        result['groups'][str(n)] = group
    result['pass'] = all(g.get('pass', False) for g in result['groups'].values())
    return result


def run(args):
    if args.manifest_only:
        print(json.dumps({'commits': COMMITS, 'argv': {k: command(v, args.model) for k, v in BINARIES.items()},
                          'schedule': SCHEDULE, 'options': OPTIONS}, indent=2))
        return 0
    if os.environ.get('NI_LOCK_HELD') != '1' or os.environ.get('NACE_GPU_LOCK_HELD') != '1':
        raise RuntimeError('parent must hold verified GPU lock and supply both inherited lock markers')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        identities = {}
        for arm, binary in BINARIES.items():
            git = capture(['git', '-C', str(binary.parent.parent), 'rev-parse', 'HEAD'])
            if git.get('returncode') != 0 or git.get('stdout', '').strip() != COMMITS[arm]:
                raise ValueError(f'{arm} worktree is not pinned commit')
            identities[arm] = {'binary': str(binary), 'sha256': file_sha(binary), 'git': git}
        metadata = {name: file_sha(Path(args.model) / name) for name in ('config.json', 'tokenizer.json')}
        save(output / 'identity.json', {'binaries': identities, 'model': args.model, 'metadata_sha256': metadata})
        fixtures = {}
        with Server(BINARIES['A'], output / 'prepare', args.startup_timeout, args.request_timeout, args.model) as server:
            for target in (4096, 32768):
                if target == 4096 and args.reuse_4096:
                    original = Path(args.reuse_4096).read_bytes()
                    source = json.loads(original)
                    try:
                        check_body(source)
                        raw = original
                    except ValueError:
                        body = request_body('')
                        body['messages'] = source['messages']
                        body['think'] = source.get('think', source.get('chat_template_kwargs', {}).get('enable_thinking', True))
                        raw = encoded(body)
                    save(output / 'reuse-origin.json', {'path': str(Path(args.reuse_4096).resolve()),
                         'sha256': sha(original), 'preserved_full_request_bytes': raw == original,
                         'messages_sha256': sha(encoded(source['messages']))})
                    if server.count(json.loads(raw)) != target:
                        raise ValueError('existing 4096 messages do not count to 4096; frozen input not modified')
                else:
                    raw = fit_prompt(server, target)
                check_body(json.loads(raw))
                fixtures[target] = raw
                with (output / f'request-{target}.json').open('xb') as f:
                    f.write(raw)
        save(output / 'fixtures.json', {'sha256': {str(n): sha(raw) for n, raw in fixtures.items()},
                                       'reuse_4096': args.reuse_4096,
                                       'frozen_before_first_measured_leg': True})
        for target, arm, rep in SCHEDULE:
            label = f'{target}-{arm}{rep}'
            record = {'label': label, 'prompt_tokens': target, 'arm': arm, 'repeat': rep, 'status': 'error'}
            records.append(record)
            try:
                with Server(BINARIES[arm], output / label, args.startup_timeout, args.request_timeout, args.model) as server:
                    record['count_preflight'] = server.count(json.loads(fixtures[target]))
                    if record['count_preflight'] != target:
                        raise ValueError('arm tokenizer count differs from frozen fixture target')
                    warmup = request_body('Zebra warmup. Explain how CPUs execute machine instructions in detail.')
                    warmup['options']['num_predict'] = 32
                    http('/api/chat', warmup, args.request_timeout, server.directory / 'warmup-response.json')
                    before = resource_snapshot()
                    save(server.directory / 'before.json', before)
                    server.check_snapshot(before)
                    raw = fixtures[target]
                    if sha((output / f'request-{target}.json').read_bytes()) != sha(raw):
                        raise ValueError('frozen fixture changed')
                    response = http('/api/chat', raw, args.request_timeout, server.directory / 'response.json')
                    after = resource_snapshot()
                    save(server.directory / 'after.json', after)
                    server.check_snapshot(after)
                    measurement = parse_result(response, target)
                    record['measurement'] = measurement
                    record['status'] = 'ok'
            except ResourceAbort as exc:
                record['status'] = 'resource_abort'
                record['error'] = str(exc)
                raise
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            finally:
                save(output / f'{label}.json', record)
                print(json.dumps(record, sort_keys=True), flush=True)
        if metadata != {name: file_sha(Path(args.model) / name) for name in metadata}:
            raise ValueError('model metadata changed during run')
        summary = summarize(records)
        save(output / 'summary.json', summary)
        return 0 if summary['pass'] else 1
    except BaseException as exc:
        save(output / 'fatal.json', {'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        for target, arm, rep in SCHEDULE:
            label = f'{target}-{arm}{rep}'
            if not any(r['label'] == label for r in records):
                save(output / f'{label}.json', {'label': label, 'prompt_tokens': target, 'arm': arm,
                                               'repeat': rep, 'status': 'not_run', 'error': 'earlier fatal error'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--output', default=str(BASE / 'ab-run'))
    parser.add_argument('--reuse-4096', default='/Users/alisheramantay/.agents/state/k754/request-4096.json')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--startup-timeout', type=float, default=900)
    parser.add_argument('--request-timeout', type=float, default=3600)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
