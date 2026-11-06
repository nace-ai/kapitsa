import datetime
import fcntl
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
import traceback
import urllib.request

P = Path(__file__).resolve().parent
ROOT = P.parent.parent
trees = {'base': P.parent / 'kflash-reduce-v8/baseline', 'candidate': ROOT / 'kflash-reduce-v8'}
model = P.parent / 'followup-1250/models/qwen3.5-4b-4bit'
cmake = '/Users/bytedance/Documents/ChatGPT/blog/work/sdk-build-tools/cmake/data/bin/cmake'
sys.path.insert(0, str(trees['candidate'] / 'bench/harness'))
import gpu_hogs

state = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': os.getpid(),
         'status': 'building', 'results': [], 'purpose': 'PR1023 missing continuation, HTTP and native benchmark coverage',
         'environment': 'Apple M4 Pro 48 GiB macOS15.7.1; actual packed 4B drafter path; remote M3 stock profile lacks it'}
def save():
    (P / 'STATE.json').write_text(json.dumps(state, indent=2) + '\n')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def run(cmd, label, env=None, timeout=1200, cwd=None):
    cmd = [str(x) for x in cmd]
    print('START', label, flush=True)
    (P / (label + '.command.json')).write_text(json.dumps({'argv': cmd, 'cwd': str(cwd or P)}, indent=2))
    with (P / (label + '.log')).open('w') as f:
        cp = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout, cwd=cwd)
    return cp.returncode
clean = {k: v for k, v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'KAPITSA_', 'MTL_', 'BENCH_'))}
clean.update(NI_LOCK_HELD='1', KAPITSA_NO_AUTO_UPDATE='1', KAPITSA_NO_PROMPT='1')
def env_for(label): return dict(clean, KAPITSA_HOME=str(P / ('home-' + label)))

save()
try:
    identities = {}
    for tag, tree in trees.items():
        assert run([cmake, '-S', tree, '-B', tree / 'build', '-DCMAKE_BUILD_TYPE=Release'], tag + '-configure', cwd=tree) == 0
        assert run([cmake, '--build', tree / 'build', '-j', '4', '--target', 'kapitsa', 'ni_run'], tag + '-build', cwd=tree) == 0
        identities[tag] = {'head': subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip(),
                           'version': subprocess.check_output([str(tree / 'build/kapitsa'), '--version'], text=True).strip(),
                           'ni_run_sha256': sha(tree / 'build/ni_run'), 'kapitsa_sha256': sha(tree / 'build/kapitsa'),
                           'source_sha256': {s: sha(tree / s) for s in ['src/model/dflash2.cpp', 'src/kernels/spec_dflash.metal']}}
    state['identities'] = identities
    state['model'] = {str(p.relative_to(model)): sha(p) for p in [model / 'config.json', model / 'tokenizer.json', *sorted((model / 'dflash2').glob('*.safetensors'))]}
    tree = trees['candidate']
    compile_cmd = ['/usr/bin/c++', '-std=c++20', '-O3', '-arch', 'arm64', '-mmacosx-version-min=14.0', '-I' + str(tree / 'src'),
                   P / 'tokenizer.cpp', tree / 'build/libnaceinfer.a', '-framework', 'Foundation', '-framework', 'CoreFoundation', '-o', P / 'tokenizer']
    assert run(compile_cmd, 'tokenizer-build') == 0
    prompt = '<|im_start|>user\nWrite a detailed tutorial explaining how a compiler transforms a small programming language into machine code. Cover lexical analysis, parsing, type checking, optimization, register allocation, and code generation. Include examples and explain each stage in depth. Write at least 2000 words.\n<|im_end|>\n<|im_start|>assistant\n'
    (P / 'prompt.txt').write_text(prompt)
    ids = subprocess.check_output([str(P / 'tokenizer'), str(model), 'encode', str(P / 'prompt.txt')], text=True)
    (P / 'prompt.ids').write_text(ids)
    state['prompt_tokens'] = len(ids.strip().split(','))
    cfg = json.loads((model / 'config.json').read_text())
    eos = cfg.get('text_config', cfg).get('eos_token_id', [248044, 248046])
    ends = {248044, 248046} | set(eos if isinstance(eos, list) else [eos])
    state['status'] = 'waiting for GPU lock'; save()
    with open('/tmp/kapitsa-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reference = None
        for arm, tag in enumerate(['base', 'candidate', 'base', 'candidate']):
            label = f'continuation-{arm}-{tag}'
            state['status'] = label; save()
            assert gpu_hogs.wait_quiet(maxpct=5, quiet_s=5, timeout_s=60, exclude_roots=[os.getpid()]), 'GPU remained busy'
            monitor = gpu_hogs.ContentionMonitor([os.getpid()], interval=0.5); monitor.start()
            try:
                rc = run([trees[tag] / 'build/ni_run', model, '@' + str(P / 'prompt.ids'), '256', '0', '--dflash', model / 'dflash2'], label, env_for(label))
            finally: monitor.stop()
            text = (P / (label + '.log')).read_text()
            match = re.search(r'^ids:(.*)$', text, re.M); assert match, label
            output = [int(x) for x in match.group(1).split()]
            (P / (label + '.ids')).write_text(' '.join(map(str, output)) + '\n')
            decoded = subprocess.check_output([str(P / 'tokenizer'), str(model), 'decode', str(P / (label + '.ids'))], text=True)
            (P / (label + '.txt')).write_text(decoded)
            hist = re.search(r'hist((?: \d+)+);', text); assert hist
            counts = [int(x) for x in hist.group(1).split()]
            row = {'label': label, 'exit': rc, 'tokens': len(output), 'eos_count': sum(x in ends for x in output),
                   'before_first_eos': next((i for i, x in enumerate(output) if x in ends), len(output)),
                   'text_nonempty': bool(decoded.strip()), 'ids_sha256': sha(P / (label + '.ids')),
                   'acceptance_histogram': counts, 'cycles': sum(counts),
                   'accepted_draft_tokens_per_cycle': sum(i * n for i, n in enumerate(counts)) / sum(counts),
                   'decode_line': next(l for l in text.splitlines() if 'dflash decode (' in l),
                   'drafter_lines': [l for l in text.splitlines() if 'drafter' in l or 'packed 39 static draft' in l],
                   'contention': gpu_hogs.summarize_samples(monitor.samples), 'samples': monitor.samples}
            state['results'].append(row); save()
            assert rc == 0 and len(output) == 256 and decoded.strip() and row['eos_count'] == 0, label
            assert 'packed 39 static draft' in text and 'DFlash2 drafter' in text, 'packed drafter not exercised'
            if reference is None: reference = output
            else: assert output == reference, 'continuations differ'
        (P / 'continuations-passed').touch()
        http = []
        for tag, tree in trees.items():
            label = 'http-' + tag
            state['status'] = label; save()
            env = env_for(label)
            with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            cmd = [str(tree / 'build/kapitsa'), 'serve', '--model', str(model), '--host', '127.0.0.1', '--port', str(port), '--private', '--slots', '2', '--context', '32768', '--disk-cache', '0']
            (P / (label + '-server.command.json')).write_text(json.dumps(cmd))
            with (P / (label + '-server.log')).open('w') as log:
                server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    url = f'http://127.0.0.1:{port}'
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    deadline = time.monotonic() + 300
                    while time.monotonic() < deadline:
                        assert server.poll() is None, 'server exited'
                        try:
                            with opener.open(url + '/health', timeout=2) as resp:
                                if resp.status == 200: break
                        except OSError: time.sleep(0.5)
                    else: raise TimeoutError(label)
                    rc = run([sys.executable, tree / 'tests/test_server.py', url], label, env, timeout=1800, cwd=tree)
                    lines = (P / (label + '.log')).read_text().splitlines()
                    row = {'label': label, 'exit': rc, 'passes': sum(l.startswith('PASS ') for l in lines),
                           'failures': [l for l in lines if l.startswith('FAIL ')]}
                    http.append(row); state['http'] = http; save()
                finally:
                    if server.poll() is None:
                        server.send_signal(signal.SIGINT)
                        try: server.wait(timeout=20)
                        except subprocess.TimeoutExpired: server.terminate(); server.wait(timeout=20)
        state['http_same_outcomes'] = http[0]['passes'] == http[1]['passes'] and [x.split('  ', 1)[0] for x in http[0]['failures']] == [x.split('  ', 1)[0] for x in http[1]['failures']]
        save()
        assert state['http_same_outcomes'], 'HTTP outcomes differ; review logs before proceeding'
        (P / 'http-complete').touch()
        for arm, tag in enumerate(['base', 'candidate', 'base', 'candidate']):
            label = f'bench-{arm}-{tag}'
            state['status'] = label; save()
            assert gpu_hogs.wait_quiet(maxpct=5, quiet_s=10, timeout_s=60, exclude_roots=[os.getpid()]), 'GPU busy before bench'
            rc = run([trees[tag] / 'build/kapitsa', 'bench', '--model', model, '--quick', '--json', '--out', P / (label + '.json')], label, env_for(label), timeout=1800)
            state.setdefault('bench', []).append({'label': label, 'exit': rc, 'file': label + '.json'}); save()
            assert rc == 0, label
    state['status'] = 'complete; review every result and benchmark flag before publication'; save()
    (P / 'complete').touch()
except BaseException as exc:
    state['status'] = 'failed'; state['error'] = repr(exc); save()
    (P / 'failed').write_text(traceback.format_exc())
    raise
