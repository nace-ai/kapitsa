import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

cfg = json.loads(Path(sys.argv[1]).read_text())
P = Path(cfg['output'])
P.mkdir(exist_ok=False)
trees = {k: Path(v) for k, v in cfg['trees'].items()}
builds = {k: Path(v) for k, v in cfg['builds'].items()}
sys.path.insert(0, str(trees['candidate'] / 'bench/harness'))
import gpu_hogs

state = {'pid': os.getpid(), 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'status': 'waiting for builds', 'config': cfg, 'commands': [], 'models': []}
def save():
    tmp = P / 'STATE.tmp'
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(P / 'STATE.json')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()
clean = {k: v for k, v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'KAPITSA_', 'MTL_', 'BENCH_'))}
clean.update(NI_LOCK_HELD='1', KAPITSA_NO_AUTO_UPDATE='1', KAPITSA_NO_PROMPT='1')
def env_for(label):
    return dict(clean, KAPITSA_HOME=str(P / ('home-' + label)))
def stop(cp):
    if cp.poll() is None:
        os.killpg(cp.pid, signal.SIGTERM)
        try: cp.wait(timeout=25)
        except subprocess.TimeoutExpired: os.killpg(cp.pid, signal.SIGKILL); cp.wait()
def run(cmd, label, cwd=None, extra=None, timeout=2400, required=True):
    cmd = list(map(str, cmd)); cwd = Path(cwd or trees['candidate'])
    env = env_for(label); env.update(extra or {})
    state['status'] = label; save()
    rec = {'label': label, 'argv': cmd, 'cwd': str(cwd), 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'environment': {k: v for k, v in env.items() if k.startswith(('NI_', 'KAPITSA_', 'MTL_', 'BENCH_'))}}
    (P / (label + '.command.json')).write_text(json.dumps(rec, indent=2) + '\n')
    with (P / (label + '.log')).open('x') as log:
        cp = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['child_pid'] = cp.pid; save()
        try: cp.wait(timeout=timeout)
        except BaseException: stop(cp); raise
    rec.update(exit=cp.returncode, completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state['commands'].append(rec); state.pop('child_pid', None); save()
    if required: assert cp.returncode == 0, (label, cp.returncode)
    return cp.returncode

def http(model, short):
    rows = []
    for tag, build in builds.items():
        label = f'{short}-http-{tag}'
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        cmd = [str(build / 'kapitsa'), 'serve', '--model', str(model), '--host', '127.0.0.1', '--port', str(port),
               '--private', '--slots', '2', '--ctx', '32768', '--disk-cache', '0']
        (P / (label + '-server.command.json')).write_text(json.dumps({'argv': cmd, 'environment': {k:v for k,v in env_for(label).items() if k.startswith(('NI_', 'KAPITSA_'))}}, indent=2) + '\n')
        with (P / (label + '-server.log')).open('x') as log:
            server = subprocess.Popen(cmd, cwd=trees[tag], env=env_for(label), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['server_pid'] = server.pid; save()
            try:
                url = f'http://127.0.0.1:{port}'
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    assert server.poll() is None, f'{label}: server exited'
                    try:
                        with opener.open(url + '/health', timeout=2) as resp:
                            if resp.status == 200: break
                    except OSError: time.sleep(1)
                else: raise TimeoutError(label)
                rc = run([sys.executable, trees[tag] / 'tests/test_server.py', url], label, cwd=trees[tag], required=False)
                lines = (P / (label + '.log')).read_text().splitlines()
                rows.append({'label': label, 'exit': rc, 'passes': sum(x.startswith('PASS ') for x in lines), 'failures': [x for x in lines if x.startswith('FAIL ')]})
            finally: stop(server); state.pop('server_pid', None); save()
    return rows

save()
try:
    deadline = time.monotonic() + 1800
    while not all(Path(x).exists() for x in cfg['ready_files']):
        if time.monotonic() > deadline: raise TimeoutError('build markers')
        time.sleep(5)
    state['identities'] = {}
    for tag in trees:
        actual = subprocess.check_output(['git', '-C', str(trees[tag]), 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == cfg['heads'][tag], (tag, actual)
        state['identities'][tag] = {'head': actual, 'version': subprocess.check_output([str(builds[tag] / 'kapitsa'), '--version'], text=True).strip(),
                                    'binaries': {x: sha(builds[tag] / x) for x in ['kapitsa', 'ni_run', 'logit_dump']},
                                    'sources': {x: sha(trees[tag] / x) for x in ['src/model/affine_pack.h', 'src/model/qwen35_spec.cpp']}}
    run(['/usr/bin/c++', '-std=c++20', '-O3', '-arch', 'arm64', '-mmacosx-version-min=14.0', '-I' + str(trees['candidate'] / 'src'),
         cfg['tokenizer_source'], builds['candidate'] / 'libnaceinfer.a', '-framework', 'Foundation', '-framework', 'CoreFoundation', '-o', P / 'tokenizer'], 'tokenizer-build')
    prompt = '<|im_start|>user\nWrite a detailed tutorial explaining how a compiler transforms a small programming language into machine code. Cover lexical analysis, parsing, type checking, optimization, register allocation, and code generation. Include examples and explain each stage in depth. Write at least 2000 words.\n<|im_end|>\n<|im_start|>assistant\n'
    (P / 'prompt.txt').write_text(prompt)
    state['status'] = 'waiting for GPU lock'; save()
    with open('/tmp/kapitsa-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state['gpu_lock_acquired_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
        if cfg.get('loader_test'):
            run([builds['candidate'] / 'load_error_test', '--mtp-gpu'], 'mtp-loader-validation', extra={'MTL_SHADER_VALIDATION':'1','MTL_DEBUG_LAYER':'1'})
        for spec in cfg['models']:
            model = Path(spec['path']); short = spec['name']
            assert shutil.disk_usage(P).free > cfg.get('min_free_bytes', 6 << 30), 'free disk guard'
            entry = {'name': short, 'path': str(model), 'metadata': {f: sha(model / f) for f in ['config.json', 'tokenizer.json']}, 'mtp_requested': spec['mtp']}
            if (model / 'mtp.safetensors').exists(): entry['mtp_sha256'] = sha(model / 'mtp.safetensors')
            state['models'].append(entry); save()
            for tag in trees:
                out = P / f'{short}-logits-{tag}'
                cmd = [builds[tag] / 'logit_dump', model, out, '--corpus', trees[tag] / 'bench/corpus.txt']
                if tag == 'candidate': cmd += ['--from', P / f'{short}-logits-base']
                run(cmd, f'{short}-logits-{tag}', cwd=trees[tag])
            report = P / f'{short}-compare.json'
            run([sys.executable, trees['candidate'] / 'tools/logit_compare.py', P / f'{short}-logits-base', P / f'{short}-logits-candidate', '--json', report], f'{short}-compare')
            data = json.loads(report.read_text()); entry['logits'] = data
            assert data['paths'] and all(v['records'] > 0 and v['records'] == v['bitwise'] and not v['non_finite'] for v in data['paths'].values()), 'inspect numeric differences'
            for tag in trees:
                for f in (P / f'{short}-logits-{tag}').rglob('*.f32'): f.unlink()
            ids = subprocess.check_output([str(P / 'tokenizer'), str(model), 'encode', str(P / 'prompt.txt')], text=True)
            (P / f'{short}-prompt.ids').write_text(ids)
            ref = None; entry['continuations'] = []
            for arm, tag in enumerate(['base', 'candidate', 'base', 'candidate']):
                label = f'{short}-continuation-{arm}-{tag}'
                assert gpu_hogs.wait_quiet(maxpct=5, quiet_s=5, timeout_s=120, exclude_roots=[os.getpid()]), 'GPU busy'
                mon = gpu_hogs.ContentionMonitor([os.getpid()], interval=0.5); mon.start()
                try:
                    run([builds[tag] / 'ni_run', model, '@' + str(P / f'{short}-prompt.ids'), '256', '0', '--dflash', 'none', '--spec', '2' if spec['mtp'] else '0'], label, cwd=trees[tag])
                finally: mon.stop()
                text = (P / (label + '.log')).read_text(); match = re.search(r'^ids:(.*)$', text, re.M); assert match
                output = [int(x) for x in match.group(1).split()]
                (P / (label + '.ids')).write_text(' '.join(map(str, output)) + '\n')
                decoded = subprocess.check_output([str(P / 'tokenizer'), str(model), 'decode', str(P / (label + '.ids'))], text=True)
                (P / (label + '.txt')).write_text(decoded)
                row = {'label': label, 'tokens': len(output), 'text_nonempty': bool(decoded.strip()), 'eos_count': sum(x in {248044,248046} for x in output),
                       'before_first_eos': next((i for i,x in enumerate(output) if x in {248044,248046}), len(output)), 'ids_sha256': sha(P / (label + '.ids')),
                       'load_lines': [x for x in text.splitlines() if 'MTP' in x or 'mtp' in x], 'decode_lines': [x for x in text.splitlines() if 'tok/s' in x],
                       'contention': gpu_hogs.summarize_samples(mon.samples), 'samples': mon.samples}
                entry['continuations'].append(row); save()
                assert len(output) == 256 and decoded.strip() and row['eos_count'] == 0, 'invalid continuation'
                if spec['mtp']: assert 'spec decode D=2:' in text, 'MTP not exercised'
                if ref is None: ref = output
                else: assert output == ref, 'continuations differ'
            entry['http'] = http(model, short); save()
            outcomes = [(x['passes'], [re.split(r'  +', z)[0] for z in x['failures']]) for x in entry['http']]
            entry['http_same_outcomes'] = outcomes[0] == outcomes[1]; save()
            if not entry['http_same_outcomes']: entry['http_needs_investigation'] = True
            save()
            entry['benchmarks'] = []
            for arm, tag in enumerate(['base','candidate','base','candidate']):
                label = f'{short}-bench-{arm}-{tag}'
                assert gpu_hogs.wait_quiet(maxpct=5, quiet_s=10, timeout_s=120, exclude_roots=[os.getpid()]), 'GPU busy'
                out = P / (label + '.json')
                rc = run([builds[tag] / 'kapitsa', 'bench', '--model', model, '--quick', '--json', '--out', out], label, cwd=trees[tag], required=False)
                data = json.loads(out.read_text()); entry['benchmarks'].append({'label':label,'exit':rc, **{k:data[k] for k in ['kapitsa','model','results','conditions','flags','invalid']}}); save()
            assert entry['metadata'] == {f:sha(model/f) for f in entry['metadata']}
    state['status'] = 'complete; inspect HTTP failures and all benchmark flags before Ready'; save(); (P / 'complete').touch()
except BaseException as exc:
    state['status'] = 'failed'; state['error'] = repr(exc); save(); (P / 'failed').write_text(traceback.format_exc()); raise
