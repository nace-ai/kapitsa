import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=False)
SHARED = Path('/Users/bytedance/kapitsa-ngram-remote-20261107')
AFFINE = Path('/Users/bytedance/kapitsa-ngram-affine-20261107/trained-1927')
MODEL = SHARED / 'model-sushi'
state = {'pid': os.getpid(), 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'status': 'waiting for preceding trained-model jobs', 'commands': [], 'comparisons': []}
def save():
    (OUT / 'STATE.json').write_text(json.dumps(state, indent=2) + '\n')
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()
def run(cmd, label, check=True):
    cmd = list(map(str, cmd))
    state['status'] = label
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'KAPITSA_', 'MTL_', 'BENCH_'))}
    env.update(NI_LOCK_HELD='1', KAPITSA_NO_AUTO_UPDATE='1', KAPITSA_NO_PROMPT='1', KAPITSA_HOME=str(OUT / 'home'))
    with (OUT / (label + '.log')).open('x') as log:
        cp = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['child_pid'] = cp.pid
        save()
        try:
            cp.wait(timeout=2400)
        except BaseException:
            if cp.poll() is None:
                os.killpg(cp.pid, signal.SIGTERM)
                try:
                    cp.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(cp.pid, signal.SIGKILL)
                    cp.wait()
            raise
    state['commands'].append({'label': label, 'argv': cmd, 'exit': cp.returncode,
                              'environment': {k: v for k, v in env.items() if k.startswith(('NI_', 'KAPITSA_'))}})
    save()
    if check:
        assert cp.returncode == 0, (label, cp.returncode)
    return cp.returncode
def fp(path):
    import numpy as np
    a = np.memmap(path, dtype='<f4', mode='r')
    nonzero = 0
    for i in range(0, a.size, 1 << 20):
        v = a[i:i+(1 << 20)]
        assert np.isfinite(v).all(), path
        nonzero += int(np.count_nonzero(v))
    assert nonzero, path
    return {'bytes': Path(path).stat().st_size, 'sha256': sha(path), 'finite': True, 'nonzero_values': nonzero}
save()
try:
    identity = json.loads((ROOT / 'identity.json').read_text())
    for name, digest in identity['files'].items():
        assert sha(ROOT / name) == digest, name
    state['identity'] = identity
    state['host'] = {key: subprocess.check_output(cmd, text=True).strip() for key, cmd in [
        ('macos', ['sw_vers']), ('chip', ['sysctl', '-n', 'machdep.cpu.brand_string']),
        ('ram_bytes', ['sysctl', '-n', 'hw.memsize'])]}
    save()
    while not (AFFINE / 'passed').exists():
        assert not (AFFINE / 'failed').exists(), 'preceding affine job failed; inspect shared fixture before continuing'
        assert not (SHARED / 'gpu-failed.json').exists(), 'shared coverage preparation failed'
        time.sleep(20)
    assert (SHARED / 'weights-passed').exists() and (SHARED / 'gpu-passed').exists()
    assert shutil.disk_usage(ROOT).free > 6 << 30, 'retain 6GiB disk guard'
    sys.path.insert(0, str(SHARED))
    from table_rows import TableRows
    table = TableRows()
    corpus = json.loads((SHARED / 'evidence/corpus-base.json').read_text())
    state['coverage'] = table.check(corpus['rows'])
    state['input_scope'] = 'Pinned actual model weights and explicitly covered corpus rows only. Sparse table is not a complete checkpoint. All model outputs below are computed by PR1023 reference and candidate binaries.'
    state['status'] = 'waiting for remote GPU lock'
    save()
    with open('/tmp/kapitsa-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (OUT / 'lock-acquired').touch()
        for tag in ['base', 'candidate']:
            cmd = [ROOT / ('logit_dump-' + tag), MODEL, OUT / ('logits-' + tag), '--paths', 'step,verify', '--corpus', ROOT / 'corpus.txt']
            if tag == 'candidate':
                cmd += ['--from', OUT / 'logits-base']
            run(cmd, 'logits-' + tag)
        for name in ['prefill', 'step', 'verify']:
            a, b = [OUT / ('logits-' + tag) / name for tag in ['base', 'candidate']]
            assert (a / 'sequences.txt').read_bytes() == (b / 'sequences.txt').read_bytes()
            sequences = [[int(x) for x in re.split(r'[,\s]+', s.strip()) if x] for s in (a / 'sequences.txt').read_text().splitlines() if s.strip()]
            assert sequences == corpus['sequences'], 'corpus does not match prepared row coverage'
            assert (a / 'records.txt').read_bytes() == (b / 'records.txt').read_bytes()
            fa, fb = fp(a / 'logits.f32'), fp(b / 'logits.f32')
            assert fa == fb, name
            state['comparisons'].append({'path': name, **fa, 'byte_equal': True,
                                          'records': json.loads((a / 'meta.json').read_text())['records']})
            save()
        state['comparator_exit'] = run([sys.executable, ROOT / 'logit_compare.py', OUT / 'logits-base', OUT / 'logits-candidate', '--json', OUT / 'compare.json'], 'compare', check=False)
        assert state['comparator_exit'] == 0, 'comparison failed'
        state['coverage_after'] = table.check(corpus['rows'])
        save()
        for tag in ['base', 'candidate']:
            for name in ['prefill', 'step', 'verify']:
                (OUT / ('logits-' + tag) / name / 'logits.f32').unlink()
    (OUT / 'lock-released').touch()
    state['status'] = 'complete; own Flash-Next covered-window comparison only, no full-checkpoint, native speed, arbitrary HTTP or packed-drafter claim'
    save()
    (OUT / 'passed').touch()
except BaseException as e:
    state['status'] = 'failed'
    state['error'] = repr(e)
    save()
    (OUT / 'failed').write_text(traceback.format_exc())
    raise
