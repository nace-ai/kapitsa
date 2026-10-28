"""Latest-head native API, opt-in abort/fallback and SSD isolation checks."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('$PREVIOUS_RUN')
REPO = Path('$REPO')
PROGRESS = ROOT / 'evidence/integration-progress.json'
progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else []


def save(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Refusing to change saved evidence: {path}'
        return
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def run(label, command, env=(), timeout=2400):
    cmd = [sys.executable, str(ROOT / 'gpu_leg_startup.py'), label, '--quiet', '0',
           '--sample-ms', '1000', '--timeout', str(timeout)]
    for value in env:
        cmd += ['--env', value]
    cmd += ['--', *map(str, command)]
    leg = ROOT / 'evidence/legs' / label
    if (leg / 'result.json').is_file():
        result = json.loads((leg / 'result.json').read_text())
        recorded = json.loads((leg / 'command.json').read_text())
        assert result['returncode'] == 0 and not (leg / 'error.json').exists()
        assert recorded['command'] == list(map(str, command))
        assert any(row['label'] == label and row['returncode'] == 0 for row in progress)
        print('KEEP successful leg', label, flush=True)
        return 0
    print('START', label, flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc == 75:
        raise SystemExit(75)
    progress.append({'label': label, 'returncode': rc, 'command': cmd,
                     'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    PROGRESS.write_text(json.dumps(progress, indent=2) + '\n')
    print(('PASS ' if rc == 0 else 'FAIL ') + label, flush=True)
    return rc


def main():
    assert (ROOT / 'evidence/functional-logits-completed.json').is_file()
    if (ROOT / 'evidence/integration-completed.json').is_file():
        assert json.loads((ROOT / 'evidence/integration-completed.json').read_text())['all_passed']
        print('Integration checks already complete; retaining all evidence.', flush=True)
        return
    save(ROOT / 'evidence/integration-plan.json', {
        'purpose': 'Functional only; no timing claims.',
        'tests': ['Unmodified current tests/test_ane_split_abort.py with NI_ANE_H4096=1',
                  'Unmodified current tests/test_server.py and tests/test_structured.py, native private server,3slots',
                  '9B on and4B default native API coverage', '9B saved-prefix namespace isolation across4real restarts'],
        'source_hashes': {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in
                          ['tests/test_ane_split_abort.py', 'tests/test_server.py', 'tests/test_structured.py']},
        'metal_debug': 'Both validation flags were used for the changed packing/staging dispatches. The full-model runs omit the debug layer because unchanged full-prefill kernels exceed its doubled-memory cap.',
        'script_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), ROOT / 'gpu_leg_startup.py', ROOT / 'ane_api_checks.py',
                           ROOT / 'check_ane_cache_modes.py', ROOT / 'observe']},
    })
    for b, on in [(9, True), (4, False)]:
        label = f'head-{b}b-native-' + ('on' if on else 'default')
        command = [sys.executable, ROOT / 'ane_api_checks.py', '--server', ROOT / 'candidate/kapitsa',
                   '--model', OLD / f'home/models/qwen3.5-{b}b-4bit', '--out', ROOT / 'evidence' / label,
                   '--cases', ROOT / 'evidence/empty-cases.json', '--native', '--slots', '3']
        if on:
            command += ['--ane']
        assert run(label, command, ('NI_ANE_H4096=1',) if on else ()) == 0
    deadline = time.monotonic() + 2400
    if not (ROOT / 'evidence/cpu/completed.json').is_file():
        print('Waiting for CPU checks before the bounded abort test.', flush=True)
    while not (ROOT / 'evidence/cpu/completed.json').is_file():
        assert time.monotonic() < deadline, 'CPU completion wait timed out'
        time.sleep(1)
    port_file = ROOT / 'evidence/abort-port.json'
    if port_file.exists():
        port = json.loads(port_file.read_text())['port']
    else:
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        save(port_file, {'port': port})
    label = 'head-9b-abort'
    rc = run(label, [sys.executable, REPO / 'tests/test_ane_split_abort.py', ROOT / 'candidate/kapitsa',
                    OLD / 'home/models/qwen3.5-9b-4bit', '--port', str(port), '--keep'],
             ('NI_ANE_H4096=1',), timeout=1800)
    text = (ROOT / 'evidence/legs' / label / 'stdout.txt').read_text(errors='replace')
    # Keep the original test's complete directory even though it is outside our evidence root.
    for path in re.findall(r'^logs in (.+)$', text, re.M):
        candidate = Path(path.strip())
        if candidate.is_dir() and any(candidate.glob('*.log')):
            dest = ROOT / 'evidence/abort-server-logs'
            if not dest.exists():
                shutil.copytree(candidate, dest)
                save(ROOT / 'evidence/abort-server-logs-origin.json', {'source': str(candidate)})
    assert rc == 0, 'Current-head abort/fallback test failed; inspect evidence'
    run('head-9b-cache-isolation', [sys.executable, ROOT / 'check_ane_cache_modes.py'], timeout=1800)
    save(ROOT / 'evidence/integration-completed.json', {'completed': True,
         'all_passed': all(row['returncode'] == 0 for row in progress), 'results': progress,
         'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    raise SystemExit(any(row['returncode'] != 0 for row in progress))


if __name__ == '__main__':
    main()
