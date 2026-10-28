"""Current-head CPU/lint validation; no model weights or GPU work."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('$PREVIOUS_RUN')
REPO = Path('$REPO')
DEST = ROOT / 'evidence/cpu'
DEST.mkdir(exist_ok=False)
env = {k: v for k, v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'KAPITSA_', 'MTL_'))}
env.update({'PATH': str(OLD / 'venv/bin') + ':$USER_ROOT/.local/bin:' + env['PATH'],
            'VK_EVIDENCE': str(ROOT / 'evidence/vk'), 'VK_RUN': 'pr775-final-cpu-20261028',
            'KAPITSA_NO_AUTO_UPDATE': '1', 'KAPITSA_NO_PROMPT': '1', 'PYTHONUNBUFFERED': '1'})
tests = [('vk-cpu-tests', ['.dsh/skills/verify-kapitsa/scripts/vk', 'cpu-tests'])]
tests += [(name, ['build/' + name]) for name in (
    'execution_session_test http_stream_test json_file_test exl3_test splash_protocol_test '
    'splash_runtime_test splash_discover_test splash_plan_test ane_segment_check').split()]
tests += [('logit-dump-cli', [sys.executable, 'tests/test_logit_dump.py', 'build']),
          ('structured-tokenizer', [sys.executable, 'tests/test_structured.py', '--cli', 'build',
                                   str(OLD / 'home/models/qwen3.5-4b-4bit')])]
for b in (4, 9):
    tests.append((f'parser-{b}b', ['build/parser_test', str(OLD / f'home/models/qwen3.5-{b}b-4bit')]))
tests += [('prek', ['$USER_ROOT/.local/bin/prek', 'run', '--all-files']),
          ('diff-check', ['git', 'diff', '--check'])]
results = []
for name, command in tests:
    started = time.time()
    print('START', name, flush=True)
    with (DEST / (name + '.log')).open('x') as log:
        p = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (DEST / 'active.json').write_text(json.dumps({'name': name, 'pid': p.pid, 'command': command}) + '\n')
        try:
            rc = p.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = 124
    row = {'name': name, 'command': command, 'returncode': rc,
           'started_utc': datetime.datetime.fromtimestamp(started, datetime.timezone.utc).isoformat(),
           'seconds': time.time() - started}
    results.append(row)
    (DEST / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(('PASS ' if rc == 0 else 'FAIL ') + name, flush=True)
manifest = json.loads((ROOT / 'candidate/manifest.json').read_text())
hashes = {name: hashlib.sha256((REPO / 'build' / name).read_bytes()).hexdigest()
          for name in manifest['files']}
assert hashes == manifest['files'], 'CPU build changed frozen candidate binaries'
(DEST / 'completed.json').write_text(json.dumps({'passed': all(r['returncode'] == 0 for r in results),
    'binary_hashes': hashes, 'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2) + '\n')
raise SystemExit(any(r['returncode'] != 0 for r in results))
