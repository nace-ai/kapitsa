import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path('/Users/bytedance/kapitsa-ngram-affine-20261107')
SHARED = Path('/Users/bytedance/kapitsa-ngram-remote-20261107')
OUT = ROOT / 'real-rows-1857'
OUT.mkdir(exist_ok=False)
assert (SHARED / 'rows-passed').exists()
MODEL = SHARED / 'model-sushi'
commands = []
def run(cmd, label, cwd=ROOT):
    cmd = [str(x) for x in cmd]
    commands.append({'label': label, 'argv': cmd, 'cwd': str(cwd)})
    (OUT / 'commands.json').write_text(json.dumps(commands, indent=2))
    with (OUT / (label + '.log')).open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, check=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

(OUT / 'probe.cpp').write_bytes((SHARED / 'probe.cpp').read_bytes())
identities = {'scope': 'PR1022 own full-library CPU row checks on pinned trained affine Q4 table; no GPU or speed measurement',
              'model_revision': 'f37e7ba46f0f2993b6ca373221530c579f258712',
              'probe_source_sha256': sha(OUT / 'probe.cpp'), 'variants': {}}
for tag in ['base', 'candidate']:
    tree = ROOT / tag; build = tree / 'build'
    flags = (build / 'CMakeFiles/ni_run.dir/flags.make').read_text()
    args = []
    for key in ['CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS']:
        args += shlex.split(re.search(r'^' + key + r' = (.*)$', flags, re.M).group(1))
    obj = OUT / (tag + '.o')
    run(['/usr/bin/clang++', *args, '-c', OUT / 'probe.cpp', '-o', obj], 'compile-' + tag, build)
    cmd = shlex.split((build / 'CMakeFiles/ni_run.dir/link.txt').read_text())
    cmd = [str(obj) if x.endswith('tools/ni_run.cpp.o') else x for x in cmd]
    cmd[cmd.index('-o') + 1] = str(OUT / ('probe-' + tag))
    run(cmd, 'link-' + tag, build)
    identities['variants'][tag] = {'source': subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip(),
                                  'library_sha256': sha(build / 'libnaceinfer.a'), 'probe_sha256': sha(OUT / ('probe-' + tag))}
    for case, input_name in [('corpus', 'corpus-base.json'), ('chat', 'chat-prompt.json')]:
        src = SHARED / 'evidence' / input_name
        (OUT / input_name).write_bytes(src.read_bytes())
        run([OUT / ('probe-' + tag), 'rows', MODEL, src, OUT / (case + '-' + tag + '.f32')], case + '-' + tag)
results = []
for case, coverage in [('corpus', 'corpus-row-coverage.json'), ('chat', 'chat-prompt-row-coverage.json')]:
    a = OUT / (case + '-base.f32'); b = OUT / (case + '-candidate.f32')
    av = a.read_bytes(); bv = b.read_bytes()
    row = {'case': case, 'bytes': len(av), 'base_sha256': sha(a), 'candidate_sha256': sha(b), 'byte_equal': av == bv}
    assert av == bv and av
    results.append(row)
    (OUT / coverage).write_bytes((SHARED / 'evidence' / coverage).read_bytes())
identities['checks'] = results
(OUT / 'result.json').write_text(json.dumps(identities, indent=2))
for p in OUT.glob('*.f32'): p.unlink()
(OUT / 'passed').touch()
print(json.dumps(identities, indent=2))
