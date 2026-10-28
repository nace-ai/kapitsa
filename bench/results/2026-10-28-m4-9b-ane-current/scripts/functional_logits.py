"""Fresh latest-main and integrated-PR functional checks. No speed claims from these runs."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('$PREVIOUS_RUN')
REPO = Path('$REPO')
PROGRESS = ROOT / 'evidence/functional-logits-progress.json'
progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else []


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Refusing to change saved evidence: {path}'
        return
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def run(label, command, env=(), timeout=2400):
    # The exclusive lock remains mandatory. Desktop activity is allowed only because
    # these are functional checks whose times are explicitly excluded from speed evidence.
    cmd = [sys.executable, str(ROOT / 'gpu_leg_startup.py'), label,
           '--quiet', '0', '--sample-ms', '1000', '--timeout', str(timeout)]
    for value in env:
        cmd += ['--env', value]
    cmd += ['--', *map(str, command)]
    leg = ROOT / 'evidence/legs' / label
    if (leg / 'result.json').is_file():
        result = json.loads((leg / 'result.json').read_text())
        recorded = json.loads((leg / 'command.json').read_text())
        assert result['returncode'] == 0 and not (leg / 'error.json').exists()
        assert recorded['command'] == list(map(str, command))
        print('KEEP successful leg', label, flush=True)
        return
    print('START', label, flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc == 75:
        raise SystemExit(75)  # The runner recorded a deferral without starting a GPU leg.
    progress.append({'label': label, 'returncode': rc, 'command': cmd,
                     'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    PROGRESS.write_text(json.dumps(progress, indent=2) + '\n')
    assert rc == 0, f'{label} failed; inspect preserved evidence before continuing'
    print('PASS', label, flush=True)


def compare(label, ref, new, expected, exact, ane_expected):
    dest = ROOT / 'evidence' / (label + '.json')
    cmd = [sys.executable, str(REPO / 'tools/logit_compare.py'), str(ref), str(new), '--json', str(dest)]
    # 9B has no measured floor. Zero bounds enforce an unchanged default or
    # document deliberate opt-in non-equivalence; they are not a relaxed floor.
    if '9b' in label:
        cmd += ['--min-top1', '1', '--max-mean-kl', '0', '--max-median-kl', '0',
                '--max-p99-kl', '0', '--near-tie', '0', '--max-far-flips', '0', '--max-nll-change', '0']
    summary_path = ROOT / 'evidence' / (label + '-summary.json')
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        assert previous['command'] == cmd
        rc = previous['compare_returncode']
    else:
        with (ROOT / 'evidence' / (label + '.txt')).open('x') as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    assert rc in (0, 1), 'logit comparison failed to execute'
    result = json.loads(dest.read_text())
    identities, hashes, compact, counters = {}, {}, {}, {}
    for path, count in expected.items():
        hashes[path], counters[path] = {}, {}
        for name, folder in [('A', ref), ('B', new)]:
            meta = json.loads((folder / path / 'meta.json').read_text())
            arm = 'baseline' if name == 'A' else 'candidate'
            manifest = json.loads((ROOT / arm / 'manifest.json').read_text())
            assert meta['build'] == manifest['version'].removeprefix('kapitsa '), 'stale logit tool'
            assert meta['records'] == count and meta['vocab'] == 248320
            assert (folder / path / 'logits.f32').stat().st_size == count * 248320 * 4
            assert meta['ane']['reruns'] == 0
            if name == 'B':
                assert (meta['ane']['evals'] > 0) == ane_expected, f'{path}: wrong ANE path'
            counters[path][name] = meta['ane']
            hashes[path][name] = {f: sha(folder / path / f)
                                   for f in ('logits.f32', 'records.txt', 'sequences.txt')}
        identities[path] = {f: hashes[path]['A'][f] == hashes[path]['B'][f]
                            for f in hashes[path]['A']}
        assert identities[path]['records.txt'] and identities[path]['sequences.txt']
        row = result['paths'][path]
        assert not row['non_finite']
        compact[path] = {k: row[k] for k in ('records', 'bitwise', 'top1', 'flips', 'kl_mean', 'kl_p99',
                                            'kl_max', 'nll_change_mean', 'max_abs_diff')}
    summary = {'command': cmd, 'compare_returncode': rc, 'identity': identities,
               'hashes': hashes, 'ane': counters, 'paths': compact,
               'classification': 'unchanged default regression' if exact else 'intentional opt-in numerical change'}
    save(summary_path, summary)
    print(json.dumps({'label': label, 'compare_returncode': rc, 'paths': compact}), flush=True)
    if exact:
        assert rc == 0 and all(all(v.values()) for v in identities.values()), 'default path changed'


def main():
    assert (ROOT / 'evidence/build-completed.json').is_file()
    for arm in ('baseline', 'candidate'):
        manifest = json.loads((ROOT / arm / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            assert sha(ROOT / arm / name) == digest
    if (ROOT / 'evidence/functional-logits-completed.json').is_file():
        print('Functional logits already complete; retaining all evidence.', flush=True)
        return
    save(ROOT / 'evidence/functional-logits-plan.json', {
        'purpose': 'Functional only; CPU checks may overlap. No timings from these runs are speed evidence.',
        'base': json.loads((ROOT / 'baseline/manifest.json').read_text()),
        'candidate': json.loads((ROOT / 'candidate/manifest.json').read_text()),
        'windows': 'Standard unmodified bench/corpus.txt: 8 prefill + 192 step + 192 verify + 192 batch rows, vocab248320.',
        'exact': ['4B shipped default (ANE enabled) vs current-main default', '9B explicit opt-out vs current-main default'],
        'approximate': '9B NI_ANE_H4096=1, all full-vocabulary drift reported; no floor-pass or output-equivalence claim.',
        'small_validation': ['MTL_SHADER_VALIDATION=1', 'MTL_DEBUG_LAYER=1'],
        'full_model_validation': 'No Metal debug layer on full prefill: unchanged attn_prefill11g uses20480B static, counted as40960B by the layer, beyond32768B. Small changed dispatch geometry is validated first.',
        'long_context': 'One fixed16384-token prefix,1prefill+24step+24verify rows, functional only.',
        'script_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                          [Path(__file__), ROOT / 'gpu_leg_startup.py', ROOT / 'observe']},
        'input_hashes': {'bench/corpus.txt': sha(REPO / 'bench/corpus.txt'),
                         'prompts/code-32768.ids': sha(ROOT / 'prompts/code-32768.ids')},
    })
    run('head-segments-debug', [ROOT / 'candidate/ane_segment_check', '--gpu'],
        ('MTL_SHADER_VALIDATION=1', 'MTL_DEBUG_LAYER=1'), timeout=600)
    expected = {'prefill': 8, 'step': 192, 'verify': 192, 'batch': 192}
    for b in (9, 4):
        model = OLD / f'home/models/qwen3.5-{b}b-4bit'
        ref = ROOT / f'dumps/{b}b-base-default'
        run(f'logit-{b}b-base-default', [ROOT / 'baseline/logit_dump', model, ref])
        modes = [('default', False), ('on', True)] if b == 9 else [('default', False)]
        for name, on in modes:
            label = f'logit-{b}b-{name}'
            dest = ROOT / 'dumps' / label
            env = ('NI_ANE_H4096=1',) if on else (('NI_ANE_H4096=0',) if b == 9 else ())
            run(label, [ROOT / 'candidate/logit_dump', model, dest, '--from', ref], env)
            compare(label, ref, dest, expected, not on, on or b == 4)
    seed = ROOT / 'dumps/16k-input'
    seed.mkdir(exist_ok=True)
    tokens = [int(x) for x in (ROOT / 'prompts/code-32768.ids').read_text().strip().split(',')]
    assert len(tokens) >= 16409
    for name, contents in {
        'sequences.txt': ' '.join(map(str, tokens[:16409])) + '\n',
        'records.txt': ''.join(f'0 {p}\n' for p in range(16385, 16409)),
    }.items():
        if (seed / name).exists():
            assert (seed / name).read_text() == contents
        else:
            (seed / name).write_text(contents)
    save(seed / 'provenance.json', {'source': str(ROOT / 'prompts/code-32768.ids'),
         'source_sha256': sha(ROOT / 'prompts/code-32768.ids'), 'prefix_tokens': 16384})
    ref = ROOT / 'dumps/9b-16k-base'
    new = ROOT / 'dumps/9b-16k-on'
    for arm, binary, dest in [('A', ROOT / 'baseline/logit_dump', ref),
                              ('B', ROOT / 'candidate/logit_dump', new)]:
        run(f'logit-9b-16k-{arm}', [binary, OLD / 'home/models/qwen3.5-9b-4bit', dest,
                                  '--from', seed, '--paths', 'step,verify'],
            ('NI_ANE_H4096=1',) if arm == 'B' else ())
    compare('logit-9b-16k', ref, new, {'prefill': 1, 'step': 24, 'verify': 24}, False, True)
    save(ROOT / 'evidence/functional-logits-completed.json', {'completed': True,
         'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('Latest-main functional logits checks complete.', flush=True)


if __name__ == '__main__':
    main()
