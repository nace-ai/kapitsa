"""Freeze the native HTTP client, telemetry reader and inputs before final GPU work."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('$REPO')
OLD = Path('$PREVIOUS_RUN')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def main():
    assert not (ROOT / 'evidence/speed-4k-plan.json').exists()
    assert not (ROOT / 'evidence/functional-logits-plan.json').exists()
    sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
    sources = {'api_probe': ROOT.parent / 'api_probe.cpp',
               'observe': OLD / 'observe.cpp', 'tokens': OLD / 'tokens.cpp'}
    manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'library_sha256': sha(REPO / 'build/libnaceinfer.a'), 'helpers': {}}
    for name, source in sources.items():
        dest = ROOT / (name + '.cpp')
        shutil.copyfile(source, dest)
        command = ['clang++', '-O3', '-std=c++20', '-arch', 'arm64', '-isysroot', sdk,
                   '-mmacosx-version-min=14.0', '-I' + str(REPO / 'src'), '-I' + str(REPO / 'include'),
                   '-I' + str(REPO / 'third_party'), str(dest), str(REPO / 'build/libnaceinfer.a')]
        for framework in ('Metal', 'Foundation', 'CoreFoundation', 'IOSurface', 'IOKit', 'ImageIO',
                          'CoreGraphics', 'Security', 'SystemConfiguration'):
            command += ['-framework', framework]
        command += ['-o', str(ROOT / name)]
        with (ROOT / 'evidence' / ('build-helper-' + name + '.log')).open('x') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest['helpers'][name] = {'command': command, 'source_sha256': sha(dest),
                                      'binary_sha256': sha(ROOT / name)}
        print('PASS rebuilt helper', name, flush=True)
    for path in list(ROOT.glob('*.py')) + list((ROOT / 'scripts').glob('*.py')):
        ast.parse(path.read_text(), filename=str(path))
    command = ['uvx', '--from', 'ruff==0.15.21', 'ruff', 'check', '--config', str(REPO / 'ruff.toml'),
               *map(str, sorted(ROOT.glob('*.py'))), *map(str, sorted((ROOT / 'scripts').glob('*.py')))]
    with (ROOT / 'evidence/helpers-ruff.log').open('x') as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest['lint_command'] = command
    manifest['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(ROOT / 'evidence/helper-manifest.json', manifest)

    inputs = {}
    for source, dest in [(OLD / 'prompts/code-32768.ids', ROOT / 'prompts/code-32768.ids'),
                         (REPO / 'bench/corpus.txt', ROOT / 'prompts/corpus.txt')]:
        shutil.copyfile(source, dest)
        inputs[dest.name] = {'source': str(source), 'sha256': sha(dest)}
    source = OLD / 'prompts/ane-quality/cases.json'
    save(ROOT / 'prompts/cache-case.json', json.loads(source.read_text())[0])
    inputs['cache-case.json'] = {'source': str(source), 'source_sha256': sha(source),
                                'selection': 'element 0, unchanged',
                                'sha256': sha(ROOT / 'prompts/cache-case.json')}
    save(ROOT / 'prompts/provenance.json', inputs)


if __name__ == '__main__':
    main()
