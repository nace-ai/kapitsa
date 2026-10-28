import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('$REPO')
BASE = 'dbf3f3b330263a93c322ca650914bfd82cdecbea'
HEAD = 'fe74a3db5db5ef4f07233c795e0b6c00a9a13f2d'
ENV = os.environ.copy()
ENV['PATH'] = '$USER_ROOT/.local/bin:' + ENV['PATH']
ENV['VK_EVIDENCE'] = str(ROOT / 'evidence/vk')
ENV['KAPITSA_NO_AUTO_UPDATE'] = '1'
ENV['KAPITSA_NO_PROMPT'] = '1'


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def run(label, argv):
    start = time.time()
    print('START', label, flush=True)
    with (ROOT / 'evidence' / (label + '.log')).open('x') as f:
        result = subprocess.run(argv, cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    save(ROOT / 'evidence' / (label + '.json'), {
        'argv': argv, 'commit': git('rev-parse', 'HEAD'), 'returncode': result.returncode,
        'seconds': time.time() - start,
    })
    result.check_returncode()
    print('PASS', label, flush=True)


def freeze(directory, commit, binaries):
    assert git('rev-parse', 'HEAD') == commit and not git('status', '--porcelain')
    files = {}
    for name in binaries:
        dest = ROOT / directory / name
        assert not dest.exists()
        shutil.copy2(REPO / 'build' / name, dest)
        files[name] = sha(dest)
    version = subprocess.check_output([str(ROOT / directory / 'kapitsa'), '--version'],
                                      env=ENV, text=True).strip()
    assert commit[:7] in version, version
    save(ROOT / directory / 'manifest.json', {
        'commit': commit, 'version': version, 'files': files,
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sources': {p: sha(REPO / p) for p in (
            'AGENTS.md', 'src/ane/ffn_split.cpp', 'src/ane/ffn_split.h',
            'src/model/qwen35_ane.cpp', 'src/model/qwen35.cpp', 'src/model/qwen35.h',
            'tests/test_ane_split_abort.py', 'tools/logit_dump.cpp')},
    })


def main():
    assert git('rev-parse', 'HEAD') == HEAD and not git('status', '--porcelain')
    run('switch-base', ['git', 'switch', '--detach', BASE])
    binaries = ['kapitsa', 'logit_dump', 'ni_run', 'ni_eval']
    run('configure-base', ['cmake', '-S', '.', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release'])
    run('build-base', ['cmake', '--build', 'build', '-j', '6', '--target', *binaries])
    freeze('baseline', BASE, binaries)
    run('switch-head', ['git', 'switch', 'codex/ane-9b-pr775-final-20261028'])
    cpu = ('api_test execution_session_test http_stream_test bench_test update_test prefix_cache_test '
           'grammar_test structured_check fn_store_test fn_dflash_test tokenizer_test iq_layout_test '
           'exl3_test remote_test cluster_test split_place_test json_file_test splash_protocol_test '
           'splash_runtime_test splash_discover_test splash_plan_test ane_segment_check parser_test').split()
    run('configure-head', ['cmake', '-S', '.', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release'])
    run('build-head', ['cmake', '--build', 'build', '-j', '6', '--target', *binaries, *cpu])
    freeze('candidate', HEAD, [*binaries, 'ane_segment_check'])
    run('required-checks', ['.dsh/skills/verify-kapitsa/scripts/vk', 'checks', BASE])
    save(ROOT / 'evidence/build-completed.json', {'complete': True, 'base': BASE, 'head': HEAD})
    print('Both builds frozen at exact PR base/head.', flush=True)


if __name__ == '__main__':
    main()
