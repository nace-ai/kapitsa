"""Verify this text archive and replay its statistics without loading a model."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_telemetry(actual, expected, label):
    assert actual.keys() == expected.keys(), label
    for key, value in actual.items():
        saved = expected[key]
        if key == 'other_gpu_including_desktop_mean_pct':
            # statistics.fmean differs by up to two binary64 ulps on Python 3.9/3.12.
            # Only this derived mean gets roundoff tolerance; eligibility and reasons stay exact.
            assert math.isfinite(value) and math.isfinite(saved), (label, key)
            assert abs(value - saved) <= 4 * max(math.ulp(value), math.ulp(saved)), (label, key)
        else:
            assert value == saved, (label, key)


def main():
    manifest = read(ROOT / 'publication.json')
    for name, entry in manifest['files'].items():
        path = ROOT / name
        assert digest(path.read_bytes()) == entry['published_sha256'], name
    for path in ROOT.rglob('*.json'):
        read(path)
    for path in ROOT.rglob('*.jsonl'):
        for line in path.read_text().splitlines():
            if line.strip():
                json.loads(line)
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))

    spec = importlib.util.spec_from_file_location('confirmation', ROOT / 'confirm_4k.py')
    confirmation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(confirmation)
    progress = read(ROOT / 'evidence/speed-4k-progress.json')
    requests = 0
    for row in progress:
        label = row['label']
        leg = ROOT / 'evidence/legs' / label
        samples = [json.loads(line)['delta'] for line in (leg / 'telemetry.jsonl').read_text().splitlines()]
        before = read(leg / 'before.json')
        result = read(leg / 'result.json')
        assert result['returncode'] == 0
        for name, metrics in row['requests'].items():
            output = ROOT / 'evidence/speed-4k' / label / name
            full = read(output / 'result.json')
            request = read(output / 'request.json')
            assert digest(full['content'].encode()) == full['content_sha256']
            assert digest(json.dumps(request, ensure_ascii=False, separators=(',', ':')).encode()) == full['request_sha256']
            assert full['completion_tokens'] == 512 and full['finish'] == 'length'
            assert full['content'].strip() and full['cached_tokens'] == 0 and not full['invalid']
            for key in ('request_sha256', 'content_sha256', 'begin_epoch', 'end_epoch', 'ttft_s', 'total_s'):
                assert metrics[key] == full[key]
            ttft = confirmation.phase(samples, metrics['begin_epoch'],
                                      metrics['begin_epoch'] + metrics['ttft_s'], before['swapouts'])
            total = confirmation.phase(samples, metrics['begin_epoch'], metrics['end_epoch'], before['swapouts'])
            check_telemetry(ttft, metrics['ttft_telemetry'], (label, name, 'TTFT'))
            check_telemetry(total, metrics['total_telemetry'], (label, name, 'HTTP'))
            requests += 1

    with tempfile.TemporaryDirectory(prefix='kapitsa-pr775-replay-') as folder:
        replay = Path(folder)
        (replay / 'evidence').mkdir()
        shutil.copy2(ROOT / 'analyze_4k.py', replay / 'analyze_4k.py')
        for name in ('speed-4k-plan.json', 'speed-4k-progress.json', 'speed-4k-completed.json'):
            shutil.copy2(ROOT / 'evidence' / name, replay / 'evidence' / name)
        subprocess.run([sys.executable, str(replay / 'analyze_4k.py')], check=True, stdout=subprocess.DEVNULL)
        expected = read(ROOT / 'evidence/speed-4k-analysis.json')
        actual = read(replay / 'evidence/speed-4k-analysis.json')
        assert actual == expected, 'Published aggregation differs from the retained evidence'

    print(f'PASS {len(manifest["files"])} file hashes, JSON/JSONL and Python syntax; '
          f'{requests} requests and exact telemetry gates (derived means within 4 ulps); '
          'aggregate analysis reproduced exactly.')


if __name__ == '__main__':
    main()
