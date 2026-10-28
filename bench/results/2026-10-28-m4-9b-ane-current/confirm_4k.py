"""Fixed latest-main 4K A/B rerun; preserve every attempt and select only by telemetry."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OLD = Path('$PREVIOUS_RUN')
REPO = Path('$REPO')
PLAN = ROOT / 'evidence/speed-4k-plan.json'


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


def phase(samples, begin, end, pre_run_swapouts):
    # Include the 1s sample overlapping each boundary, and the preceding cumulative
    # swap count. A swap before the first in-phase sample must not disappear.
    selected = [s for s in samples if begin <= s['epoch'] <= end + 1.0]
    preceding = [s for s in samples if s['epoch'] < begin]
    swap_before = preceding[-1]['swapouts'] if preceding else pre_run_swapouts
    reasons = []
    if len(selected) < 2:
        reasons.append('fewer than two telemetry samples')
    combined = [s['foreign_pct'] + s['desktop_pct'] for s in selected]
    mean = sum(combined) / max(1, len(combined))
    peak = max(combined, default=0)
    if mean > 1:
        reasons.append('other GPU including desktop mean above1%')
    if peak > 2:
        reasons.append('other GPU including desktop peak above2%')
    if any(s['thermal'] != 'nominal' for s in selected):
        reasons.append('thermal state not nominal')
    if any(s.get('user_idle_seconds', 0) < 300 for s in selected):
        reasons.append('user activity during the phase')
    epochs = [begin] + [s['epoch'] for s in selected if s['epoch'] <= end] + [end]
    if any(b - a > 2.5 for a, b in zip(epochs, epochs[1:])):
        reasons.append('telemetry gap exceeds 2.5s')
    if selected and max(s['swapouts'] for s in selected) != swap_before:
        reasons.append('swapouts during phase')
    return {'valid': not reasons, 'reasons': reasons, 'samples': len(selected),
            'other_gpu_including_desktop_mean_pct': mean,
            'other_gpu_including_desktop_peak_pct': peak, 'swapouts_before_phase': swap_before,
            'swapouts_max_in_phase': max((s['swapouts'] for s in selected), default=swap_before)}


def prepare():
    assert not PLAN.exists(), 'A protocol is already frozen; do not replace it.'
    cases = json.loads((ROOT / 'prompts/4k-suite.json').read_text())
    plan = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'kind': 'Fixed 4K performance regression rerun on freshly fetched latest main; historical prompts reused, no new workload-independence claim.',
        'machine': 'Mac mini2024, Apple M4,10GPU cores,24GiB; macOS26.5.2(25F84),SDK26.2',
        'baseline': json.loads((ROOT / 'baseline/manifest.json').read_text()),
        'candidate': json.loads((ROOT / 'candidate/manifest.json').read_text()),
        'model': 'lmstudio-community/Qwen3.5-9B-MLX-4bit@b455506b0f574c74616dbcd56879bde38fafcff3',
        'drafter': 'z-lab/Qwen3.5-9B-DFlash@5fc3b3d474760f18c516db87d84c37edbfd3ede6',
        'pairs': 4, 'arms': ['A', 'B'], 'pair_orders': [['A', 'B'], ['B', 'A'], ['A', 'B'], ['B', 'A']],
        'max_retries_per_pair_case': 1,
        'cases': cases, 'tokens': 512, 'temperature': 0,
        'primary': 'Loaded-model native HTTP TTFT after fixed warmup and completed ANE preparation. Analyze paired log(A/B) per workload.',
        'secondary': 'Complete512-token HTTP request time, eligible only when telemetry for the whole request passes. Continuations may differ; no same-work decode speed claim.',
        'eligibility': 'Two or more1s samples overlapping each phase with no gap>2.5s; non-owned GPU including WindowServer mean<=1%, peak<=2%; nominal thermal; HID idle>=300s; zero phase swapouts using the preceding sample/pre-run counter. Exactly512nonempty tokens, finish=length, no native invalid flags, cached_tokens=0.',
        'retries': 'Only TTFT-telemetry-rejected workload pairs are repeated, once, both A and B. Use the first eligible attempt; retain every first/retry result. No retries based on speed. Idle-gate timeout stops the schedule as incomplete.',
        'failure': 'Stop for invalid output, wrong path, unexpected ANE fallback or request mismatch. Fewer than3eligible pairs per workload is inconclusive.',
        'analysis': 'Paired geometric A/B ratio and two-sided95% Student-t confidence interval of log ratios per workload. Confirmation requires>=3pairs and lowerCI>1 for both4Kworkloads; all first attempts retained as sensitivity. No early stopping.',
        'controls': 'Immutable current-main/current-integration binaries, exclusive GPU lock,8s quiet gate, no builds/downloads/CPU suites, one slot, fresh KAPITSA_HOME and disk cacheoff; matching A/B requests and budgets. CPU contention checked before launch.',
        'flags_A': 'Default current-main server flags; no NI_ overrides.',
        'flags_B': 'Same flags plus NI_ANE_H4096=1; default DFlash, default async ANE, default units, context32768.',
        'display': 'User explicitly authorized display sleep for performance. Keep system awake, preserve unrelated apps/processes. Record control before launch.',
        'scope': 'New focused4Krerun only. Does not turn the historical failed four-workload/16K protocol into a pass. No16Kperformance or broad quality claim.',
        'hashes': {name: sha(ROOT / name) for name in
                   ['confirm_4k.py', 'analyze_4k.py', 'gpu_leg_startup.py', 'observe', 'api_server_leg_confirm.py', 'api_probe']},
    }
    for case in cases:
        assert sha(Path(case['messages'])) == case['sha256']
    save(PLAN, plan)
    print('Saved fixed4Kprotocol before any timed request.', flush=True)


def run():
    plan = json.loads(PLAN.read_text())
    assert (ROOT / 'evidence/functional-logits-completed.json').is_file()
    assert json.loads((ROOT / 'evidence/cpu/completed.json').read_text())['passed']
    assert json.loads((ROOT / 'evidence/integration-completed.json').read_text())['all_passed']
    for name, digest in plan['hashes'].items():
        assert sha(ROOT / name) == digest
    for folder in ('baseline', 'candidate'):
        for name, digest in plan[folder]['files'].items():
            assert sha(ROOT / folder / name) == digest
    if (ROOT / 'evidence/speed-4k-completed.json').exists():
        if not (ROOT / 'evidence/speed-4k-analysis.json').exists():
            subprocess.run([sys.executable, str(ROOT / 'analyze_4k.py')], cwd=REPO, check=True)
        print('Fixed 4K protocol already complete; retaining all evidence.', flush=True)
        return
    progress_path = ROOT / 'evidence/speed-4k-progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else []
    recorded_rows = {row['label']: row for row in progress}
    assert len(recorded_rows) == len(progress)
    first_hash, first_request = {}, {}
    for pair in range(plan['pairs']):
        pending = list(plan['cases'])
        for attempt in range(2):
            arms = {}
            subset = ROOT / f'evidence/speed-4k-suite-p{pair}-a{attempt}.json'
            save(subset, pending)
            for arm in plan['pair_orders'][pair]:
                label = f'speed-4k-p{pair}-a{attempt}-{arm}'
                out = ROOT / 'evidence/speed-4k' / label
                binary = ROOT / ('baseline' if arm == 'A' else 'candidate') / 'kapitsa'
                cmd = [sys.executable, str(ROOT / 'gpu_leg_startup.py'), label,
                       '--quiet', '8', '--max-foreign', '1', '--idle-timeout', '180',
                       '--sample-ms', '1000', '--timeout', '900', '--sleep-display']
                if arm == 'B':
                    cmd += ['--env', 'NI_ANE_H4096=1']
                cmd += ['--', sys.executable, str(ROOT / 'api_server_leg_confirm.py'), '--server', str(binary),
                        '--model', str(OLD / 'home/models/qwen3.5-9b-4bit'), '--out', str(out),
                        '--pair', str(pair), '--workloads', '', '--suite', str(subset), '--tokens', '512']
                print('START', label, flush=True)
                leg = ROOT / 'evidence/legs' / label
                if (leg / 'result.json').is_file():
                    result = json.loads((leg / 'result.json').read_text())
                    recorded = json.loads((leg / 'command.json').read_text())
                    assert result['returncode'] == 0 and not (leg / 'error.json').exists()
                    assert recorded['command'] == cmd[cmd.index('--') + 1:]
                    print('KEEP successful leg', label, flush=True)
                else:
                    proc = subprocess.run(cmd, cwd=REPO)
                    if proc.returncode == 75:
                        raise SystemExit(75)
                    assert proc.returncode == 0, f'{label} failed; preserve incomplete schedule'
                server_log = (out / 'server-stderr.txt').read_text()
                assert ('NI_ANE_H4096=1: experimental' in server_log) == (arm == 'B')
                assert 'reruns on the GPU' not in server_log and 'startup check failed' not in server_log
                assert 'DFlash2 drafter' in server_log and 'dflash2' in server_log
                samples = [json.loads(line)['delta'] for line in (leg / 'telemetry.jsonl').read_text().splitlines()]
                before = json.loads((leg / 'before.json').read_text())
                requests = {}
                for case in pending:
                    result = json.loads((out / case['name'] / 'result.json').read_text())
                    request_log = (out / case['name'] / 'server-during.txt').read_text()
                    evals = re.findall(r'prefill: Neural Engine busy .*\((\d+) evaluations\)', request_log)
                    assert bool(evals) == (arm == 'B') and all(int(n) > 0 for n in evals)
                    assert result['completion_tokens'] == 512 and result['finish'] == 'length'
                    assert result['content'].strip() and not result['invalid'] and result['cached_tokens'] == 0
                    assert abs(result['prompt_tokens'] - case['nominal_tokens']) <= 32
                    assert result['usage']['kapitsa']['kflash']['source'] == 'DFlash2'
                    key = (pair, case['name'], arm)
                    first_hash.setdefault(key, result['content_sha256'])
                    identity = (result['request_sha256'], result['prompt_tokens'])
                    request_key = (pair, case['name'])
                    first_request.setdefault(request_key, identity)
                    assert first_request[request_key] == identity, 'A/B or retry request mismatch'
                    requests[case['name']] = {k: result[k] for k in
                        ('request_sha256', 'begin_epoch', 'end_epoch', 'prompt_tokens', 'completion_tokens',
                         'finish', 'content_sha256', 'ttft_s', 'gen_time_s', 'total_s', 'native_mid_tps', 'usage')}
                    requests[case['name']].update({
                        'retry_content_repeatable': first_hash[key] == result['content_sha256'],
                        'ane_evaluations': sum(map(int, evals)),
                        'ttft_telemetry': phase(samples, result['begin_epoch'], result['begin_epoch'] + result['ttft_s'], before['swapouts']),
                        'total_telemetry': phase(samples, result['begin_epoch'], result['end_epoch'], before['swapouts']),
                    })
                row = {'pair': pair, 'attempt': attempt, 'arm': arm, 'label': label, 'requests': requests,
                       'telemetry': json.loads((leg / 'result.json').read_text())}
                if label in recorded_rows:
                    assert row == recorded_rows[label], 'Recorded timing or eligibility changed'
                else:
                    progress.append(row)
                    progress_path.write_text(json.dumps(progress, indent=2) + '\n')
                arms[arm] = requests
                print(json.dumps({'label': label, 'requests': {n: {'ttft_s': x['ttft_s'],
                    'total_s': x['total_s'], 'ttft_valid': x['ttft_telemetry']['valid'],
                    'reasons': x['ttft_telemetry']['reasons']} for n, x in requests.items()}}), flush=True)
            pending = [case for case in pending if not all(arms[a][case['name']]['ttft_telemetry']['valid'] for a in ('A', 'B'))]
            if not pending:
                break
    save(ROOT / 'evidence/speed-4k-completed.json', {'completed': True, 'recorded_arms': len(progress),
         'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    subprocess.run([sys.executable, str(ROOT / 'analyze_4k.py')], cwd=REPO, check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    prepare() if args.prepare else run()
