"""Apply the frozen ANE confirmation selection rule; never add trials or change thresholds."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def ratio_stats(pairs, metric):
    ratios = [a[metric] / b[metric] for a, b in pairs]
    n = len(ratios)
    if not n:
        return {'pairs': 0}
    logs = [math.log(r) for r in ratios]
    mean = statistics.mean(logs)
    result = {'pairs': n, 'ratios_A_over_B': ratios, 'geometric_speed_ratio': math.exp(mean),
              'time_reduction_pct': 100 * (1 - math.exp(-mean)),
              'A_seconds': [a[metric] for a, b in pairs], 'B_seconds': [b[metric] for a, b in pairs]}
    if n >= 2:
        t = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305}[n - 1]
        half = t * statistics.stdev(logs) / math.sqrt(n)
        result['ci95_ratio'] = [math.exp(mean - half), math.exp(mean + half)]
    return result


def main():
    plan = json.loads((ROOT / 'evidence/speed-4k-plan.json').read_text())
    rows = json.loads((ROOT / 'evidence/speed-4k-progress.json').read_text())
    assert plan['pairs'] == 4 and plan['max_retries_per_pair_case'] == 1
    attempts = {}
    for row in rows:
        key = (row['pair'], row['attempt'])
        assert row['arm'] not in attempts.setdefault(key, {})
        attempts[key][row['arm']] = row['requests']
    all_workloads = {}
    selections = []
    for case in plan['cases']:
        name = case['name']
        eligible, all_first, whole_request = [], [], []
        same_content = []
        for pair in range(plan['pairs']):
            selected = None
            for attempt in range(2):
                arms = attempts.get((pair, attempt), {})
                if any(name not in arms.get(arm, {}) for arm in ('A', 'B')):
                    continue
                a, b = arms['A'][name], arms['B'][name]
                assert a['request_sha256'] == b['request_sha256']
                assert a['prompt_tokens'] == b['prompt_tokens']
                if attempt == 0:
                    all_first.append((a, b))
                accepted = all(x['ttft_telemetry']['valid'] for x in (a, b))
                selections.append({'workload': name, 'pair': pair, 'attempt': attempt,
                                   'eligible': accepted, 'selected': accepted and selected is None,
                                   'A': a, 'B': b})
                if accepted and selected is None:
                    selected = (a, b)
            if selected:
                eligible.append(selected)
                a, b = selected
                same_content.append(a['content_sha256'] == b['content_sha256'])
                if all(x['total_telemetry']['valid'] for x in (a, b)):
                    whole_request.append(selected)
        primary = ratio_stats(eligible, 'ttft_s')
        confirmed = primary['pairs'] >= 3 and primary.get('ci95_ratio', [0])[0] > 1
        all_workloads[name] = {
            'primary_ttft': primary, 'primary_confirmed': confirmed,
            'complete_http_on_selected_pairs': ratio_stats(eligible, 'total_s'),
            'complete_http_with_whole_request_telemetry': ratio_stats(whole_request, 'total_s'),
            'first_attempts_ttft_sensitivity': ratio_stats(all_first, 'ttft_s'),
            'first_attempts_http_sensitivity': ratio_stats(all_first, 'total_s'),
            'identical_replies_in_selected_pairs': sum(same_content),
            'selected_reply_pairs': len(same_content),
        }
    complete = (ROOT / 'evidence/speed-4k-completed.json').is_file() and all(
        (pair, 0) in attempts and set(attempts[pair, 0]) == {'A', 'B'} for pair in range(4))
    result = {'plan': plan, 'completed_first_pairs': complete,
              'confirmed': complete and all(x['primary_confirmed'] for x in all_workloads.values()),
              'workloads': all_workloads, 'attempts': selections,
              'scope': 'Loaded-model HTTP TTFT on this M4 only. Explicit W8A8 approximation changes replies. '
                       'Full HTTP time includes possibly different continuation work; no decode-speed claim.'}
    dest = ROOT / 'evidence/speed-4k-analysis.json'
    with dest.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    lines = ['# Latest-main M4 ANE 4K rerun', '', result['scope'], '',
             f"Primary protocol confirmed: **{result['confirmed']}**. Full first-pair schedule: {complete}.", '',
             '| Workload | Eligible pairs | TTFT A/B ratio (95% interval) | TTFT time reduction | Identical replies |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, values in all_workloads.items():
        p = values['primary_ttft']
        ratio = p.get('geometric_speed_ratio')
        ci = p.get('ci95_ratio')
        text = f'{ratio:.3f}' if ratio else '—'
        if ci:
            text += f' ({ci[0]:.3f}–{ci[1]:.3f})'
        reduction = f"{p['time_reduction_pct']:.2f}%" if ratio else '—'
        lines.append(f"| {name} | {p['pairs']} | {text} | {reduction} | "
                     f"{values['identical_replies_in_selected_pairs']}/{values['selected_reply_pairs']} |")
    lines += ['', 'Every attempt, including rejected attempts:', '',
              '| Workload | Pair | Attempt | TTFT A / B (s) | HTTP A / B (s) | TTFT other GPU peak including desktop A / B (%) | Selected |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in selections:
        a, b = row['A'], row['B']
        lines.append(f"| {row['workload']} | {row['pair']} | {row['attempt']} | {a['ttft_s']:.4f} / "
                     f"{b['ttft_s']:.4f} | {a['total_s']:.4f} / {b['total_s']:.4f} | "
                     f"{a['ttft_telemetry']['other_gpu_including_desktop_peak_pct']:.3f} / {b['ttft_telemetry']['other_gpu_including_desktop_peak_pct']:.3f} | "
                     f"{row['selected']} |")
    with (ROOT / 'evidence/speed-4k-analysis.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps({'confirmed': result['confirmed'], 'workloads': all_workloads}, indent=2))


if __name__ == '__main__':
    main()
