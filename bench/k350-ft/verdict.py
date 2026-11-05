#!/usr/bin/env python3
"""VERDICT.json of the #350 fine-tune eval: per request and pass, tok/cycle and decode tok/s of ship vs ft, outputs identical."""
import json, re, sys, os
O = sys.argv[1] if len(sys.argv) > 1 else '/tmp/k350d/eval'
extra = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
pat = re.compile(r'dflash decode .*?: (\d+) tok in ([\d.]+) ms .*?, (\d+) cycles, ([\d.]+) tok/cycle')
rows, hist_on, legs = [], 0, 0
for p in (1, 2, 3, 4):
    for i in range(19):
        r = {'pass': p, 'req': i}
        for arm in ('ship', 'ft'):
            try:
                e = open(f'{O}/p{p}.r{i}.{arm}.err').read(); ids = open(f'{O}/p{p}.r{i}.{arm}.out').read()
            except FileNotFoundError:
                continue
            m = pat.search(e)
            if not m:
                continue
            legs += 1; hist_on += 'history lane on' in e
            r[arm] = {'tok': int(m[1]), 'ms': float(m[2]), 'cycles': int(m[3]), 'tpc': float(m[4])}
            r[arm + '_ids'] = ids.split('ids:')[-1].split()
        if 'ship' in r and 'ft' in r:
            rows.append(r)
def agg(rs, a):
    return (sum(r[a]['tok'] for r in rs) / sum(r[a]['cycles'] for r in rs), sum(r[a]['tok'] for r in rs) / sum(r[a]['ms'] for r in rs) * 1000)
def summ(rs):
    s, f = agg(rs, 'ship'), agg(rs, 'ft')
    return {'n_pairs': len(rs), 'outputs_identical': sum(r['ship_ids'] == r['ft_ids'] for r in rs),
            'shipped': {'tok_per_cycle': round(s[0], 4), 'decode_tok_s': round(s[1], 2)},
            'finetuned': {'tok_per_cycle': round(f[0], 4), 'decode_tok_s': round(f[1], 2)},
            'delta_tok_per_cycle_pct': round(100 * (f[0] / s[0] - 1), 2), 'delta_tok_s_pct': round(100 * (f[1] / s[1] - 1), 2),
            'requests_ft_faster_tok_s': sum(r['ft']['tok'] / r['ft']['ms'] > r['ship']['tok'] / r['ship']['ms'] for r in rs)}
v = {'task': '#350 offline fine-tune of the 35B-A3B DFlash2 drafter on self-distilled English coding turns vs the shipped drafter',
     'mac': 'MacBook Pro M5 Pro 48 GB, macOS ' + os.popen('sw_vers -productVersion').read().strip(),
     'argv': 'KAPITSA_HOME=<per arm> build/ni_run <symlinked model dir, dflash2/ = arm> @r<i>.ids <n> 0 --dflash auto --ignore-eos',
     'history_lane_on_legs': f'{hist_on}/{legs}', **extra, 'all': summ(rows)}
for p in sorted({r['pass'] for r in rows}):
    v[f'pass{p}'] = summ([r for r in rows if r['pass'] == p])
v['per_request'] = [{'pass': r['pass'], 'req': r['req'], 'n': r['ship']['tok'], 'tpc_ship': r['ship']['tpc'], 'tpc_ft': r['ft']['tpc'],
                     'tps_ship': round(r['ship']['tok'] / r['ship']['ms'] * 1000, 1), 'tps_ft': round(r['ft']['tok'] / r['ft']['ms'] * 1000, 1),
                     'identical': r['ship_ids'] == r['ft_ids']} for r in rows]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERDICT.json')
json.dump(v, open(out, 'w'), indent=1)
print(json.dumps({k: v[k] for k in v if k != 'per_request'}, indent=1))
