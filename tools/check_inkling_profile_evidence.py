#!/usr/bin/env python3
"""Replay the conditional KL in the frozen Inkling fast-default DEV fixture. CPU only."""
import argparse
import json
import math
from pathlib import Path


ARMS = ('precise1', 'fast1', 'precise2', 'fast2')


def normalized(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('empty or nonfinite logit vector')
    peak = max(values)
    shifted = [x - peak for x in values]
    logz = math.log(math.fsum(math.exp(x) for x in shifted))
    return [x - logz for x in shifted]


def kl(p, q):
    if len(p) != len(q):
        raise ValueError('unaligned support')
    value = math.fsum(math.exp(x) * (x - y) for x, y in zip(p, q))
    if value < -1e-10:
        raise ValueError('negative KL')
    return max(0.0, value)


def replay(data):
    if data['format'] != 'inkling-fast-profile-dev64-v1':
        raise ValueError('unknown fixture format')
    rows = data['rows']
    if len(rows) != 64 or [r['row'] for r in rows] != list(range(64)):
        raise ValueError('expected the 64 original aligned DEV rows')
    accepted = [r for r in rows if r['accepted_for_conditional_kl']]
    if [r['row'] for r in rows if not r['accepted_for_conditional_kl']] != [53, 54]:
        raise ValueError('the declared clipped-tail exclusions changed')
    totals = {name: [] for name in ARMS}
    self_kl = []
    for row in accepted:
        ids = row['support_token_ids']
        ref = row['reference_logp']
        if len(ids) != len(set(ids)) or len(ids) != len(ref):
            raise ValueError('duplicate or unaligned support')
        if row['clipped_union_tokens'] != 0:
            raise ValueError('clipped reference code on accepted support')
        lower = math.fsum(math.exp(x) for x in ref) * math.exp(-row['encoder_logp_error_allowance']) - 1e-12
        if lower < 0.999:
            raise ValueError('uncertified reference coverage')
        p = normalized(ref)
        arms = {}
        for name in ARMS:
            values = row['candidate_logits'][name]
            if len(values) != len(ids):
                raise ValueError('candidate support length differs')
            arms[name] = normalized(values)
            totals[name].append(kl(p, arms[name]))
        self_kl.append(kl(arms['precise1'], arms['precise2']))
    n = len(accepted)
    means = {name: math.fsum(values) / n for name, values in totals.items()}
    precise = (means['precise1'] + means['precise2']) / 2
    fast = (means['fast1'] + means['fast2']) / 2
    result = {
        'conditional_kl_rows': n,
        'excluded_zero_based_rows': [53, 54],
        'mean_conditional_kl_reference_to_arm': means,
        'fast_over_precise': fast / precise,
        'precise_self_mean_conditional_kl': math.fsum(self_kl) / n,
        'top1_rows': len(rows),
        'top1_matches': {name: sum(r['candidate_top1'][name] == r['reference_top1'] for r in rows) for name in ARMS},
        'full_reference_kl_available': False,
        'accepted_subset_ratio_below_1_5': fast <= 1.5 * precise,
    }
    expected = data['expected']
    if n != 62 or not math.isclose(result['fast_over_precise'], expected['fast_over_precise'], abs_tol=1e-12):
        raise ValueError('replayed conditional ratio differs from frozen result')
    for name in ARMS:
        if not math.isclose(means[name], expected['mean_conditional_kl_reference_to_arm'][name], abs_tol=1e-12):
            raise ValueError('replayed KL differs from frozen result')
        if result['top1_matches'][name] != expected['top1_matches'][name]:
            raise ValueError('top-1 counts differ from frozen result')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(json.loads(args.fixture.read_text())), indent=2))


if __name__ == '__main__':
    main()
