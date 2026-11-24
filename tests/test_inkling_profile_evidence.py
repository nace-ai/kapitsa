#!/usr/bin/env python3
"""CPU checks for the frozen Inkling conditional-quality replay."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('profile_evidence', ROOT / 'tools/check_inkling_profile_evidence.py')
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


def fixture():
    precise = 0.8 * math.log(0.8 / 0.78) + 0.2 * math.log(0.2 / 0.22)
    fast = 0.8 * math.log(0.8 / 0.779) + 0.2 * math.log(0.2 / 0.221)
    row = {'reference_top1': 1, 'candidate_top1': dict.fromkeys(EVIDENCE.ARMS, 1),
           'accepted_for_conditional_kl': True, 'support_token_ids': [1, 2],
           'reference_logp': [math.log(0.8), math.log(0.2)],
           'candidate_logits': {name: [math.log(0.78), math.log(0.22)] if name.startswith('precise')
                                else [math.log(0.779), math.log(0.221)] for name in EVIDENCE.ARMS},
           'encoder_logp_error_allowance': 0.0, 'clipped_union_tokens': 0}
    rows = [dict(copy.deepcopy(row), row=i) for i in range(64)]
    for i in (53, 54):
        rows[i]['accepted_for_conditional_kl'] = False
    return {'format': 'inkling-fast-profile-dev64-v1', 'rows': rows,
            'expected': {'mean_conditional_kl_reference_to_arm': {
                name: precise if name.startswith('precise') else fast for name in EVIDENCE.ARMS},
                'fast_over_precise': fast / precise, 'top1_matches': dict.fromkeys(EVIDENCE.ARMS, 64)}}


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def test_closed_form_bernoulli_kl(self):
        result = EVIDENCE.replay(self.data)
        self.assertEqual(result['conditional_kl_rows'], 62)
        self.assertEqual(result['top1_rows'], 64)
        self.assertFalse(result['full_reference_kl_available'])
        self.assertTrue(result['accepted_subset_ratio_below_1_5'])

    def test_clipped_union_is_rejected(self):
        self.data['rows'][0]['clipped_union_tokens'] = 1
        with self.assertRaisesRegex(ValueError, 'clipped'):
            EVIDENCE.replay(self.data)

    def test_uncertified_mass_is_rejected(self):
        self.data['rows'][0]['reference_logp'] = [math.log(0.7), math.log(0.2)]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            EVIDENCE.replay(self.data)

    def test_nan_is_rejected(self):
        self.data['rows'][0]['candidate_logits']['fast1'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            EVIDENCE.replay(self.data)

    def test_changed_exclusions_are_rejected(self):
        self.data['rows'][53]['accepted_for_conditional_kl'] = True
        with self.assertRaisesRegex(ValueError, 'exclusions'):
            EVIDENCE.replay(self.data)

    def test_mismatched_support_is_rejected(self):
        self.data['rows'][0]['candidate_logits']['fast1'].pop()
        with self.assertRaisesRegex(ValueError, 'length'):
            EVIDENCE.replay(self.data)

    def test_changed_expected_ratio_is_rejected(self):
        self.data['expected']['fast_over_precise'] = 1.41
        with self.assertRaisesRegex(ValueError, 'ratio differs'):
            EVIDENCE.replay(self.data)

    def test_recorded_fixture_replays(self):
        import json
        data = json.loads((ROOT / 'bench/results/inkling-fast-default-20261124/conditional-quality.json').read_text())
        result = EVIDENCE.replay(data)
        self.assertEqual(result['top1_matches']['fast1'], 64)
        self.assertEqual(result['top1_matches']['precise1'], 63)
        self.assertAlmostEqual(result['fast_over_precise'], 1.4216959293580875, places=12)


if __name__ == '__main__':
    unittest.main()
