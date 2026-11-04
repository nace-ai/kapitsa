"""CPU mapping and source contracts, not Metal execution or numerical parity."""

from collections import Counter
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
G = 6
NS = 8
BK = 16
EXTENTS = (1, 2, 3, 10, 11, 15, 16, 17, 21, 22, 31, 32, 33, 141, 2047, 2048, 2049)


def fragments(lane):
    qid = lane >> 2
    return (qid & 4) | ((lane >> 1) & 3), ((qid & 2) | (lane & 1)) * 4


def limits(t, pos0, group, pair, row, bidi):
    last = pos0 + t - 1 if bidi else pos0 + min((group * 64 + 63) // G, t - 1)
    first = pos0 + (group * 64 + pair * 16) // G
    limit = last if bidi else pos0 + min(row // G, t - 1)
    return last, first, limit


class BidiMappingTests(unittest.TestCase):
    def test_live_rows_cover_every_token_head_once(self):
        for t in EXTENTS:
            seen = Counter()
            inactive = 0
            for group in range((t * G + 63) // 64):
                for pair in range(NS // 2):
                    r0 = group * 64 + pair * 16
                    if r0 >= t * G:
                        inactive += 1
                        continue
                    for offset in range(16):
                        row = r0 + offset
                        if row // G < t:
                            seen[divmod(row, G)] += 1
            self.assertEqual(seen, Counter({(token, head): 1 for token in range(t) for head in range(G)}))
            if t <= 2:
                self.assertGreater(inactive, 0)

    def test_lane_fragments_store_each_dimension_once(self):
        seen = Counter()
        for sg in range(2):
            for lane in range(32):
                fm, fn = fragments(lane)
                for i in range(2):
                    for block in range(8):
                        for component in range(4):
                            seen[fm + 8 * i, sg * 128 + block * 16 + fn + component] += 1
        self.assertEqual(seen, Counter({(row, dim): 1 for row in range(16) for dim in range(256)}))

    def test_clamped_query_loads_stay_inside_token_and_head_domain(self):
        for t in EXTENTS:
            for group in range((t * G + 63) // 64):
                for pair in range(4):
                    for offset in range(16):
                        row = group * 64 + pair * 16 + offset
                        token, head = divmod(row, G)
                        self.assertTrue(0 <= min(token, t - 1) < t)
                        for kvh in (0, 3):
                            self.assertTrue(kvh * G <= kvh * G + head < (kvh + 1) * G)

    def test_key_domain_and_tail_masks(self):
        for t in EXTENTS:
            for pos0 in (0, 1, 15, 16, 127, 16384 - t):
                groups = (t * G + 63) // 64
                for group in {0, groups // 2, groups - 1}:
                    for pair in range(4):
                        r0 = group * 64 + pair * 16
                        if r0 >= t * G:
                            continue
                        for offset in (0, 7, 8, 15):
                            row = r0 + offset
                            if row >= t * G:
                                continue
                            for bidi in (False, True):
                                last, first, limit = limits(t, pos0, group, pair, row, bidi)
                                self.assertLess(last, pos0 + t)
                                starts = {0, (last // BK) * BK, (limit // BK) * BK}
                                for k0 in starts:
                                    if k0 > last:
                                        continue
                                    diag = k0 + BK - 1 > (last if bidi else first)
                                    for key in range(k0, k0 + BK):
                                        masked = diag and key > limit
                                        self.assertEqual(not masked, key <= limit)
                                        if not masked:
                                            self.assertLessEqual(key, last)
                                if bidi:
                                    self.assertEqual(limit, pos0 + t - 1)
                                else:
                                    self.assertEqual(limit, pos0 + row // G)

    def test_bidi_future_key_and_capacity_edge(self):
        for t in (2, 11, 33, 2049):
            pos0 = 16384 - t
            last, _, limit = limits(t, pos0, 0, 0, 0, True)
            self.assertEqual(last, 16383)
            self.assertEqual(limit, last)
            self.assertGreater(limit, pos0)
            self.assertLessEqual(((t * G + 63) // 64) * 64, t * G + 63)


class BidiSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = (ROOT / "src/kernels/tensor/qattn_t.metal").read_text()
        cls.d2 = cls.kernel.split("template <int NS, int G, bool BIDI = false>", 1)[1]
        cls.host = (ROOT / "src/model/qwen35.cpp").read_text().split("void Qwen35::enc_attn_prefill(", 1)[1]
        cls.host = cls.host.split("void Qwen35::enc_mixer_prefill(", 1)[0]

    def test_only_new_g6_wrapper_is_bidirectional(self):
        wrappers = re.findall(r"^QATTN_TD2\((\w+), (\d+), (\d+), (true|false)\)$", self.d2, re.M)
        self.assertEqual(wrappers, [
            ("qattn_t_g8d2s8", "8", "8", "false"),
            ("qattn_t_g6d2s8", "8", "6", "false"),
            ("qattn_t_g6d2s8_bidi", "8", "6", "true"),
            ("qattn_t_g4d2s8", "8", "4", "false"),
        ])
        self.assertIn("qattn_t_d2_impl<NS, G, BIDI>", self.d2)
        self.assertNotIn("BIDI", self.kernel.split("template <int NS, int G, bool BIDI = false>", 1)[0])

    def test_bidi_bounds_preserve_causal_expressions(self):
        compact = " ".join(self.d2.split())
        self.assertIn("BIDI ? p.pos0 + p.T - 1 : p.pos0 + min((tg.x * NP * 16 + NP * 16 - 1) / G, p.T - 1)", compact)
        self.assertIn("lim[i] = BIDI ? tgl : p.pos0 + min(tk[i], p.T - 1);", compact)
        self.assertIn("BIDI ? k0 + BK - 1 > tgl : k0 + BK - 1 > first", compact)
        self.assertIn("if (diag && k0 + fn + j > lim[i]) v = BIDI ? -INFINITY : -1e30f;", compact)
        self.assertIn("const bool ok = k0 + r <= tgl;", compact)
        self.assertIn("if (!live || tk[i] >= p.T) continue;", compact)
        self.assertEqual(self.d2.count("threadgroup_barrier(mem_flags::mem_threadgroup);"), 3)
        skip = self.d2.index("if (!live) continue;")
        self.assertGreater(skip, self.d2.rindex("threadgroup_barrier(mem_flags::mem_threadgroup);"))

    def test_static_memory_contract(self):
        self.assertIn("threadgroup half kt[16 * 256], vt[16 * 256];", self.d2)
        self.assertIn("threadgroup float xs[NS * 256];", self.d2)
        allocated = 2 * BK * 256 * 2 + NS * 256 * 4
        self.assertEqual(allocated, 24576)
        self.assertLessEqual(allocated, 32768)
        self.assertGreater(2 * allocated, 32768)

    def test_dispatch_eligibility_and_fallback(self):
        preamble, noncausal = self.host.split("if (!cfg_.causal) {", 1)
        noncausal = noncausal.split("// default attn_prefill11r", 1)[0]
        self.assertIn("if (T == 0) return;", preamble)
        for flag in ("NI_ATTN", "NI_QATTN", "NI_QATTN_D2", "NI_QATTN_G6"):
            self.assertIn(f'getenv("{flag}")' if flag == "NI_ATTN" else f'env_int("{flag}", 1)', preamble)
        self.assertIn("qmt_on()", preamble)
        for guard in ("qattn && d2_on && g6_on", "cfg_.head_dim == 256", "cfg_.hkv > 0", "cfg_.hq == 6 * cfg_.hkv",
                      "T > 0", "pos0 >= 0", "pos0 <= cap", "T <= cap - pos0", "uint64_t(T) * 6 + 63 <= 0xffffffffu"):
            self.assertIn(guard, noncausal)
        self.assertNotIn("static const bool qg6", self.host)
        self.assertIn('pipe("qattn_t_g6d2s8_bidi")', noncausal)
        self.assertIn('pipe("attn_prefill11_bidi_k16")', noncausal)
        self.assertIn("e.dispatch(uint32_t((uint64_t(T) * 6 + 63) / 64), cfg_.hkv, 1, 256);", noncausal)
        self.assertIn("e.dispatch((T + 31) / 32, cfg_.hq, 1, 256);", noncausal)


if __name__ == "__main__":
    unittest.main()
