#!/usr/bin/env python3
"""CPU source/layout checks, not Metal compilation or synchronization proof."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/kernels/attn.metal").read_text()


def body(marker):
    start = SOURCE.index(marker)
    opening = SOURCE.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (SOURCE[end] == "{") - (SOURCE[end] == "}")
        end += 1
    return SOURCE[opening + 1:end - 1]


class BidiExchange(unittest.TestCase):
    def test_wrapper_preserves_bidi_and_rescale_skip(self):
        wrapper = body("void attn_prefill11_bidi_k16(")
        self.assertIn("threadgroup uint4 kvm[attn11_kv<16>() / 8];", wrapper)
        self.assertIn("threadgroup float2 xs[attn11_xs<16>()];", wrapper)
        self.assertIn("attn_prefill11_impl<16, true, false, false, true, false, true>", wrapper)
        self.assertIn("(threadgroup half*)kvm, xs", wrapper)

    def test_static_memory_and_exchange_addresses(self):
        # Keep these arithmetic checks tied to the actual shader layout helpers.
        self.assertRegex(SOURCE, r"constant constexpr uint kHD = 256;")
        self.assertIn("return kHD * (BK + 8) > BK * kVLD11 ? kHD * (BK + 8) : BK * kVLD11;", SOURCE)
        self.assertIn("return 8 * (BK / 8) * 32;", SOURCE)
        self.assertRegex(SOURCE, r"kVLD11 = kHD \+ 8;")
        bk, hd, nc = 16, 256, 2
        kv_bytes = max(hd * (bk + 8), bk * (hd + 8)) * 2
        slots = 8 * nc * 32
        self.assertEqual((kv_bytes, slots * 8, kv_bytes + slots * 8), (12288, 4096, 16384))
        self.assertLessEqual(2 * (kv_bytes + slots * 8), 32768)
        written = {(sg * nc + c) * 32 + lane for sg in range(8) for c in range(nc) for lane in range(32)}
        self.assertEqual(written, set(range(slots)))
        for sg in range(8):
            for c in range(nc):
                for lane in range(32):
                    peer = ((sg ^ 1) * nc + c) * 32 + lane
                    self.assertIn(peer, written)
                    self.assertEqual((sg ^ 1) // 2, sg // 2)
                    self.assertNotEqual((sg ^ 1) % 2, sg % 2)

    def test_retained_fences_and_disjoint_exchange(self):
        impl = body("NI_INLINE void attn_prefill11_impl(")
        self.assertIn("threadgroup float2* xw = XS ? xs : (threadgroup float2*)kv;", impl)
        fence = "threadgroup_barrier(mem_flags::mem_threadgroup);"
        self.assertEqual(impl.count("if (!XS) " + fence), 2)
        # One MASK-only fence, four retained fences, and two alias-only fences.
        self.assertEqual(impl.count(fence), 7)
        exchange = impl.index("xw[(sg * NC + c) * 32 + lane] =")
        publication = impl.index(fence, exchange)
        peer_read = impl.index("const float2 o2 = xw[")
        store_v = impl.index("attn11_store_v<BK>(kv, r, tid);")
        v_visible = impl.index(fence, store_v)
        pv = impl.index("simdgroup_multiply_accumulate(o[j]")
        self.assertLess(exchange, publication)
        self.assertLess(publication, peer_read)
        self.assertLess(peer_read, store_v)
        self.assertLess(store_v, v_visible)
        self.assertLess(v_visible, pv)
        self.assertEqual(2 * ((4096 + 15) // 16), 512)

    def test_bidi_key_tails_and_padded_query_rows(self):
        impl = body("NI_INLINE void attn_prefill11_impl(")
        self.assertIn("kend = p.pos0 + (BIDI ? p.T : min(t0 + kBQ11, p.T));", impl)
        self.assertIn("k0 + c * 8 + fcol + i > (BIDI ? kend - 1 : qpos)", impl)
        for t in (1, 15, 16, 17, 31, 32, 33, 65, 4095, 4096, 4097):
            for pos0 in (0, 1, 17):
                kend = pos0 + t
                keys = {tile * 16 + c * 8 + col + i
                        for tile in range((kend + 15) // 16)
                        for c in range(2) for col in (0, 2, 4, 6) for i in range(2)
                        if tile * 16 + c * 8 + col + i < kend}
                self.assertEqual(keys, set(range(kend)))
                rows = {(block * 32 + rg * 8 + row)
                        for block in range((t + 31) // 32) for rg in range(4) for row in range(8)
                        if block * 32 + rg * 8 + row < t}
                self.assertEqual(rows, set(range(t)))
        # All fragment row indices are 0..7; lane pairs cover each 8-column fragment.
        lanes = {(((lane // 4) & 4) + (lane // 2) % 4,
                  ((lane // 4) & 2) * 2 + (lane % 2) * 2) for lane in range(32)}
        self.assertEqual(lanes, {(row, col) for row in range(8) for col in (0, 2, 4, 6)})


if __name__ == "__main__":
    unittest.main()
