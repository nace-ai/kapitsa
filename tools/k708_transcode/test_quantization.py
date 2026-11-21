"""Small CPU tests: python -m unittest tools.k708_transcode.test_quantization."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize

try:
    from . import quantization as quant
except ImportError:
    import quantization as quant


def raw_rows(qtype, rows=3, columns=512, seed=708):
    rng = np.random.default_rng(seed)
    if qtype == T.F32:
        return rng.normal(0, 0.2, (rows, columns)).astype("<f4").view(np.uint8)
    block_size, block_bytes = GGML_QUANT_SIZES[qtype]
    blocks = rng.integers(0, 256, (rows * columns // block_size, block_bytes), dtype=np.uint8)
    offset = {T.Q6_K: 208, T.Q3_K: 108}.get(qtype, 0)
    blocks[:, offset:offset + 2] = rng.uniform(-0.05, 0.05, (len(blocks), 1)).astype("<f2").view(np.uint8)
    if qtype in (T.Q4_K, T.Q5_K):
        blocks[:, 2:4] = rng.uniform(0, 0.05, (len(blocks), 1)).astype("<f2").view(np.uint8)
    return blocks.reshape(rows, -1)


def affine(packed, scales, biases, bits, group_size, dtype=np.float32):
    q = quant.unpack(packed, bits).reshape(scales.shape[0], -1, group_size)
    return (scales.astype(dtype)[..., None] * q + biases.astype(dtype)[..., None]).reshape(packed.shape[0], -1)


def metrics(reference, result):
    reference, result = reference.astype(np.float64), result.astype(np.float64)
    diff = result - reference
    sse, energy = np.sum(diff ** 2), np.sum(reference ** 2)
    return {"max_error": np.max(np.abs(diff)), "mse": sse / reference.size,
            "nmse": sse / energy if energy else (0 if sse == 0 else np.inf)}


class PackingTests(unittest.TestCase):
    def test_matches_rehearsal_bitstream(self):
        rng = np.random.default_rng(581)
        for bits in (2, 3, 4, 5, 6, 8):
            with self.subTest(bits=bits):
                q = rng.integers(0, 1 << bits, (3, 128), dtype=np.uint32)
                expanded = ((q[..., None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
                expected = np.packbits(expanded.reshape(3, -1), axis=-1, bitorder="little").view("<u4")
                packed = quant.pack(q, bits)
                self.assertEqual(packed.dtype, np.uint32)
                self.assertEqual(packed.shape, (3, 4 * bits))
                np.testing.assert_array_equal(packed, expected)
                np.testing.assert_array_equal(quant.unpack(packed, bits), q)

    def test_word_crossings_against_scalar_integer(self):
        for bits in (3, 5, 6):
            q = (np.arange(64, dtype=np.uint8) * 7) % (1 << bits)
            whole = sum(int(value) << (i * bits) for i, value in enumerate(q))
            expected = np.array([(whole >> i) & 0xffffffff for i in range(0, 64 * bits, 32)], dtype=np.uint32)
            np.testing.assert_array_equal(quant.pack(q[None], bits)[0], expected)

    def test_refuses_invalid_packing(self):
        for bits in (1, 7, True, 4.0):
            with self.assertRaises(ValueError):
                quant.pack(np.zeros((1, 32), dtype=np.uint8), bits)
        for q in (np.full((1, 32), -1), np.full((1, 32), 16), np.zeros((1, 8), dtype=np.uint8),
                  np.zeros((1, 32), dtype=np.float32)):
            with self.assertRaises(ValueError):
                quant.pack(q, 4)
        with self.assertRaises(ValueError):
            quant.unpack(np.zeros((1, 4), dtype=np.uint32), 5)


class DirectTests(unittest.TestCase):
    def test_random_legal_blocks_match_authoritative_float32(self):
        for qtype in (T.Q4_K, T.Q5_K, T.Q8_0):
            for seed in (0, 708, 1708):
                with self.subTest(qtype=qtype.name, seed=seed):
                    raw = raw_rows(qtype, rows=7, columns=1024, seed=seed)
                    q, s, b = quant.direct_affine(raw, qtype)
                    got = (s[..., None] * q.reshape(7, -1, 32) + b[..., None]).reshape(7, -1)
                    np.testing.assert_array_equal(got, dequantize(raw, qtype))
                    self.assertEqual(q.dtype, np.uint8)
                    self.assertEqual(s.dtype, np.float32)
                    self.assertEqual(b.dtype, np.float32)

    def test_codes_and_coefficients_against_scalar_block_decode(self):
        for qtype in (T.Q4_K, T.Q5_K):
            raw = raw_rows(qtype, rows=1, columns=256)
            q, scales, biases = quant.direct_affine(raw, qtype)
            block = raw[0]
            d = block[:2].copy().view("<f2").astype(np.float32)[0]
            dm = block[2:4].copy().view("<f2").astype(np.float32)[0]
            meta = block[4:16]
            for group in range(8):
                if group < 4:
                    sc, mn = int(meta[group] & 63), int(meta[group + 4] & 63)
                else:
                    sc = int(meta[group + 4] & 15) | (int(meta[group - 4] >> 6) << 4)
                    mn = int(meta[group + 4] >> 4) | (int(meta[group] >> 6) << 4)
                self.assertEqual(sc * d, scales[0, group])
                self.assertEqual(-mn * dm, biases[0, group])
                for index in range(32):
                    low_offset = (16 if qtype == T.Q4_K else 48) + group // 2 * 32 + index
                    code = (int(block[low_offset]) >> (4 * (group % 2))) & 15
                    if qtype == T.Q5_K:
                        code |= ((int(block[16 + index]) >> group) & 1) << 4
                    self.assertEqual(code, q[0, group * 32 + index])

    def test_q8_signed_extrema_and_zero_scale_codes(self):
        raw = raw_rows(T.Q8_0, rows=1, columns=32)
        raw[0, :2] = np.array([0.03125], dtype="<f2").view(np.uint8)
        raw[0, 2:] = np.resize(np.array([-128, -127, 0, 126, 127], dtype=np.int8), 32).view(np.uint8)
        q, s, b = quant.direct_affine(raw, T.Q8_0)
        np.testing.assert_array_equal(q, raw[:, 2:].view(np.int8).astype(np.int16) + 128)
        np.testing.assert_array_equal(s[:, :, None] * q.reshape(1, 1, 32) + b[:, :, None], dequantize(raw, T.Q8_0)[:, None])
        for qtype, bits in ((T.Q4_K, 4), (T.Q5_K, 5), (T.Q8_0, 8)):
            raw = raw_rows(qtype, rows=1, columns=256)
            block_size = GGML_QUANT_SIZES[qtype][1]
            raw.reshape(-1, block_size)[:, :2] = 0
            expected_q, _, _ = quant.direct_affine(raw, qtype)
            p, _, _, receipt = quant.transcode(raw, qtype, bits=bits, group_size=32, exact=True)
            np.testing.assert_array_equal(quant.unpack(p, bits), expected_q)
            self.assertTrue(receipt["integer_codes_preserved"])

    def test_zero_and_subnormal_coefficients(self):
        for qtype, bits in ((T.Q4_K, 4), (T.Q5_K, 5), (T.Q8_0, 8)):
            raw = raw_rows(qtype, rows=1, columns=256)
            blocks = raw.reshape(-1, GGML_QUANT_SIZES[qtype][1])
            blocks[:, :2] = np.array([np.nextafter(np.float16(0), np.float16(1))], dtype="<f2").view(np.uint8)
            if qtype != T.Q8_0:
                blocks[:, 2:4] = 0
            p, s, b, receipt = quant.transcode(raw, qtype, bits=bits, group_size=32, exact=True)
            np.testing.assert_array_equal(affine(p, s, b, bits, 32), dequantize(raw, qtype))
            self.assertEqual(receipt["max_error"], 0)

    def test_direct_receipt_separates_coefficient_and_algebra_errors(self):
        for qtype, bits in ((T.Q4_K, 4), (T.Q5_K, 5), (T.Q8_0, 8)):
            raw = raw_rows(qtype)
            q, sf, bf = quant.direct_affine(raw, qtype)
            p, s, b, receipt = quant.transcode(raw, qtype, bits=bits, group_size=32, exact=True)
            np.testing.assert_array_equal(s, sf.astype(np.float16))
            np.testing.assert_array_equal(b, bf.astype(np.float16))
            np.testing.assert_array_equal(quant.unpack(p, bits), q)
            reference = dequantize(raw, qtype)
            actual = affine(p, s, b, bits, 32)
            ideal = (sf.astype(np.float64)[..., None] * q.reshape(3, -1, 32)
                     + bf.astype(np.float64)[..., None]).reshape(reference.shape)
            rounded = affine(p, s, b, bits, 32, np.float64)
            for key, expected in metrics(reference, actual).items():
                self.assertEqual(receipt[key], expected)
            self.assertEqual(receipt["coefficient_rounding_error"], metrics(ideal, rounded))
            self.assertEqual(receipt["source_float32_algebra_error"], metrics(ideal, reference))
            if qtype in (T.Q4_K, T.Q5_K):
                self.assertGreater(receipt["coefficient_rounding_error"]["max_error"], 0)
            else:
                self.assertEqual(receipt["coefficient_rounding_error"]["max_error"], 0)
            # Every observed error is accounted for by coefficient casting plus
            # each side's actual float32 algebra rounding, not an allclose epsilon.
            bound = np.abs(rounded - ideal) + np.abs(reference - ideal) + np.abs(actual - rounded)
            self.assertTrue(np.all(np.abs(actual.astype(np.float64) - reference) <= bound))
            self.assertTrue(receipt["direct_float32_verified"])
            self.assertFalse(receipt["calibration_used"])
            self.assertFalse(receipt["evaluation_text_used"])
            json.dumps(receipt, allow_nan=False)

    def test_exact_never_silently_falls_back(self):
        for qtype, bits, group in ((T.Q6_K, 6, 32), (T.IQ4_NL, 4, 32), (T.IQ4_XS, 4, 32),
                                   (T.Q3_K, 3, 32), (T.Q4_K, 4, 64), (T.Q5_K, 4, 32), (T.F32, 4, 32)):
            with self.subTest(qtype=qtype, bits=bits, group=group), self.assertRaises(ValueError):
                quant.transcode(raw_rows(qtype), qtype, bits=bits, group_size=group, exact=True)

    def test_direct_coefficient_overflow_raises(self):
        raw = raw_rows(T.Q8_0, rows=1, columns=32)
        raw[0, :2] = np.array([1024], dtype="<f2").view(np.uint8)
        with np.errstate(over="ignore"), self.assertRaisesRegex(ValueError, "overflow"):
            quant.transcode(raw, T.Q8_0, bits=8, group_size=32, exact=True)

    def test_direct_verification_detects_parser_regression(self):
        original = quant.direct_affine
        def broken(*args):
            q, s, b = original(*args)
            q[0, 0] ^= 1
            return q, s, b
        with patch.object(quant, "direct_affine", broken), self.assertRaisesRegex(RuntimeError, "disagrees"):
            quant.transcode(raw_rows(T.Q4_K), T.Q4_K, bits=4, group_size=32, exact=True)


class FittingTests(unittest.TestCase):
    def test_t64_all_source_types_have_same_target_bytes(self):
        for qtype in (T.Q4_K, T.Q5_K, T.Q6_K, T.Q8_0, T.IQ4_XS, T.IQ4_NL, T.Q3_K, T.F32):
            with self.subTest(qtype=qtype.name):
                raw = raw_rows(qtype, rows=2, columns=256)
                p, s, b, receipt = quant.transcode(raw, qtype, bits=4, group_size=64)
                self.assertEqual((p.shape, s.shape, b.shape), ((2, 32), (2, 4), (2, 4)))
                self.assertEqual(p.nbytes + s.nbytes + b.nbytes, 2 * (256 * 4 // 8 + 256 // 64 * 4))
                self.assertEqual(receipt["method"], "least_squares_reround")
                self.assertFalse(receipt["integer_codes_preserved"])
                self.assertIsNone(receipt["coefficient_rounding_error"])
                expected = metrics(dequantize(raw, qtype), affine(p, s, b, 4, 64))
                for key, value in expected.items():
                    self.assertEqual(receipt[key], value)
                json.dumps(receipt, allow_nan=False)

    def test_fitting_not_worse_than_stored_minmax_baseline(self):
        rng = np.random.default_rng(708)
        w = rng.normal(0.3, 0.4, (5, 512)).astype(np.float32)
        w[0, ::31] *= 4
        for bits, group in ((4, 32), (4, 64), (5, 32), (6, 64), (8, 32)):
            with self.subTest(bits=bits, group=group):
                p, s, b, _ = quant.transcode(w.view(np.uint8), T.F32, bits=bits, group_size=group)
                wg = w.reshape(5, -1, group)
                lo, hi = wg.min(-1, keepdims=True), wg.max(-1, keepdims=True)
                bs = ((hi - lo) / ((1 << bits) - 1)).astype(np.float16).astype(np.float32)
                bb = lo.astype(np.float16).astype(np.float32)
                q = np.clip(np.rint((wg - bb) / bs), 0, (1 << bits) - 1)
                baseline = bs * q + bb
                got = affine(p, s, b, bits, group).reshape(wg.shape)
                baseline_sse = np.sum((baseline.astype(np.float64) - wg) ** 2, axis=-1)
                actual_sse = np.sum((got.astype(np.float64) - wg) ** 2, axis=-1)
                self.assertTrue(np.all(actual_sse <= baseline_sse))
                self.assertTrue(np.any(actual_sse < baseline_sse))

    def test_constant_zero_and_tiny_weights(self):
        for value in (0.0, 0.125, -0.5, float(np.nextafter(np.float32(0), np.float32(1)))):
            w = np.full((2, 64), value, dtype=np.float32)
            p, s, b, receipt = quant.transcode(w.view(np.uint8), T.F32, bits=4, group_size=32)
            self.assertTrue(np.isfinite(s).all() and np.isfinite(b).all())
            expected = np.full(w.shape, np.float16(value), dtype=np.float32)
            np.testing.assert_array_equal(affine(p, s, b, 4, 32), expected)
            self.assertEqual(receipt["nmse"], 0 if value == 0 or abs(value) >= 0.125 else 1)

    def test_q6_subblock16_uses_explicit_eight_bit_reround(self):
        raw = raw_rows(T.Q6_K, rows=2, columns=256)
        p, s, b, receipt = quant.transcode(raw, T.Q6_K, bits=8, group_size=32)
        self.assertEqual(receipt["method"], "least_squares_reround")
        self.assertFalse(receipt["integer_codes_preserved"])
        self.assertEqual(affine(p, s, b, 8, 32).shape, (2, 256))
        self.assertGreater(receipt["mse"], 0)

    def test_row_chunking_produces_identical_arrays(self):
        for qtype, bits, group, exact in ((T.Q5_K, 5, 32, True), (T.IQ4_XS, 4, 64, False), (T.F32, 4, 32, False)):
            raw = raw_rows(qtype, rows=7, columns=256)
            whole = quant.transcode(raw, qtype, bits=bits, group_size=group, exact=exact)
            with patch.object(quant, "_CHUNK_ELEMENTS", 512):
                chunks = quant.transcode(raw, qtype, bits=bits, group_size=group, exact=exact)
            for full, chunked in zip(whole[:3], chunks[:3]):
                np.testing.assert_array_equal(full, chunked)
            self.assertEqual(chunks[3]["chunk_rows"], 2)
            self.assertEqual(whole[3]["max_error"], chunks[3]["max_error"])
            self.assertAlmostEqual(whole[3]["mse"], chunks[3]["mse"], places=14)

    def test_noncontiguous_raw_rows(self):
        raw = raw_rows(T.Q5_K)
        padded = np.zeros((raw.shape[0], raw.shape[1] * 2), dtype=np.uint8)
        padded[:, ::2] = raw
        result = quant.transcode(raw, T.Q5_K, bits=5, group_size=32, exact=True)
        strided = quant.transcode(padded[:, ::2], T.Q5_K, bits=5, group_size=32, exact=True)
        for a, b in zip(result[:3], strided[:3]):
            np.testing.assert_array_equal(a, b)

    def test_numpy_integer_parameters_leave_json_receipt(self):
        _, _, _, receipt = quant.transcode(raw_rows(T.F32), T.F32, bits=np.int64(4),
                                          group_size=np.int32(64))
        json.dumps(receipt, allow_nan=False)

    def test_unrepresentable_fit_coefficients_raise(self):
        w = np.full((1, 64), 100000, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite float16"):
            quant.transcode(w.view(np.uint8), T.F32, bits=4, group_size=32)

    def test_bad_input_is_rejected(self):
        good = raw_rows(T.Q4_K)
        for raw, qtype, kwargs in ((good.astype(np.int8), T.Q4_K, {}), (good[0], T.Q4_K, {}),
                                   (good[:, :-1], T.Q4_K, {}), (good[:0], T.Q4_K, {}),
                                   (good, T.Q4_K, {"group_size": 16}), (good, T.Q4_K, {"group_size": 32.0}),
                                   (good, T.Q4_K, {"exact": "yes"}), (good, T.Q4_0, {})):
            args = {"bits": 4, "group_size": 32, **kwargs}
            with self.subTest(qtype=qtype, args=args), self.assertRaises(ValueError):
                quant.transcode(raw, qtype, **args)
        w = np.full((1, 64), np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            quant.transcode(w.view(np.uint8), T.F32, bits=4, group_size=32)
        with patch.object(quant, "_CHUNK_ELEMENTS", 128), self.assertRaisesRegex(ValueError, "workspace"):
            quant.transcode(good, T.Q4_K, bits=4, group_size=32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
