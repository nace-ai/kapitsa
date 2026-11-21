"""CPU-only GGUF rows to MLX affine weights, without calibration data.

``exact=True`` preserves source codes, not bitwise weights: MLX coefficients
are float16. Receipts measure that coefficient rounding separately. Fitted
weights minimize unweighted error against GGUF dequantization, never eval text.
"""
from __future__ import annotations

import numpy as np
from gguf import GGMLQuantizationType as QType
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize as gguf_dequantize


_BITS = (2, 3, 4, 5, 6, 8)
_SUPPORTED = {QType.Q4_K, QType.Q5_K, QType.Q6_K, QType.Q8_0,
              QType.IQ4_XS, QType.IQ4_NL, QType.Q3_K, QType.F32}
_DIRECT_BITS = {QType.Q4_K: 4, QType.Q5_K: 5, QType.Q8_0: 8}
# Fit arrays are bounded independently of the tensor's total number of rows.
_CHUNK_ELEMENTS = 1 << 20


def _bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)) or bits not in _BITS:
        raise ValueError(f"bits must be one of {_BITS}, got {bits!r}")


def pack(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack unsigned codes as MLX's row-wise, LSB-first uint32 bitstream."""
    _bits(bits)
    q = np.asarray(q)
    if q.ndim != 2 or q.dtype.kind not in "ui" or 0 in q.shape:
        raise ValueError("codes must be a nonempty two-dimensional integer array")
    if q.shape[1] % 32:
        raise ValueError("each row must contain a whole number of 32-code blocks")
    if np.any(q < 0) or np.any(q > (1 << bits) - 1):
        raise ValueError("code is outside bit width")
    # At most 32 codes occupy exactly `bits` words, including 5/6-bit cases.
    # Unlike a bit-expanded array this uses only O(number of codes) storage.
    q = q.astype(np.uint32, copy=False).reshape(q.shape[0], -1, 32)
    words = np.zeros((*q.shape[:2], bits), dtype=np.uint32)
    for i in range(32):
        word, shift = divmod(i * bits, 32)
        words[..., word] |= q[..., i] << np.uint32(shift)
        if shift + bits > 32:
            words[..., word + 1] |= q[..., i] >> np.uint32(32 - shift)
    return words.reshape(q.shape[0], -1)


def unpack(packed: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of :func:`pack`; each row must contain whole 32-code blocks."""
    _bits(bits)
    packed = np.asarray(packed)
    if packed.ndim != 2 or packed.dtype != np.uint32 or 0 in packed.shape:
        raise ValueError("packed weights must be a nonempty two-dimensional uint32 array")
    if packed.shape[1] % bits:
        raise ValueError("packed row must contain a whole number of 32-code blocks")
    words = packed.reshape(packed.shape[0], -1, bits)
    q = np.empty((*words.shape[:2], 32), dtype=np.uint8)
    for i in range(32):
        word, shift = divmod(i * bits, 32)
        value = words[..., word] >> np.uint32(shift)
        if shift + bits > 32:
            value = value | (words[..., word + 1] << np.uint32(32 - shift))
        q[..., i] = value & np.uint32((1 << bits) - 1)
    return q.reshape(packed.shape[0], -1)


def _source(data: np.ndarray, qtype: QType) -> tuple[np.ndarray, QType, int]:
    data = np.asarray(data)
    qtype = QType(qtype)
    if qtype not in _SUPPORTED:
        raise ValueError(f"unsupported GGUF tensor type: {qtype.name}")
    if data.dtype != np.uint8 or data.ndim != 2 or 0 in data.shape:
        raise ValueError("data must be nonempty raw uint8 GGUF rows [out, row_bytes]")
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    if data.shape[1] % type_size:
        raise ValueError(f"row bytes are not divisible by {qtype.name} block bytes")
    columns = data.shape[1] // type_size * block_size
    return data, qtype, columns


def direct_affine(data: np.ndarray, qtype: QType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode Q4_K/Q5_K/Q8_0 to codes and original float32 group32 coefficients.

    Returns ``(uint8[out,in], float32[out,in/32], float32[out,in/32])``.
    Q8_0 signed codes become unsigned via ``q = signed_q + 128`` and
    ``bias = -128 * scale``. This helper decodes the supplied rows in one pass;
    use :func:`transcode` for bounded-memory conversion of large tensors.
    """
    data, qtype, columns = _source(data, qtype)
    if qtype not in _DIRECT_BITS:
        raise ValueError(f"{qtype.name} has no supported direct group32 affine mapping")
    rows = data.shape[0]
    size = GGML_QUANT_SIZES[qtype][1]
    blocks = np.ascontiguousarray(data).reshape(-1, size)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    if qtype == QType.Q8_0:
        q = (blocks[:, 2:].view(np.int8).astype(np.int16) + 128).astype(np.uint8)
        s, b = d, np.float32(-128) * d
    else:
        dm = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
        meta = blocks[:, 4:16]
        sc = np.concatenate((meta[:, :4] & 63,
                             (meta[:, 8:12] & 15) | ((meta[:, :4] >> 2) & 48)), axis=1)
        mn = np.concatenate((meta[:, 4:8] & 63,
                             (meta[:, 8:12] >> 4) | ((meta[:, 4:8] >> 2) & 48)), axis=1)
        s = d * sc.astype(np.float32)
        b = -(dm * mn.astype(np.float32))
        low = blocks[:, 16 if qtype == QType.Q4_K else 48:]
        low = low.reshape(-1, 4, 32)
        q = np.stack((low & 15, low >> 4), axis=2).reshape(-1, 8, 32)
        if qtype == QType.Q5_K:
            high = (blocks[:, 16:48, None] >> np.arange(8, dtype=np.uint8)) & 1
            q |= high.transpose(0, 2, 1) << 4
    return q.reshape(rows, columns), s.reshape(rows, -1), b.reshape(rows, -1)


def _assign(w: np.ndarray, s: np.ndarray, b: np.ndarray, levels: int) -> np.ndarray:
    q = np.zeros_like(w)
    np.divide(w - b, s, out=q, where=s != 0)
    np.rint(q, out=q)
    np.clip(q, 0, levels, out=q)
    return q


def _fit(w: np.ndarray, bits: int, group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = w.reshape(w.shape[0], -1, group_size)
    levels = (1 << bits) - 1
    lo, hi = w.min(axis=-1, keepdims=True), w.max(axis=-1, keepdims=True)
    center, half = (hi + lo) * 0.5, (hi - lo) * 0.5
    sw = w.sum(axis=-1, keepdims=True, dtype=np.float64)
    best_error = np.full(lo.shape, np.inf)
    best_q = np.zeros(w.shape, dtype=np.uint8)
    best_s, best_b = np.zeros(lo.shape, dtype=np.float16), np.zeros(lo.shape, dtype=np.float16)
    for clip in (1.0, 0.97, 0.9, 0.8):
        s = (2 * half * clip / levels).astype(np.float32)
        b = lo.copy() if clip == 1.0 else (center - half * clip).astype(np.float32)
        q = _assign(w, s, b, levels)
        for step in range(5):
            # Score the actual stored half coefficients and recode against them.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                sh, bh = s.astype(np.float16), b.astype(np.float16)
                sf, bf = sh.astype(np.float32), bh.astype(np.float32)
                qh = _assign(w, sf, bf, levels)
                diff = (sf * qh + bf).astype(np.float64) - w
                error = np.sum(diff * diff, axis=-1, keepdims=True)
            valid = np.isfinite(sf) & np.isfinite(bf)
            np.copyto(qh, 0, where=~valid)
            better = valid & (error < best_error)
            np.copyto(best_q, qh.astype(np.uint8), where=better)
            np.copyto(best_s, sh, where=better)
            np.copyto(best_b, bh, where=better)
            np.copyto(best_error, error, where=better)
            if step == 4:
                break
            sq = q.sum(axis=-1, keepdims=True, dtype=np.float64)
            sqq = (q * q).sum(axis=-1, keepdims=True, dtype=np.float64)
            sqw = (q.astype(np.float64) * w).sum(axis=-1, keepdims=True)
            var = sqq - sq * sq / group_size
            cov = sqw - sq * sw / group_size
            fitted_s = np.zeros_like(var)
            np.divide(cov, var, out=fitted_s, where=var > 0)
            s = np.maximum(fitted_s, 0).astype(np.float32)
            b = ((sw - s * sq) / group_size).astype(np.float32)
            q = _assign(w, s, b, levels)
    if not np.isfinite(best_error).all():
        raise ValueError("weights cannot be represented with finite float16 affine coefficients")
    return best_q.reshape(w.shape[0], -1), best_s[..., 0], best_b[..., 0]


def _accumulate(stats: list, reference: np.ndarray, actual: np.ndarray) -> None:
    ref = reference.astype(np.float64)
    delta = actual.astype(np.float64) - ref
    stats[0] = max(stats[0], float(np.max(np.abs(delta))))
    stats[1] += float(np.sum(delta * delta))
    stats[2] += float(np.sum(ref * ref))
    stats[3] += ref.size


def _metrics(stats: list) -> dict:
    maximum, sse, energy, count = stats
    return {"max_error": maximum, "mse": sse / count,
            "nmse": sse / energy if energy else (0.0 if sse == 0 else float("inf"))}


def transcode(data: np.ndarray, qtype: QType, *, bits: int, group_size: int,
              exact: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Convert one raw GGUF matrix to MLX affine arrays and an error receipt.

    ``exact=True`` only accepts Q4_K/4-bit, Q5_K/5-bit or Q8_0/8-bit,
    all at group32. Other requests raise, rather than silently fitting.
    ``exact=False`` always fits, including the all-4-bit/group64 T64 recipe.
    Errors use float32 affine reconstruction of the stored half coefficients;
    NMSE is summed squared error divided by GGUF weight energy, with zero/zero=0.
    All source decoding, fitting and receipt calculations run in row chunks.
    """
    _bits(bits)
    bits = int(bits)
    data, qtype, columns = _source(data, qtype)
    if (isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer))
            or group_size not in (32, 64, 128)):
        raise ValueError("group_size must be 32, 64 or 128")
    group_size = int(group_size)
    if columns % group_size:
        raise ValueError("dequantized columns must be divisible by group_size")
    if not isinstance(exact, (bool, np.bool_)):
        raise ValueError("exact must be a boolean")
    if exact and (group_size != 32 or _DIRECT_BITS.get(qtype) != bits):
        raise ValueError(f"no exact mapping for {qtype.name} to {bits}-bit/group{group_size}")
    if columns > _CHUNK_ELEMENTS:
        raise ValueError("a single row exceeds the conversion workspace budget")
    rows = data.shape[0]
    chunk_rows = max(1, _CHUNK_ELEMENTS // columns)
    packed = np.empty((rows, columns * bits // 32), dtype=np.uint32)
    scales = np.empty((rows, columns // group_size), dtype=np.float16)
    biases = np.empty_like(scales)
    error, coeff_error, algebra_error = [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0]
    for first in range(0, rows, chunk_rows):
        last = min(first + chunk_rows, rows)
        raw = np.ascontiguousarray(data[first:last])
        w = gguf_dequantize(raw, qtype)
        if not np.isfinite(w).all():
            raise ValueError(f"nonfinite GGUF weights in rows {first}:{last}")
        if exact:
            q, sf, bf = direct_affine(raw, qtype)
            qg = q.reshape(last - first, -1, 32).astype(np.float32)
            source_affine = sf[..., None] * qg + bf[..., None]
            if not np.array_equal(source_affine.reshape(w.shape), w):
                raise RuntimeError(f"direct {qtype.name} parser disagrees with gguf dequantize")
            with np.errstate(over="ignore"):
                s, b = sf.astype(np.float16), bf.astype(np.float16)
            if not (np.isfinite(s).all() and np.isfinite(b).all()):
                raise ValueError("source affine coefficients overflow float16")
            ideal = sf[..., None].astype(np.float64) * qg + bf[..., None].astype(np.float64)
            rounded = s[..., None].astype(np.float64) * qg + b[..., None].astype(np.float64)
            _accumulate(coeff_error, ideal, rounded)
            _accumulate(algebra_error, ideal, w.reshape(ideal.shape))
        else:
            q, s, b = _fit(w, bits, group_size)
        packed[first:last] = pack(q, bits)
        scales[first:last], biases[first:last] = s, b
        reconstructed = (s[..., None].astype(np.float32) * q.reshape(last - first, -1, group_size)
                         + b[..., None].astype(np.float32)).reshape(w.shape)
        _accumulate(error, w, reconstructed)
    metrics = _metrics(error)
    receipt = {"source_qtype": qtype.name, "shape": [rows, columns],
               "bits": bits, "group_size": group_size,
               "method": "direct_affine" if exact else "least_squares_reround",
               "integer_codes_preserved": bool(exact), "calibration_used": False,
               "calibration_data": None, "evaluation_text_used": False,
               "objective": "unweighted_weight_mse_to_gguf_dequantized",
               "reconstruction_dtype": "float32", "coefficient_dtype": "float16",
               "chunk_rows": chunk_rows, **metrics,
               "direct_float32_verified": bool(exact),
               "coefficient_rounding_error": _metrics(coeff_error) if exact else None,
               "source_float32_algebra_error": _metrics(algebra_error) if exact else None}
    if not exact:
        receipt.update({"clip_initializations": [1.0, 0.97, 0.9, 0.8], "least_squares_steps": 4})
    return packed, scales, biases, receipt
