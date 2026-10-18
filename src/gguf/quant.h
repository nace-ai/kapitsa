// GGUF block formats decoded on the CPU, and MLX affine encodings (codes packed little-endian along each row, a bf16
// scale and bias per group), for converting GGUF tensors into Kapitsa's weight formats.
#pragma once
#include <cstddef>
#include <cstdint>

namespace ni::gguf {

// Whether dequantize() reads a type: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K .. Q6_K, IQ4_NL, IQ4_XS,
// MXFP4 and NVFP4.
bool can_dequantize(uint32_t type);
// n values (a whole number of the type's blocks) into floats, as llama.cpp computes them.
void dequantize(uint32_t type, const uint8_t* src, float* dst, size_t n);

// Q4_0, Q4_1 and Q4_K hold 4-bit codes with an affine map per 32 values (the format of a 4-bit, group-32 MLX
// linear): the codes as stored, and per group of 32 the scale and offset (w = scale * code + offset).
bool exact4(uint32_t type);
void decode4(uint32_t type, const uint8_t* src, uint8_t* codes, float* scale, float* offset, size_t n);

uint16_t to_bf16(float x);  // round to nearest even (NaN stays NaN)
float from_bf16(uint16_t h);
float from_f16(uint16_t h);

// n codes of `bits` bits -> MLX's packing: value j at bits [bits*j, bits*j + bits) of a little-endian stream of
// uint32 words (n * bits must be a multiple of 32).
void pack_codes(const uint8_t* codes, size_t n, int bits, uint32_t* words);
// Affine quantization of n values (n a multiple of gs) to `bits` bits per group of gs: codes, bf16 scale and bias.
// Per group the best (least squared error, after rounding scale and bias to bf16) of the min-max grid, a symmetric
// grid and least-squares refits of the min-max grid.
void quantize(const float* x, size_t n, int bits, int gs, uint8_t* codes, uint16_t* scales, uint16_t* biases);

}  // namespace ni::gguf
