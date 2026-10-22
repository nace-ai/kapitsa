// GGUF block formats decoded on the CPU, and MLX affine encodings (codes packed little-endian along each row, a bf16
// scale and bias per group), for converting GGUF tensors into Kapitsa's weight formats.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace ni::gguf {

// Whether dequantize() reads a type: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K .. Q6_K, IQ1_S, IQ1_M,
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4 and NVFP4.
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
// llama.cpp's IQ1_S / IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S blocks in the GPU layout of src/kernels/wfmt_iq.metal (routed
// experts kept as the GGUF stores them): a matrix of `rows` rows of K values (K % 256 == 0) as sections Q (a uint2 per
// 32 values, IQ1_S a uint), X1-X3 (the types' other bits) and D (llama.cpp's half d per 256 values), the GGUF's bytes
// per 256 values. The bits are moved, not changed.
struct IqLayout {
  size_t x1 = 0, x2 = 0, x3 = 0, d = 0;  // byte offsets of the sections X1, X2, X3, D (Q at 0)
  size_t bytes = 0;                      // the whole matrix
};
bool iq_gpu_type(uint32_t type);  // a type with that layout: IQ1_S, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S
size_t iq_block_bytes(uint32_t type);  // the type's bytes per 256 values
int iq_bits(uint32_t type);            // its nominal bits (1, 2 or 3), as config.json's quantization names it
// the kernel-name suffix (wfmt_iq.metal's instantiations: iq1s, iq2xxs, iq2s, iq3xxs, iq3s), and the quantization mode
// the importer writes in config.json (iq1_s, iq2_xxs, ...) back to the type (0: another mode)
std::string iq_suffix(uint32_t type);
uint32_t iq_mode_type(const std::string& mode);
IqLayout iq_layout(uint32_t type, size_t rows, size_t K);
// Rows [r0, r1) of the matrix from their GGUF blocks (src: the bytes of row r0 onwards) into dst (the matrix's buffer,
// iq_layout(type, rows, K).bytes long).
void iq_repack(uint32_t type, const uint8_t* src, size_t rows, size_t K, size_t r0, size_t r1, uint8_t* dst);

// Affine quantization of n values (n a multiple of gs) to `bits` bits per group of gs: codes, bf16 scale and bias.
// Per group the best (least squared error, after rounding scale and bias to bf16) of the min-max grid, a symmetric
// grid and least-squares refits of the min-max grid.
void quantize(const float* x, size_t n, int bits, int gs, uint8_t* codes, uint16_t* scales, uint16_t* biases);

}  // namespace ni::gguf
