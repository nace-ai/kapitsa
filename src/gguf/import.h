// `kapitsa import`: a GGUF file of a Qwen3.5-family model (llama.cpp's qwen35 / qwen35moe, as LM Studio, Ollama,
// llama.cpp and Hugging Face GGUF repos ship them) converted once into a Kapitsa model folder in the models dir:
// config.json and tokenizer files from the GGUF's metadata, MLX-format safetensors in the formats Kapitsa's kernels
// run, and mtp.safetensors when the GGUF (or an mtp-*.gguf beside it) holds the MTP head. llama.cpp's tensor
// transforms are undone (RMSNorm weights stay 1 + w as MLX keeps them, A_log from -exp(A_log), the DeltaNet value
// heads back from tiled to grouped order, conv1d to [C, 4, 1]).
//
// Formats, per GGUF type: Q4_0 / Q4_1 / Q4_K keep their codes as 4-bit group 32 (scales and offsets rounded to bf16);
// Q8_0 and Q6_K become 8-bit group 64, Q5_0 / Q5_1 / Q5_K 6-bit group 64, IQ4_NL / IQ4_XS / MXFP4 / NVFP4 5-bit group
// 64, IQ3_XXS / IQ3_S / Q3_K 4-bit group 32, IQ1 / IQ2 / Q2_K 3-bit group 64 (all re-quantized from the GGUF's values);
// F32 / F16 / BF16 become bf16. In a file that is mostly 4-bit or better (unsloth's UD mixes), its 1-3-bit tensors
// take the file's majority format instead. MoE experts (whose kernels have no group-32 forms) take 5-bit group 64 from
// 4-bit types, and a layer's experts share one format, as do a dense layer's gate and up. Ternary types are refused.
// Recursive's Flint (arch "flint", build_flint): only a file whose routed experts are IQ blocks, kept as stored (an
// APFS clone of the GGUF); attention, the dense MLP and the shared expert become 4-bit group 64, the embedding 3-bit
// group 64, the LM head 6-bit group 64, and the router stays bf16.
#pragma once
#include <string>

namespace ni::gguf {

struct ImportOptions {
  std::string file;          // the GGUF (any part of a split file, or an Ollama blob)
  std::string mtp;           // a separate MTP GGUF ("": the file's own MTP block, else an mtp-*.gguf beside it)
  std::string name;          // folder name in the models dir ("": from the file name)
  int threads = 0;           // 0: this Mac's performance cores, at most 8
  bool force = false;        // convert again over an existing folder of that name
  bool dry_run = false;      // print what would be written (formats per GGUF type, size, folder) and write nothing
  // hub::plan_dir: write only what a memory plan reads into this folder (config.json, each safetensors file's header,
  // no tensor data) and import.json ({"write": bytes the conversion writes, "disk": the folder's size once the GGUF is
  // gone}); nothing in the models dir
  std::string plan_dir;
  bool progress = true;
};

// Converts the GGUF, or finds its earlier conversion (same files, sizes and dates), and returns the model folder.
// Throws Error (gguf.h) with a message for the user.
std::string import_model(const ImportOptions& opt);

// Whether `arg` names a GGUF file (by extension or by its first four bytes), for `kapitsa serve --model`.
bool is_gguf_file(const std::string& arg);

// The `kapitsa import` command (argv[0] = "kapitsa", argv[1] = "import").
int run_import(int argc, char** argv);

}  // namespace ni::gguf
