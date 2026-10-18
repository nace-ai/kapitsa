// What a language model's passes see of images (shared by every model): the prompt rows that take image embeddings
// instead of their placeholder tokens' embeddings, and where each image sits in the prompt (for models whose
// positions depend on it, like Qwen's M-RoPE).
#pragma once
#include "metal/gpu.h"

#include <vector>

namespace ni::vision {

// Prompt rows [pos, pos + n) take rows [row, row + n) of `src` (bf16, `src_ld` values per row) as their embeddings.
struct EmbedRun {
  int pos = 0, n = 0;
  Buffer src;
  int row = 0;
  int src_ld = 0;
};

// One image in a prompt: its rows [pos, pos + n) and its token grid (n = grid_t * grid_h * grid_w).
struct ImageSpan {
  int pos = 0, n = 0;
  int grid_t = 1, grid_h = 0, grid_w = 0;
};

// Writes the run rows inside [pos0, pos0 + T) into x (fp32 rows of D values, row t = prompt position pos0 + t),
// after the pass embedded the chunk's tokens there. No-op when no run reaches the chunk.
void enc_embed_runs(Encoder& e, const std::vector<EmbedRun>& runs, int pos0, int T, const Buffer& x, int D);

}  // namespace ni::vision
