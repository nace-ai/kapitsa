// The models `kapitsa serve --model <name>[:<variant>]` knows by name: the one table to edit to add a model, add a
// variant (another precision of the same model) or change a model's default variant. Nothing else needs to change:
// the downloader, `pull`, `list`, `--help` and the folder names (<name>-<variant>) follow this table.
//
// - Models are listed as the README and the picker show them: Qwen from small to large, then Nemotron from small to
//   large, then Recursive's Flint. `kapitsa serve` without --model takes the most capable one that fits this Mac (best_model's order,
//   src/server/main.cpp).
// - `--model <name>` takes the model's default variant where the engine's memory plan of its headers leaves room for
//   one --ctx conversation with its drafter, else the first of its fallbacks that does (smaller variants, best
//   first). The default is the highest-quality variant whose decode with speculative decoding stays within ~10% of
//   the 4-bit variant's on a 48 GB M4 Pro (measured: quality against
//   the bf16 model, decode and prefill speed); today that is 4-bit for every model, as each step up in precision
//   reads 22% (5-bit) to 89% (8-bit) more bytes per token.
// - A variant pins its weights to a revision (a commit), so a download is always the same bytes; every file of the
//   repo but .gitattributes is fetched. Its companions are fetched with it: an MTP head ("mtp": files taken from a
//   repo; "mtp-sidecar": built from an official checkpoint's mtp.* tensors, with the result's sha256 and size) and a
//   DFlash2 drafter ("dflash2", in the dflash2/ folder that `--dflash auto` looks in first; "dflash": a DFlash (v1)
//   drafter, Qwen3.5-4B's, in the same folder). Both run with a trunk of any precision (the drafter reads the trunk's
//   embedding and output layer in any format). Nemotron 3.5 Bolt's and Super-VL's companion is NVIDIA's LiLiCorr
//   drafter for each ("lilicorr", in dflash/ as oMLX lays it out), Lightning's NVIDIA's DSpark drafter ("dspark", in
//   dflash/ too).
// - A GGUF variant (`gguf`: one file of the repo, pinned by sha256 and size) downloads that file alone and converts it
//   with `kapitsa import` into the variant's folder, which keeps the conversion, not the file. It has no companions
//   (the import replaces the folder).
// - download_gb is shown before a download (weights and companions). Memory needs no entry: the engine plans it
//   from the variant's headers (a GGUF variant's: the import's plan of the file's header).
// - The 4-bit variants' repos and revisions are the Nace Agent app's, so the app and the CLI download the same bytes.
#include "hub/hub.h"

namespace ni::hub {

namespace {

// The companions every precision of a model shares
Companion mtp35() {
  // the MTP head MTPLX ships with mlx-community's 4-bit 35B (its own file in the repo)
  return {.kind = "mtp",
          .repo = "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
          .revision = "95460516fbfe34c48f577f67c9ed191f5d61fbaf",
          .files = {"mtp.safetensors", "mtplx_runtime.json"}};
}
Companion dflash35() {
  return {.kind = "dflash2",
          .repo = "incoai/Qwen3.6-35B-A3B-DFlash2",
          .revision = "51ef7b6923ad6c14cb1bb41c37a9041446496ab6",
          .files = {"config.json", "model.safetensors", "LICENSE"},
          .subdir = "dflash2"};
}
Companion mtp27() {
  // the MLX packs have no MTP head; MTPLX's 27B pack ships Qwen's (mtplx_runtime.json: it reads the post-norm hidden
  // state)
  return {.kind = "mtp",
          .repo = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
          .revision = "1d5087d2062c02b279180a53e4016cf9cd7a3d7e",
          .files = {"mtp.safetensors", "mtplx_runtime.json"}};
}
Companion dflash27() {
  return {.kind = "dflash2",
          .repo = "incoai/Qwen3.8-27B-DFlash2",
          .revision = "015e795645c74b1a0eeef3b570031fb62e769bc5",
          .files = {"config.json", "model.safetensors"},
          .subdir = "dflash2"};
}
Companion mtp4() {
  // the MLX packs drop the MTP head: built from the official checkpoint's 15 mtp.* tensors (241 MB by range requests,
  // not the 9.3 GB checkpoint), byte for byte what tools/mtp_sidecar.py makes; bf16, quantized to 4-bit at load
  return {.kind = "mtp-sidecar",
          .repo = "Qwen/Qwen3.5-4B",
          .revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
          .sha256 = "4f491dd27975520ec9c9bfa970ca215cc7f12a0328761d3656ab7b9f388e472a",
          .size = 241200816};
}

Companion dflash4() {
  // z-lab's DFlash (v1) block drafter for Qwen3.5-4B (Apache-2.0; 1.27 GB of bf16, 4-bit at load), in the dflash2/
  // folder that --dflash auto looks in first: the DFlash2 path runs DFlashDraftModel checkpoints too (no block conv,
  // no candidate selector)
  return {.kind = "dflash",
          .repo = "z-lab/Qwen3.5-4B-DFlash",
          .revision = "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf",
          .files = {"config.json", "model.safetensors"},
          .subdir = "dflash2"};
}

Companion mtp9() {
  // as the 4B's: built from the official checkpoint's 15 mtp.* tensors (487 MB by range requests, not the 19 GB
  // checkpoint), byte for byte what tools/mtp_sidecar.py makes; bf16, quantized to 4-bit at load
  return {.kind = "mtp-sidecar",
          .repo = "Qwen/Qwen3.5-9B",
          .revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
          .sha256 = "5f7c656b3930220c704c52d6c9ad0ab82cacf8da4b66c2f2361cbc3acee25a1f",
          .size = 486582960};
}

Companion dflash9() {
  // z-lab's DFlash (v1) block drafter for Qwen3.5-9B (2.58 GB of bf16, 4-bit at load), in dflash2/ as the 4B's
  return {.kind = "dflash",
          .repo = "z-lab/Qwen3.5-9B-DFlash",
          .revision = "5fc3b3d474760f18c516db87d84c37edbfd3ede6",
          .files = {"config.json", "model.safetensors"},
          .subdir = "dflash2"};
}

Companion lilicorr_bolt() {
  // NVIDIA's LiLiCorr block drafter for Bolt (1.11B, NVFP4 / BF16), in dflash/ where --dflash auto finds it
  return {.kind = "lilicorr",
          .repo = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr",
          .revision = "abd478065678ee83a93821c4b3f512da24e4cecb",
          .files = {"config.json", "model.safetensors", "LICENSE"},
          .subdir = "dflash"};
}

Companion lilicorr_super() {
  // NVIDIA's LiLiCorr block drafter for Super-VL (NVFP4 / BF16, with its own NVFP4 copy of the LM head), in dflash/
  return {.kind = "lilicorr",
          .repo = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr",
          .revision = "292bf1efc06ba39567da7351f2ad7a522afa3959",
          .files = {"config.json", "model.safetensors"},
          .subdir = "dflash"};
}

Companion dspark_lightning() {
  // NVIDIA's DSpark block drafter for Lightning (NVFP4 MLP and Markov head, BF16 attention), in dflash/
  return {.kind = "dspark",
          .repo = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
          .revision = "6d127b50ecd98562cea85d0087680b7d86091fb0",
          .files = {"config.json", "model.safetensors", "LICENSE"},
          .subdir = "dflash"};
}

}  // namespace

const std::vector<CatalogModel>& catalog() {
  static const std::vector<CatalogModel> c = {
      {.name = "qwen3.5-4b",
       .aliases = {"4b"},
       .title = "Qwen3.5-4B",
       .about = "small and quick: for 16-24 GB Macs and simple tasks",
       .brief = "small and quick, for 16-24 GB Macs",
       // 4-bit: no other precision decodes within 10% of it with MTP (6bit: KL 0.022 from bf16 against 0.113,
       // decode -28%: the quality choice)
       .default_variant = "4bit",
       .fallbacks = {},
       .variants = {
           {.name = "4bit",
            .quant = "MLX affine 4-bit, group 64",
            .repo = "lmstudio-community/Qwen3.5-4B-MLX-4bit",
            .revision = "c43ee1d65576a5d98de1e8405cac93c371a655c1",
            .companions = {mtp4(), dflash4()},
            .download_gb = 4.57},
           {.name = "6bit",
            .quant = "MLX affine 6-bit, group 64",
            .repo = "mlx-community/Qwen3.5-4B-6bit",
            .revision = "fbb1092aa9daef007603ea9273cd24c225a409f1",
            .companions = {mtp4(), dflash4()},
            .download_gb = 5.62},
           {.name = "8bit",
            .quant = "MLX affine 8-bit, group 64",
            .repo = "lmstudio-community/Qwen3.5-4B-MLX-8bit",
            .revision = "59098e5fc57e3b33fd18fd3eb6deffa42b320f25",
            .companions = {mtp4(), dflash4()},
            .download_gb = 6.67},
           {.name = "bf16",
            .quant = "bf16, unquantized",
            .repo = "mlx-community/Qwen3.5-4B-bf16",
            .revision = "491fdc7c087ba7fb48adcb1253f8e76d011db783",
            .companions = {mtp4(), dflash4()},
            .download_gb = 10.62},
       }},
      {.name = "qwen3.5-9b",
       .aliases = {"9b"},
       .title = "Qwen3.5-9B",
       .about = "the 4B's bigger sibling: the most capable model for a 16 GB Mac",
       .brief = "more capable than the 4B, for 16-24 GB Macs",
       .default_variant = "4bit",
       .fallbacks = {},
       .variants = {
           {.name = "4bit",
            .quant = "MLX affine 4-bit, group 64",
            .repo = "lmstudio-community/Qwen3.5-9B-MLX-4bit",
            .revision = "b455506b0f574c74616dbcd56879bde38fafcff3",
            .companions = {mtp9(), dflash9()},
            .download_gb = 9.05},
       }},
      {.name = "qwen3.8-27b",
       .aliases = {"27b"},
       .title = "Qwen3.8-27B",
       .about = "dense: the strongest coder of the three, slower to write",
       .brief = "dense: the strongest coder, slower",
       .default_variant = "4bit",
       .fallbacks = {},
       .variants = {
           {.name = "4bit",
            .quant = "MLX affine 4-bit, group 64",
            .repo = "lmstudio-community/Qwen3.8-27B-MLX-4bit",
            .revision = "6067b15cf581666a4aecf6af3afaba4bb5efc20c",
            .companions = {mtp27(), dflash27()},
            .download_gb = 20.17},
           {.name = "5bit",
            .quant = "MLX affine 5-bit, group 64",
            .repo = "lmstudio-community/Qwen3.8-27B-MLX-5bit",
            .revision = "b9ad7b92aa2fd07b041f4b8b3aa459941f8abbf7",
            .companions = {mtp27(), dflash27()},
            .download_gb = 23.53},
           {.name = "6bit",
            .quant = "MLX affine 6-bit, group 64",
            .repo = "lmstudio-community/Qwen3.8-27B-MLX-6bit",
            .revision = "05f38c1f39e2f39619653616fe8e96a5d676358e",
            .companions = {mtp27(), dflash27()},
            .download_gb = 26.90},
           {.name = "8bit",
            .quant = "MLX affine 8-bit, group 64",
            .repo = "lmstudio-community/Qwen3.8-27B-MLX-8bit",
            .revision = "241ebb5f1d60b122fd653da658836a55feb9e2b0",
            .companions = {mtp27(), dflash27()},
            .download_gb = 33.62},
           {.name = "bf16",
            .quant = "bf16, unquantized",
            .repo = "mlx-community/Qwen3.8-27B-bf16",
            .revision = "6f265714824f3c38d4452baa1628aef3d9b9aae9",
            .companions = {mtp27(), dflash27()},
            .download_gb = 58.83},
       }},
      {.name = "qwen3.6-35b-a3b",
       .aliases = {"qwen3.6-35b", "35b", "35b-a3b"},
       .title = "Qwen3.6-35B-A3B",
       .about = "MoE, 3B active per token: fast replies and fast reading",
       .brief = "MoE: fast replies, fast reading",
       // 4-bit: no other precision decodes within 10% of it with DFlash2 (6bit: KL 0.024 from bf16 against 0.114,
       // decode -26%: the quality choice). Where it does not fit (16-24 GB Macs): the 2-bit experts of 16gb.
       .default_variant = "4bit",
       .fallbacks = {"16gb"},
       .variants = {
           {.name = "4bit",
            .quant = "MLX affine 4-bit, group 64",
            // mlx-community/Qwen3.6-35B-A3B-4bit with its MTP head
            .repo = "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
            .revision = "95460516fbfe34c48f577f67c9ed191f5d61fbaf",
            .companions = {dflash35()},
            .download_gb = 22.07},
           {.name = "5bit",
            .quant = "MLX affine 5-bit, group 64",
            .repo = "mlx-community/Qwen3.6-35B-A3B-5bit",
            .revision = "c90c1a8b7e7a6a9e39e549f0361a96eb05bbb499",
            .companions = {mtp35(), dflash35()},
            .download_gb = 26.40},
           {.name = "6bit",
            .quant = "MLX affine 6-bit, group 64",
            // byte for byte lmstudio-community/Qwen3.6-35B-A3B-MLX-6bit
            .repo = "mlx-community/Qwen3.6-35B-A3B-6bit",
            .revision = "cb7e092ef8efe540bc3672c8929c4adbe5f4f759",
            .companions = {mtp35(), dflash35()},
            .download_gb = 30.73},
           {.name = "8bit",
            .quant = "MLX affine 8-bit, group 64",
            .repo = "mlx-community/Qwen3.6-35B-A3B-8bit",
            .revision = "e06a74e6236a60c8367e1a3214e83d8b61b637b0",
            .companions = {mtp35(), dflash35()},
            .download_gb = 39.39},
           {.name = "bf16",
            .quant = "bf16, unquantized",
            .repo = "mlx-community/Qwen3.6-35B-A3B-bf16",
            .revision = "04bbe07c3ed2d93cbf832f523fea344bca1d0fcf",
            .companions = {mtp35(), dflash35()},
            .download_gb = 71.88},
           // a 16 GB Mac's: unsloth's UD-IQ2_XXS GGUF imported (11.0 GB of weights in a 12.2 GB budget; M1 Pro 16 GB:
           // 52.6 tok/s), its routed experts read in the file's own blocks. No MTP head in the file, and a drafter
           // would not fit beside it: plain decoding. Text only (the file has no vision tower).
           {.name = "16gb",
            .quant = "IQ2_XXS / IQ2_S / IQ3_S routed experts (as stored), 6-bit attention and DeltaNet, 8-bit shared "
                     "experts, 4-bit embedding / head",
            .label = "2-bit experts (unsloth UD-IQ2_XXS GGUF)",
            .repo = "unsloth/Qwen3.6-35B-A3B-GGUF",
            .revision = "a483e9e6cbd595906af30beda3187c2663a1118c",
            .gguf = {.path = "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
                     .sha256 = "2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef",
                     .size = 10756586464},
            .companions = {},
            .download_gb = 10.76},
       }},
      // a 128 GB Mac's model, not in by_preference (src/server/main.cpp): never picked without --model
      {.name = "qwen3.8-flash-next",
       .aliases = {"flash-next", "flash"},
       .title = "Qwen3.8-Flash-Next 180B-A7B",
       .about = "Qwen's 180B MoE, 7B active: hyper-connections, sparse attention, a 32 GB n-gram table read from the SSD",
       .brief = "Qwen 180B MoE: for 128 GB Macs",
       .default_variant = "4bit",
       .fallbacks = {},
       .variants = {
           // Nace AI's quantization of Qwen/Qwen3.8-Flash-Next @ de4b8e4d (bf16; tools/convert_qwen4_exp.py): the MTP
           // head in mtp.safetensors, the n-gram table in ple.safetensors (mapped, outside GPU memory), vision bf16
           {.name = "4bit",
            .quant = "MLX affine 4-bit experts / DeltaNet / head, 6-bit attention / hyper-connections, 8-bit router / "
                     "indexer / embedding, 4-bit n-gram table",
            .repo = "nace-ai/Qwen3.8-Flash-Next-MLX-4bit",
            .revision = "5d423f9f563c1f1b7ca7945b268e65c0b92b0eae",
            .companions = {},
            .download_gb = 105.86},
       }},
      {.name = "nemotron-labs-dory1.0-4b",
       .aliases = {"dory", "dory-4b", "dory1.0-4b"},
       .title = "Nemotron Dory1.0 4B",
       .about = "NVIDIA's looped 4B: a shared block run 1-4 times per token (--loops; 2 by default), small and fast",
       .brief = "NVIDIA looped 4B: small and fast",
       // NVIDIA's checkpoints as they are: NVFP4 (its native pre-training format; W4A16 with BF16 embedding and LM
       // head) and BF16. No drafter: plain decoding.
       .default_variant = "nvfp4",
       .fallbacks = {},
       .variants = {
           {.name = "nvfp4",
            .quant = "NVFP4 linears (ModelOpt W4A16), BF16 embedding / head",
            .label = "NVFP4 (NVIDIA)",
            .repo = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4",
            .revision = "41f02335e1a1ad3289b8b32485e2265cbe97cfdf",
            .companions = {},
            .download_gb = 3.22},
           {.name = "bf16",
            .quant = "bf16, unquantized",
            .repo = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
            .revision = "b366b87f762ea334ba3eb55912031baba1515810",
            .companions = {},
            .download_gb = 7.97},
       }},
      {.name = "nemotron-3-nano-4b",
       .aliases = {"nano-4b", "nemotron-nano-4b"},
       .title = "Nemotron 3 Nano 4B",
       .about = "NVIDIA's small hybrid Mamba-2 model, text only: prompt lookup drafts from the conversation",
       .brief = "NVIDIA hybrid 4B, text only",
       // NVIDIA's checkpoints as they are: ModelOpt FP8 (W8A16; NVIDIA keeps 39 of its projections and the LM head in
       // BF16, the head loading as 8-bit rows) by default, as it decodes 35 against 24 tok/s for BF16 on an M1 Pro
       // 16 GB in 3 GB less memory, and BF16. No drafter: prompt lookup.
       .default_variant = "fp8",
       .fallbacks = {},
       .variants = {
           {.name = "fp8",
            .quant = "FP8 E4M3 linears (ModelOpt W8A16), some projections and the LM head BF16",
            .label = "FP8 (NVIDIA)",
            .repo = "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
            .revision = "3fe6dab75665a93884214ad4b1b95cf02717d081",
            .companions = {},
            .download_gb = 5.29},
           {.name = "bf16",
            .quant = "bf16, unquantized",
            .repo = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            .revision = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
            .companions = {},
            .download_gb = 7.97},
       }},
      {.name = "nemotron-nano-9b-v2",
       .aliases = {"nano-9b", "nemotron-nano-9b"},
       .title = "Nemotron Nano 9B v2",
       .about = "NVIDIA's hybrid Mamba-2 9B in NVFP4, text only: reasoning switched with /think or enable_thinking",
       .brief = "NVIDIA hybrid 9B (NVFP4), text only",
       // NVIDIA's NVFP4 checkpoint (ModelOpt W4A16, BF16 embedding / head; the head loads as 8-bit rows). No
       // drafter: prompt lookup with --spec.
       .default_variant = "nvfp4",
       .fallbacks = {},
       .variants = {
           {.name = "nvfp4",
            .quant = "NVFP4 linears (ModelOpt W4A16), BF16 embedding / head",
            .label = "NVFP4 (NVIDIA)",
            .repo = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4",
            .revision = "8556c9164ddb43fe1f4f4ad730593b3c5e3f7328",
            .companions = {},
            .download_gb = 7.85},
       }},
      {.name = "nemotron-3.5-lightning",
       .aliases = {"nemotron-lightning", "lightning"},
       .title = "Nemotron 3.5 Lightning 30B-A3B",
       .about = "NVIDIA's hybrid Mamba-2 MoE, 3B active, NVFP4 experts: fast replies on 32 GB Macs and up",
       // one variant: NVIDIA's NVFP4 checkpoint, its safetensors as they are (its MTP head is not used)
       .default_variant = "nvfp4",
       .fallbacks = {},
       .variants = {
           {.name = "nvfp4",
            .quant = "NVFP4 experts / shared expert / head, FP8 Mamba, BF16 attention / embedding, F32 router",
            .repo = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            .revision = "bee7596271d1495f6992ae224aefde4410e816b8",
            .companions = {dspark_lightning()},
            .download_gb = 22.93},
       }},
      {.name = "nemotron-3.5-bolt",
       .aliases = {"nemotron-bolt", "bolt"},
       .title = "Nemotron 3.5 Bolt 74B-A9B",
       .about = "NVIDIA's hybrid Mamba-2 MoE, 9B active, 2-bit experts: an agentic model in 24 GB",
       .brief = "NVIDIA hybrid MoE: agentic, 24 GB",
       // one variant: NVIDIA's own 24 GB checkpoint (quantization-aware distilled), its safetensors as they are
       .default_variant = "24gb",
       .fallbacks = {},
       .variants = {
           {.name = "24gb",
            .quant = "IQ2_XXS experts, NVFP4 Mamba, Q8_0 attention / latent / head, BF16 embedding",
            .label = "NVIDIA 24 GB (2-bit experts, NVFP4, 8-bit)",
            .repo = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
            .revision = "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a",
            .companions = {lilicorr_bolt()},
            .download_gb = 25.08},
       }},
      {.name = "nemotron-3.5-super-vl",
       .aliases = {"nemotron-super", "super-vl", "super"},
       .title = "Nemotron 3.5 Super-VL 120B-A12B",
       .about = "NVIDIA's hybrid Mamba-2 MoE, 12B active, NVFP4 experts (96 GB+ Macs, 128 GB for full context)",
       // one variant: NVIDIA's NVFP4 checkpoint, its safetensors as they are
       .default_variant = "nvfp4",
       .fallbacks = {},
       .variants = {
           {.name = "nvfp4",
            .quant = "NVFP4 experts and shared expert, FP8 Mamba / attention / head, BF16 latent / embedding",
            .repo = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
            .revision = "59b8721646c38111a242da18218859fb2b454322",
            .companions = {lilicorr_super()},
            .download_gb = 76.38},
       }},
      {.name = "flint",
       .aliases = {"recursive-flint", "flint-40b"},
       .title = "Recursive Flint 40B-A2.5B",
       .about = "Recursive's MoE for coding agents, 2.5B active per token: reasoning and tool calls, prompt lookup drafts",
       .brief = "Recursive coding-agent MoE, 2.5B active",
       // Recursive's MLX export as it is (the weights its own engine, rec, runs): affine 4-bit group-64 projections and
       // experts, 8-bit embedding, bf16 router. No drafter: prompt lookup.
       .default_variant = "4bit",
       .fallbacks = {},
       .variants = {
           {.name = "4bit",
            .quant = "MLX affine 4-bit, group 64 (8-bit embedding, bf16 router)",
            .label = "4-bit (Recursive)",
            .repo = "RecursiveSI/flint-int4",
            .revision = "3b93f9053b530488405ae9fca51d4900d6ce21f0",
            .companions = {},
            .download_gb = 22.58},
       }}
  };
  return c;
}

}  // namespace ni::hub
