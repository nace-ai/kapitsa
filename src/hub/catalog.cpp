// The models `kapitsa serve --model <name>[:<variant>]` knows by name: the one table to edit to add a model, add a
// variant (another precision of the same model) or change a model's default variant. Nothing else needs to change:
// the downloader, `pull`, `list`, `--help` and the folder names (<name>-<variant>) follow this table.
//
// - Models are listed best first: `kapitsa serve` without --model takes the first with a variant that fits this Mac.
// - `--model <name>` takes the model's default variant where the engine's memory plan of its headers leaves room for
//   one --ctx conversation with its drafter, else the first of its fallbacks that does (smaller variants, best
//   first): so each RAM size gets the best variant it holds. The defaults and fallbacks are chosen from measurements
//   (docs/PERF.md, "Weight formats and quantization"): quality against the bf16 model, decode and prefill speed.
// - A variant pins its weights to a revision (a commit), so a download is always the same bytes; every file of the
//   repo but .gitattributes is fetched. Its companions are fetched with it: an MTP head ("mtp": files taken from a
//   repo; "mtp-sidecar": built from an official checkpoint's mtp.* tensors, with the result's sha256 and size) and a
//   DFlash2 drafter ("dflash2", in the dflash2/ folder that `--dflash auto` looks in first). Both run with a trunk of
//   any precision (the drafter reads the trunk's embedding and output layer in any format).
// - download_gb is shown before a download (weights and companions). Memory needs no entry: the engine plans it
//   from the variant's headers.
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
  // not the 9.3 GB checkpoint), byte for byte what tools/mtp_sidecar.py makes; bf16, quantized to 8-bit at load
  return {.kind = "mtp-sidecar",
          .repo = "Qwen/Qwen3.5-4B",
          .revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
          .sha256 = "4f491dd27975520ec9c9bfa970ca215cc7f12a0328761d3656ab7b9f388e472a",
          .size = 241200816};
}

}  // namespace

const std::vector<CatalogModel>& catalog() {
  static const std::vector<CatalogModel> c = {
      {.name = "qwen3.6-35b-a3b",
       .aliases = {"qwen3.6-35b", "35b", "35b-a3b"},
       .title = "Qwen3.6-35B-A3B",
       .about = "MoE, 3B active per token: fast replies and fast reading",
       // 6-bit: KL 0.024 from bf16 (4-bit 0.114), decode 0.74x of 4-bit; 5-bit / 4-bit where 6-bit does not fit
       .default_variant = "6bit",
       .fallbacks = {"5bit", "4bit"},
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
       }},
      {.name = "qwen3.8-27b",
       .aliases = {"27b"},
       .title = "Qwen3.8-27B",
       .about = "dense: the strongest coder of the three, slower to write",
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
      {.name = "qwen3.5-4b",
       .aliases = {"4b"},
       .title = "Qwen3.5-4B",
       .about = "small and quick: for 16-24 GB Macs and simple tasks",
       // 6-bit: KL 0.022 from bf16 (4-bit 0.113); 4-bit where 6-bit does not fit
       .default_variant = "6bit",
       .fallbacks = {"4bit"},
       .variants = {
           {.name = "4bit",
            .quant = "MLX affine 4-bit, group 64",
            .repo = "lmstudio-community/Qwen3.5-4B-MLX-4bit",
            .revision = "c43ee1d65576a5d98de1e8405cac93c371a655c1",
            .companions = {mtp4()},
            .download_gb = 3.30},
           {.name = "6bit",
            .quant = "MLX affine 6-bit, group 64",
            .repo = "mlx-community/Qwen3.5-4B-6bit",
            .revision = "fbb1092aa9daef007603ea9273cd24c225a409f1",
            .companions = {mtp4()},
            .download_gb = 4.35},
           {.name = "8bit",
            .quant = "MLX affine 8-bit, group 64",
            .repo = "lmstudio-community/Qwen3.5-4B-MLX-8bit",
            .revision = "59098e5fc57e3b33fd18fd3eb6deffa42b320f25",
            .companions = {mtp4()},
            .download_gb = 5.40},
           {.name = "bf16",
            .quant = "bf16, unquantized",
            .repo = "mlx-community/Qwen3.5-4B-bf16",
            .revision = "491fdc7c087ba7fb48adcb1253f8e76d011db783",
            .companions = {mtp4()},
            .download_gb = 9.35},
       }},
  };
  return c;
}

}  // namespace ni::hub
