// The models `kapitsa serve --model <name>[:<variant>]` knows by name: the one table to edit to add a model, add a
// variant (another precision of the same model) or change a model's default variant. Nothing else needs to change:
// the downloader, `pull`, `list`, `--help` and the folder names (<name>-<variant>) follow this table.
//
// - Models are listed best first: `kapitsa serve` without --model takes the first whose default variant's memory
//   plan (the engine's own, on its headers) leaves room for one --ctx conversation on this Mac.
// - A variant pins its weights to a revision (a commit), so a download is always the same bytes; every file of the
//   repo but .gitattributes is fetched. Its companions are fetched with it: an MTP head ("mtp": files taken from a
//   repo; "mtp-sidecar": built from an official checkpoint's mtp.* tensors, with the result's sha256 and size) and a
//   DFlash2 drafter ("dflash2", in the dflash2/ folder that `--dflash auto` looks in first). A drafter runs only with
//   a 4-bit embedding table, so higher-precision variants carry the MTP head alone.
// - download_gb is shown before a download. Memory needs no entry: the engine plans it from the variant's headers.
// - The repos and revisions are the Nace Agent app's, so the app and the CLI download the same bytes.
#include "hub/hub.h"

namespace ni::hub {

const std::vector<CatalogModel>& catalog() {
  static const std::vector<CatalogModel> c = {
      {.name = "qwen3.6-35b-a3b",
       .aliases = {"qwen3.6-35b", "35b", "35b-a3b"},
       .title = "Qwen3.6-35B-A3B",
       .about = "MoE, 3B active per token: fast replies and fast reading",
       .default_variant = "4bit",
       .variants = {
           {.name = "4bit",
            .quant = "MLX affine 4-bit, group 64",
            // mlx-community/Qwen3.6-35B-A3B-4bit with its MTP head
            .repo = "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
            .revision = "95460516fbfe34c48f577f67c9ed191f5d61fbaf",
            .companions = {{.kind = "dflash2",
                            .repo = "incoai/Qwen3.6-35B-A3B-DFlash2",
                            .revision = "51ef7b6923ad6c14cb1bb41c37a9041446496ab6",
                            .files = {"config.json", "model.safetensors", "LICENSE"},
                            .subdir = "dflash2"}},
            .download_gb = 22.07},
       }},
      {.name = "qwen3.8-27b",
       .aliases = {"27b"},
       .title = "Qwen3.8-27B",
       .about = "dense: the strongest coder of the three, slower to write",
       .default_variant = "4bit",
       .variants = {
           {.name = "4bit",
            .quant = "MLX affine 4-bit, group 64",
            .repo = "lmstudio-community/Qwen3.8-27B-MLX-4bit",
            .revision = "6067b15cf581666a4aecf6af3afaba4bb5efc20c",
            // the pack has no MTP head; MTPLX's 27B pack ships Qwen's (mtplx_runtime.json: it reads the post-norm
            // hidden state)
            .companions = {{.kind = "mtp",
                            .repo = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
                            .revision = "1d5087d2062c02b279180a53e4016cf9cd7a3d7e",
                            .files = {"mtp.safetensors", "mtplx_runtime.json"}},
                           {.kind = "dflash2",
                            .repo = "incoai/Qwen3.8-27B-DFlash2",
                            .revision = "015e795645c74b1a0eeef3b570031fb62e769bc5",
                            .files = {"config.json", "model.safetensors"},
                            .subdir = "dflash2"}},
            .download_gb = 20.17},
       }},
      {.name = "qwen3.5-4b",
       .aliases = {"4b"},
       .title = "Qwen3.5-4B",
       .about = "small and quick: for 16-24 GB Macs and simple tasks",
       .default_variant = "4bit",
       .variants = {
           {.name = "4bit",
            .quant = "MLX affine 4-bit, group 64",
            .repo = "lmstudio-community/Qwen3.5-4B-MLX-4bit",
            .revision = "c43ee1d65576a5d98de1e8405cac93c371a655c1",
            // the MLX pack drops the MTP head: built from the official checkpoint's 15 mtp.* tensors (241 MB by range
            // requests, not the 9.3 GB checkpoint), byte for byte what tools/mtp_sidecar.py makes
            .companions = {{.kind = "mtp-sidecar",
                            .repo = "Qwen/Qwen3.5-4B",
                            .revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                            .sha256 = "4f491dd27975520ec9c9bfa970ca215cc7f12a0328761d3656ab7b9f388e472a",
                            .size = 241200816}},
            .download_gb = 3.30},
       }},
  };
  return c;
}

}  // namespace ni::hub
