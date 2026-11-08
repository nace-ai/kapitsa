# Day-0 support playbook

Internal playbook for adding a new Qwen or Nemotron release to Kapitsa within 24 hours.

## 1. Convert the pack

### MLX affine (Qwen, standard path)

If the release ships MLX safetensors (most Qwen releases via `mlx-community` or `lmstudio-community`):

```bash
# Import directly from the HF repo
kapitsa import <hf-org>/<ModelName>-MLX-4bit --name <short-name>

# Or from a local directory / GGUF
kapitsa import /path/to/model-dir --name <short-name>
kapitsa import /path/to/model.gguf --name <short-name>
```

### Flash-Next pack conversion

For a Flash-Next model (hyper-connections, n-gram table, sparse attention):

```bash
python3 tools/convert_qwen4_exp.py \
  --input /path/to/bf16-checkpoint \
  --output /path/to/output-dir \
  --bits 4 --group 64
```

This produces the MLX affine pack with the MTP head in `mtp.safetensors` and the n-gram table in `ple.safetensors`.

### Nemotron (NVFP4 / FP8)

NVIDIA ships NVFP4 and FP8 checkpoints directly. Import the safetensors:

```bash
kapitsa import nvidia/<ModelName>-NVFP4 --name <short-name>
```

## 2. Attach or train the drafter

### DFlash2 drafter (Qwen models)

Check if `incoai` has a DFlash2 drafter for the new model on Hugging Face:

```bash
# Search for existing drafters
curl -s "https://huggingface.co/api/models?search=incoai+DFlash2+<model-name>" | python3 -m json.tool
```

If a drafter exists, add it to `src/hub/catalog.cpp` as a `CatalogCompanion` entry:

```cpp
static CatalogCompanion dflash_new() {
    return {.kind = "dflash2",
            .repo = "incoai/<ModelName>-DFlash2",
            .revision = "<commit-sha>",
            .files = {"config.json", "model.safetensors"},
            .size = <bytes>};
}
```

If no drafter exists yet, the model runs with prompt lookup (`--spec`) until one appears. File an issue to track it.

### Other drafter types

- **MTP** (multi-token prediction): built from the model's own `mtp.*` tensors. Added as a `mtp-sidecar` companion.
- **DSpark**: for Nemotron Lightning. Already in the catalog.
- **LiLiCorr**: for Nemotron Bolt and Super-VL. Already in the catalog.

## 3. Add to the catalog

Edit `src/hub/catalog.cpp`. Follow the existing pattern:

```cpp
{.name = "<short-name>",
 .aliases = {"<alias>"},
 .title = "<Full Model Name>",
 .about = "<one line: what it is and who it's for>",
 .brief = "<shorter version for tables>",
 .default_variant = "4bit",
 .fallbacks = {},
 .variants = {
     {.name = "4bit",
      .quant = "MLX affine 4-bit, group 64",
      .repo = "<hf-org>/<repo>",
      .revision = "<commit-sha>",
      .companions = {dflash_new()},
      .download_gb = <size>},
 }},
```

Build and run the CPU tests:

```bash
cmake --build build -j --target kapitsa
build/api_test
python3 tests/test_server.py --cli build/kapitsa
```

## 4. Run the benchmark harness

```bash
# Start the server with the new model
kapitsa serve --model <short-name> --port 8000 --ctx 32768 --slots 1 --disk-cache 0

# Run the benchmark
kapitsa bench --model <short-name> --json

# Full benchmark (all prompt sizes, multiple runs)
kapitsa bench --model <short-name>

# Quick mode for a first look
kapitsa bench --model <short-name> --quick
```

Record the numbers: prefill tok/s at 4K and 16K, decode tok/s (median of 5 runs), greedy decode tok/s.

## 5. Publish the speed table

Update these files with the new numbers:

- `docs/benchmarks.md`: add a section with the model's results
- `README.md`: add the model to the Models tables if it belongs in the poster
- `bench/results/`: commit the raw benchmark output

Format: `prefill / decode` in tok/s, with the Mac, macOS version, and commit.

```bash
# Generate the community results table
python3 tools/community_table.py
```

## 6. Ship the update

```bash
# One PR with the catalog entry, drafter, and benchmark results
git checkout -b add-<model-name>
git add src/hub/catalog.cpp docs/benchmarks.md bench/results/
git commit -m "Add <Model Name> to the catalog (<decode> tok/s on <Mac>)"
```

## 7. Post on r/LocalLLaMA

Within 24 hours of the model release. Template:

```
Title: Kapitsa runs <Model Name> at <N> tok/s on <Mac> (Apple silicon)

<Model Name> just dropped. Kapitsa has day-0 support:

    brew install nace-ai/tap/kapitsa
    kapitsa serve --model <short-name>

<N> tok/s decode on <Mac chip> (<RAM> GB), <M> tok/s prefill at 4K context.
[Benchmark numbers / comparison table]

Kapitsa is a native inference engine for Apple silicon.
GitHub: https://github.com/nace-ai/kapitsa
```

Include a comparison against other engines if `kapitsa bench compare` data is available.

## Timeline

| Hour | Task |
|------|------|
| 0 | Model drops. Start conversion. |
| 1 | Pack converted, import tested. |
| 2 | Catalog entry added, drafter attached (if available). |
| 3 | Benchmark run on at least one Mac. |
| 4 | PR opened with catalog + benchmarks. |
| 6 | PR merged after review. |
| 8 | `kapitsa update` picks up the new model. |
| 24 | r/LocalLLaMA post with final numbers. |
