# README images for Kapitsa

Renders the README's benchmark poster (`docs/assets/benchmarks-{light,dark}.webp`) and header
(`docs/assets/header-{light,dark,narrow}.svg`) from one data file, plus a 16:9 benchmark grid for posts.

## Needs

- macOS, for the system fonts New York, SF Pro and SF Mono in `/System/Library/Fonts`
- Python 3 (standard library only)
- Google Chrome, run headless on the CPU (`CHROME=...` if it's not in /Applications)
- `cwebp`: `brew install webp`

## Render

```bash
python3 tools/readme_images/readme_images.py             # all five files, into docs/assets
python3 tools/readme_images/readme_images.py --poster    # only the poster (webp)
python3 tools/readme_images/readme_images.py --headers   # only the header (svg, no Chrome needed)
python3 tools/readme_images/reddit_grid.py ~/Downloads   # the 16:9 grid (1920x1080 and 2x)
```

It prints the README alt texts for the poster and header; paste them into README.md if they changed.

## Edit the numbers

Everything is in `readme_images.json`:

- **`engines`:** column order (Kapitsa first), logo, tile color, and the version shown under the name. A `\n`
  splits a version onto two lines.
- **`poster.macs[].rows[].cells`:** one cell per engine. It is one of:
  - `{"value": tok/s, "method": "KFlash" | "DFlash2" | "MTP" | "no spec" | ...}` (`"no spec"`, plain decoding, draws no method label, on every engine); a cell can add
    `"prefill": tok/s` (a 4K prompt, the same engine's stored run); when any cell has it, every cell reads `prefill / decode` in one size
    (Kapitsa's column is wider), and a cell without it shows a dash before the slash;
  - `null`, drawn as a dash: not measured;
  - `"x"`, drawn as ×: can't run on that Mac.
- **`header`:** which Mac and model row the header's bars show, and which engines, in order (sorted by speed).

`readme_images.json` is the README's only source of speed numbers. `python3 tools/readme_images/readme_images.py --tables` writes the README's per-Mac tables under Models from it: each Mac's poster rows plus its `tables.macs[].rows` (the models the poster leaves out), named by `tables.models`; `--tables --check` fails if README.md differs. A cell's `"detail"` (a weight format or engine build) shows in the tables only. The 16:9 grid keeps its data inline in `reddit_grid.py`.

## The Watch → Custom → Releases GIF

`watch_releases.html?N` draws step N (0 to 3) of the walkthrough; `watch_releases.txt` is the ffmpeg concat list (frames `f0.png` to `f3.png`, from headless Chrome at 880x480).
`ffmpeg -f concat -i watch_releases.txt -vf "scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=none" docs/assets/watch-releases.gif` gives the 47 KB file.

## Assets

- `assets/logos/`: the engines' logos.
- `assets/geist-mono.json`: the header's Geist Mono letter outlines, with the font's license note.
