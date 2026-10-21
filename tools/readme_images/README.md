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
  - `{"value": tok/s, "method": "KFlash" | "DFlash2" | "MTP" | "no spec" | ...}`; Kapitsa's cell can add
    `"prefill": tok/s` (a 4K prompt), drawn as a third line;
  - `null`, drawn as a dash: not measured;
  - `"x"`, drawn as ×: can't run on that Mac.
- **`header`:** which Mac and model row the header's bars show, and which engines, in order (sorted by speed).

The README's tables are written by hand in README.md; keep them in sync with the poster. The 16:9 grid keeps its data inline in `reddit_grid.py`.

## Assets

- `assets/logos/`: the engines' logos.
- `assets/geist-mono.json`: the header's Geist Mono letter outlines, with the font's license note.
