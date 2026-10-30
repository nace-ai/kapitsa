#!/usr/bin/env python3
"""Turn what `brew bottle --json` made into the release assets and the formula's `bottle do` block.

    add_bottle.py FORMULA BOTTLE_JSON OUTDIR

Why a bottle: a formula with no `bottle do` block is a source build to Homebrew, which runs its build-environment checks
before `install` and refuses a Mac whose Command Line Tools are stale or missing ("Your Command Line Tools (CLT) does not
support macOS 26"), though Kapitsa compiles nothing. A formula with a bottle is poured, and those checks never run.

The one bottle serves every tag in TAGS: it is a single arm64 binary that links only system frameworks (formula
`depends_on macos: :sonoma`), so the file is the same on each. It is copied to OUTDIR as
`kapitsa-VERSION.arm64_TAG.bottle.tar.gz` (the name Homebrew fetches under the bottle's `root_url`) for the workflow to
upload, and FORMULA gets the block just before `head do`.
"""
import json
import pathlib
import re
import shutil
import sys

TAGS = ("arm64_tahoe", "arm64_sequoia", "arm64_sonoma")


def main(formula_path: str, json_path: str, out_dir: str) -> None:
    formula = pathlib.Path(formula_path)
    text = formula.read_text()
    if re.search(r"^  bottle do$", text, re.M):
        sys.exit(f"{formula} already has a bottle block")
    if "\n  head do\n" not in text:
        sys.exit(f"{formula} has no head block to put the bottle block before")

    # one formula per file; one bottle per formula, built on the runner's own tag
    [entry] = json.loads(pathlib.Path(json_path).read_text()).values()
    bottle = entry["bottle"]
    [built] = bottle["tags"].values()
    sha = built["sha256"]
    if bottle["cellar"] != "any_skip_relocation":
        sys.exit(f"the bottle is relocatable ({bottle['cellar']}); the block below says any_skip_relocation")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        sys.exit(f"bad bottle sha256 {sha!r}")

    source = pathlib.Path(json_path).parent / built["local_filename"]
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for tag in TAGS:
        shutil.copyfile(source, out / re.sub(r"\.arm64_\w+\.bottle", f".{tag}.bottle", built["filename"]))

    lines = ["  bottle do", f'    root_url "{bottle["root_url"]}"']
    lines += [f'    sha256 cellar: :any_skip_relocation, {tag + ":":<14} "{sha}"' for tag in TAGS]
    lines += ["  end", ""]
    formula.write_text(text.replace("\n  head do\n", "\n" + "\n".join(lines) + "\n  head do\n", 1))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
