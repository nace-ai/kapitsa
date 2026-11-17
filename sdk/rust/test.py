#!/usr/bin/env python3
"""Rust binding CPU checks against the shared ABI double and the real library."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library_dir", type=Path, help="directory containing libkapitsa_sdk.dylib")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    library_dir = args.library_dir.resolve()
    if not (library_dir / "libkapitsa_sdk.dylib").is_file():
        parser.error("build the C SDK shared library first")
    cargo = ["cargo", "--manifest-path", str(here / "Cargo.toml")]

    def run(subcommand, options, directory):
        env = dict(os.environ, KAPITSA_LIB_DIR=str(directory), DYLD_LIBRARY_PATH=str(directory))
        subprocess.run([cargo[0], subcommand, *cargo[1:], *options], env=env, check=True)

    with tempfile.TemporaryDirectory(prefix="kapitsa-rust-fixture-") as tmp:
        fixture = Path(tmp)
        subprocess.run(["c++", "-std=c++20", "-shared", "-fPIC", "-I", str(root / "sdk/c"),
                        "-I", str(root / "third_party"), str(root / "sdk/node/test/fixture.cpp"),
                        "-o", str(fixture / "libkapitsa_sdk.dylib")], check=True)
        run("fmt", ["--check"], fixture)
        run("clippy", ["--all-targets", "--features", "fixture", "--", "-D", "warnings"], fixture)
        run("test", ["--features", "fixture"], fixture)
    run("test", [], library_dir)


if __name__ == "__main__":
    main()
