# Contributing

Thanks for helping make Kapitsa faster. A few rules keep it fast and correct.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python3 tests/test_server.py --cli build/kapitsa           # the command line, offline
python3 tests/test_sidecar.py build/kapitsa                 # the MTP sidecar builder against tools/mtp_sidecar.py
build/kapitsa serve --model 4b --port 8090 &                # then, against a running server:
python3 tests/test_server.py http://127.0.0.1:8090
```

[DEVELOPMENT.md](DEVELOPMENT.md) explains the layout, the switches and the tools.

## Speed changes

- Measure on the GPU only when nothing else runs on it, and interleave the runs of the old and the new build (A B A B
  ...). Report every run, not the best one, with the Mac, the macOS version and the commit.
- A default changes only after an interleaved A/B and a check that the output stays the same: greedy tokens and
  logits against the old path, with differences allowed only at near-ties.
- Put the numbers in [docs/PERF.md](docs/PERF.md) with the change that produced them.

## Pull requests

- One change per pull request, with its measurements when it touches speed or memory.
- Keep new code in the style of the code around it. No new dependencies without a strong reason: Kapitsa builds with
  the Xcode Command Line Tools and CMake alone.
- By contributing you agree that your contribution is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

Report security issues privately: see [SECURITY.md](SECURITY.md).
