# CPU tokenizer comparison

This measures complete `Tokenizer::encode` calls, including normalization and pre-tokenization.
It does not load neural weights or measure inference. Each input case is a JSON object with
`name`, `text`, and positive `iterations`; the input file is an array of these cases.

Build the same driver against each source version (from each checkout):

```sh
c++ -std=c++20 -O3 -DNDEBUG -Isrc -Ithird_party \
  bench/tokenizer/encode.cpp src/tokenizer/tokenizer.cpp \
  -framework CoreFoundation -o "$ENCODE_DRIVER"
```

Keep the driver and all other sources identical between arms; change only the tokenizer implementation.
Then run:

```sh
python3 bench/tokenizer/make_cases.py > "$CASES"
python3 bench/tokenizer/compare.py --reference "$REF_DRIVER" --candidate "$CAND_DRIVER" \
  --tokenizer "$TOKENIZER" --cases "$CASES" --output "$NEW_RESULTS"
```

The order is A1/B1/A2/B2, with a fresh process and tokenizer cache per arm. Loading the tokenizer JSON is
outside the timer. `first_ms` is the first encoding of each case in that process, with any cache entries
from earlier cases retained. `warm_ms` averages subsequent encodings **and their complete-array equality
checks**. The harness also compares every case's complete token array across all four arms and verifies
that the input files and binaries did not change. Keep the raw JSONL files; the summary omits token arrays.

Use both ordinary prose/code and long pre-tokenized pieces. Random letters and unpunctuated CJK are stress
inputs, not a claim about typical chat latency. Repeated runs of a single character exercise the cheap
linear-pass path. Short pieces retain that path; a piece still longer than 64 symbols after eight passes
uses ranked local edge updates. The heap needs additional temporary O(n) storage for a long piece.
