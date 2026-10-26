#!/usr/bin/env python3
"""Writes tests/data/nanbeige43_tokenizer.json: Hugging Face tokenizers' ids and decode of ~100 texts on Nanbeige 4.3's
tokenizer.json (the Metaspace BPE with byte fallback), and HF's decode of ~60 id lists (byte tokens that are, and are
not, well-formed UTF-8; a reply cut in the middle of a character), which tokenizer_test checks Kapitsa's encode,
decode and StreamDecoder against. Also writes tests/data/nanbeige43_tokenizer_small.json, the same file cut to the
tokens and merges these texts use, which `build/tokenizer_test` checks with no arguments (CI): the 256 byte tokens,
every token on the way to a text's tokens (all merges giving it, down to characters) and the merges among them, with
the real file's ids and the rest of its config; the added tokens are in the vocabulary too, at their ids (HF numbers
an added token outside it from the vocabulary's size). Both files give HF the same ids and decodes (asserted).

usage: uv run --no-project --with tokenizers==0.23.3 --with jinja2 python tests/nanbeige43_tokenizer_fixture.py DIR
  DIR: the model folder (its tokenizer.json, and tokenizer_config.json's chat template, rendered with jinja2 as
  transformers renders it). Ids are encode(text, add_special_tokens=False), the decode is
  decode(ids, skip_special_tokens=False).
"""
import hashlib
import json
import os
import random
import sys

import jinja2
import jinja2.sandbox
import tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "nanbeige43_tokenizer.json")
SMALL = os.path.join(HERE, "data", "nanbeige43_tokenizer_small.json")


def render(template, messages, **kw):
    # transformers' chat template environment (trim_blocks, lstrip_blocks, its tojson and raise_exception)
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, **kw)


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, e.g. 北京"},
        "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]


def chats(template):
    user = [{"role": "user", "content": "What's the weather in Beijing? 北京天气怎么样？"}]
    call = [{"role": "assistant", "content": "", "reasoning_content": "The user wants the weather.",
             "tool_calls": [{"type": "function", "function": {
                 "name": "get_weather", "arguments": {"city": "Beijing", "unit": "c"}}}]},
            {"role": "tool", "content": '{"temp": 21, "sky": "clear ☀️"}'}]
    return [
        render(template, user, add_generation_prompt=True),
        render(template, user, add_generation_prompt=True, enable_thinking=False),
        render(template, [{"role": "system", "content": "  You are terse.  "}] + user + [
            {"role": "assistant", "content": "<think>\nshort\n</think>\n\nSunny, 21 °C."},
            {"role": "user", "content": " And tomorrow?"}], add_generation_prompt=True),
        render(template, user, tools=TOOLS, add_generation_prompt=True),
        render(template, user + call, tools=TOOLS, add_generation_prompt=True),
        render(template, user + call + [{"role": "assistant", "content": "It is 21 °C and clear."}], tools=TOOLS,
               tool_call_format="json"),
    ]


FIXED = [
    # spaces
    "", " ", "  ", "   ", "\t", "\n", "\r\n", " \n ", "hello", " hello", "  hello", "hello ", "hello  ",
    "a  b   c    d", "  leading and trailing  ", "\u3000full-width space\u3000", "nbsp\u00a0here",
    "\u2581", "\u2581\u2581x", "a\u2581b", " \u2581 ",
    # English and punctuation
    "The quick brown fox jumps over the lazy dog.", "It's 3:45 p.m.; don't panic!", "e-mail: a.b+c@example.com",
    "Numbers 0 1 22 333 4444 55555 3.14159 -2e10 1,000,000", "\"quoted\" 'single' (paren) [brack] {brace}",
    "UPPER lower MiXeD camelCase snake_case kebab-case",
    # CJK
    "南北阁是一款大语言模型。", "你好，世界！今天天气很好。", "中文English混合Mixed文本123",
    "東京タワーは333メートルです。", "ひらがなとカタカナ", "한국어 텍스트입니다.", "《三体》——刘慈欣",
    "繁體中文與簡體中文", "１２３ＡＢＣ全角", "。，、；：？！",
    # emoji, ZWJ, variation selectors, flags, skin tones
    "👍", "👨\u200d👩\u200d👧\u200d👦 family", "🏳\ufe0f\u200d🌈", "🇨🇳🇺🇸🇯🇵", "👋🏽 hi 👋🏿", "✅\ufe0f done ✔",
    "I ❤️ 🍜 and 🐼!", "🧑\u200d💻\u200d", "😀😃😄😁😆😅🤣😂",
    # supplementary planes
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙 𝑥² + 𝑦²", "𠀀𠮷𪚥𫝀 CJK Ext B", "𐌰𐌱𐌲 Gothic", "𓀀𓁐 hieroglyphs", "\U0010fffd\U000f0000 private",
    # byte fallback and rare characters
    "Ꭰ Cherokee ᏣᎳᎩ", "ꙮ multiocular", "\u0000nul\u0001\u007f", "e\u0301 combining é", "Z\u0351\u0352\u0353algo",
    "\ufffd replacement", "\ue000 private use", "ǅ title-case digraph", "ﬁ ligature", "\u202eRTL override\u202c",
    "עברית العربية हिन्दी ไทย", "Ελληνικά Кириллица",
    # code with tabs
    "def f(x):\n\treturn x * 2\n", "int main() {\n\tprintf(\"%d\\n\", 42);\n\treturn 0;\n}\n",
    "\t\tdouble\ttab\t\t", "if (a && b) { c = d >> 2; }  // comment", '{"key": [1, 2.5, null, true], "k2": "v"}',
    "    four spaces\n        eight spaces\n", "<div class=\"x\">&amp;</div>", "SELECT * FROM t WHERE id = 1;",
    # added tokens and the text after them
    "<think>", "<think>hello", "<think> hello", "<think>\nreasoning\n</think>\n\nanswer",
    "</think>  two spaces", "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
    "<|im_start|>user\nhi<|im_end|>\n", "<|im_start|> space", "text<|im_end|> after end", "<|endoftext|>x",
    "<s> bos</s>eos", "<unk>", "hi<think>hi hi", "a<think>\u2581b", "<|im_end|><|im_start|>", "<think",
]


def fuzz(n, seed=4343):
    pieces = ["the", " ", "  ", "\t", "\n", "中文", "🙂", "\u200d", "é", "\u2581", "<think>", "</think>",
              "<|im_end|>", "x", "42", "𝑥", "ꙮ", "Ꭰ", ",", ".", "'s", "\u00a0"]
    rng = random.Random(seed)
    return ["".join(rng.choice(pieces) for _ in range(rng.randint(1, 12))) for _ in range(n)]


def id_lists(tok, cases, seed=603):
    # HF's ByteFallback decodes a run of adjacent byte tokens whole: its text if all of it is well-formed UTF-8, else
    # one U+FFFD per byte (a valid character in the run included)
    b = lambda *xs: [tok.token_to_id(f"<0x{x:02X}>") for x in xs]  # noqa: E731
    w = lambda *ts: [tok.token_to_id(t) for t in ts]  # noqa: E731
    lists = [b(0xE4, 0xB8), b(0xE4, 0xB8, 0x80), b(0xE4, 0xB8, 0x80, 0xE4), b(0x61, 0xE4), b(0x20) + w("hello"),
             w("hello") + b(0xE4, 0xB8) + w("\u2581hello"), b(0xE4) + w("<think>") + b(0xB8, 0x80),
             b(0xE2, 0x96, 0x81), b(0xED, 0xA0, 0x80), b(0xC0, 0x80), b(0xF4, 0x90, 0x80, 0x80), b(0x80),
             w("<think>") + b(0x20, 0x20) + w("hello")]
    # a reply cut by max_tokens in the middle of a character given as byte tokens
    cont = set(b(*range(0x80, 0xC0)))  # a UTF-8 continuation byte: the ids before it end within a character
    for c in cases:
        ids = c["ids"]
        for cut in [k for k in range(len(ids)) if ids[k] in cont][:2]:  # after one byte, after two or a token
            if ids[:cut] not in lists and len(lists) < 21:
                lists.append(ids[:cut])
    rng = random.Random(seed)
    pool = sorted({i for c in cases for i in c["ids"]})
    for _ in range(48):
        lists.append([rng.choice(b(rng.choice([rng.randrange(256), rng.randrange(0x80, 0x100)])))
                      if rng.random() < 0.6 else rng.choice(pool) for _ in range(rng.randint(1, 10))])
    assert all(None not in ids for ids in lists)
    return lists


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = os.path.join(sys.argv[1], "tokenizer.json")
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(sys.argv[1], "tokenizer_config.json"), encoding="utf-8") as f:
        template = json.load(f)["chat_template"]
    with open(path, encoding="utf-8") as f:
        full = json.load(f)
    tok = tokenizers.Tokenizer.from_file(path)
    texts = chats(template) + FIXED + fuzz(16)
    assert len(set(texts)) == len(texts)
    dec = lambda t, ids: t.decode(ids, skip_special_tokens=False)  # noqa: E731
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in texts]
    for c in cases:
        c["decode"] = dec(tok, c["ids"])
    id_cases = [{"ids": ids, "decode": dec(tok, ids)} for ids in id_lists(tok, cases)]

    vocab, model = full["model"]["vocab"], full["model"]
    token = {i: t for t, i in vocab.items()}
    makes = {}  # token -> the merges giving it
    for a, b in model["merges"]:
        makes.setdefault(a + b, []).append((a, b))
    keep = {f"<0x{b:02X}>" for b in range(256)}
    todo = [token[i] for c in cases + id_cases for i in c["ids"] if i in token]
    while todo:
        t = todo.pop()
        if t not in keep:
            keep.add(t)
            todo += [p for m in makes.get(t, []) for p in m]
    small = dict(full, model=dict(model, vocab={t: i for t, i in vocab.items() if t in keep},
                                  merges=[[a, b] for a, b in model["merges"] if a in keep and b in keep and a + b in keep]))
    small["model"]["vocab"].update({a["content"]: a["id"] for a in full["added_tokens"]})
    with open(SMALL, "w", encoding="utf-8") as f:
        f.write(json.dumps(small, ensure_ascii=False, separators=(",", ":")) + "\n")
    st = tokenizers.Tokenizer.from_file(SMALL)
    for c in cases:
        assert st.encode(c["text"], add_special_tokens=False).ids == c["ids"], c["text"]
    for c in cases + id_cases:
        assert dec(st, c["ids"]) == c["decode"], c["ids"]
    with open(SMALL, "rb") as f:
        small_sha = hashlib.sha256(f.read()).hexdigest()

    head = {"tokenizer_sha256": sha, "small_sha256": small_sha, "tokenizers_version": tokenizers.__version__,
            "generator": "tests/nanbeige43_tokenizer_fixture.py"}
    line = lambda c: json.dumps(c, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "cases": [\n')
        f.write(",\n".join(map(line, cases)) + '\n], "id_cases": [\n')
        f.write(",\n".join(map(line, id_cases)) + "\n]}\n")
    print(f"{OUT}: {len(cases)} texts, {len(id_cases)} id lists, {os.path.getsize(OUT)} bytes")
    print(f"{SMALL}: {len(small['model']['vocab'])} tokens, {len(small['model']['merges'])} merges, "
          f"{os.path.getsize(SMALL)} bytes")


if __name__ == "__main__":
    main()
