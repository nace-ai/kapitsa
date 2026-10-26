#!/usr/bin/env python3
"""Writes tests/data/nanbeige43_tokenizer.json: Hugging Face tokenizers' ids and decode of ~100 texts on Nanbeige 4.3's
tokenizer.json (the Metaspace BPE with byte fallback), which `build/tokenizer_test path/to/tokenizer.json` checks
Kapitsa's encode, decode and StreamDecoder against.

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


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = os.path.join(sys.argv[1], "tokenizer.json")
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(sys.argv[1], "tokenizer_config.json"), encoding="utf-8") as f:
        template = json.load(f)["chat_template"]
    tok = tokenizers.Tokenizer.from_file(path)
    texts = chats(template) + FIXED + fuzz(16)
    assert len(set(texts)) == len(texts)
    cases = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False).ids
        cases.append({"text": t, "ids": ids, "decode": tok.decode(ids, skip_special_tokens=False)})
    head = {"tokenizer_sha256": sha, "tokenizers_version": tokenizers.__version__,
            "generator": "tests/nanbeige43_tokenizer_fixture.py"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "cases": [\n')
        f.write(",\n".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in cases))
        f.write("\n]}\n")
    print(f"{OUT}: {len(cases)} cases, {os.path.getsize(OUT)} bytes")


if __name__ == "__main__":
    main()
