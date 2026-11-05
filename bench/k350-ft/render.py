#!/usr/bin/env python3
"""Renders each answered conversation (items.jsonl + results.jsonl) through the 35B's chat_template.jinja (thinking off)
and tokenizes it: <out>/ids/<id>.ids (prompt + answer, at most 2047 tokens) and <out>/meta.jsonl {id, n_prompt, n}.
Writes <out>/capt.list for k350_capt (mode c). Drops a conversation whose prompt tokens are not a prefix of the whole."""
import json, os, sys
import jinja2
from tokenizers import Tokenizer

M = os.path.expanduser('~/Documents/nace/models/qwen3.6-35b-a3b-4bit')
items_f, res_f, out = sys.argv[1], sys.argv[2], sys.argv[3]
feat = sys.argv[4]  # where k350_capt writes the c rows
os.makedirs(out + '/ids', exist_ok=True)
tk = Tokenizer.from_file(M + '/tokenizer.json')
env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
def raise_exception(m): raise ValueError(m)
env.globals['raise_exception'] = raise_exception
env.filters['tojson'] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
tpl = env.from_string(open(M + '/chat_template.jinja').read())
items = {}
for l in open(items_f):
    it = json.loads(l); items[it['id']] = it
bad = kept = 0
meta = open(out + '/meta.jsonl', 'w')
lst = open(out + '/capt.real.list', 'w')
open(out + '/capt.list', 'w').close()  # job 1 then captures nothing and frees the lock; capture runs as its own job
for l in open(res_f):
    r = json.loads(l)
    it = items[r['id']]
    msg = r['message']
    a = {'role': 'assistant', 'content': msg.get('content') or ''}
    if msg.get('tool_calls'):
        a['tool_calls'] = [{'type': 'function', 'function': {'name': c['function']['name'],
                            'arguments': json.loads(c['function']['arguments']) if isinstance(c['function']['arguments'], str) else c['function']['arguments']}}
                           for c in msg['tool_calls']]
    kw = {'tools': it['tools']} if it.get('tools') else {}
    p = tpl.render(messages=it['messages'], add_generation_prompt=True, enable_thinking=False, **kw)
    full = tpl.render(messages=it['messages'] + [a], add_generation_prompt=False, enable_thinking=False, **kw)
    if not full.startswith(p):
        bad += 1; continue
    pid = tk.encode(p, add_special_tokens=False).ids
    fid = tk.encode(full.rstrip('\n'), add_special_tokens=False).ids
    if fid[:len(pid)] != pid or len(fid) - len(pid) < 16 or len(pid) > 1900:
        bad += 1; continue
    fid = fid[:2047]
    open(f"{out}/ids/{r['id']}.ids", 'w').write(','.join(map(str, fid)))
    meta.write(json.dumps({'id': r['id'], 'n_prompt': len(pid), 'n': len(fid), 'style': it['style']}) + '\n')
    lst.write(f"c {out}/ids/{r['id']}.ids {feat}/{r['id']}.c\n")
    kept += 1
print('render: kept', kept, 'dropped', bad)
