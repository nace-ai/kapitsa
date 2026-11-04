#!/usr/bin/env python3
"""REPORT.md -> REPORT.html (headings, bullets, tables, bold, code). usage: md2html.py REPORT.md > REPORT.html"""
import html, re, sys
md = open(sys.argv[1]).read().split('\n'); out = []; tbl = []; inlist = False
def inline(t):
    t = html.escape(t); t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t); t = re.sub(r'`(.+?)`', r'<code>\1</code>', t); return t.replace('\\*', '*')
def flush():
    global tbl
    if not tbl: return
    rows = [r for r in tbl if not re.match(r'^\|[-| ]+\|$', r)]
    o = '<table>'
    for i, r in enumerate(rows):
        c = [x.strip() for x in r.strip('|').split('|')]; t = 'th' if i == 0 else 'td'
        o += '<tr>' + ''.join('<%s>%s</%s>' % (t, inline(x), t) for x in c) + '</tr>'
    out.append(o + '</table>'); tbl = []
for l in md:
    if l.startswith('|'): tbl.append(l); continue
    flush()
    if l.startswith('- '):
        if not inlist: out.append('<ul>'); inlist = True
        out.append('<li>%s</li>' % inline(l[2:])); continue
    if inlist: out.append('</ul>'); inlist = False
    for p, tg in (('### ', 'h3'), ('## ', 'h2'), ('# ', 'h1')):
        if l.startswith(p): out.append('<%s>%s</%s>' % (tg, inline(l[len(p):]), tg)); break
    else:
        if l.strip(): out.append('<p>%s</p>' % inline(l))
flush()
css = 'body{font:15px/1.5 -apple-system,Helvetica,sans-serif;max-width:860px;margin:2em auto;padding:0 1em;color:#1d1d1f}table{border-collapse:collapse;margin:1em 0}td,th{border:1px solid #d2d2d7;padding:4px 10px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#f5f5f7}code{background:#f5f5f7;padding:1px 4px;border-radius:3px}h1{font-size:1.5em}h2{margin-top:1.6em}'
print('<!doctype html><meta charset=utf-8><title>#325 Kapitsa vs MLX, M5 Pro, 3 Nov 2026</title><style>%s</style>%s' % (css, '\n'.join(out)), end='')
