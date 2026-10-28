#!/usr/bin/env python3
"""Summarize preserved per-row evidence and document-cluster uncertainty."""
import json
import math
from pathlib import Path
import numpy as np
import argparse


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def summary(rows):
    k=np.array([r['kl'] for r in rows],dtype=float)
    n=np.array([r['nll_change'] for r in rows],dtype=float)
    flips=[r for r in rows if r['top_ref']!=r['top_cand']]
    return {'rows':len(rows),'kl_mean':float(k.mean()),'kl_median':float(np.median(k)),
            'kl_p99':float(np.quantile(k,.99)),'kl_max':float(k.max()),'flips':len(flips),
            'flip_margin_max':max((r['margin'] for r in flips),default=0),
            'nll_change_mean':float(n.mean()),'sample_ppl_change_percent':float(100*np.expm1(n.mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.results
    if args.output.exists():
        parser.error("output already exists; choose a fresh path")
    result={}
    grouped={}
    for model in ('4b','9b','27b'):
        r={}
        rowfiles=sorted((ROOT/'runs'/model).glob('*-comparison/rows.json'))
        if not rowfiles: continue
        for f in rowfiles:
            rows=json.loads(f.read_text())
            name=f.parent.name.replace('-comparison','')
            r[name]={path:summary(v) for path,v in rows.items()}
            if name=='long-1':
                r['long_by_context']={label:summary([x for x in rows['step'] if x['sequence'] in seqs]) for label,seqs in [('4K',{0,1}),('16K',{2,3})]}
            if name=='primary-1':
                grouped[model]=[[x for x in rows['step'] if x['sequence']==s] for s in range(8)]
        paths=[ROOT/'runs'/model/f'primary-{i}-{arm}/logit-sha256.json' for i in (1,2) for arm in ('A','B')]
        if all(f.exists() for f in paths):
            h=[json.loads(f.read_text()) for f in paths]
            r['repeat_bitwise']={'GPU':h[0]==h[2],'ANE':h[1]==h[3]}
        f=ROOT/'runs'/model/'ppl-summary.json'
        if f.exists():
            vals=json.loads(f.read_text())
            tot=vals[-1]
            deltas=np.array([math.log(x['ppl']/x['ppl_ref']) for x in vals[:-1]])
            rng=np.random.default_rng(20261028)
            boot=deltas[rng.integers(0,8,(20000,8))].mean(axis=1)
            r['ppl']={**tot,'relative_change_percent':100*(tot['ppl']/tot['ppl_ref']-1),
                      'delta_nll':math.log(tot['ppl']/tot['ppl_ref']),
                      'delta_nll_document_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
                      'ppl_relative_percent_document_bootstrap_95':(100*np.expm1(np.quantile(boot,[.025,.975]))).tolist()}
        result[model]=r
    comparisons={}
    if len(grouped)==3:
        rng=np.random.default_rng(20261028)
        indices=rng.integers(0,8,(20000,8))
        means={m:np.array([np.mean([r['kl'] for r in rows]) for rows in docs]) for m,docs in grouped.items()}
        for other in ('4b','27b'):
            boot=means['9b'][indices].mean(axis=1)/means[other][indices].mean(axis=1)
            comparisons['9b_over_'+other]={'mean_kl_ratio':float(means['9b'].mean()/means[other].mean()),
                 'paired_document_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
        for m,docs in grouped.items():
            result[m]['primary_by_document']=[summary(rows) for rows in docs]
    save(args.output,{'models':result,'comparisons':comparisons,
        'bootstrap_note':'20,000 paired resamples of eight source documents; limited code-only sample, not token-independent CI; same repeated rows counted once.'})
    print(json.dumps({'models':result,'comparisons':comparisons},indent=2))

if __name__=='__main__': main()
