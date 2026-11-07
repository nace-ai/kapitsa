from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;cases=json.load(open(root/'long-quality/cases.json'))
rows=[json.loads(l[9:]) for l in (root/'long-results/tf-fast-vs-precise.log').read_text().splitlines() if l.startswith('POSITION ')]
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for sid,ax in enumerate(axes.flat):
 rr=[r for r in rows if r['sequence']==sid];ax.plot([r['position'] for r in rr],[max(r['kl_topk'],1e-8) for r in rr],lw=.7,color='#ae4d12')
 ax.axvline(cases[sid]['prompt_tokens']-1,color='#405a72',ls='--',lw=1,label='Reference answer begins')
 ax.set_yscale('log');ax.set_ylim(1e-8,40);ax.set_title(cases[sid]['id'].replace('test/',''),fontsize=10);ax.set_xlabel('Teacher-forced token position');ax.set_ylabel('KL lower bound (nats)');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('M4 Pro: fast vs precise on all four naturally long MATH-500 prompts\nTop-4096 plus residual bucket; prompt and reference answer scored in prefill',fontsize=12)
fig.savefig(root/'long-results/kl-tail.png',dpi=170)
