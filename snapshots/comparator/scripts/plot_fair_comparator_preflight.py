"""Plot pricing/feasibility timing only; never plot pilot rejection rates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True)
ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--images',type=Path,required=True)
a=ap.parse_args()
meta=json.loads((a.archive/'metadata.json').read_text())
result=json.loads((a.analysis/'analysis.json').read_text())
assert result['stage']=='preflight' and result['performance_evidence'] is False
assert meta['stage']=='preflight' and meta['status']=='complete'
rows=[json.loads(s) for s in (a.archive/'raw.jsonl').read_text().splitlines()]
assert all(r['status']=='OK' for r in rows)
a.images.mkdir(parents=True,exist_ok=True)
methods=meta['protocol']['methods'];dims=meta['protocol']['dimensions']
labels=['Ranks (1D)','Ranks (2D)','SCF median','SCF train grid','RFF median','RFF train grid','C2ST logistic','C2ST RBF']
fig,axes=plt.subplots(1,len(dims),figsize=(12,4.7),sharey=True)
for ax,d in zip(np.atleast_1d(axes),dims):
    means=[]
    for name in methods:
        group=[r for r in rows if r['d']==d and r['method']==name]
        means.append([np.mean([r[k] for r in group]) for k in ('common_prep_seconds','training_seconds','evaluation_seconds')])
    means=np.asarray(means);bottom=np.zeros(len(methods))
    for j,label in enumerate(['Common preprocessing','Fit / training grid','Evaluation / calibration']):
        ax.bar(np.arange(len(methods)),means[:,j],bottom=bottom,label=label);bottom+=means[:,j]
    ax.set_xticks(range(len(methods)),labels,rotation=40,ha='right');ax.set_title(f'd={d} — local preflight')
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Mean seconds / call (equal pilot case weights)')
axes[-1].legend(fontsize=8)
fig.suptitle('Feasibility/pricing timings only — not comparative power or cluster prices')
fig.tight_layout()
for suffix in ('pdf','png'):fig.savefig(a.images/f'fair_comparator_preflight_timings.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
forecast=[]
for d in dims:
    group=[r for r in result['pricing'] if r['d']==d]
    forecast.append([sum(r['mean_forecast_core_hours'] for r in group),sum(r['twice_max_forecast_core_hours'] for r in group)])
forecast=np.asarray(forecast);x=np.arange(len(dims))
ax.bar(x-.18,forecast[:,0],.36,label='Mean-time extrapolation')
ax.bar(x+.18,forecast[:,1],.36,label='2 × slowest pilot block (planning allowance)')
ax.set_xticks(x,[f'd={d}' for d in dims]);ax.set_ylabel('CPU-core-hours at measured local speed')
ax.set_title('Frozen full-panel forecast — not approval or a statistical upper bound')
ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
for suffix in ('pdf','png'):fig.savefig(a.images/f'fair_comparator_preflight_forecast.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
print('Saved two PDF+PNG preflight-only figures; no pilot power/size display.')
