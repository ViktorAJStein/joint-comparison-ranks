"""Plot every protected axis cell/contrast, with explicit uncertainty/timing scopes."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'plot-inputs/axis';IMAGES=ROOT/'reproduced-figures'
a=json.loads((OUT/'results.json').read_text());p=json.loads((ROOT/'experiments/axis_mixture_v1/protocol.json').read_text())
methods=p['methods'];labels=['Rank G\npairs','ME64\nbridge','Fourier64\nbridge','SCF10\ngrid','RFF256\ngrid','RBF\nC2ST']
colors=['#4477aa','#999999','#bbbbbb','#228833','#ee7733','#aa3377']
lookup={(r['case'],r['d'],r['method']):r for r in a['summary']}

def save(fig,name):
    fig.savefig(IMAGES/f'{name}.pdf');fig.savefig(IMAGES/f'{name}.png',dpi=175);plt.close(fig)

fig,axes=plt.subplots(3,3,figsize=(13,11),layout='constrained',sharey=True)
for i,d in enumerate(p['dimensions']):
    for j,case in enumerate(p['cases'][1:]):
        ax=axes[i,j];rows=[lookup[case['id'],d,m] for m in methods];y=np.array([r['rate'] for r in rows]);interval=np.array([r['pointwise_exact_95_interval'] for r in rows])
        ax.bar(np.arange(6),y,color=colors,yerr=np.vstack([y-interval[:,0],interval[:,1]-y]),capsize=2,error_kw={'lw':1})
        ax.axhline(.05,color='black',ls=':',lw=.8);ax.set_ylim(0,1.07);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks(range(6),labels,fontsize=7);s=case['params']['sparse_dims'];delta=min(case['params']['delta'],.98*np.sqrt(s))
        ax.set_title(f'd={d}, s={s}, actual a={delta:.3f} | '+('TU EPYC7302' if d<50 else 'LRZ Gold6148'),fontsize=9)
        if j==0:ax.set_ylabel('Observed rejection frequency')
fig.suptitle('All nine coordinate-aligned mixtures: R=500, pointwise exact95% frequency intervals\nFixed six-method budgets; not universal dominance or power-one claims',fontsize=12)
save(fig,'axis_mixture_complete_power')

fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained',sharey=True)
for ax,d in zip(axes,p['dimensions']):
    rows=[lookup['null_gaussian',d,m] for m in methods];y=np.arange(6)
    for k,r in enumerate(rows):
        color={'resolved':'#228833','inconclusive':'#ee7733','overreject':'#cc3311'}[r['size_status']]
        ax.plot([r['rate'],r['simultaneous_upper']],[k,k],color=color,lw=2)
        ax.scatter(r['rate'],k,color=color,s=35);ax.scatter(r['simultaneous_upper'],k,color=color,marker='|',s=80)
    ax.axvline(.05,c='black',ls='--',lw=1,label='Nominal .05');ax.axvline(.065,c='#cc3311',ls=':',lw=1,label='Protected upper criterion .065')
    ax.set_yticks(y,[s.replace('\n',' ') for s in labels]);ax.set_xlim(.035,.071);ax.set_ylim(5.7,-.7)
    ax.set_xticks([.04,.05,.06,.065]);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=1))
    ax.set_title(f'd={d} | '+('TU EPYC7302' if d<50 else 'LRZ Gold6148'));ax.set_xlabel('Frequency → simultaneous upper bound')
axes[-1].legend(fontsize=7,loc='lower right')
fig.suptitle('All18 Gaussian matched-null screens resolved | R=4000 per dimension\nBonferroni18 simultaneous one-sided upper bounds; empirical screen, not a universal level theorem',fontsize=11)
save(fig,'axis_mixture_complete_null')

fig,axes=plt.subplots(1,3,figsize=(13,7),layout='constrained',sharex=True)
for ax,d in zip(axes,p['dimensions']):
    rows=[r for r in a['paired_contrasts'] if r['d']==d]
    for k,r in enumerate(rows):
        lo,hi=r['paired_pointwise_normal_95_interval'];ax.plot([100*lo,100*hi],[k,k],c=colors[methods.index(r['comparator'])],lw=1.5)
        ax.scatter(100*r['difference'],k,c=colors[methods.index(r['comparator'])],s=23)
    ax.axvline(0,c='black',ls=':',lw=1);ax.set_xlim(-3,102);ax.set_ylim(14.8,-.8)
    ax.set_yticks(range(15),[r['case'].replace('axis_','')+' vs '+labels[methods.index(r['comparator'])].replace('\n',' ') for r in rows],fontsize=7)
    ax.set_xlabel('Rank minus comparator (percentage points)');ax.set_title(f'd={d} | '+('TU EPYC7302' if d<50 else 'LRZ Gold6148'))
fig.suptitle('All45 predeclared paired contrasts: calibrated, Holm45-significant rank gains\nDisplayed intervals are paired pointwise normal95%, NOT multiplicity-adjusted intervals',fontsize=12)
save(fig,'axis_mixture_complete_paired')

runtimes=list(csv.DictReader((OUT/'all_runtime_distributions.csv').open()))
lookup_t={(r['case'],int(r['d']),r['method']):{k:float(v) for k,v in r.items() if k not in ('case','d','method')} for r in runtimes}
fig,axes=plt.subplots(3,4,figsize=(14,10),layout='constrained',sharey=True)
for i,d in enumerate(p['dimensions']):
    for j,case in enumerate(p['cases']):
        ax=axes[i,j];rows=[lookup_t[case['id'],d,m] for m in methods];med=np.array([r['median_total'] for r in rows]);low=np.array([r['q25_total'] for r in rows]);high=np.array([r['q75_total'] for r in rows])
        ax.bar(range(6),med,color=colors,yerr=np.vstack([med-low,high-med]),capsize=2,error_kw={'lw':1})
        ax.set_yscale('log');ax.set_xticks(range(6),labels,fontsize=6);ax.set_title(f'd={d} | '+case['id'].replace('axis_','')+' | '+('TU EPYC7302' if d<50 else 'LRZ Gold6148'),fontsize=8)
        if j==0:ax.set_ylabel('Median total seconds (IQR bars)')
fig.suptitle('All72 timing distributions: newly batched standalone implementation, one numerical thread\nIncludes common data/preprocessing + call/training work; not comparable to older external rank-pipeline timings',fontsize=11)
save(fig,'axis_mixture_complete_runtime')
print('Saved all four complete PDF+PNG evidence figures; no selective cells.')
