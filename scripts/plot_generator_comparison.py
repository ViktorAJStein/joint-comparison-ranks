"""Resource figures now; complete calibrated generator comparison figures later."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
KINDS=('chi2','gtest','freeman_tukey','cr:2')
NAMES=('Pearson','G/KL','Freeman–Tukey','CR λ=2')
COLORS=('tab:blue','tab:orange','tab:green','tab:red')

def save(fig,name):
    out=ROOT/'reproduced-figures';out.mkdir(exist_ok=True)
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

def resource():
    p=json.loads((ROOT/'artifacts/generator_comparison_20261007/pricing.json').read_text())
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.bar(['Mean forecast','2×max planning','All reservations'],[p['mean_forecast_core_hours'],p['twice_max_planning_core_hours'],p['reservation_core_hours']],color=['tab:blue','tab:orange','tab:green'])
    ax.axhline(8,color='black',ls='--',label='Inclusive8-hour ceiling')
    ax.set_ylabel('Allocated CPU-core-hours');ax.set_ylim(0,9);ax.legend();ax.set_title('Generator-study admission: resource estimates, not power evidence')
    save(fig,'generator_comparison_v1_budget')

def results(path):
    a=json.loads(Path(path).read_text())
    if not a['complete_primary']:raise RuntimeError('No complete-primary plot from survivor groups')
    # Null calibration: pointwise intervals; simultaneous upper bounds in analysis CSV.
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,d in zip(axes,(20,50)):
        strata=[('null_gaussian','raw'),('null_gaussian','rotated'),('null_rounded','raw'),('null_ar1','raw')]
        for i,(kind,name,color) in enumerate(zip(KINDS,NAMES,COLORS)):
            rs=[next(r for r in a['summary'] if r['case']==c and r['orientation']==o and r['d']==d and r['generator']==kind) for c,o in strata]
            ys=np.array([r['rate'] for r in rs]);lo=np.array([r['ci95'][0] for r in rs]);hi=np.array([r['ci95'][1] for r in rs])
            for j,r in enumerate(rs):
                ax.errorbar(j+(i-1.5)*.12,ys[j],yerr=[[ys[j]-lo[j]],[hi[j]-ys[j]]],fmt='o',color=color,
                            markerfacecolor=color if r['size_status']=='resolved' else 'white',label=name if j==0 else None,capsize=2)
        ax.axhline(.05,color='black',ls='--');ax.set_xticks(range(4),['Gaussian','Rotated','Rounded','AR1']);ax.set_title(f'd={d},R=4000; hollow = unresolved screen',fontsize=10);ax.set_ylim(0,.09)
    axes[0].set_ylabel('Null rejection frequency (pointwise95% intervals)');axes[1].legend(handles=[Line2D([],[],color=color,marker='o',linestyle='None',label=name) for kind,name,color in zip(KINDS,NAMES,COLORS)],fontsize=8)
    save(fig,'generator_comparison_v1_nulls')
    # All methods shown; no selected winners. Equal N,K,views and calibration.
    fig,axes=plt.subplots(2,4,figsize=(13,6),sharey=True)
    alternatives=[('alt_locshift','raw','AR1 shift'),('alt_sparse_scale','raw','Sparse scale'),('alt_sparse_scale','rotated','Rotated scale'),('alt_gmm_matched','raw','Matched GMM')]
    for j,d in enumerate((20,50)):
        for k,(case,orientation,title) in enumerate(alternatives):
            ax=axes[j,k]
            for i,(kind,name,color) in enumerate(zip(KINDS,NAMES,COLORS)):
                r=next(r for r in a['summary'] if r['case']==case and r['d']==d and r['orientation']==orientation and r['generator']==kind)
                ax.errorbar(i,r['rate'],yerr=[[r['rate']-r['ci95'][0]],[r['ci95'][1]-r['rate']]],fmt='o',color=color,capsize=3)
            ax.set_xticks(range(4),['Pearson','G','FT','CR2'],rotation=30);ax.set_ylim(0,1);ax.set_title(f'{title},d={d}',fontsize=9)
    axes[0,0].set_ylabel('Rejection frequency');axes[1,0].set_ylabel('Rejection frequency');fig.suptitle('Matched generator comparison,R=500; pointwise95% intervals')
    fig.tight_layout();save(fig,'generator_comparison_v1_power')
    fig,axes=plt.subplots(2,4,figsize=(13,6),sharey=True)
    for j,d in enumerate((20,50)):
        for k,(case,orientation,title) in enumerate(alternatives):
            ax=axes[j,k]
            for kind,name,color in zip(KINDS,NAMES,COLORS):
                r=next(r for r in a['summary'] if r['case']==case and r['d']==d and r['orientation']==orientation and r['generator']==kind)
                ax.scatter(1000*r['standalone_mean_seconds'],r['rate'],color=color,label=name)
            ax.set_title(f'{title},d={d}',fontsize=9);ax.set_ylim(0,1);ax.set_xlabel('Test mean ms (32 fixed blocks)',fontsize=8)
    axes[0,0].legend(fontsize=7,loc='upper left')
    axes[0,0].set_ylabel('Rejection frequency');axes[1,0].set_ylabel('Rejection frequency');fig.suptitle('All generator test-only runtime/rejection-frequency points (descriptive)')
    fig.tight_layout();save(fig,'generator_comparison_v1_power_runtime')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path);a=ap.parse_args()
    if a.analysis:results(a.analysis)
    else:resource()
