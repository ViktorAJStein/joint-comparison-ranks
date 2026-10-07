"""Plots only complete audited decision results; exports underlying CSVs."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('analysis',type=Path)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    result=json.loads(args.analysis.read_text())
    if result['status']!='complete' or result['blocks']!=46000 or result['method_results']!=144000:
        raise RuntimeError('No plots of incomplete/survivor evidence')
    args.out.mkdir(parents=True,exist_ok=True)
    rows=result['rates']
    with (args.out/'dependence_rates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['study','d','n','law','rho','null','method','repetitions','rejections','rate','ci_low','ci_high','null_upper','null_resolved','component_seconds'])
        writer.writeheader()
        for r in rows:
            s=r['stratum'];writer.writerow(dict(study=s['study'],d=s['d'],n=s['n'],law=s['law'],rho=s['rho'],null=s['null'],method=r['method'],repetitions=r['repetitions'],rejections=r['rejections'],rate=r['rate'],ci_low=r['interval95'][0],ci_high=r['interval95'][1],null_upper=r.get('simultaneous_null_upper'),null_resolved=r.get('null_resolved'),component_seconds=r['mean_call_seconds']))
    def save(fig,name):
        fig.tight_layout()
        for extension in ('pdf','png'):fig.savefig(args.out/f'{name}.{extension}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    for ax,rho in zip(axes,(.2,.4)):
        for method in result['protocol']['gaussian']['methods']:
            group=sorted([r for r in rows if r['stratum']['study']=='gaussian' and not r['stratum']['null'] and r['stratum']['rho']==rho and r['method']==method],key=lambda r:r['stratum']['n'])
            x=[r['stratum']['n'] for r in group];y=np.array([r['rate'] for r in group]);ci=np.array([r['interval95'] for r in group])
            ax.errorbar(x,y,yerr=np.vstack([y-ci[:,0],ci[:,1]-y]),marker='o',label=method)
        ax.axhline(.05,color='gray',ls=':');ax.set_xscale('log',base=2)
        ax.set_xlabel('N = M');ax.set_title(f'Gaussian copula rho={rho}; d=2, K=2')
        ax.set_ylim(0,1.04)
    axes[0].set_ylabel('Rejection frequency (pointwise exact 95% CI)');axes[1].legend(fontsize=7)
    save(fig,'gaussian_copula_empirical_power')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    methods=result['protocol']['copula']['methods']
    for ax,d in zip(axes,(4,20)):
        for j,method in enumerate(methods):
            group=[next(r for r in rows if r['stratum']['study']=='copula' and r['stratum']['d']==d and not r['stratum']['null'] and r['stratum']['law']==law and r['method']==method) for law in ('clayton','gumbel')]
            y=np.array([r['rate'] for r in group]);ci=np.array([r['interval95'] for r in group]);x=np.arange(2)+(j-1.5)*.16
            ax.errorbar(x,y,yerr=np.vstack([y-ci[:,0],ci[:,1]-y]),fmt='o',label=method)
        ax.set_xticks([0,1],['Clayton','Gumbel']);ax.set_title(f'Matched-marginal alternatives d={d}, N=M=1000')
        ax.set_ylim(0,1.04);ax.axhline(.05,color='gray',ls=':')
    axes[0].set_ylabel('Rejection frequency (pointwise exact 95% CI)');axes[1].legend(fontsize=7)
    save(fig,'copula_corrected_calibrated_power')
    nulls=[r for r in rows if r['stratum']['null']]
    fig,ax=plt.subplots(figsize=(12,4.5));x=np.arange(len(nulls));y=np.array([r['rate'] for r in nulls]);ci=np.array([r['interval95'] for r in nulls])
    colors=['C0' if r['null_resolved'] else 'C3' for r in nulls]
    ax.errorbar(x,y,yerr=np.vstack([y-ci[:,0],ci[:,1]-y]),fmt='none',color='gray')
    ax.scatter(x,y,c=colors,label='Blue: size resolved; red: unresolved')
    ax.scatter(x,[r['simultaneous_null_upper'] for r in nulls],marker='_',color='black',label='Simultaneous one-sided upper bound')
    ax.axhline(.05,color='gray',ls=':');ax.axhline(.065,color='black',ls='--')
    ax.set_xticks(x,[f"{r['stratum']['study']} d{r['stratum']['d']} N{r['stratum']['n']}\n{r['stratum']['law']} {r['method']}" for r in nulls],rotation=90,fontsize=6)
    ax.set_ylabel('Null rejection frequency');ax.legend(fontsize=8)
    save(fig,'dependence_null_calibration')
    # Timings are components with shared allocation cost charged, not an old/new speed claim.
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,d in zip(axes,(4,20)):
        for law,marker in [('clayton','o'),('gumbel','s')]:
            group=[next(r for r in rows if r['stratum']['study']=='copula' and r['stratum']['d']==d and not r['stratum']['null'] and r['stratum']['law']==law and r['method']==method) for method in methods]
            ax.plot(np.arange(4),[r['mean_call_seconds'] for r in group],marker=marker,label=law)
        ax.set_xticks(range(4),methods,rotation=25,ha='right',fontsize=7);ax.set_yscale('log');ax.set_title(f'd={d}');ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean component seconds + shared allocation charge')
    save(fig,'copula_runtime_components')
    print(json.dumps(dict(status='pass',figure_pairs=4,rate_rows=len(rows))))


if __name__=='__main__':main()
