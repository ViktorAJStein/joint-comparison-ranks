#!/usr/bin/env python3
"""Exact retained-signal ratios; analytic illustration, not empirical power.

uv run --project joint-comparison-ranks --extra plots python \
  overleaf_submission/theory_verification/plot_subspace_interaction_signal.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
r=np.arange(2,13)
resolutions=(1,2,4,8)
ratios=np.column_stack([(k/(k+2))**r for k in resolutions])
np.savetxt(root/'theory_verification/subspace_interaction_retained_signal.csv',np.column_stack((r,ratios)),delimiter=',',header='interaction_order,K1,K2,K4,K8',comments='',fmt='%.12g')
fig,ax=plt.subplots(figsize=(5.6,2.9),constrained_layout=True)
for i,k in enumerate(resolutions):
    ax.semilogy(r,ratios[:,i],marker='o',markersize=3,label=f'K={k}')
ax.set(xlabel='Active interaction order r',ylabel='Retained Pearson divergence\n(rank / continuum)',xticks=r[::2])
ax.legend(ncol=2,fontsize=8)
ax.grid(alpha=0.2)
purple='#800080'
ax.tick_params(colors=purple)
ax.xaxis.label.set_color(purple);ax.yaxis.label.set_color(purple)
for ext in ('pdf','png'):
    output=root/'imgs'/f'theory_subspace_interaction_signal.{ext}'
    metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None
    fig.savefig(output,dpi=180,metadata=metadata)
    print(output)
plt.close(fig)
