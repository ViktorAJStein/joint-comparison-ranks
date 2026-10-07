#!/usr/bin/env python3
"""Illustrate an exact fixed-K population counterexample; not experiment results.

uv run --project joint-comparison-ranks --extra plots python \
  overleaf_submission/theory_verification/plot_rank_scope_counterexample.py
"""
from pathlib import Path
from math import comb
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
k=4
density=lambda u: 1+0.5*Legendre.basis(k+1)(2*u-1)
u=np.linspace(0,1,1001)
probabilities=np.array([quad(lambda u: comb(k,j)*u**j*(1-u)**(k-j)*density(u),0,1,epsabs=1e-12)[0] for j in range(k+1)])
assert np.max(np.abs(probabilities-1/(k+1))) < 1e-11
purple='#800080'
fig,axes=plt.subplots(1,2,figsize=(8.4,2.8),constrained_layout=True)
axes[0].plot(u,np.ones_like(u),color='gray',linestyle='--',label='Uniform reference')
axes[0].plot(u,density(u),color=purple,label='Non-null target')
axes[0].set(xlabel='u',ylabel='Density',title='Different population laws')
axes[0].legend(fontsize=8)
ranks=np.arange(k+1)
axes[1].bar(ranks-0.16,np.full(k+1,1/(k+1)),width=0.32,color='gray',label='Null rank law')
axes[1].bar(ranks+0.16,probabilities,width=0.32,color=purple,label='Target rank law')
axes[1].set(xlabel='Rank cell n',ylabel='Cell probability',title='Identical rank laws at K=4',xticks=ranks,ylim=(0,0.27))
axes[1].legend(fontsize=8)
for axis in axes:
    axis.tick_params(colors=purple)
    axis.xaxis.label.set_color(purple)
    axis.yaxis.label.set_color(purple)
    axis.title.set_color(purple)
np.savetxt(root/'theory_verification/fixed_K_density.csv',np.column_stack((u,np.ones_like(u),density(u))),delimiter=',',header='u,reference_density,target_density',comments='',fmt='%.12g')
np.savetxt(root/'theory_verification/fixed_K_rank_probabilities.csv',np.column_stack((ranks,np.full(k+1,1/(k+1)),probabilities)),delimiter=',',header='rank,null_probability,target_probability',comments='',fmt='%.12g')
for ext in ('pdf','png'):
    output=root/'imgs'/f'theory_fixed_K_counterexample.{ext}'
    metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None
    fig.savefig(output,dpi=180,metadata=metadata)
    print(output)
plt.close(fig)
