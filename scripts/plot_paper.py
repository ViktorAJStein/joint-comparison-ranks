"""Rebuild all retained empirical PDF/PNG plots from frozen archived inputs."""
import argparse
from pathlib import Path
import runpy
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'reproduced-figures');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    (ROOT/'reproduced-figures').mkdir(exist_ok=True)
    sys.path.insert(0,str(ROOT/'scripts'))
    oldsave=Figure.savefig;oldtext=Text.set_text;oldticks=Axes.set_xticks
    def save(self,path,*a,**kw):return oldsave(self,args.out/Path(path).name,*a,**kw)
    def text(self,value):
        if isinstance(value,str):
            value=value.replace('TU EPYC7302','AMD EPYC7302').replace('LRZ Gold6148','Intel Gold6148')
            value=value.replace('ME64\nbridge','ME64').replace('Fourier64\nbridge','Fourier64')
            value=value.replace('newly batched standalone','batched standalone').replace('older external rank-pipeline timings','other backend/cohort timing scopes')
        return oldtext(self,value)
    def ticks(self,positions,labels=None,*a,**kw):
        if labels is not None:labels=[str(v).replace('Legacy-style combined','Combined switches') for v in labels]
        return oldticks(self,positions,labels,*a,**kw)
    Figure.savefig=save;Text.set_text=text;Axes.set_xticks=ticks
    jobs=[
        ('scripts/plot_calibration_audit.py',['--analysis',str(ROOT/'plot-inputs/calibration'),'--output',str(args.out)]),
        ('scripts/plot_comparator_main_results.py',['--analysis',str(ROOT/'plot-inputs/comparator/d50.json')]),
        ('scripts/plot_comparator_d20_descriptive.py',[]),
        ('scripts/plot_generator_comparison.py',['--analysis',str(ROOT/'plot-inputs/generator/analysis.json')]),
        ('scripts/plot_axis_complete_evidence.py',[]),
        ('experiments/dependence_power_v1/plot.py',[str(ROOT/'plot-inputs/dependence/results.json'),'--out',str(args.out)]),
        ('theory_verification/plot_rank_scope_counterexample.py',[]),
        ('theory_verification/plot_subspace_interaction_signal.py',[])]
    for name,arguments in jobs:
        sys.argv=[name,*arguments];runpy.run_path(str(ROOT/name),run_name='__main__')
    print('Plots regenerated. Gaussian-copula analytic illustration is supplied as a rendered reference asset; its original plotting source is unavailable.')

if __name__=='__main__':main()
