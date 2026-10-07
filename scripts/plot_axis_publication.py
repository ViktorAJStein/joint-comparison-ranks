"""Publication-only label/anonymity wrapper; same complete protected plotted rows."""
import argparse
from pathlib import Path
import runpy
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.text import Text

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
old_text=Text.set_text;old_save=Figure.savefig

def set_text(self,text):
    if isinstance(text,str):
        text=text.replace('TU EPYC7302','AMD EPYC7302').replace('LRZ Gold6148','Intel Gold6148')
        text=text.replace('ME64\nbridge','ME64').replace('Fourier64\nbridge','Fourier64')
        text=text.replace('ME64 bridge','ME64').replace('Fourier64 bridge','Fourier64')
        text=text.replace('newly batched standalone','batched standalone')
        text=text.replace('older external rank-pipeline timings','other backend/cohort timing scopes')
    return old_text(self,text)

def save(self,path,*a,**kw):return old_save(self,args.out/Path(path).name,*a,**kw)

Text.set_text=set_text;Figure.savefig=save
runpy.run_path(str(ROOT/'scripts/plot_axis_complete_evidence.py'),run_name='__main__')
