"""Independently verify full raw coverage, seeds, counts and paired inference inputs."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import math

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'snapshots/comparator/src'))
import numpy as np
from scipy.stats import beta, binomtest
from scripts.fair_comparator_bench import seed_for


def read_json(path):return json.loads(path.read_text())


def asset_rows(asset,study):
    with tarfile.open(asset,'r|gz') as tar:
        for member in tar:
            if not member.isfile() or '..' in Path(member.name).parts or Path(member.name).is_absolute():
                raise ValueError('Unsafe/unexpected asset member')
            if not member.name.startswith(study+'/'):raise ValueError('Wrong study asset')
            with tar.extractfile(member) as f:
                if study=='calibration':yield from csv.DictReader(line.decode('utf8') for line in f)
                else:
                    for line in f:yield json.loads(line)


def check(study,rows):
    groups=defaultdict(dict)
    if study=='comparator':p=read_json(ROOT/'snapshots/comparator/experiments/fair_comparators_v1.json')
    elif study!='calibration':
        folder={'axis':'axis_mixture','generator':'generator_comparison','dependence':'dependence_power'}[study]
        p=read_json(ROOT/f'experiments/{folder}_v1/protocol.json')
    def add(group,rep,value):
        if rep in groups[group]:raise ValueError(('Duplicate key',group,rep))
        groups[group][rep]=bool(value)
    count=0
    for row in rows:
        rep=int(row['rep']);count+=1
        if study=='calibration':
            case=row['case'];caseid={'gaussian_balanced':0,'rounded_balanced':1,'gaussian_200_800':2,'gaussian_800_200':3}[case]
            key=(rep,) if caseid==0 else (caseid,rep)
            seed=int(np.random.SeedSequence(6060,spawn_key=key).generate_state(1)[0])
            assert int(row['data_seed'])==seed and int(row['method_seed'])==seed+17
            assert int(row['d'])==20 and int(row['effective_L'])==20 and int(row['requested_L'])==64
            assert row['reuse_certificate']==''
            method=row['method'];decision='p_mid' if method in ('midp_only','legacy_combined') else 'p_perm'
            assert int(row['reject'])==int(float(row[decision])<=.05)
            assert (50<=int(row['perms_used'])<=100) if method in ('wilson_only','legacy_combined') else int(row['perms_used'])==100
            add((case,int(row['alternative']),method,row['orientation']),rep,int(row['reject']))
            continue
        assert row['status']=='OK' and not row['warnings']
        if study=='dependence':
            s=row['stratum'];sid=s['id']
            assert row['data_seed']==int(np.random.SeedSequence([p['base_seed'],sid,rep,1]).generate_state(1)[0])
            assert row['method_seed']==int(np.random.SeedSequence([p['base_seed'],sid,rep,10]).generate_state(1)[0])
            values=row['results'];group=(sid,)
        else:
            case=next(c for c in p['cases'] if c['id']==row['case'])
            assert row['data_seed']==seed_for(p,case,row['d'],rep,1)
            group=(row['case'],row['d'],row.get('orientation','raw'))
            values=row['results'] if study=='generator' else {row['method']:row}
        for method,value in values.items():
            assert math.isfinite(value['stat_obs']) and 0<=value['pvalue']<=1
            assert value['reject']==(value['pvalue']<=p['alpha'])
            b=p['B'] if study=='dependence' else p['evaluation_permutations']
            assert value['perms_used']==(0 if method.startswith('C2ST') else b)
            if 'permutation_statistics' in value:
                scores=value['permutation_statistics'];assert len(scores)==b and np.isfinite(scores).all()
                tail=sum(v>=value['stat_obs'] for v in scores)
                assert value['pvalue']==(1+tail)/(b+1)
            add((*group,method),rep,value['reject'])
    # The published group list defines all retained groups, never a survivor subset.
    refs=[];contrasts=[]
    if study=='calibration':
        for r in csv.DictReader((ROOT/'results/calibration/summary.csv').open()):
            refs.append(((r['case'],int(r['alternative']),r['method'],r['orientation']),int(r['R']),int(r['rejections'])))
    elif study=='comparator':
        a=read_json(ROOT/'plot-inputs/comparator/d50.json')
        for r in a['rates']:
            if r['d']==50 and r['complete']:refs.append(((r['case'],50,r['orientation'],r['method']),4000 if r['case'].startswith('null') else 500,r['rejections']))
        for r in read_json(ROOT/'plot-inputs/comparator/d20.json')['rates']:
            refs.append(((r['case'],20,r['orientation'],r['method']),r['repetitions'],r['rejections']))
        contrasts=[c for c in a['paired_contrasts'] if c['d']==50 and c['complete']]
    elif study=='generator':
        a=read_json(ROOT/'plot-inputs/generator/analysis.json')
        refs=[((r['case'],r['d'],r['orientation'],r['generator']),r['expected_blocks'],r['rejections']) for r in a['summary']]
        contrasts=a['paired_contrasts']
    elif study=='axis':
        a=read_json(ROOT/'plot-inputs/axis/results.json')
        refs=[((r['case'],r['d'],'raw',r['method']),r['n'],r['rejections']) for r in a['summary']]
        contrasts=a['paired_contrasts']
    else:
        a=read_json(ROOT/'plot-inputs/dependence/results.json')
        refs=[((r['stratum']['id'],r['method']),r['repetitions'],r['rejections']) for r in a['rates']]
        contrasts=a['contrasts']
    assert set(groups)=={g for g,n,k in refs},'Missing/extra scientific groups'
    for group,n,k in refs:
        assert set(groups[group])==set(range(n)),('Incomplete repetitions',group)
        assert sum(groups[group].values())==k,('Rejection-count discrepancy',group)
    for c in contrasts:
        if study=='dependence':ga=(c['stratum']['id'],c['method_a']);gb=(c['stratum']['id'],c['method_b'])
        else:
            head=(c['case'],c['d'],c.get('orientation','raw'))
            ga=(*head,c.get('a',c.get('rank',p['analysis'].get('primary_rank'))))
            gb=(*head,c.get('b',c.get('comparator')))
        av,bv=groups[ga],groups[gb];plus=sum(av[i] and not bv[i] for i in av);minus=sum(bv[i] and not av[i] for i in av)
        assert plus==c.get('a_only',c.get('rank_only')) and minus==c.get('b_only',c.get('comparator_only'))
        pv=binomtest(plus,plus+minus,.5).pvalue if plus+minus else 1.
        expected=c.get('exact_mcnemar_pvalue',c.get('exact_pvalue',c.get('pvalue')))
        assert np.isclose(pv,expected,rtol=1e-11,atol=1e-300)
    # Coupled identity check from independent actual rows, not the displayed certificate.
    if study=='calibration':
        for alternative,n in ((0,8000),(1,500)):
            for orientation in ('original','rotated'):
                assert groups['gaussian_balanced',alternative,'canonical_pca',orientation]==groups['gaussian_balanced',alternative,'canonical_whiten',orientation]
    expected_rows=dict(calibration=203000,comparator=288000,generator=36000,axis=99000,dependence=46000)
    assert count==expected_rows[study]
    return dict(study=study,status='pass',raw_rows=count,groups=len(refs),paired_contrasts=len(contrasts))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path)
    ap.add_argument('--rerun-dir',type=Path,nargs='+')
    ap.add_argument('--study',choices=('calibration','comparator','generator','axis','dependence'))
    args=ap.parse_args()
    if args.rerun_dir:
        if not args.study:ap.error('--rerun-dir needs --study')
        def records():
            for directory in args.rerun_dir:
                meta=read_json(directory/'metadata.json');assert meta['status']=='complete'
                raw=directory/('raw.csv' if args.study=='calibration' else 'raw.jsonl')
                assert hashlib.sha256(raw.read_bytes()).hexdigest()==meta['raw_sha256']
                if args.study=='calibration':yield from csv.DictReader(raw.open())
                else:
                    with raw.open() as f:
                        for line in f:yield json.loads(line)
        print(json.dumps(check(args.study,records())))
    else:
        if not args.assets:ap.error('--assets required')
        for item in read_json(ROOT/'release-assets.json'):
            asset=args.assets/item['file'];h=hashlib.sha256()
            with asset.open('rb') as f:
                for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
            assert h.hexdigest()==item['sha256'],'Asset checksum mismatch'
            print(json.dumps(check(item['study'],asset_rows(asset,item['study']))),flush=True)


if __name__=='__main__':main()
