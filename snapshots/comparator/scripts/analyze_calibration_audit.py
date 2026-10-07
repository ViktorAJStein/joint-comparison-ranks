"""Validate complete raw blocks and summarize the predeclared calibration audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from collections import defaultdict
import numpy as np
from scipy.stats import beta, binomtest


CASES = {'gaussian_balanced': (0, 500, 500), 'rounded_balanced': (1, 500, 500),
         'gaussian_200_800': (2, 200, 800), 'gaussian_800_200': (3, 800, 200)}


def validate_row(row, meta):
    """Check provenance against the frozen metadata, not just plausible scores."""
    case_id, n, m = CASES[row['case']]
    assert (int(row['N']), int(row['M'])) == (n, m)
    rep = int(row['rep'])
    key = (rep,) if case_id == 0 else (case_id, rep)
    seed = int(np.random.SeedSequence(meta['arguments']['seed'], spawn_key=key).generate_state(1)[0])
    assert int(row['data_seed']) == seed and int(row['method_seed']) == seed + 17
    assert row['cohort'] == meta['arguments']['cohort']
    method = row['method']
    cfg = meta['configurations'][method if method in meta['configurations'] else 'conservative']
    expected_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert row['config_hash'] == expected_hash
    assert row['orientation'] in ('original', 'rotated')
    if case_id != 0:
        assert row['orientation'] == 'original' and not int(row['alternative'])
    for field in ('data_hash', 'score_input_hash', 'views_hash', 'reference_positions_hash'):
        assert len(row[field]) == 64 and all(c in '0123456789abcdef' for c in row[field])
    if row['orientation'] == 'original' and method in meta['configurations']:
        assert row['score_input_hash'] == row['data_hash']
    assert int(row['effective_L']) == 20


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    meta=json.loads((args.input/'metadata.json').read_text())
    assert meta['status']=='complete' and meta['complete_blocks']==32500
    assert meta['arguments']['reps'] == 8000 and meta['arguments']['power_reps'] == 500
    assert meta['arguments']['rep_start'] == 0
    accounting = json.loads((args.input / 'accounting.json').read_text())
    assert accounting['status'] == 'complete' and accounting['exit_code'] == 0
    assert accounting['gpu_hours'] == 0 and accounting['workers'] == 4
    assert accounting['elapsed_seconds'] <= accounting['wall_cap_seconds'] == 5400
    # Historical wrapper reads time.time() twice; its core-hour record is
    # slightly more conservative than its wall record, not bitwise identical.
    difference_seconds = accounting['reserved_cpu_core_hours']*3600/4 - accounting['elapsed_seconds']
    assert 0 <= difference_seconds < .01
    raw=args.input/'raw.csv'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==meta['raw_sha256']
    root=Path(__file__).resolve().parents[1]
    for name,checksum in meta['source_hashes'].items():
        source=subprocess.check_output(['git','-C',str(root),'show','rank-calibration-20261004-source:'+name])
        assert hashlib.sha256(source).hexdigest()==checksum,(name,'source hash')
    groups=defaultdict(list);blocks=defaultdict(dict);keys=set();identities=[]
    for row in csv.DictReader(raw.open()):
        validate_row(row, meta)
        key=(row['case'],int(row['rep']),int(row['alternative']),row['method'],row['orientation'])
        assert key not in keys;keys.add(key)
        assert row['reuse_certificate']=='', 'No cached/debug-only variant allowed'
        assert int(row['effective_L'])<=20 and int(row['effective_L'])>0
        assert int(row['requested_L'])==64 and int(row['d'])==20
        assert np.isfinite(float(row['stat_obs']))
        decision='p_mid' if row['method'] in ('midp_only','legacy_combined') else 'p_perm'
        assert int(row['reject'])==int(float(row[decision])<=.05)
        assert 0<=float(row['p_perm'])<=1 and 0<=float(row['p_mid'])<=1
        used=int(row['perms_used'])
        if row['method'] not in ('wilson_only','legacy_combined'): assert used==100
        else: assert 50<=used<=100
        g=(row['case'],int(row['alternative']),row['method'],row['orientation'])
        groups[g].append(row)
        block=(row['case'],int(row['rep']),int(row['alternative']))
        blocks[block][(row['method'],row['orientation'])]=row
        if row.get('whitening_identity'):identities.append(int(row['whitening_identity']))
    assert len(keys)==203000 and len(blocks)==32500 and len(groups)==31
    paired=[]
    for (case,rep,alternative),rows in blocks.items():
        expected=6 if alternative else (10 if case=='gaussian_balanced' else 5)
        assert len(rows)==expected
        assert len({r['data_hash'] for r in rows.values()})==1
        assert len({r['data_seed'] for r in rows.values()})==1
        assert len({r['method_seed'] for r in rows.values()})==1
        assert len({r['views_hash'] for r in rows.values()})==1
        assert len({r['reference_positions_hash'] for r in rows.values()})==1
        if not alternative:
            base=rows[('conservative','original')]
            assert all(rows[(method,'original')]['stat_obs']==base['stat_obs']
                       for method in ('midp_only','wilson_only','antithetic_only','legacy_combined'))
            # Verified threshold/control-flow identity, not a cached run.
            assert rows[('wilson_only','original')]['reject']==base['reject']
        if case=='gaussian_balanced':
            for orientation in ('original','rotated'):
                p,w=rows[('canonical_pca',orientation)],rows[('canonical_whiten',orientation)]
                actual=all(p[k]==w[k] for k in ('stat_obs','p_perm','p_mid','reject','perms_used'))
                assert int(p['whitening_identity'])==int(w['whitening_identity'])==actual
                paired.append({'rep':rep,'alternative':alternative,'orientation':orientation,
                               'whitening_identity':int(actual)})
    summary=[]
    for (case,alternative,method,orientation),rows in sorted(groups.items()):
        n=len(rows);expected=500 if alternative else 8000;assert n==expected
        assert {int(r['rep']) for r in rows}==set(range(expected))
        k=sum(int(r['reject']) for r in rows);rate=k/n
        lower=0 if k==0 else beta.ppf(.025,k,n-k+1)
        upper=1 if k==n else beta.ppf(.975,k+1,n-k)
        p_over=binomtest(k,n,.05,alternative='greater').pvalue if not alternative else None
        summary.append({'cohort':rows[0]['cohort'],'case':case,'alternative':alternative,
                        'N':rows[0]['N'],'M':rows[0]['M'],'d':20,'method':method,'orientation':orientation,
                        'R':n,'rejections':k,'rate':rate,'lo95':lower,'hi95':upper,
                        'se':np.sqrt(rate*(1-rate)/n),'p_over':p_over,
                        'p_over_bonf50':min(1,50*p_over) if p_over is not None else None,
                        'mean_B_used':np.mean([int(r['perms_used']) for r in rows]),
                        'min_L':min(int(r['effective_L']) for r in rows),
                        'max_L':max(int(r['effective_L']) for r in rows)})
    args.output.mkdir(parents=True,exist_ok=True)
    output=args.output/'summary.csv'
    with output.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
    with (args.output/'paired_whitening.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(paired[0]),lineterminator='\n');writer.writeheader();writer.writerows(paired)
    audit={'status':'verified','raw_rows':len(keys),'blocks':len(blocks),'summary_rows':len(summary),
           'raw_sha256':meta['raw_sha256'],'summary_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
           'whitening_pairs':len(paired),'whitening_equal_pairs':sum(r['whitening_identity'] for r in paired),
           'family_size':50,'alpha':.05,'source_tag':'rank-calibration-20261004-source',
           'all_variants_actually_executed':True,'pilot_rows_included':0,
           'provenance_rows_verified':len(keys),
           'main_reserved_cpu_core_hours':accounting['reserved_cpu_core_hours'],
           'gpu_hours':0}
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit))
    for row in summary:
        print(f"{row['case']:22} alt={row['alternative']} {row['method']:18} {row['orientation']:8}: {row['rejections']}/{row['R']}={row['rate']:.5f}, CI=({row['lo95']:.5f},{row['hi95']:.5f}), p_adj={row['p_over_bonf50']}")

if __name__=='__main__':main()
